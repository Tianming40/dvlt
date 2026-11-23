#!/usr/bin/env python3
"""invariant_part.py -- does the collapse destroy the VIEW-INVARIANT ("which 3D point") part of the tokens?

Stored states out/tokendump/K{16,64}/step_XXX.npy (state after s steps; s=0 is z0). Patch tokens only.
Tracks (classes) from GT: anchor frames every --anchor-every frames, anchor patches on a stride-2 grid, projected with GT
depth/pose/K into frames within +-max_gap (occlusion-checked) -> the patches of one 3D point in several views.
Per state:
  invariant subspace = LDA on the tracks: S_b v = lam (S_w + eps I) v, top --dim directions, fitted on half of the tracks;
  everything below is measured on the OTHER half (held-out 3D points):
    fisher     tr(W'S_b W) / tr(W'S_w W)                      (are different 3D points still separated across views?)
    retrieval  for held-out anchors: nearest patch of a neighbouring frame (cosine in the subspace) == GT patch?
               (top-1 and within +-1 patch), averaged over gaps 1..max_gap
    cka_z0     linear CKA between this state's and z0's representation of the SAME held-out 3D points (track means,
               each in its own invariant subspace -> basis-free comparison of the point-to-point similarity structure)
    keff_inv   k_eff (dvlt_probe.cluster_stats) of 1024 random patches projected into the subspace
  full-space references: retrieval with mean-removed full tokens, k_eff of full tokens, CKA of full-token track means
  accuracy   per-patch decoded log-depth (targets_XXX.npz) vs GT -> AbsRel (median scale)

  PYTHONPATH=$(pwd) python shovel/invariant_part.py --out out/invariant
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlt_probe import load_gt, cluster_stats  # noqa: E402

PM = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="out/tokendump")
    ap.add_argument("--gt", default="out/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--Ks", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--anchor-every", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--occ", type=float, default=0.05)
    ap.add_argument("--out", default="out/invariant")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = np.load(f"{a.dump}/K16/meta.npz")
    gh, gw = [int(v) for v in meta["grid"]]
    H, W = gh * 14, gw * 14
    S = int(meta["S"])
    Np = gh * gw
    gt_d, c2w, K = load_gt(a.gt, max(H, W))
    gt_d, c2w = gt_d[:S].float(), c2w[:S].float()
    gt_ld = torch.from_numpy(meta["gt_logdepth"]).float()                       # (S,Np) nan = invalid

    # ---------------- tracks from GT
    jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
    uc, vc = torch.from_numpy(jj * 14 + 7.0).float().flatten(), torch.from_numpy(ii * 14 + 7.0).float().flatten()
    Kinv = torch.linalg.inv(K)

    def project(fa, pidx, fb):
        u, v = uc[pidx], vc[pidx]
        d = gt_d[fa, v.long(), u.long()]
        Xc = (torch.stack([u, v, torch.ones_like(u)], -1) @ Kinv.T) * d[:, None]
        Xw = Xc @ c2w[fa, :3, :3].T + c2w[fa, :3, 3]
        w2c = torch.linalg.inv(c2w[fb])
        Xb = Xw @ w2c[:3, :3].T + w2c[:3, 3]
        zb = Xb[:, 2]; p = Xb @ K.T
        ub, vb = p[:, 0] / zb.clamp_min(1e-6), p[:, 1] / zb.clamp_min(1e-6)
        ok = (d > 1e-6) & (zb > 1e-3) & (ub >= 0) & (ub <= W - 1) & (vb >= 0) & (vb <= H - 1)
        db = gt_d[fb, vb.clamp(0, H - 1).long(), ub.clamp(0, W - 1).long()]
        ok = ok & (db > 1e-6) & ((zb / db.clamp_min(1e-6) - 1).abs() < a.occ)
        return ok, (vb.clamp(0, H - 1).long() // 14).clamp(max=gh - 1) * gw + (ub.clamp(0, W - 1).long() // 14).clamp(max=gw - 1)

    anchors = (ii % 2 == 0) & (jj % 2 == 0)
    apat = torch.from_numpy(np.nonzero(anchors.flatten())[0])
    tracks = []                                                                  # list of [(frame, patch), ...]
    for fa in range(0, S, a.anchor_every):
        mem = {int(p): [(fa, int(p))] for p in apat}
        for g in range(-a.max_gap, a.max_gap + 1):
            fb = fa + g
            if g == 0 or not 0 <= fb < S:
                continue
            ok, idx = project(fa, apat, fb)
            for p, o, q in zip(apat.tolist(), ok.tolist(), idx.tolist()):
                if o:
                    mem[p].append((fb, q))
        tracks += [m for m in mem.values() if len(m) >= 3]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tracks))
    tr_ids, te_ids = perm[: len(perm) // 2], perm[len(perm) // 2:]
    print(f"{len(tracks)} tracks (>=3 views), {sum(len(t) for t in tracks)} memberships; fit on {len(tr_ids)}, test on {len(te_ids)}", flush=True)
    rand_idx = torch.from_numpy(rng.choice(S * Np, 1024, replace=False))

    def gather(Z, ids):
        X, y = [], []
        for c, t in enumerate(ids):
            for f, p in tracks[t]:
                X.append((f, p)); y.append(c)
        f = torch.tensor([x[0] for x in X]); p = torch.tensor([x[1] for x in X])
        return Z[f, p], torch.tensor(y)

    def scatter(X, y, n):
        mu = X.mean(0)
        M = torch.zeros(n, X.shape[1], device=X.device).index_add_(0, y.to(X.device), X)
        cnt = torch.bincount(y, minlength=n).float().to(X.device)[:, None]
        M = M / cnt
        Xc = X - M[y.to(X.device)]
        Sw = Xc.T @ Xc
        Mc = (M - mu) * cnt.sqrt()
        Sb = Mc.T @ Mc
        return Sw, Sb, M

    def cka(A, B):
        A = A - A.mean(0); B = B - B.mean(0)
        return float((A.T @ B).norm() ** 2 / ((A.T @ A).norm() * (B.T @ B).norm()))

    def keff(X):
        return float(cluster_stats(torch.nn.functional.normalize(X, dim=-1).float(), [0.9])[0])

    def absrel(ld):
        m = ~torch.isnan(gt_ld)
        p, g = ld[m].exp(), gt_ld[m].exp()
        s = (g / p).median()
        return float(((p * s - g).abs() / g).mean())

    res = {}
    for Kn in a.Ks:
        rows, ref = [], {}
        for s in range(Kn + 1):
            Z = torch.from_numpy(np.load(f"{a.dump}/K{Kn}/step_{s:03d}.npy").astype(np.float32)[:, PM:]).to(dev)   # (S,Np,C)
            ld = torch.from_numpy(np.load(f"{a.dump}/K{Kn}/targets_{s:03d}.npz")["logdepth"]).float()
            Xtr, ytr = gather(Z, tr_ids)
            Sw, Sb, _ = scatter(Xtr, ytr, len(tr_ids))
            Sw = Sw + 1e-3 * Sw.trace() / Sw.shape[0] * torch.eye(Sw.shape[0], device=dev)
            L = torch.linalg.cholesky(Sw.double())
            Li = torch.linalg.inv(L)
            ev, V = torch.linalg.eigh(Li @ Sb.double() @ Li.T)
            Wm = (Li.T @ V[:, -a.dim:]).float()                                   # (C,dim) LDA directions
            Xte, yte = gather(Z, te_ids)
            Sw_te, Sb_te, M_te = scatter(Xte @ Wm, yte, len(te_ids))
            fisher = float(Sb_te.trace() / Sw_te.trace())
            _, _, Mfull = scatter(Xte - Z.mean(1).mean(0), yte, len(te_ids))
            # retrieval on held-out anchors
            hits = np.zeros(3); n = 0
            Zp = Z @ Wm
            Zm = Z - Z.mean(1, keepdim=True)
            for t in te_ids:
                fa, pa = tracks[t][0]
                for fb, pb in tracks[t][1:]:
                    for k_, ZZ in enumerate([Zp, Zm]):
                        q = torch.nn.functional.normalize(ZZ[fa, pa], dim=0)
                        sim = torch.nn.functional.normalize(ZZ[fb], dim=-1) @ q
                        b = int(sim.argmax())
                        if k_ == 0:
                            hits[0] += b == pb
                            hits[1] += abs(b // gw - pb // gw) <= 1 and abs(b % gw - pb % gw) <= 1
                        else:
                            hits[2] += b == pb
                    n += 1
            row = dict(step=s, fisher=fisher, ret_top1_inv=hits[0] / n, ret_win1_inv=hits[1] / n, ret_top1_full=hits[2] / n,
                       keff_inv=keff((Z.reshape(-1, Z.shape[-1])[rand_idx.to(dev)] - Z.mean((0, 1))) @ Wm),
                       keff_full=keff(Z.reshape(-1, Z.shape[-1])[rand_idx.to(dev)] - Z.mean((0, 1))),
                       absrel=absrel(ld), lda_top=float(ev[-1]))
            if s == 0:
                ref = dict(inv=M_te.clone(), full=Mfull.clone())
            row["cka_z0_inv"] = cka(M_te, ref["inv"])
            row["cka_z0_full"] = cka(Mfull, ref["full"])
            rows.append(row)
            if s in (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64):
                print(f"K{Kn} step {s:>2}: AbsRel {row['absrel']:.3f} | fisher {fisher:.3f} | retrieval inv top1 {row['ret_top1_inv']:.3f}"
                      f" (±1 {row['ret_win1_inv']:.3f}) full top1 {row['ret_top1_full']:.3f} | CKA(z0) inv {row['cka_z0_inv']:.3f}"
                      f" full {row['cka_z0_full']:.3f} | k_eff inv {row['keff_inv']:.2f} full {row['keff_full']:.2f}", flush=True)
        res[f"K{Kn}"] = rows
    json.dump(res, open(f"{a.out}/invariant.json", "w"), indent=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = [("absrel", "AbsRel (decoded depth)"), ("fisher", "Fisher ratio (invariant subspace)"),
            ("ret_top1_inv", "retrieval top-1 (inv) / dashed: full"), ("cka_z0_inv", "CKA to z0 (inv) / dashed: full"),
            ("keff_inv", "k_eff (inv) / dashed: full")]
    dash = {"ret_top1_inv": "ret_top1_full", "cka_z0_inv": "cka_z0_full", "keff_inv": "keff_full"}
    fig, ax = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    for c, (k, t) in enumerate(keys):
        for j, (Kn, rows) in enumerate(res.items()):
            x = [r["step"] / int(Kn[1:]) for r in rows]
            ax[c].plot(x, [r[k] for r in rows], "-", color=f"C{j}", label=Kn)
            if k in dash:
                ax[c].plot(x, [r[dash[k]] for r in rows], "--", color=f"C{j}", alpha=.6)
        ax[c].set_title(t, fontsize=9); ax[c].set_xlabel("progress  step / K"); ax[c].grid(alpha=.3)
    ax[0].set_yscale("log"); ax[0].legend()
    fig.tight_layout(); fig.savefig(f"{a.out}/invariant.png", dpi=130); plt.close(fig)
    lines = []
    for Kn, rows in res.items():
        for r in rows:
            if r["step"] in (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64):
                lines.append(f"{Kn} step {r['step']:>2}: AbsRel {r['absrel']:.3f} fisher {r['fisher']:.3f} ret_inv {r['ret_top1_inv']:.3f}"
                             f" (±1 {r['ret_win1_inv']:.3f}) ret_full {r['ret_top1_full']:.3f} CKA_inv {r['cka_z0_inv']:.3f}"
                             f" CKA_full {r['cka_z0_full']:.3f} keff_inv {r['keff_inv']:.2f} keff_full {r['keff_full']:.2f}")
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
