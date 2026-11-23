#!/usr/bin/env python3
"""z0_similarity.py -- how similar are CORRESPONDING patches in z0 (encoder output, before any loop step / global attn)?

z0 = out/tokendump/K16/step_000.npy (S, 5+Np, 768); only patch tokens are used.
GT correspondences: patch centre of frame a, GT depth -> 3D (GT c2w, K) -> project into frame b -> patch of b;
kept if inside b and not occluded (|z_proj / gt_depth_b - 1| < --occ).
Reports (raw tokens, and per-frame-mean-removed tokens):
  cosine of GT-corresponding pairs vs random cross-frame pairs vs same-frame 4-neighbours; ROC-AUC (GT vs random);
  retrieval: is the GT patch the most similar patch of frame b (top-1 / top-5 / within 1 patch), by frame gap.
Figures: cosine histograms, retrieval vs gap, similarity heatmaps of 3 query patches in another frame.

  PYTHONPATH=$(pwd) python shovel/z0_similarity.py --out out/z0sim
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlt_probe import load_gt  # noqa: E402

PM = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="out/tokendump/K16")
    ap.add_argument("--gt", default="out/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--max-gap", type=int, default=6)
    ap.add_argument("--occ", type=float, default=0.05)
    ap.add_argument("--out", default="out/z0sim")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = np.load(f"{a.dump}/meta.npz")
    gh, gw = [int(v) for v in meta["grid"]]
    H, W = gh * 14, gw * 14
    z = torch.from_numpy(np.load(f"{a.dump}/step_000.npy").astype(np.float32))[:, PM:]      # (S,Np,768)
    S, Np, C = z.shape
    gt_d, c2w, K = load_gt(a.gt, max(H, W))
    gt_d, c2w = gt_d[:S].float(), c2w[:S].float()
    assert gt_d.shape[1:] == (H, W), (gt_d.shape, H, W)

    jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
    uc, vc = torch.from_numpy(jj * 14 + 7.0).float().flatten(), torch.from_numpy(ii * 14 + 7.0).float().flatten()
    Kinv = torch.linalg.inv(K)

    def corr(fa, fb):
        d = gt_d[fa, vc.long(), uc.long()]
        ok = d > 1e-6
        rays = torch.stack([uc, vc, torch.ones_like(uc)], -1) @ Kinv.T
        Xc = rays * d[:, None]
        Xw = Xc @ c2w[fa, :3, :3].T + c2w[fa, :3, 3]
        w2c = torch.linalg.inv(c2w[fb])
        Xb = Xw @ w2c[:3, :3].T + w2c[:3, 3]
        zb = Xb[:, 2]
        p = Xb @ K.T
        u, v = p[:, 0] / zb.clamp_min(1e-6), p[:, 1] / zb.clamp_min(1e-6)
        inb = ok & (zb > 1e-3) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        ui, vi = u.clamp(0, W - 1).long(), v.clamp(0, H - 1).long()
        db = gt_d[fb, vi, ui]
        vis = inb & (db > 1e-6) & ((zb / db.clamp_min(1e-6) - 1).abs() < a.occ)
        idx = (vi // 14).clamp(max=gh - 1) * gw + (ui // 14).clamp(max=gw - 1)
        return vis, idx, (u / 14 - 0.5), (v / 14 - 0.5)

    variants = {"raw": z, "mean-removed": z - z.mean(1, keepdim=True)}
    rng = np.random.default_rng(0)
    lines = [f"z0 = encoder output (before any loop step), {S} frames x {Np} patches; GT correspondences, occlusion tol {a.occ}"]
    figdata = {}
    for vn, zz in variants.items():
        zn = zz / zz.norm(dim=-1, keepdim=True)
        cos_gt, cos_rand, cos_nb = [], [], []
        ret = {g: [0, 0, 0, 0] for g in range(1, a.max_gap + 1)}          # top1, top5, within-1-patch, count
        for fa in range(S):
            # same-frame right/down neighbours
            g2 = zn[fa].reshape(gh, gw, C)
            cos_nb += (g2[:, :-1] * g2[:, 1:]).sum(-1).flatten().tolist() + (g2[:-1] * g2[1:]).sum(-1).flatten().tolist()
            for gap in range(1, a.max_gap + 1):
                fb = fa + gap
                if fb >= S:
                    continue
                vis, idx, uf, vf = corr(fa, fb)
                if vis.sum() == 0:
                    continue
                sim = zn[fa] @ zn[fb].T                                     # (Np,Np)
                q = torch.nonzero(vis).flatten()
                cos_gt += sim[q, idx[q]].tolist()
                cos_rand += sim[q, torch.from_numpy(rng.integers(0, Np, len(q)))].tolist()
                top = sim[q].topk(5, dim=1).indices
                ret[gap][0] += int((top[:, 0] == idx[q]).sum())
                ret[gap][1] += int((top == idx[q][:, None]).any(1).sum())
                br, bc = top[:, 0] // gw, top[:, 0] % gw
                ret[gap][2] += int((((br - idx[q] // gw).abs() <= 1) & ((bc - idx[q] % gw).abs() <= 1)).sum())
                ret[gap][3] += len(q)
                if vn == "mean-removed" and fa == 4 and gap == 2:
                    figdata.update(sim=sim, q=q, idx=idx, fa=fa, fb=fb, uf=uf, vf=vf)
        cg, cr, cn = np.array(cos_gt), np.array(cos_rand), np.array(cos_nb)
        allv = np.concatenate([cg, cr]); lab = np.concatenate([np.ones_like(cg), np.zeros_like(cr)])
        order = np.argsort(allv); ranks = np.empty_like(order); ranks[order] = np.arange(len(allv))
        auc = (ranks[lab == 1].sum() - len(cg) * (len(cg) - 1) / 2) / (len(cg) * len(cr))
        lines.append(f"\n[{vn}] cosine: GT-corresponding {cg.mean():.3f} ± {cg.std():.3f} (n={len(cg)}) | random cross-frame "
                     f"{cr.mean():.3f} ± {cr.std():.3f} | same-frame neighbours {cn.mean():.3f} ± {cn.std():.3f} | AUC(GT vs random) {auc:.3f}")
        lines.append("  retrieval of the GT patch among all patches of frame b:  gap: top-1 / top-5 / best match within 1 patch")
        for g, (t1, t5, w1, n) in ret.items():
            if n:
                lines.append(f"    gap {g}: {t1 / n:.3f} / {t5 / n:.3f} / {w1 / n:.3f}   (n={n})")
        figdata.setdefault("hist", {})[vn] = (cg, cr, cn)
        figdata.setdefault("ret", {})[vn] = ret
    print("\n".join(lines))
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for c, vn in enumerate(variants):
        cg, cr, cn = figdata["hist"][vn]
        b = np.linspace(min(cr.min(), cg.min()), 1, 80)
        ax[c].hist(cr, b, alpha=.5, density=True, label="random cross-frame")
        ax[c].hist(cg, b, alpha=.5, density=True, label="GT corresponding")
        ax[c].hist(cn, b, alpha=.3, density=True, label="same-frame neighbour")
        ax[c].set_title(f"z0 cosine ({vn})"); ax[c].legend(fontsize=8)
    for vn in variants:
        ret = figdata["ret"][vn]
        gs = [g for g in ret if ret[g][3]]
        ax[2].plot(gs, [ret[g][0] / ret[g][3] for g in gs], "-o", label=f"top-1 {vn}")
        ax[2].plot(gs, [ret[g][2] / ret[g][3] for g in gs], "--o", label=f"within 1 patch {vn}")
    ax[2].set_xlabel("frame gap"); ax[2].set_title("retrieval of the GT patch"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{a.out}/z0_cosine.png", dpi=130); plt.close(fig)

    if figdata.get("sim") is not None:
        rgb = meta["rgb"]                                                    # (S,Np,3) per-patch mean colour
        sim, q, idx, fa, fb = figdata["sim"], figdata["q"], figdata["idx"], figdata["fa"], figdata["fb"]
        picks = q[np.linspace(0, len(q) - 1, 3).astype(int)]
        fig, ax = plt.subplots(3, 3, figsize=(15, 9))
        for r, p in enumerate(picks.tolist()):
            ax[r, 0].imshow(rgb[fa].reshape(gh, gw, 3).clip(0, 1)); ax[r, 0].plot(p % gw, p // gw, "r*", ms=14)
            ax[r, 0].set_title(f"frame {fa}: query patch")
            ax[r, 1].imshow(rgb[fb].reshape(gh, gw, 3).clip(0, 1))
            ax[r, 1].plot(figdata["uf"][p], figdata["vf"][p], "g+", ms=16, mew=2); ax[r, 1].set_title(f"frame {fb}: GT location (+)")
            im = ax[r, 2].imshow(sim[p].reshape(gh, gw).numpy(), cmap="magma")
            ax[r, 2].plot(figdata["uf"][p], figdata["vf"][p], "c+", ms=16, mew=2)
            b = int(sim[p].argmax()); ax[r, 2].plot(b % gw, b // gw, "wx", ms=10, mew=2)
            ax[r, 2].set_title("cosine to every patch of frame b (+ GT, x best)"); plt.colorbar(im, ax=ax[r, 2], fraction=.03)
        fig.tight_layout(); fig.savefig(f"{a.out}/z0_heatmaps.png", dpi=110); plt.close(fig)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
