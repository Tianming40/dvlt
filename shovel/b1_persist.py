#!/usr/bin/env python3
"""b1_persist.py -- B1: does a token edit SURVIVE further loop steps?  (frozen DVLT, stored K16 tokens, fp32)

At stored state z_s (s in --steps) edit the tokens of --frames (others untouched) so that the DECODED geometry changes by
a target dy while all other decoded outputs stay put; then continue the ORIGINAL K16 schedule for n = 0,1,2,4 steps and
compare against the unedited run from the same z_s (same fp32 pipeline).

Targets (per edited frame):
  region : log-depth +alpha on the central 6x6 patches
  frame  : log-depth +alpha on all patches
  camera : camera centre (ray origin) + 0.1*scene_scale along x; directions and depth unchanged
Write methods:
  jac_euc : dz from 0 by L-BFGS on |D(z+dz) - y*|^2           (linear regime -> Euclidean min-norm  J^T(JJ^T)^-1 dy)
  jac_man : same in w with dz = w Sigma^1/2 (token covariance) (-> Mahalanobis min-norm  Sigma J^T(J Sigma J^T)^-1 dy)
  cov     : Cov(z,y)Var(y)^-1 dy on the target patches (token_rw E2/E4)
  learnedG: (camera only) shared rank-32 linear map G trained on the OTHER frames of z_s (latent_dyn A4 recipe), z += G z
Metrics after n steps (standardized outputs, edited frames):
  retained  = <dY_n, dy> / |dy|^2 on target outputs   (1 kept, 0 erased, >1 amplified)
  leak_self = rms(dY_n on non-target outputs) / rms(dy on target)
  leak_other= rms(dY_n on unedited frames)    / rms(dy on target)
  latent    = |z_edit,n - z_base,n| / |dz|  (edited frames)

  PYTHONPATH=$(pwd) python shovel/b1_persist.py --steps 4 8 12 --frames 3 12 20 --out out/stage3_latent/b1persist
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvlt_probe import Probe  # noqa: E402
from token_rw import Dec, PM, fit_ridge, cov_write_mat  # noqa: E402
from latent_dyn import Loop, load_state, DEV  # noqa: E402

NS = [0, 1, 2, 4]
OTHER = [0, 7, 16, 25]


def outs(dec, z, frames, sd, grad=False):
    """decode frames (chunks of 4) -> standardized (n, Np, 7): [logdepth, dir(3), origin(3)]"""
    parts = []
    for i in range(0, len(frames), 4):
        f = frames[i:i + 4]
        o = dec(z[i:i + 4] if z.shape[0] == len(frames) else z[f], f, grad=grad)
        parts.append(torch.cat([o["logdepth"][..., None] / sd[0], o["dir"] / sd[1], o["origin"] / sd[2]], -1))
    return torch.cat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="out/stage3_latent/tokendump")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--frames", type=int, nargs="+", default=[3, 12, 20])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--cam", type=float, default=0.1, help="camera shift in units of scene scale")
    ap.add_argument("--lbfgs-iters", type=int, default=60)
    ap.add_argument("--g-iters", type=int, default=300)
    ap.add_argument("--out", default="out/stage3_latent/b1persist")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    P = Probe("nvidia/dvlt", max_frames=27)
    K = a.K
    L = Loop(P, a.dump, K)
    meta = np.load(f"{a.dump}/K{K}/meta.npz"); dec = Dec(P, meta)
    gh, gw = dec.gh, dec.gw
    Np = gh * gw
    cy, cx = gh // 2, gw // 2
    region = [(cy - 3 + i) * gw + (cx - 3 + j) for i in range(6) for j in range(6)]
    fr = a.frames
    others = [f for f in OTHER if f not in fr]
    log, res = [], {}

    def say(s=""):
        print(s, flush=True); log.append(s)

    for s in a.steps:
        n_max = min(max(NS), K - s)
        z = load_state(a.dump, K, s)
        S = z.shape[0]
        tg = np.load(f"{a.dump}/K{K}/targets_{s:03d}.npz")
        sd = torch.tensor([tg["logdepth"].std(), tg["dir"].std(), tg["origin"].std()], device=DEV)
        c_all = torch.from_numpy(tg["origin"].mean(1)).to(DEV)
        scale = float((c_all - c_all.mean(0)).norm(dim=-1).mean())
        say(f"\n################ K{K} edit at step {s} (frames {fr}), scene scale {scale:.3f} ################")

        # ---- baseline trajectory from z_s (fp32), decoded at n in NS
        def run(z0):
            zs, x = {0: z0}, z0
            for i in range(1, n_max + 1):
                x = L.F(x, s + i - 1)
                if i in NS:
                    zs[i] = x
            return zs
        t0 = time.time()
        base = run(z)
        Ybase = {n: (outs(dec, base[n], fr, sd), outs(dec, base[n], others, sd)) for n in base}
        say(f"baseline run {time.time() - t0:.0f}s")
        Y0 = Ybase[0][0]                                                    # (3,Np,7)

        # ---- targets: full desired output change dy (standardized), mask of target outputs
        targets = {}
        m = torch.zeros(len(fr), Np, 7, dtype=torch.bool, device=DEV)
        dy = torch.zeros(len(fr), Np, 7, device=DEV)
        dy_r, m_r = dy.clone(), m.clone(); dy_r[:, region, 0] = a.alpha / sd[0]; m_r[:, region, 0] = True
        targets["region"] = (dy_r, m_r)
        dy_f, m_f = dy.clone(), m.clone(); dy_f[:, :, 0] = a.alpha / sd[0]; m_f[:, :, 0] = True
        targets["frame"] = (dy_f, m_f)
        dy_c, m_c = dy.clone(), m.clone(); dy_c[:, :, 4] = a.cam * scale / sd[2]; m_c[:, :, 4:7] = True
        targets["camera"] = (dy_c, m_c)

        # ---- ingredients for cov / manifold writes
        Xp = z[:, PM:].reshape(-1, 768).cpu().numpy()
        Xc = Xp - Xp.mean(0)
        Sig = torch.from_numpy(Xc.T @ Xc / len(Xc)).float().to(DEV)
        Sig = Sig + 1e-3 * Sig.trace() / 768 * torch.eye(768, device=DEV)
        ev, U = torch.linalg.eigh(Sig)
        Shalf = (U * ev.clamp_min(0).sqrt()) @ U.T
        Shalf = Shalf / Shalf.diagonal().mean()                             # keep w on a similar scale as dz
        ld = tg["logdepth"].reshape(-1, 1); org = tg["origin"].reshape(-1, 3)
        Wd, _ = fit_ridge(Xp, ld); CWd = cov_write_mat(Xp, ld)
        CWo = cov_write_mat(Xp, org)
        wd = torch.from_numpy(Wd[0]).float().to(DEV); wc = torch.from_numpy(CWd[0]).float().to(DEV)

        def solve(dyt, metric):
            Yt = Y0 + dyt
            w = torch.zeros(len(fr), z.shape[1], 768, device=DEV, requires_grad=True)
            opt = torch.optim.LBFGS([w], lr=1, max_iter=a.lbfgs_iters, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-12, tolerance_change=1e-14)
            def closure():
                opt.zero_grad()
                dz = w if metric == "euc" else w @ Shalf
                loss = (outs(dec, z[fr] + dz, fr, sd, grad=True) - Yt).pow(2).sum()
                loss.backward()
                return loss
            opt.step(closure)
            with torch.no_grad():
                return (w if metric == "euc" else w @ Shalf).detach()

        def learned_G(dyt):
            trn = [f for f in range(S) if f not in fr]
            g = torch.Generator(device=DEV).manual_seed(11)
            Ug = (1e-3 * torch.randn(768, 32, generator=g, device=DEV)).requires_grad_(True)
            Vg = (torch.randn(768, 32, generator=g, device=DEV) / 768 ** 0.5 / 25).requires_grad_(True)
            opt = torch.optim.Adam([Ug, Vg], lr=3e-3)
            rng = np.random.default_rng(0)
            Yall = {f: outs(dec, z[[f]], [f], sd) for f in trn}
            for _ in range(a.g_iters):
                b = sorted(rng.choice(trn, 4, replace=False).tolist())
                zb = z[b]
                Yp = outs(dec, zb + (zb @ Vg) @ Ug.T, b, sd, grad=True)
                Yt = torch.cat([Yall[f] for f in b]) + dyt[:1].expand(len(b), -1, -1)
                loss = (Yp - Yt).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                zf = z[fr]
                return ((zf @ Vg) @ Ug.T).detach()

        R = res.setdefault(f"s{s}", {})
        for tname, (dyt, mt) in targets.items():
            methods = ["jac_euc", "jac_man", "cov"] + (["learnedG"] if tname == "camera" else [])
            for meth in methods:
                t0 = time.time()
                if meth == "jac_euc":
                    dz = solve(dyt, "euc")
                elif meth == "jac_man":
                    dz = solve(dyt, "man")
                elif meth == "cov":
                    dz = torch.zeros(len(fr), z.shape[1], 768, device=DEV)
                    if tname == "camera":
                        dc = np.array([a.cam * scale, 0.0, 0.0])
                        dz[:, PM:] = torch.from_numpy(dc @ CWo).float().to(DEV)
                    else:
                        idx = region if tname == "region" else list(range(Np))
                        dz[:, PM + np.array(idx)] = a.alpha * wc / (wd @ wc)
                else:
                    dz = learned_G(dyt)
                ze = z.clone(); ze[fr] = z[fr] + dz
                ed = run(ze)
                rows = []
                for n in sorted(ed):
                    Ye, Yo = outs(dec, ed[n], fr, sd), outs(dec, ed[n], others, sd)
                    dY, dYo = Ye - Ybase[n][0], Yo - Ybase[n][1]
                    tgt_rms = dyt[mt].pow(2).mean().sqrt()
                    row = dict(n=n,
                               retained=float((dY[mt] * dyt[mt]).sum() / dyt[mt].pow(2).sum()),
                               leak_self=float(dY[~mt].pow(2).mean().sqrt() / tgt_rms),
                               leak_other=float(dYo.pow(2).mean().sqrt() / tgt_rms),
                               latent=float((ed[n][fr] - base[n][fr]).norm() / dz.norm()))
                    rows.append(row)
                R.setdefault(tname, {})[meth] = dict(dz_norm=float(dz.norm()), dz_rel=float(dz.norm() / z[fr].norm()),
                                                    rows=rows)
                say(f"[{tname:>6}] {meth:>8} |dz|/|z_f| {dz.norm() / z[fr].norm():.4f} ({time.time() - t0:.0f}s): " +
                    " | ".join(f"n={r['n']}: kept {r['retained']:+.2f} leak {r['leak_self']:.2f} other {r['leak_other']:.3f}"
                               f" lat {r['latent']:.2f}" for r in rows))
        json.dump(res, open(f"{a.out}/b1persist.json", "w"), indent=1)

    with open(f"{a.out}/report.txt", "w") as fh:
        fh.write(" ".join(sys.argv) + "\n" + "\n".join(log) + "\n")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        tn = ["region", "frame", "camera"]
        fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
        for j, t in enumerate(tn):
            for si, (sk, R) in enumerate(res.items()):
                for mi, (meth, d) in enumerate(R[t].items()):
                    ns = [r["n"] for r in d["rows"]]
                    ls = ["-", "--", ":"][si % 3]
                    ax[0, j].plot(ns, [r["retained"] for r in d["rows"]], ls, marker="o", ms=3, color=f"C{mi}",
                                  label=f"{meth} {sk}")
                    ax[1, j].plot(ns, [r["leak_self"] for r in d["rows"]], ls, marker="o", ms=3, color=f"C{mi}")
            ax[0, j].set_title(f"{t}: retained fraction"); ax[0, j].axhline(1, color="gray", ls=":"); ax[0, j].axhline(0, color="k", lw=.6)
            ax[1, j].set_title(f"{t}: leakage on non-target outputs"); ax[1, j].set_xlabel("loop steps after the edit")
            for x in ax[:, j]: x.grid(alpha=.3)
        ax[0, 2].legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(f"{a.out}/b1persist.png", dpi=130); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("plot failed:", e)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
