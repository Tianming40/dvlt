#!/usr/bin/env python3
"""b4_oracle.py -- B4: per-step ORACLE (GT) guidance on the frozen DVLT loop, FIXED step size (true fixed point).

Every loop step (from --i-start on):
    z <- F(z)                    (one DVLT step; base "sched" = DVLT time schedule, base "BR" = fixed gates B + RMS norm)
    z <- z - lam * P * grad_z E(z)
E = E_depth/E_depth0 + w_cam * (E_centre/E_centre0 + E_dir/E_dir0)      (each term normalised by its value at i-start)
  E_depth : scale-invariant L1 depth error (median scale detached)         (= stage-2 gtdepth_residual, / mean GT depth)
  E_centre: camera-centre shape error after removing translation+scale, GT expressed in frame-0 camera coordinates
            (stage-2 gtcam_residual compared against WORLD-frame GT, i.e. without the rotation between the two frames)
  E_dir   : 1 - cos(pred ray dir, GT ray dir); GT expressed in frame-0 camera coordinates (DVLT's world)   [new]
Gradient goes ONLY to the current state's tokens of each frame (decoder is per frame), never through F, never to weights;
all alignment quantities (median scale, centre mean, centre scale) are computed in a no-grad pass and held constant.
P: "euc" = identity; "man" = token covariance Sigma of the current state (per token, mean diag 1).
lam is FIXED: calibrated once at i-start so |lam P g| / |z| = rho, then kept (unlike stage-2's normalised step, the force
vanishes when the residual vanishes).
--partial: residual only on EVEN frames; metrics also reported on ODD frames (does a correction spread to unguided views?).

Runs (default): sched K16 base (reference), sched K64 base + euc (reproduce stage-2 rescue), BR K128 base,
                BR K128 x {euc, man} x rho {1e-3, 3e-3, 1e-2}, BR K128 partial (best of the previous six).
  PYTHONPATH=$(pwd) python shovel/b4_oracle.py --out out/b4oracle
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
from dvlt_probe import Probe, load_gt, depth_vs_gt, sim3_ate  # noqa: E402
from gate_fix import FixedGates, rms_normalize  # noqa: E402

PM = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--gt", default="out/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--i-start", type=int, default=3)
    ap.add_argument("--w-cam", type=float, default=0.1)
    ap.add_argument("--rhos", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
    ap.add_argument("--K-br", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None, help="run only these run names (debug)")
    ap.add_argument("--out", default="out/b4oracle")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    P = Probe("nvidia/dvlt", max_frames=a.max_frames)
    images = P.load(a.input)
    with torch.no_grad():
        x0, rope, dims = P.init_state(images)
    x0 = x0.float()
    B, S, H, W = dims
    dev = x0.device
    gt_d, gt_c2w, gt_K = load_gt(a.gt, P.img_size, device=dev)
    gt_d, gt_c2w = gt_d[:S].float(), gt_c2w[:S].float()
    # GT ray directions in frame-0 camera coordinates (DVLT's world frame)
    rel = torch.linalg.inv(gt_c2w[0])[None] @ gt_c2w                     # (S,4,4)
    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    pix = torch.stack([u, v, torch.ones_like(u)], -1).reshape(-1, 3)
    cam_dirs = pix @ torch.linalg.inv(gt_K).T
    gt_dir = torch.einsum("sij,nj->sni", rel[:, :3, :3], cam_dirs)
    gt_dir = (gt_dir / gt_dir.norm(dim=-1, keepdim=True)).reshape(S, H, W, 3)
    gt_c = gt_c2w[:, :3, 3]                                             # world frame (only for the Sim3 ATE metric)
    gt_c_rel = rel[:, :3, 3]                                            # frame-0 camera coords = DVLT's world frame
    gc = gt_c_rel - gt_c_rel.mean(0)
    even, odd = list(range(0, S, 2)), list(range(1, S, 2))
    fg = FixedGates(P.m, mode="B")

    def F(z, i, K, base):
        ts = torch.linspace(0.0, 1.0, K).tolist()
        t_now = ts[i] if base == "sched" else 0.5
        t_next = (ts[i + 1] if i + 1 < K else 1.0) if base == "sched" else 0.5
        with torch.no_grad():
            z = P.m._interval_step(z, t_now, t_next, rope, B, S).float()
            if base == "BR":
                z = rms_normalize(z)
        return z

    def decode(z, frames):
        pr = P.m._decode(z[frames], H, W, 1, len(frames), rope[frames])
        return pr["depth"][0, ..., 0].float(), pr["rays"][0].float()      # (n,H,W), (n,H,W,6)

    @torch.no_grad()
    def evaluate(z):
        ds, rs = [], []
        for i in range(0, S, 4):
            d, r = decode(z, list(range(i, min(i + 4, S))))
            ds.append(d); rs.append(r)
        d, r = torch.cat(ds), torch.cat(rs)
        m = depth_vs_gt(d, gt_d)
        cen = r[..., 3:6].mean((1, 2))
        dirp = r[..., :3] / r[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-9)
        ang = torch.rad2deg(torch.arccos((dirp * gt_dir).sum(-1).clamp(-1, 1))).mean((1, 2))
        mk = gt_d > 1e-6
        sc = (gt_d[mk].median() / d[mk].median().clamp_min(1e-6))
        oc = cen - cen.mean(0); s_c = gc.norm() / oc.norm().clamp_min(1e-6)
        out = dict(absrel=m["absrel"], d1=m["d1"], ate=sim3_ate(cen, gt_c),
                   absrel_even=depth_vs_gt(d[even], gt_d[even])["absrel"], absrel_odd=depth_vs_gt(d[odd], gt_d[odd])["absrel"],
                   ate_odd=sim3_ate(cen[odd], gt_c[odd]), dir_deg=float(ang.mean()), dir_deg_odd=float(ang[odd].mean()),
                   tok_norm=float(z[:, PM:].norm(dim=-1).mean()))
        const = dict(sc=sc, cmean=cen.mean(0), s_c=s_c)
        return out, const

    def energy_grad(z, const, frames, norm0):
        """sum of normalised energy terms over `frames`, gradient wrt z (chunked; constants from the no-grad pass)"""
        g = torch.zeros_like(z)
        terms = dict(depth=0.0, centre=0.0, dir=0.0)
        mean_gt = gt_d[gt_d > 1e-6].mean()
        n_pix = sum(int((gt_d[f] > 1e-6).sum()) for f in frames)
        for i in range(0, len(frames), a.chunk):
            fr = frames[i:i + a.chunk]
            zc = z[fr].clone().requires_grad_(True)
            with torch.enable_grad():
                pr = P.m._decode(zc, H, W, 1, len(fr), rope[fr])
                d = pr["depth"][0, ..., 0].float(); r = pr["rays"][0].float()
                mk = gt_d[fr] > 1e-6
                Ed = (d * const["sc"] - gt_d[fr]).abs()[mk].sum() / n_pix / mean_gt
                cen = r[..., 3:6].mean((1, 2))
                Ec = ((cen - const["cmean"]) * const["s_c"] - gc[fr]).norm(dim=-1).sum() / len(frames) / gc.norm(dim=-1).mean()
                dp = r[..., :3] / r[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-9)
                Er = (1 - (dp * gt_dir[fr]).sum(-1)).mean((1, 2)).sum() / len(frames)
                parts = dict(depth=Ed, centre=Ec, dir=Er)
                if norm0 is None:                        # only measuring
                    L = Ed + Ec + Er
                else:
                    L = Ed / norm0["depth"] + a.w_cam * (Ec / norm0["centre"] + Er / norm0["dir"])
                gz, = torch.autograd.grad(L, zc)
            g[fr] = gz.detach()
            for k in terms:
                terms[k] += float(parts[k])
        return g, terms

    def precond(g, z, method):
        if method == "euc":
            return g
        X = z[:, PM:].reshape(-1, 768)
        Xc = X - X.mean(0)
        Sig = Xc.T @ Xc / len(Xc)
        Sig = Sig / Sig.diagonal().mean()
        return g @ Sig

    def run(name, base, K, method=None, rho=None, partial=False):
        t0 = time.time()
        z = x0.clone()
        if base == "BR":
            fg.enable()
        rows, lam, norm0 = [], None, None
        frames = even if partial else list(range(S))
        try:
            for i in range(K):
                z = F(z, i, K, base)
                m, const = evaluate(z)
                row = dict(step=i + 1, **m)
                if method is not None and i + 1 >= a.i_start:
                    if norm0 is None:
                        _, t = energy_grad(z, const, frames, None)
                        norm0 = {k: max(v, 1e-12) for k, v in t.items()}
                    g, t = energy_grad(z, const, frames, norm0)
                    pg = precond(g, z, method)
                    if lam is None:
                        lam = rho * float(z.norm()) / float(pg.norm())
                    z = z - lam * pg
                    row.update(E_depth=t["depth"], E_centre=t["centre"], E_dir=t["dir"],
                               force_rel=float(lam * pg.norm() / z.norm()))
                rows.append(row)
                if (i + 1) % 8 == 0 or i + 1 == K or i < 4:
                    print(f"  [{name}] step {i + 1:>3}: AbsRel {m['absrel']:.4f} (odd {m['absrel_odd']:.4f}) ATE {m['ate']:.4f}"
                          f" dir {m['dir_deg']:.2f}° |z| {m['tok_norm']:.2f}"
                          + (f" | E d/c/r {row['E_depth']:.4f}/{row['E_centre']:.4f}/{row['E_dir']:.5f} force {row['force_rel']:.2e}"
                             if "E_depth" in row else ""), flush=True)
            fin, _ = evaluate(z)
        finally:
            fg.restore()
        print(f"[{name}] done in {time.time() - t0:.0f}s: final AbsRel {fin['absrel']:.4f} ATE {fin['ate']:.4f} dir {fin['dir_deg']:.2f}°", flush=True)
        return dict(name=name, base=base, K=K, method=method, rho=rho, partial=partial, lam=lam, rows=rows, final=fin)

    plan = [("sched_K16_base", "sched", 16, None, None, False),
            ("sched_K64_base", "sched", 64, None, None, False),
            ("sched_K64_euc_3e-3", "sched", 64, "euc", 3e-3, False),
            (f"BR_K{a.K_br}_base", "BR", a.K_br, None, None, False)]
    for meth in ["euc", "man"]:
        for rho in a.rhos:
            plan.append((f"BR_K{a.K_br}_{meth}_{rho:g}", "BR", a.K_br, meth, rho, False))
    res = {}
    if os.path.exists(f"{a.out}/b4oracle.json"):
        res = json.load(open(f"{a.out}/b4oracle.json"))
    for (name, base, K, meth, rho, part) in plan:
        if a.only is not None and name not in a.only:
            continue
        if name in res:
            print(f"skip {name} (done)"); continue
        res[name] = run(name, base, K, meth, rho, part)
        json.dump(res, open(f"{a.out}/b4oracle.json", "w"), indent=1)
    # partial oracle with the best guided BR run
    guided = [v for k, v in res.items() if k.startswith("BR_") and v["method"] is not None and not v["partial"]]
    if guided and (a.only is None or "partial" in a.only):
        best = min(guided, key=lambda v: v["final"]["absrel"])
        pname = f"BR_K{a.K_br}_{best['method']}_{best['rho']:g}_partial"
        if pname not in res:
            res[pname] = run(pname, "BR", a.K_br, best["method"], best["rho"], True)
            json.dump(res, open(f"{a.out}/b4oracle.json", "w"), indent=1)

    lines = ["run | final AbsRel (odd) | ATE (odd) | dir° | best-step AbsRel | lam"]
    for k, v in res.items():
        f = v["final"]; ba = min(r["absrel"] for r in v["rows"])
        lines.append(f"{k:>28}: AbsRel {f['absrel']:.4f} ({f['absrel_odd']:.4f}) | ATE {f['ate']:.4f} ({f['ate_odd']:.4f}) | dir {f['dir_deg']:.2f}"
                     f" | best {ba:.4f} | lam {v['lam']}")
    print("\n" + "\n".join(lines))
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        keys = [("absrel", "AbsRel", True), ("ate", "ATE (Sim3, rel.)", True), ("dir_deg", "ray-direction error (deg)", True),
                ("tok_norm", "patch |z|", False), ("absrel_odd", "AbsRel on ODD frames", True)]
        fig, ax = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.2))
        for c, (k, t, logy) in enumerate(keys):
            for j, (nm, v) in enumerate(res.items()):
                ls = "--" if v["method"] is None else (":" if v["partial"] else "-")
                ax[c].plot([r["step"] for r in v["rows"]], [r[k] for r in v["rows"]], ls, lw=1.1, color=f"C{j % 10}", label=nm)
            ax[c].set_title(t); ax[c].set_xlabel("step"); ax[c].grid(alpha=.3)
            if logy: ax[c].set_yscale("log")
        ax[0].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(f"{a.out}/b4oracle.png", dpi=130); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("plot failed:", e)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
