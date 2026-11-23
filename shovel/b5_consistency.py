#!/usr/bin/env python3
"""b5_consistency.py -- B5: per-step guidance of the frozen DVLT loop by GT-FREE multi-view consistency residuals.

Same loop / write rule as B4 (shovel/b4_oracle.py): base = fixed gates B + RMS norm, Euclidean write, FIXED step size
    z <- F(z);  z <- z - lam * grad_z E(z)        (lam calibrated once at --i-start so |lam g|/|z| = rho)
Only the residual changes. Geometry is DECODED from the current state, then each frame A is reprojected into its
neighbours B (|i-j| in --nbrs, both directions):
    X_A(u)  = decoded world point of pixel u in A            (= ray origin + depth * ray dir, with gradient)
    pi_B(X) = projection into B with B's camera, fitted (no grad) from B's decoded ray field: A_B d ~ [u,v,1], c_B = mean origin
Versions (no GT anywhere in the residual):
  geo  : dense geometric consistency   r = log|X_A(u)-c_B| - log(range_B at pi_B(X_A(u)))          (Huber)
  photo: photometric consistency       0.85*(1-SSIM)/2 + 0.15*|I_A(u) - I_B(pi(u))|, scales 2/4/8 px stride, monodepth2
                                       auto-mask (keep pixels where warping beats identity)
  feat : feature-metric consistency    1 - cos( x0_A(patch) , x0_B(pi(patch centre)) ), x0 = frozen encoder (DINOv2)
                                       patch features, per-frame mean removed (raw DINO cos has a ~0.87 DC)
All versions: in-image and in-front masks + visibility mask |log(range_A/range_B)| < --vis (occlusion).
Gradient flows into frame A only (B's maps, camera fit and the images / x0 are held fixed in that term); every frame is
an A for its own pairs, so every frame is guided. GT is used ONLY for evaluation and for a diagnostic: at a few steps the
cosine between the consistency gradient and the B4 ORACLE gradient (depth + 0.1*(centre+dir)) is reported.

  PYTHONPATH=$(pwd) python shovel/b5_consistency.py --out out/b5consistency
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvlt_probe import Probe, load_gt, depth_vs_gt, sim3_ate  # noqa: E402
from gate_fix import FixedGates, rms_normalize  # noqa: E402

PM = 5


def ssim(x, y):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    p = lambda t: Fn.avg_pool2d(Fn.pad(t, (1, 1, 1, 1), mode="reflect"), 3, 1)
    mx, my = p(x), p(y)
    sx, sy, sxy = p(x * x) - mx ** 2, p(y * y) - my ** 2, p(x * y) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sx + sy + C2))
    return ((1 - s) / 2).clamp(0, 1)


def huber(r, d):
    a = r.abs()
    return torch.where(a < d, 0.5 * a ** 2 / d, a - 0.5 * d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--gt", default="out/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--i-start", type=int, default=3)
    ap.add_argument("--versions", nargs="+", default=["geo", "photo", "feat"])
    ap.add_argument("--rhos", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
    ap.add_argument("--nbrs", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--vis", type=float, default=0.15)
    ap.add_argument("--chunk", type=int, default=3)
    ap.add_argument("--diag-steps", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="out/b5consistency")
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
    img = images[0].float()                                               # (S,3,H,W) in [0,1]
    gh, gw = H // 14, W // 14
    feat = x0[:, PM:].reshape(S, gh, gw, 768)
    feat = feat - feat.mean((1, 2), keepdim=True)
    feat = (feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)).permute(0, 3, 1, 2).contiguous()   # (S,768,gh,gw)
    gt_d, gt_c2w, gt_K = load_gt(a.gt, P.img_size, device=dev)
    gt_d, gt_c2w = gt_d[:S].float(), gt_c2w[:S].float()
    rel = torch.linalg.inv(gt_c2w[0])[None] @ gt_c2w
    vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    pix = torch.stack([uu, vv, torch.ones_like(uu)], -1)                  # (H,W,3)
    gt_dir = torch.einsum("sij,hwj->shwi", rel[:, :3, :3], pix @ torch.linalg.inv(gt_K).T)
    gt_dir = gt_dir / gt_dir.norm(dim=-1, keepdim=True)
    gt_c = gt_c2w[:, :3, 3]
    gc = rel[:, :3, 3] - rel[:, :3, 3].mean(0)
    pairs = {i: [i + o for o in sorted(set(a.nbrs) | {-n for n in a.nbrs}) if 0 <= i + o < S] for i in range(S)}
    fg = FixedGates(P.m, mode="B")
    pc = torch.stack([(torch.arange(gw, device=dev) * 14 + 7).float().expand(gh, gw),
                      (torch.arange(gh, device=dev) * 14 + 7).float()[:, None].expand(gh, gw)], -1)   # patch centres (gh,gw,2)

    def F(z):
        with torch.no_grad():
            return rms_normalize(P.m._interval_step(z, 0.5, 0.5, rope, B, S).float())

    def decode(z, frames):
        pr = P.m._decode(z[frames], H, W, 1, len(frames), rope[frames])
        return pr["depth"][0, ..., 0].float(), pr["rays"][0].float(), pr["world_points"][0].float()

    @torch.no_grad()
    def fit_camera(r):
        """r: (H,W,6) decoded rays of ONE frame -> (A 3x3 with A d ~ [u,v,1], centre c). DLT on 3000 pixels."""
        d = r[..., :3] / r[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-9)
        idx = torch.randperm(H * W, device=dev, generator=torch.Generator(device=dev).manual_seed(0))[:3000]
        dd = d.reshape(-1, 3)[idx]; p = pix.reshape(-1, 3)[idx]
        su, sv = 2.0 / W, 2.0 / H                                         # Hartley-style normalisation
        un, vn = p[:, 0] * su - 1, p[:, 1] * sv - 1
        z0 = torch.zeros_like(dd)
        M = torch.cat([torch.cat([dd, z0, -un[:, None] * dd], 1), torch.cat([z0, dd, -vn[:, None] * dd], 1)])
        _, V = torch.linalg.eigh((M.T @ M).double())
        An = V[:, 0].float().reshape(3, 3)
        T = torch.tensor([[1 / su, 0, 1 / su], [0, 1 / sv, 1 / sv], [0, 0, 1.0]], device=dev)   # back to pixels
        A = T @ An
        if (A @ d[H // 2, W // 2])[2] < 0:
            A = -A
        return A, r[..., 3:6].mean((0, 1))

    @torch.no_grad()
    def evaluate(z):
        ds, rs, xs = [], [], []
        for i in range(0, S, 4):
            d, r, x = decode(z, list(range(i, min(i + 4, S))))
            ds.append(d); rs.append(r); xs.append(x)
        d, r, X = torch.cat(ds), torch.cat(rs), torch.cat(xs)
        cen = r[..., 3:6].mean((1, 2))
        dirp = r[..., :3] / r[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-9)
        ang = torch.rad2deg(torch.arccos((dirp * gt_dir).sum(-1).clamp(-1, 1))).mean()
        cams = [fit_camera(r[f]) for f in range(S)]
        rng = torch.stack([(X[f] - cams[f][1]).norm(dim=-1) for f in range(S)])          # (S,H,W) range maps
        mk = gt_d > 1e-6
        out = dict(absrel=depth_vs_gt(d, gt_d)["absrel"], ate=sim3_ate(cen, gt_c), dir_deg=float(ang),
                   tok_norm=float(z[:, PM:].norm(dim=-1).mean()))
        oc = cen - cen.mean(0)
        const = dict(cams=cams, rng=rng, sc=(gt_d[mk].median() / d[mk].median().clamp_min(1e-6)), cmean=cen.mean(0),
                     s_c=gc.norm() / oc.norm().clamp_min(1e-6))
        return out, const

    def project(Xa, cam):
        A, c = cam
        p = (Xa - c) @ A.T
        zc = p[..., 2]
        uv = p[..., :2] / zc[..., None].clamp_min(1e-6)
        return uv, zc

    def to_grid(uv, fullres=True):
        if fullres:
            return torch.stack([uv[..., 0] / (W - 1) * 2 - 1, uv[..., 1] / (H - 1) * 2 - 1], -1)
        return torch.stack([(uv[..., 0] + 0.5) / W * 2 - 1, (uv[..., 1] + 0.5) / H * 2 - 1], -1)

    def pair_energy(version, Xa, fa, fb, const):
        """energy of A=fa (Xa with grad, (H,W,3)) against B=fb (fixed). returns (sum, count)"""
        camB, rngB = const["cams"][fb], const["rng"][fb]
        if version == "feat":
            Xs = Xa[pc[..., 1].long(), pc[..., 0].long()]                                 # (gh,gw,3) patch centres
        elif version == "geo":
            Xs = Xa[::2, ::2]
        else:
            Xs = None
        if version in ("geo", "feat"):
            uv, zc = project(Xs, camB)
            g = to_grid(uv)
            rB = Fn.grid_sample(rngB[None, None], g[None], align_corners=True, padding_mode="border")[0, 0]
            rA = (Xs - camB[1]).norm(dim=-1)
            lr = rA.clamp_min(1e-6).log() - rB.clamp_min(1e-6).log()
            with torch.no_grad():
                valid = (g.abs() <= 1).all(-1) & (zc > 1e-3) & (lr.abs() < a.vis)
            if version == "geo":
                e = huber(lr, 0.05)
            else:
                fB = Fn.grid_sample(feat[fb][None], to_grid(uv, fullres=False)[None], align_corners=False, padding_mode="border")[0]
                fB = fB / fB.norm(dim=0, keepdim=True).clamp_min(1e-6)
                e = 1 - (feat[fa] * fB).sum(0)
            return (e * valid).sum(), valid.sum()
        # photometric, multi-scale strides
        tot, cnt = 0.0, 0
        for st in (2, 4, 8):
            Xs = Xa[::st, ::st]
            uv, zc = project(Xs, camB)
            g = to_grid(uv)
            IB = Fn.grid_sample(img[fb][None], g[None], align_corners=True, padding_mode="border")
            IA = img[fa][None][..., ::st, ::st]
            rB = Fn.grid_sample(rngB[None, None], g[None], align_corners=True, padding_mode="border")[0, 0]
            rA = (Xs - camB[1]).norm(dim=-1)
            with torch.no_grad():
                lr = rA.clamp_min(1e-6).log() - rB.clamp_min(1e-6).log()
                valid = (g.abs() <= 1).all(-1) & (zc > 1e-3) & (lr.abs() < a.vis)
                IBid = img[fb][None][..., ::st, ::st]                                       # auto-mask: identity warp
                e_id = (0.85 * ssim(IA, IBid) + 0.15 * (IA - IBid).abs()).mean(1)[0]
            e = (0.85 * ssim(IA, IB) + 0.15 * (IA - IB).abs()).mean(1)[0]
            with torch.no_grad():
                valid = valid & (e < e_id)
            tot = tot + (e * valid).sum() / max(st, 1) ** 0                                # equal weight per pixel and scale
            cnt += int(valid.sum())
        return tot, cnt

    def energy_grad(version, z, const):
        g = torch.zeros_like(z)
        total, count = 0.0, 0
        n_terms = sum(len(v) for v in pairs.values())
        for i in range(0, S, a.chunk):
            fr = list(range(i, min(i + a.chunk, S)))
            zc = z[fr].clone().requires_grad_(True)
            with torch.enable_grad():
                pr = P.m._decode(zc, H, W, 1, len(fr), rope[fr])
                X = pr["world_points"][0].float()
                L = 0.0
                for k, fa in enumerate(fr):
                    for fb in pairs[fa]:
                        e, c = pair_energy(version, X[k], fa, fb, const)
                        L = L + e / max(int(c), 1) / n_terms
                        count += int(c)
                gz, = torch.autograd.grad(L, zc)
            g[fr] = gz.detach(); total += float(L)
        return g, total, count

    def oracle_grad(z, const):
        g = torch.zeros_like(z)
        mean_gt = gt_d[gt_d > 1e-6].mean(); n_pix = int((gt_d > 1e-6).sum())
        for i in range(0, S, a.chunk):
            fr = list(range(i, min(i + a.chunk, S)))
            zc = z[fr].clone().requires_grad_(True)
            with torch.enable_grad():
                pr = P.m._decode(zc, H, W, 1, len(fr), rope[fr])
                d = pr["depth"][0, ..., 0].float(); r = pr["rays"][0].float()
                mk = gt_d[fr] > 1e-6
                Ed = (d * const["sc"] - gt_d[fr]).abs()[mk].sum() / n_pix / mean_gt
                cen = r[..., 3:6].mean((1, 2))
                Ec = ((cen - const["cmean"]) * const["s_c"] - gc[fr]).norm(dim=-1).sum() / S / gc.norm(dim=-1).mean()
                dp = r[..., :3] / r[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-9)
                Er = (1 - (dp * gt_dir[fr]).sum(-1)).mean((1, 2)).sum() / S
                gz, = torch.autograd.grad(Ed + 0.1 * (Ec + Er), zc)
            g[fr] = gz.detach()
        return g

    def run(name, version, rho):
        t0 = time.time()
        z = x0.clone(); fg.enable()
        rows, lam, E0 = [], None, None
        try:
            for i in range(a.K):
                z = F(z)
                m, const = evaluate(z)
                row = dict(step=i + 1, **m)
                if version is not None and i + 1 >= a.i_start:
                    g, E, cnt = energy_grad(version, z, const)
                    if lam is None:
                        lam = rho * float(z.norm()) / float(g.norm())
                        E0 = E
                    if i + 1 in a.diag_steps:
                        go = oracle_grad(z, const)
                        row["cos_oracle"] = float((g * go).sum() / (g.norm() * go.norm()).clamp_min(1e-12))
                        row["cos_oracle_patch"] = float((g[:, PM:] * go[:, PM:]).sum() / (g[:, PM:].norm() * go[:, PM:].norm()).clamp_min(1e-12))
                    z = z - lam * g
                    row.update(E=E, E_rel=E / E0, n_valid=cnt, force_rel=float(lam * g.norm() / z.norm()))
                rows.append(row)
                if (i + 1) % 4 == 0 or i + 1 == a.K or i < 4:
                    print(f"  [{name}] step {i + 1:>3}: AbsRel {m['absrel']:.4f} ATE {m['ate']:.4f} dir {m['dir_deg']:.2f}°"
                          + (f" | E/E0 {row['E_rel']:.3f} valid {row['n_valid']} force {row['force_rel']:.1e}" if "E" in row else "")
                          + (f" | cos(g, oracle) {row['cos_oracle']:+.3f} (patch {row['cos_oracle_patch']:+.3f})" if "cos_oracle" in row else ""),
                          flush=True)
            fin, _ = evaluate(z)
        finally:
            fg.restore()
        print(f"[{name}] done in {time.time() - t0:.0f}s: final AbsRel {fin['absrel']:.4f} ATE {fin['ate']:.4f} dir {fin['dir_deg']:.2f}°", flush=True)
        return dict(name=name, version=version, rho=rho, lam=lam, rows=rows, final=fin)

    # sanity: camera fit reprojects B's own points onto their pixels
    with torch.no_grad():
        z = x0.clone(); fg.enable()
        for _ in range(16):
            z = F(z)
        fg.restore()
        _, cst = evaluate(z)
        _, _, Xb = decode(z, [5])
        uv, _ = project(Xb[0], cst["cams"][5])
        err = (uv - pix[..., :2]).norm(dim=-1)
        print(f"camera-fit sanity (frame 5, B+RMS step 16): self-reprojection error median {err.median():.2f}px, 90% {err.flatten().quantile(.9):.2f}px", flush=True)

    plan = [(f"base_K{a.K}", None, None)] + [(f"{v}_{r:g}", v, r) for v in a.versions for r in a.rhos]
    res = {}
    if os.path.exists(f"{a.out}/b5.json"):
        res = json.load(open(f"{a.out}/b5.json"))
    for name, v, r in plan:
        if (a.only is not None and name not in a.only) or name in res:
            continue
        res[name] = run(name, v, r)
        json.dump(res, open(f"{a.out}/b5.json", "w"), indent=1)

    lines = ["run | AbsRel @8/16/32 | best AbsRel (step) | ATE @16/32 | dir @32 | cos(g,oracle) @4/8/16/32"]
    for k, v in res.items():
        rw = {r["step"]: r for r in v["rows"]}
        b = min(v["rows"], key=lambda r: r["absrel"])
        g = lambda s, key: (f"{rw[s][key]:.4f}" if s in rw and key in rw[s] else "  -  ")
        lines.append(f"{k:>12}: {g(8, 'absrel')} {g(16, 'absrel')} {g(32, 'absrel')} | best {b['absrel']:.4f} ({b['step']}) | "
                     f"{g(16, 'ate')} {g(32, 'ate')} | {g(32, 'dir_deg')} | " + " ".join(g(s, "cos_oracle") for s in a.diag_steps))
    lines.append("reference (B4): DVLT K16 AbsRel 0.0309 ATE 0.0083 | B+RMS base 0.037@16 0.049@32 | B+RMS ORACLE euc 3e-3 0.026@16 0.027@32")
    print("\n" + "\n".join(lines))
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        keys = [("absrel", "AbsRel", True), ("ate", "ATE (Sim3, rel.)", True), ("dir_deg", "ray-dir error (deg)", True), ("E_rel", "E / E0", False)]
        fig, ax = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.2))
        for c, (k, t, logy) in enumerate(keys):
            for j, (nm, v) in enumerate(res.items()):
                xs = [r["step"] for r in v["rows"] if k in r]; ys = [r[k] for r in v["rows"] if k in r]
                ax[c].plot(xs, ys, "--" if v["version"] is None else "-", lw=1.1, color=f"C{j % 10}", label=nm)
            if c == 0:
                ax[c].axhline(0.0309, color="k", ls=":", lw=.8); ax[c].axhline(0.026, color="g", ls=":", lw=.8)
            ax[c].set_title(t); ax[c].set_xlabel("step"); ax[c].grid(alpha=.3)
            if logy: ax[c].set_yscale("log")
        ax[0].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(f"{a.out}/b5.png", dpi=130); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("plot failed:", e)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
