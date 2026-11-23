#!/usr/bin/env python3
"""photo_ba.py -- classical DIRECT (photometric) bundle adjustment on EXPLICIT geometry, initialised from DVLT.
No training, no tokens: does the image contain information that DVLT did not use?

Variables per clip (cached by residual_opt/cache_dvlt.py):
  pose of frames 1..S-1 (rotation delta so(3) + centre delta; frame 0 fixed = DVLT's world frame),
  shared intrinsics (fx, fy, cx, cy), per-patch log depth (27x36 per frame, bilinearly upsampled to pixels).
Initialisation: each frame's camera from a DLT fit to DVLT's per-patch ray directions (A d ~ [u,v,1], A = K R^T),
  RQ-decomposed into K and R; centre = mean ray origin; depth = DVLT's per-patch log depth (z-depth).
Energy (MAP):  E = photometric(i -> j, |i-j| <= 2)  +  lambda * mean (logD - logD_dvlt)^2
  photometric: every 2nd pixel of frame i is back-projected with its depth, projected into frame j, colour sampled;
  0.85 * (1-SSIM)/2 + 0.15 * L1 on images blurred with sigma 4 -> 2 -> 1 px (coarse to fine, --iters per stage);
  masks: inside image, in front, depth-consistent with frame j (|log ratio| < --occ), auto-mask (warp beats identity).
Configs: cam+depth (lambda 1), cam+depth (lambda 10), camera only. Depth lr 1e-3 (1e-2 wrecked the depth in a smoke test).
Metrics (patch level, identical to residual_opt/train.py): AbsRel, ray-direction error, camera-centre ATE.
Also: photometric energy of the GT geometry vs DVLT's (is the residual able to tell better from worse?).

  PYTHONPATH=$(pwd) python residual_opt/photo_ba.py --splits val ood_bf --out out/resopt/photo_ba
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from residual_opt.train import metrics  # noqa: E402

PS = 14


def rq3(A):
    P = np.flipud(np.eye(3))
    Q, R = np.linalg.qr((P @ A).T)
    R = P @ R.T @ P; Q = P @ Q.T
    D = np.diag(np.sign(np.diag(R)))
    return R @ D, D @ Q


def fit_cameras(x, gh, gw):
    """x (S,Np,7) -> K (3,3) shared (mean of per-frame), R (S,3,3) camera-to-world rotations, c (S,3)"""
    S = x.shape[0]
    jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
    u = (jj.flatten() * PS + PS / 2 - 0.5).astype(np.float64); v = (ii.flatten() * PS + PS / 2 - 0.5).astype(np.float64)
    W, H = gw * PS, gh * PS
    su, sv = 2.0 / W, 2.0 / H
    T = np.array([[1 / su, 0, 1 / su], [0, 1 / sv, 1 / sv], [0, 0, 1.0]])
    Ks, Rs = [], []
    for s in range(S):
        d = x[s, :, 1:4].astype(np.float64); d /= np.linalg.norm(d, axis=-1, keepdims=True)
        un, vn = u * su - 1, v * sv - 1
        z = np.zeros_like(d)
        M = np.concatenate([np.concatenate([d, z, -un[:, None] * d], 1), np.concatenate([z, d, -vn[:, None] * d], 1)])
        A = T @ np.linalg.svd(M)[2][-1].reshape(3, 3)
        Ku, Q = rq3(A)
        if np.linalg.det(Q) < 0:
            Q = -Q; Ku = -Ku
        Ku = Ku / Ku[2, 2]
        if Ku[0, 0] < 0:                                           # A defined up to sign
            Ku = -Ku; Ku = Ku / Ku[2, 2]
        Ks.append(Ku); Rs.append(Q.T)                              # A = K R^T -> R = Q^T (camera-to-world)
    K = np.mean(Ks, 0); K[0, 1] = 0; K[1, 0] = K[2, 0] = K[2, 1] = 0; K[2, 2] = 1
    return K, np.stack(Rs), x[..., 4:7].mean(1)


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1], Kx[..., 0, 2], Kx[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    Kx = Kx - Kx.transpose(-1, -2)
    th = th[..., None]
    return torch.eye(3, device=w.device) + th.sin() * Kx + (1 - th.cos()) * Kx @ Kx


def blur(img, sigma):
    if sigma <= 0:
        return img
    r = int(3 * sigma); xs = torch.arange(-r, r + 1, device=img.device).float()
    g = torch.exp(-xs ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    C = img.shape[1]
    img = F.conv2d(F.pad(img, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(img, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def ssim(x, y):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    p = lambda t: F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="reflect"), 3, 1)
    mx, my = p(x), p(y)
    sx, sy, sxy = p(x * x) - mx ** 2, p(y * y) - my ** 2, p(x * y) - mx * my
    return (((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sx + sy + C2))).clamp(-1, 1)


class Clip:
    def __init__(self, f, dev):
        z = np.load(f)
        self.name = os.path.basename(f)[:-4]
        self.img = torch.from_numpy(z["image"]).to(dev).float() / 255.0                  # (S,3,H,W)
        self.xd, self.xg, self.mask = z["x_dvlt"], z["x_gt"], z["mask"]
        self.gh, self.gw = [int(v) for v in z["grid"]]
        self.S = self.img.shape[0]
        self.H, self.W = self.img.shape[-2:]


class Geometry(torch.nn.Module):
    def __init__(self, K, R, c, logd, gh, gw):
        super().__init__()
        dev = logd.device
        self.register_buffer("R0", torch.as_tensor(R, dtype=torch.float32, device=dev))
        self.register_buffer("c0", torch.as_tensor(c, dtype=torch.float32, device=dev))
        self.register_buffer("K0", torch.as_tensor(K, dtype=torch.float32, device=dev))
        self.register_buffer("logd0", logd.clone())
        S = R.shape[0]
        self.w = torch.nn.Parameter(torch.zeros(S - 1, 3, device=dev))
        self.t = torch.nn.Parameter(torch.zeros(S - 1, 3, device=dev))
        self.kf = torch.nn.Parameter(torch.zeros(4, device=dev))                    # log fx, log fy, dcx, dcy
        self.logd = torch.nn.Parameter(logd.clone())
        self.gh, self.gw = gh, gw

    def cams(self):
        Rd = torch.cat([torch.eye(3, device=self.w.device)[None], so3_exp(self.w)])
        R = self.R0 @ Rd
        c = self.c0 + torch.cat([torch.zeros(1, 3, device=self.w.device), self.t])
        K = self.K0.clone()
        K = torch.stack([torch.stack([K[0, 0] * self.kf[0].exp(), K[0, 1], K[0, 2] + self.kf[2]]),
                         torch.stack([K[1, 0], K[1, 1] * self.kf[1].exp(), K[1, 2] + self.kf[3]]), K[2]])
        return K, R, c

    def depth(self, H, W):                                                            # (S,1,H,W) z-depth
        return F.interpolate(self.logd.reshape(-1, 1, self.gh, self.gw), size=(H, W), mode="bilinear", align_corners=False).exp()

    def to_patch(self):
        """per-patch [logdepth, unit dir, origin] in the same convention as x_dvlt / x_gt"""
        K, R, c = self.cams()
        jj, ii = torch.meshgrid(torch.arange(self.gw, device=K.device), torch.arange(self.gh, device=K.device), indexing="xy")
        pix = torch.stack([jj.flatten() * PS + PS / 2 - 0.5, ii.flatten() * PS + PS / 2 - 0.5, torch.ones(self.gh * self.gw, device=K.device)], -1)
        d = torch.einsum("sab,pb->spa", R, pix @ torch.linalg.inv(K).T)
        d = d / d.norm(dim=-1, keepdim=True)
        S = R.shape[0]
        return torch.cat([self.logd.reshape(S, -1, 1), d, c[:, None].expand(S, d.shape[1], 3)], -1)


def photometric(geo, img, sigma, stride=2, nbrs=(-2, -1, 1, 2), occ=0.1, return_count=False):
    S, _, H, W = img.shape
    im = blur(img, sigma)
    K, R, c = geo.cams()
    D = geo.depth(H, W)
    vv, uu = torch.meshgrid(torch.arange(0, H, stride, device=img.device).float(), torch.arange(0, W, stride, device=img.device).float(), indexing="ij")
    pix = torch.stack([uu, vv, torch.ones_like(uu)], -1)                              # (h,w,3)
    ray = pix @ torch.linalg.inv(K).T
    Di = D[:, 0, ::stride, ::stride]                                                  # (S,h,w)
    Xw = torch.einsum("sab,shwb->shwa", R, ray[None] * Di[..., None]) + c[:, None, None]
    Ii = im[:, :, ::stride, ::stride]
    tot, cnt = 0.0, 0
    for o in nbrs:
        i = torch.arange(max(0, -o), min(S, S - o), device=img.device); j = i + o
        Xc = torch.einsum("sba,shwb->shwa", R[j], Xw[i] - c[j][:, None, None])         # R^T (X - c)
        z = Xc[..., 2]
        p = Xc @ K.T
        u, v = p[..., 0] / z.clamp_min(1e-4), p[..., 1] / z.clamp_min(1e-4)
        g = torch.stack([u / (W - 1) * 2 - 1, v / (H - 1) * 2 - 1], -1)
        Ij = F.grid_sample(im[j], g, align_corners=True, padding_mode="border")
        Dj = F.grid_sample(D[j], g, align_corners=True, padding_mode="border")[:, 0]
        e = 0.85 * (1 - ssim(Ii[i], Ij)) / 2 + 0.15 * (Ii[i] - Ij).abs()
        e = e.mean(1)
        with torch.no_grad():
            Iid = im[j][:, :, ::stride, ::stride]
            eid = (0.85 * (1 - ssim(Ii[i], Iid)) / 2 + 0.15 * (Ii[i] - Iid).abs()).mean(1)
            m = (g.abs() <= 1).all(-1) & (z > 1e-3) & ((z.clamp_min(1e-4) / Dj.clamp_min(1e-4)).log().abs() < occ) & (e < eid)
        tot = tot + (e * m).sum(); cnt += int(m.sum())
    return (tot / max(cnt, 1), cnt) if return_count else tot / max(cnt, 1)


def run_clip(clip, cfg, args, dev):
    K, R, c = fit_cameras(clip.xd, clip.gh, clip.gw)
    logd0 = torch.from_numpy(clip.xd[..., 0]).to(dev)
    geo = Geometry(K, R, c, logd0, clip.gh, clip.gw)
    init_fit = metrics(geo.to_patch().detach().cpu().numpy(), clip.xg, clip.mask)
    params = [{"params": [geo.w], "lr": args.lr_rot}, {"params": [geo.t], "lr": args.lr_trans * float(np.linalg.norm(c - c.mean(0), axis=1).mean() + 1e-6)},
              {"params": [geo.kf], "lr": args.lr_k}]
    if cfg["depth"]:
        params.append({"params": [geo.logd], "lr": args.lr_depth})
    opt = torch.optim.Adam(params)
    traj = []
    it = 0
    for sigma in args.sigmas:
        for _ in range(args.iters):
            E = photometric(geo, clip.img, sigma)
            loss = E + (cfg["lam"] * (geo.logd - geo.logd0).pow(2).mean() if cfg["depth"] else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
            it += 1
            if it % 50 == 0:
                traj.append(dict(it=it, sigma=sigma, E=float(E), **metrics(geo.to_patch().detach().cpu().numpy(), clip.xg, clip.mask)))
    final = metrics(geo.to_patch().detach().cpu().numpy(), clip.xg, clip.mask)
    return init_fit, final, traj


@torch.no_grad()
def energy_check(clip, dev):
    """photometric energy (sigma 1) of DVLT's geometry vs GT geometry (cameras fitted the same way)"""
    out = {}
    for name, x in [("dvlt", clip.xd), ("gt", np.where(clip.mask[..., None], clip.xg, np.concatenate([clip.xd[..., :1], clip.xg[..., 1:]], -1)))]:
        K, R, c = fit_cameras(x, clip.gh, clip.gw)
        geo = Geometry(K, R, c, torch.from_numpy(x[..., 0]).float().to(dev), clip.gh, clip.gw)
        out[name] = float(photometric(geo, clip.img, 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/storage/user/zhtia/resopt_cache/small")
    ap.add_argument("--splits", nargs="+", default=["val", "ood_bf"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[4.0, 2.0, 1.0])
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lr-rot", type=float, default=1e-3)
    ap.add_argument("--lr-trans", type=float, default=1e-3)
    ap.add_argument("--lr-k", type=float, default=1e-3)
    ap.add_argument("--lr-depth", type=float, default=1e-3)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--out", default="out/resopt/photo_ba")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    cfgs = {"cam+depth_lam1": dict(depth=True, lam=1.0), "cam+depth_lam10": dict(depth=True, lam=10.0), "cam_only": dict(depth=False, lam=0.0)}
    res, lines = {}, []
    for split in a.splits:
        files = sorted(glob.glob(f"{a.cache}/{split}/*.npz"))[: a.max_clips]
        R = res.setdefault(split, {"dvlt": [], "init_fit": [], "energy": [], **{k: [] for k in cfgs}, "traj": {k: [] for k in cfgs}})
        for f in files:
            t0 = time.time()
            clip = Clip(f, dev)
            R["dvlt"].append(metrics(clip.xd, clip.xg, clip.mask))
            R["energy"].append(energy_check(clip, dev))
            for k, cfg in cfgs.items():
                init_fit, final, traj = run_clip(clip, cfg, a, dev)
                R[k].append(final); R["traj"][k].append(traj)
            R["init_fit"].append(init_fit)
            e = R["energy"][-1]
            print(f"[{split}] {clip.name}: DVLT {R['dvlt'][-1]['absrel']:.4f}/{R['dvlt'][-1]['ate']:.4f} | fitted-camera init "
                  f"{init_fit['absrel']:.4f}/{init_fit['ate']:.4f} | " + " | ".join(f"{k} {R[k][-1]['absrel']:.4f}/{R[k][-1]['ate']:.4f}" for k in cfgs)
                  + f" | E(dvlt) {e['dvlt']:.4f} E(gt) {e['gt']:.4f} | {time.time() - t0:.0f}s", flush=True)
        mean = lambda L, key: float(np.mean([m[key] for m in L]))
        lines.append(f"\n[{split}] {len(files)} clips   (AbsRel / ray-dir deg / ATE, patch level, mean over clips)")
        for k in ["dvlt", "init_fit"] + list(cfgs):
            lines.append(f"  {k:>18}: {mean(R[k], 'absrel'):.4f} / {mean(R[k], 'dir_deg'):.3f} / {mean(R[k], 'ate'):.4f}")
        for k in cfgs:
            best = [min(t, key=lambda r: r["absrel"]) for t in R["traj"][k]]
            lines.append(f"  {k:>18} (GT-picked best iteration, reference only): {mean(best, 'absrel'):.4f} / {mean(best, 'dir_deg'):.3f} / {mean(best, 'ate'):.4f}")
        eg = np.array([[e["dvlt"], e["gt"]] for e in R["energy"]])
        lines.append(f"  photometric energy (sigma 1): DVLT {eg[:, 0].mean():.4f}, GT {eg[:, 1].mean():.4f}; GT lower on {(eg[:, 1] < eg[:, 0]).sum()}/{len(eg)} clips")
        print("\n".join(lines[-(len(cfgs) * 2 + 4):]), flush=True)
    json.dump(res, open(f"{a.out}/photo_ba.json", "w"), indent=1)
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
