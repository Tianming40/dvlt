#!/usr/bin/env python3
"""token_dump.py -- store the FULL DVLT state at every loop step (original model, DVLT time schedule), for later
read/write (probe, decoder-Jacobian, steering) analyses.

For each K in --Ks: out/tokendump/K{K}/
  step_{s:03d}.npy      state after s loop steps (s = 0 is the encoder state x0), (S, P, C) float16, P = 5 + gh*gw
  targets_{s:03d}.npz   DVLT's own decode of that state, per patch (mean over the 14x14 pixels):
                        logdepth (S,Np) | dir (S,Np,3) unit ray direction | origin (S,Np,3) | point (S,Np,3) | pose_enc (S,9)
  meta.npz              K, S, P, C, grid, patch_start, t_schedule, rope positions, per-patch RGB, GT log-depth per patch,
                        GT c2w (S,4,4), GT intrinsics (3,3) at model resolution
Size: ~36 MB per state for 27 frames.

  PYTHONPATH=$(pwd) python shovel/token_dump.py --Ks 16 64 --out out/tokendump
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlt_probe import Probe, load_gt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--gt", default="out/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--Ks", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--out", default="out/tokendump")
    a = ap.parse_args()

    P = Probe("nvidia/dvlt", max_frames=a.max_frames)
    images = P.load(a.input)
    x0, rope, dims = P.init_state(images)
    B, S, H, W = dims
    ps = P.m.patch_size; gh, gw = H // ps, W // ps
    pm = P.m.patch_start_idx
    gd, c2w, Kin = load_gt(a.gt, P.img_size, device=P.acc.device)
    gd, c2w = gd[:S].float(), c2w[:S].float()

    def per_patch(t):                                   # (S,H,W,C) -> (S, gh*gw, C)
        return t.reshape(t.shape[0], gh, ps, gw, ps, t.shape[-1]).mean((2, 4)).reshape(t.shape[0], gh * gw, t.shape[-1])

    @torch.no_grad()
    def targets(x):
        pr = P.decode(x, rope, dims)
        rays = pr["rays"][0].float(); dep = pr["depth"][0].float()
        d = rays[..., :3] / rays[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-9)
        dd = per_patch(d); dd = dd / dd.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return dict(logdepth=per_patch(dep.clamp_min(1e-6).log())[..., 0].cpu().numpy().astype(np.float32),
                    dir=dd.cpu().numpy().astype(np.float32),
                    origin=per_patch(rays[..., 3:6]).cpu().numpy().astype(np.float32),
                    point=per_patch(rays[..., 3:6] + rays[..., :3] * dep).cpu().numpy().astype(np.float32),
                    pose_enc=pr["pose_enc"][0].float().cpu().numpy().astype(np.float32))

    rgb = per_patch(images[0].permute(0, 2, 3, 1).float()).cpu().numpy()
    gdp = per_patch(torch.where(gd > 1e-6, gd.log(), torch.full_like(gd, float("nan")))[..., None])[..., 0].cpu().numpy()

    for K in a.Ks:
        d = f"{a.out}/K{K}"
        os.makedirs(d, exist_ok=True)
        np.savez(f"{d}/meta.npz", K=K, S=S, P=x0.shape[1], C=x0.shape[2], grid=np.array([gh, gw]), patch_start=pm,
                 t_schedule=np.linspace(0, 1, K).astype(np.float32), rope=rope.cpu().numpy(), rgb=rgb, gt_logdepth=gdp,
                 gt_c2w=c2w.cpu().numpy(), gt_K=Kin.cpu().numpy(), input=a.input, gt=a.gt)
        with torch.no_grad():
            x = x0.clone()
            for s in range(K + 1):
                if s > 0:
                    x = P.step(x, s - 1, K, rope, B, S)
                np.save(f"{d}/step_{s:03d}.npy", x.cpu().numpy().astype(np.float16))
                np.savez(f"{d}/targets_{s:03d}.npz", **targets(x))
                if s % 8 == 0 or s == K:
                    print(f"K={K}: saved step {s}/{K}", flush=True)
        print(f"K={K} done -> {d}")


if __name__ == "__main__":
    main()
