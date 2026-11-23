#!/usr/bin/env python3
"""Run the FROZEN DVLT (original K=16 schedule, fp32) on ScanNet clips and cache everything the refiner needs.

Per clip (<cache>/<split>/<scene>_<k>.npz):
  tokens  (S, P, 768) fp16   DVLT state after K steps (cam + 4 reg + patches)
  image   (S, 3, H, W) uint8 the preprocessed model input
  x_dvlt  (S, Np, 7)  fp32   DVLT per-patch geometry: [log depth, ray dir (unit, 3), ray origin (3)]
  x_gt    (S, Np, 7)  fp32   GT in the SAME frame and scale: frame-0 camera coordinates (DVLT's world), scaled by
                             median(pred depth / GT depth) over valid pixels
  mask    (S, Np)     bool   patch has >= 50% valid GT depth (depth channel supervised only there)
  grid, scene, frames, scale

  PYTHONPATH=$(pwd) python residual_opt/cache_dvlt.py --n-train 40 --n-val 10 --clips 2 --out /storage/user/zhtia/resopt_cache/small
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dvlt_probe import Probe  # noqa: E402
from dvlt.util.preprocess import preprocess_images  # noqa: E402
from dvlt.common.constants import DataField  # noqa: E402
from residual_opt.data_scannet import list_scenes, sample_clip, sample_clip_bf, BF_SCENES, to_model_res  # noqa: E402

PM = 5


def per_patch(t, gh, gw, ps=14):
    S = t.shape[0]
    return t.reshape(S, gh, ps, gw, ps, -1).mean((2, 4)).reshape(S, gh * gw, -1)


@torch.no_grad()
def process(P, clip, K_steps=16):
    dev = P.acc.device
    batch = preprocess_images(clip.images, img_size=P.img_size, patch_size=14, device=dev)
    images = batch[DataField.IMAGES]                                                   # (1,S,3,H,W)
    x, rope, dims = P.init_state(images)
    x = x.float(); B, S, H, W = dims
    ts = torch.linspace(0.0, 1.0, K_steps).tolist()
    for i in range(K_steps):
        x = P.m._interval_step(x, ts[i], ts[i + 1] if i + 1 < K_steps else 1.0, rope, B, S).float()
    ds, rs = [], []
    for j in range(0, S, 4):
        pr = P.m._decode(x[j:j + 4], H, W, 1, min(4, S - j), rope[j:j + 4])
        ds.append(pr["depth"][0, ..., 0].float()); rs.append(pr["rays"][0].float())
    d, r = torch.cat(ds), torch.cat(rs)
    gh, gw = H // 14, W // 14
    dirs = r[..., :3] / r[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-9)
    pd = per_patch(dirs, gh, gw); pd = pd / pd.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x_dvlt = torch.cat([per_patch(d.clamp_min(1e-6).log()[..., None], gh, gw), pd, per_patch(r[..., 3:6], gh, gw)], -1)

    gd, Km = to_model_res(clip.depth, clip.K, img_size=P.img_size)
    gd = torch.from_numpy(gd).to(dev)
    assert gd.shape[1:] == (H, W), (gd.shape, H, W)
    valid = gd > 1e-3
    scale = float((d[valid] / gd[valid]).median())                                     # GT -> DVLT scale
    c2w = torch.from_numpy(clip.c2w).to(dev)
    rel = torch.linalg.inv(c2w[0])[None] @ c2w                                          # frame-0 camera coords
    vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    pix = torch.stack([uu, vv, torch.ones_like(uu)], -1)
    cam = pix @ torch.linalg.inv(torch.from_numpy(Km).to(dev)).T
    gdir = torch.einsum("sij,hwj->shwi", rel[:, :3, :3], cam)
    gdir = gdir / gdir.norm(dim=-1, keepdim=True)
    pgd = per_patch(gdir, gh, gw); pgd = pgd / pgd.norm(dim=-1, keepdim=True)
    logd = torch.where(valid, (gd * scale).clamp_min(1e-6).log(), torch.zeros_like(gd))
    cnt = per_patch(valid.float()[..., None], gh, gw)[..., 0]
    gld = per_patch(logd[..., None], gh, gw)[..., 0] / cnt.clamp_min(1e-6)
    gorg = (rel[:, :3, 3] * scale)[:, None].expand(S, gh * gw, 3)
    x_gt = torch.cat([gld[..., None], pgd, gorg], -1)
    img = (images[0].clamp(0, 1) * 255).round().to(torch.uint8)
    return dict(tokens=x.cpu().numpy().astype(np.float16), image=img.cpu().numpy(), x_dvlt=x_dvlt.cpu().numpy(),
                x_gt=x_gt.cpu().numpy(), mask=(cnt >= 0.5).cpu().numpy(), grid=np.array([gh, gw]), scale=scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--clips", type=int, default=2, help="clips per scene")
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="scannet", choices=["scannet", "bundlefusion"])
    ap.add_argument("--out", default="/storage/user/zhtia/resopt_cache/small")
    ap.add_argument("--no-tokens", action="store_true", help="do not store the (large) DVLT state tokens")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(a.seed)
    if a.dataset == "bundlefusion":                       # out-of-distribution evaluation set: all 8 scenes
        splits, sampler = {"ood_bf": BF_SCENES}, sample_clip_bf
    else:
        splits, sampler = None, sample_clip
    scenes = list_scenes() if splits is None else []
    # split by scene id (sceneXXXX), so the different scans of one room never cross train/val
    if splits is None:
        rooms = sorted({s[:9] for s in scenes}); rng.shuffle(rooms)
        val_rooms = set(rooms[: a.n_val]); train_rooms = rooms[a.n_val: a.n_val + a.n_train]
        pick = lambda rs: [sorted(s for s in scenes if s[:9] == r)[0] for r in rs]
        splits = {"val": pick(sorted(val_rooms)), "train": pick(train_rooms)}
    P = Probe("nvidia/dvlt", max_frames=a.S)
    for split, ss in splits.items():
        os.makedirs(f"{a.out}/{split}", exist_ok=True)
        for sc in ss:
            for k in range(a.clips):
                f = f"{a.out}/{split}/{sc}_{k}.npz"
                if os.path.exists(f):
                    continue
                t0 = time.time()
                clip = sampler(sc, S=a.S, stride=a.stride, rng=rng)
                if clip is None:
                    print(f"skip {sc} (no valid clip)", flush=True); break
                out = process(P, clip)
                if a.no_tokens:
                    out.pop("tokens")
                np.savez(f, scene=sc, frames=np.array(clip.frames), **out)
                xd, xg, m = out["x_dvlt"], out["x_gt"], out["mask"]
                ar = np.mean(np.abs(np.exp(xd[..., 0][m]) - np.exp(xg[..., 0][m])) / np.exp(xg[..., 0][m]))
                print(f"[{split}] {sc}_{k}: frames {clip.frames[0]}..{clip.frames[-1]} | DVLT per-patch AbsRel {ar:.4f} | {time.time() - t0:.1f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
