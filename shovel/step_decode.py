#!/usr/bin/env python3
"""Step-wise decode dump: run the trained DVLT loop for K steps, and AFTER EACH STEP decode the
current state once (repo code) to get every frame's prediction. Saves depth + rays + pose_enc
(+ depth_conf) as fp16, ONE FOLDER PER K, ONE FILE PER STEP, for later offline visualization.

Pure repo compute: the step is DVLTModel._interval_step, the decode is DVLTModel._decode, called
directly (no Probe wrapper). This script only orchestrates "step -> decode -> save".

Output layout (everything clearly separable):
  out/stepdecode_<tag>_K<K>/
    meta.npz                 # K, S, H, W, input, ckpt, t-schedule, gt intrinsics/names/c2w
    step_00.npz              # one step: depth (S,H,W,1), rays (S,H,W,6), pose_enc (S,9), depth_conf (S,H,W)
    step_01.npz              #   -> index a frame inside with [frame_i]
    ...
    step_<K-1>.npz

  PYTHONPATH=$(pwd) python shovel/step_decode.py \
      --input out/gt_00777c41d4_s10/probe_input --gt out/gt_00777c41d4_s10/gt.npz \
      --max-frames 27 --K 16 --tag s10
"""
import argparse
import os

import numpy as np
import torch

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import load_model, load_images, init_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="probe_input dir of (undistorted pinhole) images")
    ap.add_argument("--gt", default=None, help="optional gt.npz, copied into meta (intrinsics/names/c2w)")
    ap.add_argument("--ckpt", default="nvidia/dvlt")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--K", type=int, default=16, help="number of loop steps (settable; try 8/16/32)")
    ap.add_argument("--tag", default="scene", help="short scene tag used in the folder name")
    ap.add_argument("--out-root", default="out", help="top-level output folder")
    ap.add_argument("--decode-chunk", type=int, default=4, help="frames per decode chunk (memory)")
    ap.add_argument("--no-conf", action="store_true", help="skip depth_conf (store only depth+rays+pose)")
    a = ap.parse_args()

    out_dir = os.path.join(a.out_root, f"stepdecode_{a.tag}_K{a.K}")
    os.makedirs(out_dir, exist_ok=True)

    model, m, acc = load_model(a.ckpt, img_size=504)
    m.decode_chunk_size = max(1, min(getattr(m, "decode_chunk_size", None) or 128, a.decode_chunk))
    images = load_images(a.input, 504, a.max_frames, acc.device)

    ts = torch.linspace(0.0, 1.0, a.K).tolist()
    with torch.no_grad(), acc.autocast():
        x, rope, dims = init_state(m, images)
        B, S, H, W = dims
        assert B == 1
        for k in range(a.K):
            t_now = ts[k]
            x = m._interval_step(x, ts[k], ts[k + 1] if k + 1 < a.K else 1.0, rope, B, S)  # repo
            pred = m._decode(x, H, W, B, S, rope)                                           # repo
            rec = dict(
                step=k, t=float(t_now),
                depth=pred["depth"][0].float().cpu().half().numpy(),   # (S,H,W,1)
                rays=pred["rays"][0].float().cpu().half().numpy(),     # (S,H,W,6)
            )
            if "pose_enc" in pred:
                rec["pose_enc"] = pred["pose_enc"][0].float().cpu().half().numpy()  # (S,9)
            if not a.no_conf:
                rec["depth_conf"] = pred["depth_conf"][0].float().cpu().half().numpy()  # (S,H,W)
            np.savez_compressed(os.path.join(out_dir, f"step_{k:02d}.npz"), **rec)
            print(f"step {k+1}/{a.K} decoded + saved  ({S} frames @ {H}x{W})")

    # --- meta (shared across steps) ---
    meta = dict(K=a.K, S=S, H=H, W=W, input=a.input, ckpt=a.ckpt,
                t_schedule=np.array(ts, dtype=np.float32), resolution=f"{H}x{W}", dtype="float16")
    if a.gt and os.path.exists(a.gt):
        g = np.load(a.gt)
        for key in ("intrinsics", "names", "c2w"):
            if key in g.files:
                meta[f"gt_{key}"] = g[key]
    np.savez_compressed(os.path.join(out_dir, "meta.npz"), **meta)

    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 1e6
    print(f"\nwrote {out_dir}/  ({a.K} step files + meta.npz, {total:.0f} MB total)")
    print(f"  each step_XX.npz: depth(S,H,W,1) rays(S,H,W,6) pose_enc(S,9) depth_conf(S,H,W), fp16")


if __name__ == "__main__":
    main()
