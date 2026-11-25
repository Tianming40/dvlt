#!/usr/bin/env python3
"""single_frame.py -- how much of DVLT's depth is the (monocular) prior?  Run the ORIGINAL DVLT (K=16 schedule, fp32)
on each frame ALONE (S=1) and on all frames together (S=27); compare depth AbsRel per frame.

Per-frame AbsRel uses a per-frame median scale (single-frame depth has a free scale), for both settings, so the
comparison is fair; the multi-view run additionally reports the usual sequence-level AbsRel (one shared scale).
Reported at steps 0 (decode of the encoder state), 1, 2, 4, 8, 16.

  PYTHONPATH=$(pwd) python shovel/single_frame.py --out out/stage3_latent/singleframe
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlt_probe import Probe, load_gt, depth_vs_gt  # noqa: E402

STEPS = [0, 1, 2, 4, 8, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/data/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--gt", default="out/data/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--out", default="out/stage3_latent/singleframe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    P = Probe("nvidia/dvlt", max_frames=a.max_frames)
    images = P.load(a.input)
    S = images.shape[1]
    gt_d, _, _ = load_gt(a.gt, P.img_size, device=images.device)
    gt_d = gt_d[:S].float()
    ts = torch.linspace(0.0, 1.0, a.K).tolist()

    @torch.no_grad()
    def run(imgs):
        x, rope, dims = P.init_state(imgs)
        x = x.float()
        B, s, H, W = dims
        out = {}
        for i in range(a.K + 1):
            if i > 0:
                x = P.m._interval_step(x, ts[i - 1], ts[i] if i < a.K else 1.0, rope, B, s).float()
            if i in STEPS:
                d = []
                for j in range(0, s, 4):
                    pr = P.m._decode(x[j:j + 4], H, W, 1, min(4, s - j), rope[j:j + 4])
                    d.append(pr["depth"][0, ..., 0].float())
                out[i] = torch.cat(d)
        return out

    multi = run(images)
    single = {i: [] for i in STEPS}
    for f in range(S):
        o = run(images[:, f:f + 1])
        for i in STEPS:
            single[i].append(o[i][0])
        print(f"frame {f} done", flush=True)
    single = {i: torch.stack(v) for i, v in single.items()}

    res, lines = {}, ["step | single-frame AbsRel (per-frame scale) | multi-view AbsRel (per-frame scale) | multi-view (sequence scale)"]
    for i in STEPS:
        sf = [depth_vs_gt(single[i][f], gt_d[f])["absrel"] for f in range(S)]
        mf = [depth_vs_gt(multi[i][f], gt_d[f])["absrel"] for f in range(S)]
        ms = depth_vs_gt(multi[i], gt_d)["absrel"]
        res[i] = dict(single_per_frame=sf, multi_per_frame=mf, multi_seq=ms)
        lines.append(f"{i:>4} | {np.mean(sf):.4f} (median {np.median(sf):.4f}) | {np.mean(mf):.4f} (median {np.median(mf):.4f}) | {ms:.4f}")
    fs = res[a.K]
    better = int(np.sum(np.array(fs["multi_per_frame"]) < np.array(fs["single_per_frame"])))
    lines.append(f"step {a.K}: multi-view better than single-frame on {better}/{S} frames; "
                 f"mean ratio single/multi = {np.mean(np.array(fs['single_per_frame']) / np.array(fs['multi_per_frame'])):.2f}")
    print("\n" + "\n".join(lines))
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    json.dump({str(k): v for k, v in res.items()}, open(f"{a.out}/singleframe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
