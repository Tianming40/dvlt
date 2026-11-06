#!/usr/bin/env python3
"""Quantify how much the per-step Umeyama Sim3 (pred->GT) VARIES across loop steps, for each K.
If it barely varies, the rrd per-step and fixed views look identical (expected, not a bug): the
collapse is in fine/relative structure, not in the global gauge. Reports scale + rotation/translation
of each step's Sim3 relative to the final step's Sim3 (= what 'fixed' freezes)."""
import argparse, glob, os, re as _re
def _snum(p):
    return int(_re.search(r'step_(\d+)',os.path.basename(p)).group(1))
import numpy as np, torch

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import load_model, load_images, load_gt
from dvlt.common.constants import DataField, PredictionField
from dvlt.common.geometry import depth_to_world_coords_points
from dvlt.callbacks.util import align_predictions_to_gt, index_batch_and_predictions


def rot_angle(Ra, Rb):
    """geodesic angle (deg) between two 3x3 rotations."""
    R = Ra.T @ Rb
    c = (np.trace(R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--max-frames", type=int, default=27)
    a = ap.parse_args()

    model, m, acc = load_model("nvidia/dvlt", img_size=504)
    gd_all, gc_all, gK = load_gt(a.gt, 504, device="cpu")
    images = load_images(a.input, 504, a.max_frames, acc.device).cpu().float()
    S = images.shape[1]
    gd = gd_all[:S].float(); gc = gc_all[:S].float(); gK = gK.float()
    gt_world = torch.stack([depth_to_world_coords_points(gd[s], gc[s], gK)[0] for s in range(S)])
    batch = {DataField.IMAGES: images, DataField.EXTRINSICS_C2W: gc[None],
             DataField.INTRINSICS: gK[None, None].repeat(1, S, 1, 1),
             DataField.WORLD_POINTS: gt_world[None], DataField.POINT_MASKS: (gd > 1e-6)[None]}

    def step_sim3(sp):
        z = np.load(sp)
        rays = torch.from_numpy(z["rays"].astype("f4"))[None]
        depth = torch.from_numpy(z["depth"].astype("f4"))[None]
        conf = torch.from_numpy(z["depth_conf"].astype("f4"))[None]
        wp = rays[..., 3:6] + rays[..., :3] * depth
        raw = {"rays": rays, "depth": depth, "depth_conf": conf, "world_points": wp, "world_points_conf": conf}
        pred = model._postprocess_predictions(batch, raw)
        ib, ip = index_batch_and_predictions(batch, pred, batch_idx=0, seq_idxs=None, inplace=False)
        ip = align_predictions_to_gt(ib, ip, inplace=False)
        T = np.asarray(ip["transform"].detach().cpu())       # (3,4) [R|t]
        s = float(ip["scale_factor"])
        return s, T[:3, :3], T[:3, 3]

    for d in a.dirs:
        steps = sorted(glob.glob(f"{d}/step_*.npz"), key=_snum)
        rows = [step_sim3(sp) for sp in steps]
        scales = np.array([r[0] for r in rows])
        Rf, tf = rows[-1][1], rows[-1][2]                    # final-step Sim3 = what 'fixed' uses
        dang = np.array([rot_angle(r[1], Rf) for r in rows])
        dt = np.array([np.linalg.norm(r[2] - tf) for r in rows])
        tscale = np.linalg.norm(tf) + 1e-9
        K = len(steps)
        print(f"\n=== {os.path.basename(d)}  (K={K}) ===")
        print(f"  scale:   min={scales.min():.4f} max={scales.max():.4f} "
              f"spread={(scales.max()-scales.min())/np.median(scales)*100:.1f}% of median")
        print(f"  rot vs final:  max={dang.max():.2f} deg   mean={dang.mean():.2f} deg")
        print(f"  trans vs final: max={dt.max()/tscale*100:.1f}%  mean={dt.mean()/tscale*100:.1f}%  of |t_final|")
        print(f"  {'step':>4} {'scale':>8} {'rot°vsFin':>10} {'t%vsFin':>9}")
        for i, sp in enumerate(steps):
            k = _snum(sp)
            if k % max(1, K // 8) == 0 or k == K - 1:
                print(f"  {k:>4} {scales[i]:>8.4f} {dang[i]:>10.2f} {dt[i]/tscale*100:>8.1f}%")


if __name__ == "__main__":
    main()
