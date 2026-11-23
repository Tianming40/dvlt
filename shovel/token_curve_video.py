#!/usr/bin/env python3
"""token_curve_video.py -- one PNG per loop step of ONE token's 768 values, then an mp4 per K (16, 64).

Frame layout:
  top    : this step's 768 values (black), z0 in light grey; fixed y-range over all steps of both runs; the massive
           channel is clipped and its value printed; the 3 largest-|value| other channels are labelled.
  bottom : scene depth AbsRel (per-patch decoded log-depth vs GT, median scale) and this token's norm vs step,
           current step marked.
Output: out/tokencurve/frames_K{K}/step_XXX.png and out/tokencurve/token_K{K}.mp4

  PYTHONPATH=$(pwd) python shovel/token_curve_video.py --frame 3 --row 12 --col 18
"""
import argparse
import os
import shutil
import subprocess

import numpy as np

PM = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="out/tokendump")
    ap.add_argument("--frame", type=int, default=3)
    ap.add_argument("--row", type=int, default=12)
    ap.add_argument("--col", type=int, default=18)
    ap.add_argument("--token", type=int, default=None, help="raw token index (0 = cam, 1-4 = reg); overrides row/col")
    ap.add_argument("--fps", type=float, nargs=2, default=[2, 4], help="fps for K16, K64")
    ap.add_argument("--out", default="out/tokencurve")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    meta = np.load(f"{a.dump}/K16/meta.npz")
    gh, gw = [int(v) for v in meta["grid"]]
    tok = a.token if a.token is not None else PM + a.row * gw + a.col
    name = f"frame {a.frame}, " + (["cam", "reg1", "reg2", "reg3", "reg4"][tok] if tok < PM else f"patch (row {a.row}, col {a.col})")
    gt = meta["gt_logdepth"]; m = ~np.isnan(gt)

    def absrel(K, s):
        ld = np.load(f"{a.dump}/K{K}/targets_{s:03d}.npz")["logdepth"]
        p, g = np.exp(ld[m]), np.exp(gt[m]); sc = np.median(g / p)
        return float(np.mean(np.abs(p * sc - g) / g))

    traj, err = {}, {}
    for K in (16, 64):
        traj[K] = np.stack([np.load(f"{a.dump}/K{K}/step_{s:03d}.npy", mmap_mode="r")[a.frame, tok].astype(np.float32) for s in range(K + 1)])
        err[K] = [absrel(K, s) for s in range(K + 1)]
    allv = np.concatenate(list(traj.values()))
    ch = int(np.abs(allv).mean(0).argmax())
    lim = np.percentile(np.abs(np.delete(allv, ch, axis=1)), 99.95) * 1.15
    nmax = max(np.linalg.norm(t, axis=1).max() for t in traj.values()) * 1.05
    emax = max(max(e) for e in err.values()) * 1.1
    x = np.arange(768)
    ff = shutil.which("ffmpeg")

    for (K, T), fps in zip(traj.items(), a.fps):
        d = f"{a.out}/frames_K{K}"
        os.makedirs(d, exist_ok=True)
        norms = np.linalg.norm(T, axis=1)
        for s in range(K + 1):
            fig = plt.figure(figsize=(16, 7.5))
            ax = fig.add_axes([0.05, 0.38, 0.92, 0.55])
            ax.plot(x, traj[16][0], color="0.8", lw=0.8, label="z0 (step 0)")
            ax.plot(x, T[s], color="k", lw=0.8, label=f"step {s}")
            ax.axvline(ch, color="r", lw=0.8, ls=":")
            ax.text(ch + 4, lim * 0.85, f"ch{ch} = {T[s, ch]:.1f}", color="r", fontsize=10)
            other = np.argsort(-np.abs(np.where(x == ch, 0, T[s])))[:3]
            for c in other:
                v = float(np.clip(T[s, c], -lim, lim))
                ax.annotate(f"ch{c}: {T[s, c]:.1f}", (c, v), xytext=(c + 6, v * 0.9), fontsize=8, color="C0")
            ax.set_xlim(0, 767); ax.set_ylim(-lim, lim); ax.set_xlabel("channel"); ax.set_ylabel("value")
            ax.legend(loc="lower left", fontsize=9)
            ax.set_title(f"K={K}   step {s}/{K}   |token| {norms[s]:.1f}   scene AbsRel {err[K][s]:.3f}   ({name})", fontsize=12)
            b1 = fig.add_axes([0.05, 0.07, 0.42, 0.22]); b2 = fig.add_axes([0.55, 0.07, 0.42, 0.22])
            b1.plot(range(K + 1), err[K], color="C3"); b1.plot(s, err[K][s], "o", color="C3")
            b1.set_xlim(0, K); b1.set_ylim(0, emax); b1.set_title("scene depth AbsRel", fontsize=9); b1.grid(alpha=.3)
            b2.plot(range(K + 1), norms, color="C2"); b2.plot(s, norms[s], "o", color="C2")
            b2.set_xlim(0, K); b2.set_ylim(0, nmax); b2.set_title("|token|", fontsize=9); b2.grid(alpha=.3)
            fig.savefig(f"{d}/step_{s:03d}.png", dpi=100); plt.close(fig)
        if ff:
            mp4 = f"{a.out}/token_K{K}.mp4"
            subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{d}/step_%03d.png",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4], check=True)
            print(f"saved {mp4} ({K + 1} frames, {fps} fps)")
    print(f"token {tok} ({name}); massive channel {ch}; y-range ±{lim:.2f}")


if __name__ == "__main__":
    main()
