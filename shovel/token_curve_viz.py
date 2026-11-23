#!/usr/bin/env python3
"""token_curve_viz.py -- plot ONE token's 768 values at every loop step (K16 and K64), from out/tokendump.

Figures (out/tokencurve/):
  overlay_K{K}.png   all steps overlaid (x = channel 0..767, y = value), colour = step
  steps_K{K}.png     small multiples for selected steps
  heatmap.png        step x channel heatmaps for K16 and K64 (shared colour scale)
The massive channel (largest mean |value| over the stored states) is marked; y-limits are set from the OTHER channels
so it does not flatten the rest (its own values are printed in the titles).

  PYTHONPATH=$(pwd) python shovel/token_curve_viz.py --frame 3 --row 12 --col 18
"""
import argparse
import os

import numpy as np

PM = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="out/tokendump")
    ap.add_argument("--frame", type=int, default=3)
    ap.add_argument("--row", type=int, default=12)
    ap.add_argument("--col", type=int, default=18)
    ap.add_argument("--token", type=int, default=None, help="raw token index in the frame (0 = cam, 1-4 = reg); overrides row/col")
    ap.add_argument("--out", default="out/tokencurve")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    meta = np.load(f"{a.dump}/K16/meta.npz")
    gh, gw = [int(v) for v in meta["grid"]]
    tok = a.token if a.token is not None else PM + a.row * gw + a.col
    name = (f"frame {a.frame}, " + (["cam", "reg1", "reg2", "reg3", "reg4"][tok] if tok < PM else f"patch (row {a.row}, col {a.col})"))

    traj = {}
    for K in (16, 64):
        traj[K] = np.stack([np.load(f"{a.dump}/K{K}/step_{s:03d}.npy", mmap_mode="r")[a.frame, tok].astype(np.float32)
                            for s in range(K + 1)])                                         # (K+1, 768)
    allv = np.concatenate([traj[16], traj[64]])
    ch = int(np.abs(allv).mean(0).argmax())
    others = np.delete(allv, ch, axis=1)
    lim = np.percentile(np.abs(others), 99.9) * 1.1
    x = np.arange(768)

    for K, T in traj.items():
        fig, ax = plt.subplots(figsize=(16, 5))
        cmap = plt.get_cmap("viridis")
        for s in range(K + 1):
            ax.plot(x, T[s], lw=0.6, color=cmap(s / K), alpha=0.8)
        ax.axvline(ch, color="r", lw=0.8, ls=":")
        ax.set_ylim(-lim, lim); ax.set_xlim(0, 767)
        ax.set_xlabel("channel"); ax.set_ylabel("value")
        ax.set_title(f"{name}: all {K + 1} states of K={K} (colour = step, light = late); channel {ch} (red, clipped): "
                     f"{T[0, ch]:.1f} → {T[-1, ch]:.1f};  |token| {np.linalg.norm(T[0]):.1f} → {np.linalg.norm(T[-1]):.1f}")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, K)); fig.colorbar(sm, ax=ax, label="step", fraction=.02)
        fig.tight_layout(); fig.savefig(f"{a.out}/overlay_K{K}.png", dpi=130); plt.close(fig)

        sel = [0, 1, 2, 4, 8, 12, 16] if K == 16 else [0, 2, 4, 8, 16, 24, 32, 48, 64]
        fig, axs = plt.subplots(len(sel), 1, figsize=(16, 1.7 * len(sel)), sharex=True)
        for axx, s in zip(axs, sel):
            axx.plot(x, T[s], lw=0.6, color="k")
            axx.axvline(ch, color="r", lw=0.8, ls=":")
            axx.set_ylim(-lim, lim)
            axx.text(0.995, 0.9, f"step {s}  |token| {np.linalg.norm(T[s]):.1f}  ch{ch} {T[s, ch]:.1f}", transform=axx.transAxes,
                     ha="right", va="top", fontsize=8)
        axs[-1].set_xlabel("channel"); axs[0].set_title(f"{name}, K={K}")
        fig.tight_layout(); fig.savefig(f"{a.out}/steps_K{K}.png", dpi=120); plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw=dict(height_ratios=[1, 2.5]))
    for axx, K in zip(axs, (16, 64)):
        im = axx.imshow(traj[K], aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        axx.set_ylabel(f"step (K={K})"); axx.axvline(ch, color="k", lw=0.5, ls=":")
    axs[-1].set_xlabel("channel"); axs[0].set_title(f"{name}: value per channel (x) and step (y); channel {ch} clipped")
    fig.colorbar(im, ax=axs, fraction=.02)
    fig.savefig(f"{a.out}/heatmap.png", dpi=120); plt.close(fig)
    print(f"token {tok} ({name}); massive channel {ch}; y-limit ±{lim:.2f}; saved {a.out}")


if __name__ == "__main__":
    main()
