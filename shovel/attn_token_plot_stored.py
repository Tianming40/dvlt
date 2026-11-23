#!/usr/bin/env python3
"""attn_token_plot_stored.py -- OFFLINE: plot the raw global-attention row of the 3 tracked frame-0 patches over ALL
27*869 tokens (x = global token index, y = attention), from Stage-2 stored data (out/track_s10_K{K}/step_XX.npz).
Each frame's span is shaded: own frame (blue), GT-visible (green), in view but OCCLUDED (orange), out of view (grey).
GT target patch in visible frames marked with a red tick. Heads are averaged (default 0 and 9).

  one image : PYTHONPATH=$(pwd) python shovel/attn_token_plot_stored.py --K 16 --step 4
  video     : PYTHONPATH=$(pwd) python shovel/attn_token_plot_stored.py --K 16 64 --all-steps --video --fps 2
              -> out/attnmatch_stored/token_video/K{K}/step_XXX.png + token_attn_K{K}_h0_9.mp4
In video mode the y-range of each patch row is FIXED over all steps of that K (max over steps), so heights compare.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_gt  # noqa: E402
from attn_headmatch_stored import gt_pairs, PATCH_SLOTS  # noqa: E402


def load_run(a, K):
    d = f"{a.track_root}/track_{a.tag}_K{K}"
    meta = np.load(f"{d}/meta.npz")
    S, P, patch, pm = int(meta["S"]), int(meta["P"]), int(meta["patch"]), int(meta["patch_start_idx"])
    gw = int(meta["grid"][1])
    names = [str(n) for n in meta["patch_names"]]
    rc = [divmod(int(t) - pm, gw) for t in meta["tok_index"][PATCH_SLOTS]]
    gd, c2w, Km = load_gt(a.gt, 504, patch=patch, device="cpu")
    tgt, geo, occv = gt_pairs(gd[:S].numpy(), c2w[:S].numpy(), Km.numpy(), rc, patch, a.occ)
    return dict(d=d, S=S, P=P, pm=pm, gw=gw, names=names, rc=rc, tgt=tgt, geo=geo, occv=occv,
                t_sched=meta["t_schedule"])


def rows_at(R, step, heads):
    z = np.load(f"{R['d']}/step_{step:02d}.npz")
    return z["global_rows"][PATCH_SLOTS][:, heads].astype(np.float32).mean(1), float(z["t_now"])   # (3,S,P)


def render(R, G, K, step, t_now, heads, out_png, ylims=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    S, P, pm, gw = R["S"], R["P"], R["pm"], R["gw"]
    N = S * P
    x = np.arange(N)
    fig, ax = plt.subplots(3, 1, figsize=(22, 9), sharex=True)
    for pi, name in enumerate(R["names"]):
        y = G[pi].reshape(N)
        ymax = ylims[pi] if ylims is not None else y.max() * 1.08
        for j in range(S):
            col = ("#cfe0ff" if j == 0 else "#c9f0c9" if R["occv"][pi, j] else "#ffd9a8" if R["geo"][pi, j] else "#e6e6e6")
            ax[pi].axvspan(j * P, (j + 1) * P, color=col, lw=0)
            ax[pi].axvline(j * P, color="k", lw=0.3, alpha=0.4)
            ax[pi].text(j * P + P / 2, ymax * 0.98, f"{G[pi, j, pm:].sum():.3f}", ha="center", va="top", fontsize=6.5)
            if R["occv"][pi, j]:
                tx = j * P + pm + R["tgt"][pi, j, 0] * gw + R["tgt"][pi, j, 1]
                ax[pi].plot([tx], [ymax * 0.88], "rv", ms=5)
        ax[pi].plot(x, np.minimum(y, ymax), color="k", lw=0.5)
        ax[pi].set_ylim(0, ymax)
        r, c = R["rc"][pi]
        ax[pi].set_ylabel(f"{name}\n(r{r},c{c})")
        vis = [G[pi, j, pm:].sum() for j in range(1, S) if R["occv"][pi, j]]
        inv = [G[pi, j, pm:].sum() for j in range(1, S) if not R["occv"][pi, j]]
        ax[pi].set_title(f"{name}: mean mass per frame  visible={np.mean(vis):.3f}  not-visible={np.mean(inv):.3f}  "
                         f"(own frame {G[pi, 0, pm:].sum():.3f}, specials total {G[pi, :, :pm].sum():.3f})",
                         fontsize=9, loc="left")
    ax[-1].set_xticks([j * P + P / 2 for j in range(S)])
    ax[-1].set_xticklabels([f"f{j}" for j in range(S)], fontsize=7)
    ax[-1].set_xlabel(f"global token index (27 frames x {P} tokens = {N}); number above each frame = attention mass on its patches")
    fig.legend(handles=[Patch(color="#cfe0ff", label="own frame (f0)"), Patch(color="#c9f0c9", label="visible (GT)"),
                        Patch(color="#ffd9a8", label="in view but occluded"), Patch(color="#e6e6e6", label="out of view"),
                        plt.Line2D([], [], color="r", marker="v", ls="", label="GT target patch")],
               loc="upper right", ncol=5, fontsize=8)
    hs = "+".join(str(h) for h in heads)
    fig.suptitle(f"global attention of 3 frame-0 patches over all tokens  |  K={K}  step {step}/{K - 1}  (t={t_now:.3f})"
                 f"  heads {hs} averaged", y=0.995, x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="out/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--track-root", default="out")
    ap.add_argument("--tag", default="s10")
    ap.add_argument("--K", type=int, nargs="+", default=[16])
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--all-steps", action="store_true")
    ap.add_argument("--heads", type=int, nargs="+", default=[0, 9])
    ap.add_argument("--occ", type=float, default=0.1)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--out", default="out/attnmatch_stored")
    a = ap.parse_args()
    hs = "_".join(str(h) for h in a.heads)

    for K in a.K:
        R = load_run(a, K)
        if not a.all_steps:
            os.makedirs(a.out, exist_ok=True)
            G, tn = rows_at(R, a.step, a.heads)
            out = f"{a.out}/token_attn_K{K}_s{a.step}_h{hs}.png"
            render(R, G, K, a.step, tn, a.heads, out)
            print("saved", out)
            continue
        vdir = f"{a.out}/token_video/K{K}"
        os.makedirs(vdir, exist_ok=True)
        allG = [rows_at(R, s, a.heads) for s in range(K)]
        ylims = [max(G[pi].max() for G, _ in allG) * 1.08 for pi in range(3)]       # fixed per patch row
        for s, (G, tn) in enumerate(allG):
            render(R, G, K, s, tn, a.heads, f"{vdir}/step_{s:03d}.png", ylims=ylims)
        print(f"K={K}: {K} frames in {vdir}/  (fixed y-max per row: {[round(v, 3) for v in ylims]})")
        if a.video:
            mp4 = f"{a.out}/token_attn_K{K}_h{hs}.mp4"
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{vdir}/step_%03d.png",
                   "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4]
            subprocess.run(cmd, check=True)
            print("video", mp4)


if __name__ == "__main__":
    main()
