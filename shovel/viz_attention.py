#!/usr/bin/env python3
"""Overlay the per-head GLOBAL attention of a tracked frame-0 patch onto EVERY frame, with alpha,
one PNG per head (+ a 13th 'mean' over heads), per step. Reads track_token's dump.

Layout (option B, most detailed):
  out/attn_viz_<tag>_K<K>/<token>/step_<kk>/frame_<jj>/head_<hh>.png
                                                        mean.png
Each PNG = frame j's RGB image with that head's attention (global_rows[token,h,j,patches] -> 24x36,
nearest-upsampled to the patch grid) blended on top (alpha ~ normalized attention). A red cross marks
the GT correspondence in frame j (when visible); a cyan box marks the query patch itself in frame 0.
Normalisation is per (token, head, step) across ALL frames, so brighter frame = more attended.

  PYTHONPATH=$(pwd) python shovel/viz_attention.py --dir out/track_s10_K16 \
      --input out/gt_00777c41d4_s10/probe_input --tokens near_tex,flat,far
  # sample first:  --steps 1 --frames 0,2 --tokens near_tex
"""
import argparse, glob, os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import load_images


def overlay(img, attn_grid, gmax, patch, cmap, alpha=0.55, spotlight=False):
    """img: (H,W,3) uint8; attn_grid: (gh,gw); gmax: norm scalar. Returns blended (H,W,3) uint8.
    Full-coverage heatmap: EVERY patch is coloured by its attention magnitude (low=blue, high=red),
    blended at a constant alpha so the whole frame is covered. spotlight=True instead fades low-attn
    patches toward transparent (alpha scales with attention)."""
    a = np.clip(attn_grid / (gmax + 1e-12), 0, 1)
    a_up = np.kron(a, np.ones((patch, patch)))                 # nearest upsample, patch-aligned
    H, W = img.shape[:2]
    a_up = a_up[:H, :W]
    heat = (cmap(a_up)[..., :3] * 255)                         # (H,W,3) full colour field
    al = (a_up * alpha)[..., None] if spotlight else alpha     # flat alpha -> whole image covered
    return (img * (1 - al) + heat * al).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="track_<tag>_K<K> folder")
    ap.add_argument("--input", required=True, help="probe_input images dir")
    ap.add_argument("--img-size", type=int, default=504)
    ap.add_argument("--attn", default="global", choices=["global", "frame"],
                    help="global: global_attn, every frame (cross-view); frame: frame_attn, the query's OWN frame only (intra-view, RoPE)")
    ap.add_argument("--tokens", default="near_tex,flat,far", help="comma list of tracked token names (or 'all')")
    ap.add_argument("--steps", default="all", help="'all' or comma list of step indices")
    ap.add_argument("--frames", default="all", help="'all' or comma list of frame indices")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--cmap", default="hot", help="matplotlib colormap: hot (black->red->yellow->white, intuitive), turbo, inferno, jet")
    ap.add_argument("--spotlight", action="store_true", help="fade low-attention patches to transparent (default: full-coverage heatmap)")
    ap.add_argument("--norm", default="head", choices=["frame", "head", "token"],
                    help="head (default, FAITHFUL across frames): per-(head,step) one scale over all frames -> "
                         "a frame truly attended less stays dim; token: one scale over all heads+frames (compare heads too); "
                         "frame: each frame self-normalised (pattern always visible but NOT true cross-frame magnitude)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    m = np.load(f"{a.dir}/meta.npz", allow_pickle=True)
    ps = int(m["patch_start_idx"]); gh, gw = [int(x) for x in m["grid"]]
    patch = int(m["patch"]); S = int(m["S"])
    names = list(m["tok_name"]); pnames = list(m["patch_names"])
    tgt, val = m["gt_target"], m["gt_valid"]                   # (nP,S,2),(nP,S)
    tag = os.path.basename(a.dir.rstrip("/")).replace("track_", "")
    own = int(m["frame"]) if "frame" in m.files else 0           # the query's own frame (frame_attn target)
    out = a.out or (f"out/attn_frameattn_{tag}" if a.attn == "frame" else f"out/attn_viz_{tag}")
    def _stepnum(p):   # numeric step index from "step_123.npz" (lexicographic sort breaks at >=100)
        return int(os.path.basename(p).split("_")[1].split(".")[0])
    steps = sorted(glob.glob(f"{a.dir}/step_*.npz"), key=_stepnum)

    want = names if a.tokens == "all" else a.tokens.split(",")
    sel_steps = None if a.steps == "all" else set(int(x) for x in a.steps.split(","))
    sel_frames = range(S) if a.frames == "all" else [int(x) for x in a.frames.split(",")]

    cmap = matplotlib.colormaps[a.cmap]
    images = load_images(a.input, a.img_size, S, "cpu")[0]     # (S,3,H,W) in [0,1]
    imgs = (images.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)  # (S,H,W,3)
    H, W = imgs.shape[1:3]

    def draw_marker(im, r, c, color, box=False):
        d = ImageDraw.Draw(im); y, x = r * patch + patch // 2, c * patch + patch // 2
        if box:
            d.rectangle([c * patch, r * patch, (c + 1) * patch, (r + 1) * patch], outline=color, width=2)
        else:
            d.line([x - 6, y, x + 6, y], fill=color, width=2); d.line([x, y - 6, x, y + 6], fill=color, width=2)

    n_png = 0
    for tname in want:
        if tname not in names:
            print(f"  !! unknown token '{tname}', skip"); continue
        ti = names.index(tname)
        pi = pnames.index(tname) if tname in pnames else None   # patch -> has GT target; cam/reg -> None
        for sp in steps:
            si = _stepnum(sp)                 # TRUE step number (output dir + filter key), not enumerate index
            if sel_steps is not None and si not in sel_steps:
                continue
            z = np.load(sp)
            if a.attn == "frame":
                # frame_attn: ONLY the query's own frame; frame_rows[ti] is (12, P)
                fr = z["frame_rows"][ti]                        # (12, 869)
                for lbl, row in [(f"head_{h:02d}", fr[h]) for h in range(fr.shape[0])] + [("mean", fr.mean(0))]:
                    grid = row[ps:].reshape(gh, gw)
                    blended = overlay(imgs[own], grid, grid.max(), patch, cmap, a.alpha, a.spotlight)
                    im = Image.fromarray(blended)
                    if pi is not None:                          # query's own cell (white box)
                        draw_marker(im, int(tgt[pi, own, 0]), int(tgt[pi, own, 1]), (255, 255, 255), box=True)
                    od = f"{out}/{tname}/step_{si:02d}"
                    os.makedirs(od, exist_ok=True)
                    im.save(f"{od}/{lbl}.png"); n_png += 1
                print(f"  {tname} step {si:02d}: frame-attn heads+mean (own frame {own})")
                continue
            g = z["global_rows"][ti]                            # (12, S, 869)
            heads = list(range(g.shape[0]))
            token_max = g[:, :, ps:].max()                      # for --norm token (one scale, all heads+frames)
            for lbl, row in [(f"head_{h:02d}", g[h]) for h in heads] + [("mean", g.mean(0))]:
                head_max = row[:, ps:].max()                    # for --norm head (this head, all frames)
                for j in sel_frames:
                    grid = row[j, ps:].reshape(gh, gw)
                    gmax = {"frame": grid.max(), "head": head_max, "token": token_max}[a.norm]
                    blended = overlay(imgs[j], grid, gmax, patch, cmap, a.alpha, a.spotlight)
                    im = Image.fromarray(blended)
                    # white markers: they must stand out against the rainbow turbo heatmap (red/green/cyan)
                    if pi is not None and val[pi, j]:
                        draw_marker(im, int(tgt[pi, j, 0]), int(tgt[pi, j, 1]), (255, 255, 255))   # GT corresp = white cross
                    if j == 0 and pi is not None:
                        draw_marker(im, int(tgt[pi, 0, 0]), int(tgt[pi, 0, 1]), (255, 255, 255), box=True)  # query = white box
                    od = f"{out}/{tname}/step_{si:02d}/frame_{j:02d}"
                    os.makedirs(od, exist_ok=True)
                    im.save(f"{od}/{lbl}.png")
                    n_png += 1
            print(f"  {tname} step {si:02d}: wrote heads+mean x {len(list(sel_frames))} frames")
    print(f"\nwrote {n_png} PNGs under {out}/")


if __name__ == "__main__":
    main()
