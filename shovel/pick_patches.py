#!/usr/bin/env python3
"""Pick 3 representative patches on frame 0 (for the single-token tracking experiment) and
draw them as 3 coloured boxes on the RGB + GT-depth images.

Uses the SAME preprocessing as the model (_common.load_images / load_gt) so the 14x14 patch
grid (36x36 at img_size 504) lines up EXACTLY with the token indices the loop will see.

Selection (frame 0 only), each patch = one 14x14 cell:
  1) RED    high-texture + near   : among near patches (depth < median), the max image-gradient cell
  2) GREEN  low-texture / flat     : min image-gradient cell (with enough valid depth)
  3) BLUE   far / background        : the farthest (max mean-depth) valid cell
Patches are forced to be spatially distinct (>= --min-sep cells apart).

Token index of a picked (row,col) in frame 0:  idx = patch_start_idx(5) + row*GW + col

  PYTHONPATH=$(pwd) python shovel/pick_patches.py \
      --input out/gt_00777c41d4_s10/probe_input --gt out/gt_00777c41d4_s10/gt.npz \
      --out out/gt_00777c41d4_s10/patch_pick_frame0.png
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import load_images, load_gt

PATCH = 14
PATCH_START_IDX = 5        # 1 cam + 4 register, per frame (model.py:287)
COLORS = {"near_tex": "red", "flat": "lime", "far": "deepskyblue"}
LABELS = {"near_tex": "1 high-texture+near", "flat": "2 low-texture/flat", "far": "3 far/background"}


def patch_stats(img_gray, depth, gh, gw):
    """Per-patch mean image-gradient (texture) and mean valid GT depth + valid fraction."""
    gy, gx = np.gradient(img_gray.astype(np.float32))
    grad = np.sqrt(gy ** 2 + gx ** 2)
    tex = np.zeros((gh, gw), np.float32)
    dep = np.full((gh, gw), np.nan, np.float32)
    valid = np.zeros((gh, gw), np.float32)
    for r in range(gh):
        for c in range(gw):
            sl = (slice(r * PATCH, (r + 1) * PATCH), slice(c * PATCH, (c + 1) * PATCH))
            tex[r, c] = grad[sl].mean()
            dp = depth[sl]
            m = dp > 1e-6
            valid[r, c] = m.mean()
            if m.any():
                dep[r, c] = dp[m].mean()
    return tex, dep, valid


def far_enough(rc, chosen, min_sep):
    return all(abs(rc[0] - r) + abs(rc[1] - c) >= min_sep for r, c in chosen)


def pick(tex, dep, valid, min_sep, vmin=0.5, border=0, near_region=None):
    gh, gw = tex.shape
    ok = valid >= vmin
    if border > 0:                                   # drop the outer `border` ring of cells
        edge = np.ones((gh, gw), bool)
        edge[border:gh - border, border:gw - border] = False
        ok &= ~edge
    depmed = np.nanmedian(dep[ok])
    chosen = {}

    # 1) high-texture + near: near half by depth, among those the strongest gradient
    near = ok & (dep < depmed)
    if near_region is not None:                      # restrict patch-1 search to a (r0,r1,c0,c1) box
        r0, r1, c0, c1 = near_region
        box = np.zeros((gh, gw), bool); box[r0:r1, c0:c1] = True
        near = near & box
    order = np.argsort(-tex[near].ravel())
    rr, cc = np.where(near)
    for idx in order:
        rc = (int(rr[idx]), int(cc[idx]))
        if far_enough(rc, chosen.values(), min_sep):
            chosen["near_tex"] = rc
            break

    # 3) far / background: largest valid mean-depth
    rr, cc = np.where(ok)
    for idx in np.argsort(-dep[ok].ravel()):
        rc = (int(rr[idx]), int(cc[idx]))
        if far_enough(rc, chosen.values(), min_sep):
            chosen["far"] = rc
            break

    # 2) low-texture / flat: weakest gradient (any valid depth)
    for idx in np.argsort(tex[ok].ravel()):
        rc = (int(rr[idx]), int(cc[idx]))
        if far_enough(rc, chosen.values(), min_sep):
            chosen["flat"] = rc
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--img-size", type=int, default=504)
    ap.add_argument("--frame", type=int, default=0, help="which frame to pick patches on (anchor)")
    ap.add_argument("--min-sep", type=int, default=6, help="min L1 distance (in patch cells) between picks")
    ap.add_argument("--border-margin", type=int, default=0, help="exclude the outer N-cell ring (keeps picks off the edges)")
    ap.add_argument("--near-region", default=None, help="restrict the high-texture+near pick to a box 'r0,r1,c0,c1' (cells)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    near_region = tuple(int(v) for v in a.near_region.split(",")) if a.near_region else None
    out = a.out or os.path.join(os.path.dirname(a.gt), f"patch_pick_frame{a.frame}.png")

    images = load_images(a.input, a.img_size, a.frame + 1, "cpu")  # (1,S,3,H,W)
    gd, _, _ = load_gt(a.gt, a.img_size, device="cpu")             # (S,H,W)
    img = images[0, a.frame].permute(1, 2, 0).numpy()              # (H,W,3) in [0,1]
    depth = gd[a.frame].numpy()                                    # (H,W)
    H, W = img.shape[:2]
    gh, gw = H // PATCH, W // PATCH
    gray = img.mean(-1)

    tex, dep, valid = patch_stats(gray, depth, gh, gw)
    chosen = pick(tex, dep, valid, a.min_sep, border=a.border_margin, near_region=near_region)

    # --- report ---
    print(f"frame {a.frame}: image {H}x{W}  patch grid {gh}x{gw} (={gh*gw} patches)  valid depth "
          f"{(valid>=0.5).mean():.0%} of cells")
    rec = {}
    for key in ("near_tex", "flat", "far"):
        if key not in chosen:
            print(f"  !! {key}: no cell satisfied the criteria"); continue
        r, c = chosen[key]
        idx = PATCH_START_IDX + r * gw + c
        rec[key] = dict(row=r, col=c, token_index=idx, depth_m=float(dep[r, c]),
                        grad=float(tex[r, c]), valid_frac=float(valid[r, c]))
        print(f"  {LABELS[key]:<22} (row={r:>2}, col={c:>2})  token_idx={idx:>4}  "
              f"depth={dep[r,c]:5.2f}m  grad={tex[r,c]:.4f}")

    # --- figure: RGB + depth, each with the 3 boxes ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(img); ax[0].set_title(f"frame {a.frame} RGB")
    dm = np.where(depth > 1e-6, depth, np.nan)
    im = ax[1].imshow(dm, cmap="turbo"); ax[1].set_title(f"frame {a.frame} GT depth (m)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    for A in ax:
        for key, (r, c) in chosen.items():
            A.add_patch(Rectangle((c * PATCH, r * PATCH), PATCH, PATCH, linewidth=2.5,
                                  edgecolor=COLORS[key], facecolor="none"))
            A.text(c * PATCH, r * PATCH - 3, LABELS[key].split()[0], color=COLORS[key],
                   fontsize=11, fontweight="bold")
        A.set_xticks([]); A.set_yticks([])
    handles = [Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in LABELS]
    fig.legend(handles, LABELS.values(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"3 tracked patches on frame {a.frame}  (+ cam[0] + 4 register[0] = 8 tracked tokens)")
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")

    # --- save picks for reproducibility / the tracking script ---
    js = out.rsplit(".", 1)[0] + ".json"
    json.dump(dict(frame=a.frame, img_hw=[H, W], grid=[gh, gw], patch=PATCH,
                   patch_start_idx=PATCH_START_IDX, patches=rec), open(js, "w"), indent=2)
    print(f"saved {js}  (token indices for shovel/track_token.py)")


if __name__ == "__main__":
    main()
