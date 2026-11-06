#!/usr/bin/env python3
"""Stitch the per-step attention overlays (from viz_attention.py) into a video: fix a patch and a
frame, play the K loop steps left-to-right. Two modes:
  --head 09 (or mean)  -> single-head video (one overlay per step)
  --head grid (default) -> montage of all 12 heads + mean, each playing the K steps together

Reads  <dir>/<patch>/step_<kk>/frame_<jj>/head_<hh>.png  and writes an mp4 (via the ffmpeg binary;
falls back to an animated GIF). A step label is burned into each video frame.

  PYTHONPATH=$(pwd) python shovel/attn_video.py --dir out/attn_viz_s10_K16 \
      --patch near_tex --frame 0 --head grid --fps 3
"""
import argparse, glob, os, subprocess, shutil, tempfile
from PIL import Image, ImageDraw, ImageFont


def label(im, text, xy=(6, 4), fg=(255, 255, 255), bg=(0, 0, 0)):
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        f = ImageFont.load_default()
    x, y = xy
    bb = d.textbbox((x, y), text, font=f)
    d.rectangle([bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2], fill=bg)
    d.text((x, y), text, fill=fg, font=f)
    return im


def montage(paths_labels, cols, cell):
    """paths_labels: list of (png_path, caption). Returns one PIL grid image."""
    cw, ch = cell
    rows = (len(paths_labels) + cols - 1) // cols
    cap_h = 22
    canvas = Image.new("RGB", (cols * cw, rows * (ch + cap_h)), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        f = ImageFont.load_default()
    for i, (p, cap) in enumerate(paths_labels):
        r, c = divmod(i, cols)
        x, y = c * cw, r * (ch + cap_h)
        if p and os.path.exists(p):
            canvas.paste(Image.open(p).convert("RGB").resize((cw, ch)), (x, y + cap_h))
        d.text((x + 4, y + 2), cap, fill=(255, 255, 255), font=f)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="attn_viz_<tag>_K<K> folder")
    ap.add_argument("--patch", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--head", default="grid", help="'grid' (all heads+mean), or 'NN' / 'mean' for a single panel")
    ap.add_argument("--attn", default="global", choices=["global", "frame"],
                    help="global: <patch>/step/frame_jj/head.png; frame: <patch>/step/head.png (no frame subdir)")
    ap.add_argument("--fps", type=float, default=3)
    ap.add_argument("--cols", type=int, default=4, help="grid columns")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = f"{a.dir}/{a.patch}"
    steps = sorted(glob.glob(f"{base}/step_*"))
    assert steps, f"no steps under {base}"
    K = len(steps)
    fj = "" if a.attn == "frame" else f"frame_{a.frame:02d}"
    outdir = f"{a.dir}/videos"; os.makedirs(outdir, exist_ok=True)
    tagname = f"frameattn_{a.head}" if a.attn == "frame" else f"frame{a.frame:02d}_{a.head}"
    out = a.out or f"{outdir}/{a.patch}_{tagname}.mp4"

    tmp = tempfile.mkdtemp()
    nframes = 0
    for k in range(K):
        sd = f"{base}/step_{k:02d}/{fj}".rstrip("/")
        where = "frame-attn (own frame)" if a.attn == "frame" else f"frame {a.frame}"
        title = f"{a.patch}  {where}   |   step {k+1}/{K}"
        if a.head == "grid":
            panels = [(f"{sd}/head_{h:02d}.png", f"h{h}") for h in range(12)] + [(f"{sd}/mean.png", "mean")]
            grid = montage(panels, a.cols, (320, 213))
            hdr = 34                                            # header band so the title never covers h0
            frame = Image.new("RGB", (grid.width, grid.height + hdr), (0, 0, 0))
            frame.paste(grid, (0, hdr))
            label(frame, title, xy=(8, 5))
        else:
            name = "mean" if a.head == "mean" else f"head_{int(a.head):02d}"
            p = f"{sd}/{name}.png"
            if not os.path.exists(p):
                print(f"  missing {p}, skip"); continue
            frame = Image.open(p).convert("RGB")
            label(frame, f"{title}   {name}", xy=(8, 6))
        # even dimensions for yuv420p
        w, h = frame.size
        if w % 2 or h % 2:
            frame = frame.crop((0, 0, w - (w % 2), h - (h % 2)))
        frame.save(f"{tmp}/{nframes:04d}.png")
        nframes += 1

    ff = shutil.which("ffmpeg")
    if ff:
        cmd = [ff, "-y", "-framerate", str(a.fps), "-i", f"{tmp}/%04d.png",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-800:]); out = out.rsplit(".", 1)[0] + ".gif"; ff = None
    if not ff:  # GIF fallback
        frames = [Image.open(f"{tmp}/{i:04d}.png") for i in range(nframes)]
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / a.fps), loop=0)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {out}  ({nframes} frames @ {a.fps} fps, mode={a.head})")


if __name__ == "__main__":
    main()
