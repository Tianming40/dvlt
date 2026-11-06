#!/usr/bin/env python3
"""Per-pixel convergence & error per frame, across loop steps (from step_decode dump).
Where does the decoded depth converge, and where does its final error concentrate (edges vs flat)?
Runs one or all frames into a folder + a cross-frame summary.

  PYTHONPATH=$(pwd) python shovel/perpixel_conv.py --dir out/stepdecode_s10_K16 \
      --gt out/gt_00777c41d4_s10/gt.npz --input out/gt_00777c41d4_s10/probe_input --frames all
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import load_gt, load_images


def one_frame(j, Dsteps, gd_all, imgs, thr, outdir, K):
    gd = gd_all[j]; v = gd > 1e-6
    if v.sum() < 50:
        return None
    img = (imgs[j].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    H, W = gd.shape
    D = np.empty((K, H, W))
    for k in range(K):
        dep = Dsteps[k][j]
        D[k] = dep * (np.median(gd[v]) / np.median(dep[v]))     # per-frame global median scale align
    final = D[-1]
    absrel_final = np.where(v, np.abs(final - gd) / np.maximum(gd, 1e-6), np.nan)
    rel = np.abs(D - final[None]) / np.maximum(np.abs(final[None]), 1e-6)
    conv = np.full((H, W), K, dtype=float)
    for k in range(K - 1, -1, -1):
        conv[(rel[k:] < thr).all(0)] = k + 1
    conv = np.where(v, conv, np.nan)
    gy, gx = np.gradient(np.where(v, gd, np.nan)); edge = np.hypot(gy, gx)

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    ax[0, 0].imshow(img); ax[0, 0].set_title(f"frame {j} RGB")
    im = ax[0, 1].imshow(np.where(v, gd, np.nan), cmap="turbo"); ax[0, 1].set_title("GT depth (m)"); fig.colorbar(im, ax=ax[0, 1], fraction=.046)
    im = ax[0, 2].imshow(edge, cmap="inferno", vmax=np.nanpercentile(edge, 98)); ax[0, 2].set_title("GT depth edges |∇depth|"); fig.colorbar(im, ax=ax[0, 2], fraction=.046)
    im = ax[1, 0].imshow(absrel_final, cmap="hot", vmax=np.nanpercentile(absrel_final, 95)); ax[1, 0].set_title("FINAL error vs GT (absrel)"); fig.colorbar(im, ax=ax[1, 0], fraction=.046)
    im = ax[1, 1].imshow(conv, cmap="viridis"); ax[1, 1].set_title(f"convergence step (≤{thr:.0%} of final)"); fig.colorbar(im, ax=ax[1, 1], fraction=.046)
    m = v & np.isfinite(absrel_final) & np.isfinite(edge)
    sc = ax[1, 2].scatter(edge[m][::20], absrel_final[m][::20], c=conv[m][::20], s=2, cmap="viridis", alpha=.4)
    ax[1, 2].set_xlabel("GT depth edge"); ax[1, 2].set_ylabel("final absrel"); ax[1, 2].set_yscale("log"); ax[1, 2].set_title("error vs edge (color=conv step)"); fig.colorbar(sc, ax=ax[1, 2], fraction=.046, label="conv step")
    for a_ in [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1]]: a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"Per-pixel convergence & error — frame {j}")
    fig.tight_layout(); fig.savefig(f"{outdir}/frame_{j:02d}.png", dpi=110); plt.close(fig)

    ed, er, cv = edge[m], absrel_final[m], conv[m]
    hi = ed > np.percentile(ed, 90); lo = ed < np.percentile(ed, 50)
    return dict(frame=j, edge_absrel=float(er[hi].mean()), flat_absrel=float(er[lo].mean()),
                edge_conv=float(cv[hi].mean()), flat_conv=float(cv[lo].mean()),
                corr_edge_err=float(np.corrcoef(ed, er)[0, 1]), corr_edge_conv=float(np.corrcoef(ed, cv)[0, 1]),
                overall_absrel=float(er.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--frames", default="all", help="'all' or comma list")
    ap.add_argument("--img-size", type=int, default=504)
    ap.add_argument("--conv-thr", type=float, default=0.02)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    tag = os.path.basename(a.dir.rstrip("/")).replace("stepdecode_", "")
    outdir = a.out_dir or f"out/track_{tag}/perpixel"
    os.makedirs(outdir, exist_ok=True)

    gd_all = load_gt(a.gt, a.img_size, device="cpu")[0].numpy()   # (S,H,W)
    S = gd_all.shape[0]
    imgs = load_images(a.input, a.img_size, S, "cpu")[0]
    steps = sorted(glob.glob(f"{a.dir}/step_*.npz"), key=_snum); K = len(steps)
    print(f"loading {K} step depths ...")
    Dsteps = [np.load(sp)["depth"][:, :, :, 0].astype(np.float64) for sp in steps]   # list of (S,H,W)
    frames = range(S) if a.frames == "all" else [int(x) for x in a.frames.split(",")]

    rows = []
    for j in frames:
        r = one_frame(j, Dsteps, gd_all, imgs, a.conv_thr, outdir, K)
        if r:
            rows.append(r)
            print(f"frame {j:>2}: edge={r['edge_absrel']:.4f} flat={r['flat_absrel']:.4f} "
                  f"({r['edge_absrel']/max(r['flat_absrel'],1e-9):.1f}x)  corr(edge,err)={r['corr_edge_err']:+.3f} "
                  f"conv edge/flat={r['edge_conv']:.1f}/{r['flat_conv']:.1f}")

    json.dump(rows, open(f"{outdir}/summary.json", "w"), indent=1)
    # cross-frame summary figure
    fr = [r["frame"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(fr, [r["edge_absrel"] for r in rows], "-o", ms=3, label="edge (top10%)")
    ax[0].plot(fr, [r["flat_absrel"] for r in rows], "-o", ms=3, label="flat (bot50%)")
    ax[0].plot(fr, [r["overall_absrel"] for r in rows], "-o", ms=3, label="overall")
    ax[0].set_title("final absrel per frame"); ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
    ax[1].plot(fr, [r["corr_edge_err"] for r in rows], "-o", ms=3, label="corr(edge, err)")
    ax[1].plot(fr, [r["corr_edge_conv"] for r in rows], "-o", ms=3, label="corr(edge, conv)")
    ax[1].set_title("edge correlations per frame"); ax[1].axhline(0, color="k", lw=.5); ax[1].legend(fontsize=8)
    ax[2].plot(fr, [r["edge_conv"] for r in rows], "-o", ms=3, label="edge conv step")
    ax[2].plot(fr, [r["flat_conv"] for r in rows], "-o", ms=3, label="flat conv step")
    ax[2].set_title("convergence step per frame"); ax[2].legend(fontsize=8)
    for a_ in ax: a_.set_xlabel("frame"); a_.grid(alpha=.3)
    e = np.mean([r["edge_absrel"] for r in rows]); f = np.mean([r["flat_absrel"] for r in rows]); c = np.mean([r["corr_edge_err"] for r in rows])
    fig.suptitle(f"Cross-frame summary ({len(rows)} frames) — edge {e:.3f} vs flat {f:.3f} ({e/f:.1f}x), mean corr(edge,err)={c:+.3f}")
    fig.tight_layout(); fig.savefig(f"{outdir}/summary.png", dpi=120)
    print(f"\n=== {len(rows)} frames ===  mean edge absrel={e:.4f}  flat={f:.4f}  ({e/f:.1f}x)  mean corr(edge,err)={c:+.3f}")
    print(f"saved {outdir}/  (per-frame PNGs + summary.png + summary.json)")


if __name__ == "__main__":
    main()
