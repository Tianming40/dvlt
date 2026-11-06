#!/usr/bin/env python3
"""PCA of a tracked token's STATE trajectory across loop steps (from track_token's dump).

Measures how low-dimensional the 768-dim journey is (variance per PC + effective dimension) and
plots the trajectory in its own top-2 PC plane. With several tokens it fits ONE joint PCA so their
paths are comparable in a common frame.

  PYTHONPATH=$(pwd) python shovel/pca_trace.py --dir out/track_s10_K16 --tokens near_tex
  #   all 8 tokens, joint PCA:   --tokens all
  #   include the per-sub-block checkpoints (z_frame too): --points all
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def trajectory(d, ti, points):
    """Return (labels, Z) for one tracked token. points='out' -> z0 + z_out per step;
    'all' -> z0, then (z_frame, z_out) per step."""
    m = np.load(f"{d}/meta.npz", allow_pickle=True)
    z0 = m["z0"][ti].astype(np.float64)
    steps = sorted(glob.glob(f"{d}/step_*.npz"), key=_snum)
    Z = [z0]; lab = ["z0"]
    for k, sp in enumerate(steps):
        z = np.load(sp)
        if points == "all":
            Z.append(z["z_frame"][ti].astype(np.float64)); lab.append(f"{k}f")
        Z.append(z["z_out"][ti].astype(np.float64)); lab.append(f"{k}")
    return lab, np.stack(Z)


def pca(Z):
    mu = Z.mean(0); Zc = Z - mu
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    eff = (S ** 2).sum() ** 2 / (S ** 4).sum()
    coords = Zc @ Vt.T                                  # (n, k) PC coordinates
    return coords, var, eff, Vt, mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="track_<tag>_K<K> folder")
    ap.add_argument("--tokens", default="near_tex", help="comma list of tracked token names, or 'all'")
    ap.add_argument("--points", default="out", choices=["out", "all"], help="out: z0+z_out/step; all: also z_frame")
    ap.add_argument("--layout", default="overlay", choices=["overlay", "grid"],
                    help="overlay: all trajectories on one PC1-PC2 plot (they can overlap); grid: one subplot per token (shared axes)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = np.load(f"{a.dir}/meta.npz", allow_pickle=True)
    names = list(m["tok_name"])
    want = names if a.tokens == "all" else a.tokens.split(",")
    tis = [names.index(t) for t in want if t in names]
    outdir = a.out or f"{a.dir}/pca"; os.makedirs(outdir, exist_ok=True)

    # gather all trajectories, fit ONE joint PCA (comparable frame)
    all_lab, segs, Zlist, off = [], [], [], 0
    for t, ti in zip(want, tis):
        lab, Z = trajectory(a.dir, ti, a.points)
        segs.append((t, off, len(Z))); Zlist.append(Z); all_lab.append(lab); off += len(Z)
    Zall = np.concatenate(Zlist)
    coords, var, eff, Vt, mu = pca(Zall)

    print(f"joint PCA over {len(tis)} token(s), {Zall.shape[0]} states, {Zall.shape[1]}-dim:")
    print(f"  variance PC1..6: {np.round(var[:6], 3).tolist()}")
    print(f"  cumulative     : {np.round(np.cumsum(var[:6]), 3).tolist()}")
    print(f"  effective dimension = {eff:.2f}  (out of {Zall.shape[1]})")

    tag = os.path.basename(a.dir.rstrip("/")).replace("track_", "")
    cmap = plt.cm.tab10
    sfx = '_'.join(want) if len(want) <= 3 else "all"
    if a.layout == "grid" and len(segs) > 1:
        # one subplot per token (shared PC1-PC2 axes so they are comparable, but never overlap)
        xlim = (coords[:, 0].min(), coords[:, 0].max()); ylim = (coords[:, 1].min(), coords[:, 1].max())
        ncol = min(4, len(segs)); nrow = (len(segs) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow), squeeze=False)
        for i, (t, s0, n) in enumerate(segs):
            ax = axes[i // ncol][i % ncol]; c = coords[s0:s0 + n]
            ax.plot(c[:, 0], c[:, 1], "-", color="gray", alpha=0.5, zorder=1)
            sc = ax.scatter(c[:, 0], c[:, 1], c=np.arange(n), cmap="viridis", s=22, zorder=3)
            ax.annotate("s", c[0, :2], fontsize=8, color="k"); ax.annotate("e", c[-1, :2], fontsize=8, color="r")
            ax.set_title(f"{t}  (|z|:{np.linalg.norm(Zall[s0]):.0f}→{np.linalg.norm(Zall[s0+n-1]):.0f})", fontsize=9)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.grid(alpha=.3); ax.set_xticks([]); ax.set_yticks([])
        for j in range(len(segs), nrow * ncol): axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"PCA token trajectories (shared PC1={var[0]:.0%}/PC2={var[1]:.0%} axes, eff dim {eff:.2f}) — {tag}")
        fig.tight_layout(); png = f"{outdir}/pca_{sfx}_grid.png"
    else:
        fig, ax2 = plt.subplots(1, 2, figsize=(13, 5.5))
        for i, (t, s0, n) in enumerate(segs):
            c = coords[s0:s0 + n]
            ax2[0].plot(c[:, 0], c[:, 1], "-", color=cmap(i), alpha=0.5)
            sc = ax2[0].scatter(c[:, 0], c[:, 1], c=np.arange(n), cmap="viridis", s=30, zorder=3,
                                edgecolors=cmap(i), linewidths=1.2, label=t)
            ax2[0].annotate("start", c[0, :2], fontsize=8); ax2[0].annotate("end", c[-1, :2], fontsize=8)
        ax2[0].set_xlabel(f"PC1 ({var[0]:.0%})"); ax2[0].set_ylabel(f"PC2 ({var[1]:.0%})")
        ax2[0].set_title(f"state trajectory in PC1-PC2  (eff dim = {eff:.2f})"); ax2[0].legend(fontsize=8); ax2[0].grid(alpha=.3)
        fig.colorbar(sc, ax=ax2[0], label="step")
        ax2[1].bar(range(1, 11), var[:10]); ax2[1].plot(range(1, 11), np.cumsum(var[:10]), "-o", color="C3")
        ax2[1].set_xlabel("PC"); ax2[1].set_ylabel("variance explained"); ax2[1].set_title("scree (bars) + cumulative (line)")
        ax2[1].grid(alpha=.3)
        fig.suptitle(f"PCA of token trajectory — {tag} — tokens: {','.join(want)}")
        png = f"{outdir}/pca_{sfx}.png"
    fig.tight_layout(); fig.savefig(png, dpi=130); print(f"saved {png}")

    json.dump({"tokens": want, "points": a.points, "var": var[:10].tolist(), "eff_dim": float(eff),
               "coords": {t: coords[s0:s0 + n].tolist() for t, s0, n in segs}},
              open(png.replace(".png", ".json"), "w"), indent=1)
    print(f"saved {png.replace('.png', '.json')}")


if __name__ == "__main__":
    main()
