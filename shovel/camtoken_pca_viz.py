#!/usr/bin/env python3
"""camtoken_pca_viz.py -- visualize the 2026-11-10 cam-token PCA finding (re-implements the inline analysis).

Per K, per step: PCA over the 27 frames' cam tokens (centred across frames); linear fit (with bias) of the top-k PC
scores -> the camera centres DVLT itself decodes at that step (mean ray origin). NOT GT; only camera POSITION.

Row per K: (1) R^2 vs step for top-3 / top-6  (2) decoded camera centres, 2-D view  (3) cam tokens' PC1-PC2
(4) top-6 linear fit: predicted vs decoded centre coordinates.  Colour = frame index (same in panels 2-4).

  python shovel/camtoken_pca_viz.py --Ks 16 64 128 --step-plot last --out out/camtoken
"""
import argparse
import glob
import re

import numpy as np


def stepnum(p):
    return int(re.search(r"step_(\d+)", p).group(1))


def pca_scores(V, k):
    Vc = V - V.mean(0)
    U, S, _ = np.linalg.svd(Vc, full_matrices=False)
    return U[:, :k] * S[:k], (S ** 2) / (S ** 2).sum()


def fit(X, Y):
    X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
    beta = np.linalg.lstsq(X1, Y, rcond=None)[0]
    pred = X1 @ beta
    r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    return pred, r2


def centres(path):
    z = np.load(path)
    return z["rays"][..., 3:6].astype(np.float32).reshape(z["rays"].shape[0], -1, 3).mean(1)   # (27,3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--camtok", default="out/camtoken/camtoken_allframes_allK.npz")
    ap.add_argument("--stepdecode", default="out/stepdecode_s10_K{K}")
    ap.add_argument("--step-plot", default="last", help="'last' or an integer step for panels 2-4")
    ap.add_argument("--out", default="out/camtoken")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    cam = np.load(a.camtok)
    fig, ax = plt.subplots(len(a.Ks), 4, figsize=(19, 4.6 * len(a.Ks)))
    fig.subplots_adjust(left=.05, right=.90, wspace=.32, hspace=.38)
    ax = np.atleast_2d(ax)
    for r, K in enumerate(a.Ks):
        camT = cam[f"K{K}"]                                    # (K+1, 27, 768): row s = after s steps
        files = {stepnum(f): f for f in glob.glob(a.stepdecode.format(K=K) + "/step_*.npz")}
        steps = sorted(s for s in files if s < camT.shape[0])
        r3, r6 = [], []
        for s in steps:
            pos = centres(files[s]); V = camT[s].astype(np.float32)
            r3.append(fit(pca_scores(V, 3)[0], pos)[1]); r6.append(fit(pca_scores(V, 6)[0], pos)[1])
        ax[r, 0].plot(steps, r3, "o-", ms=3, label="top-3 PC")
        ax[r, 0].plot(steps, r6, "s-", ms=3, label="top-6 PC")
        ax[r, 0].set_ylim(0, 1.02); ax[r, 0].grid(alpha=.3); ax[r, 0].legend(fontsize=8)
        ax[r, 0].set_title(f"K={K}: R²(PCs → decoded centre)", fontsize=10); ax[r, 0].set_xlabel("step")

        s = steps[-1] if a.step_plot == "last" else min(steps, key=lambda q: abs(q - int(a.step_plot)))
        pos = centres(files[s]); V = camT[s].astype(np.float32)
        sc6, evr = pca_scores(V, 6)
        pred, r2 = fit(sc6, pos)
        col = np.arange(len(pos))
        # 2-D view of decoded centres: their own two principal axes
        Pc = pos - pos.mean(0); _, _, Wt = np.linalg.svd(Pc, full_matrices=False); p2 = Pc @ Wt[:2].T
        ax[r, 0].axvline(s, color="k", ls=":", lw=.8)
        ax[r, 1].plot(p2[:, 0], p2[:, 1], "-", color="0.7", lw=.8)
        sc = ax[r, 1].scatter(p2[:, 0], p2[:, 1], c=col, cmap="viridis", s=40, edgecolors="k", linewidths=.4)
        for i in (0, len(pos) - 1): ax[r, 1].annotate(f"f{i}", p2[i], fontsize=8)
        ax[r, 1].set_title(f"K={K} step {s}: decoded centres", fontsize=10); ax[r, 1].set_aspect("equal", "datalim")
        ax[r, 2].plot(sc6[:, 0], sc6[:, 1], "-", color="0.7", lw=.8)
        ax[r, 2].scatter(sc6[:, 0], sc6[:, 1], c=col, cmap="viridis", s=40, edgecolors="k", linewidths=.4)
        for i in (0, len(pos) - 1): ax[r, 2].annotate(f"f{i}", sc6[i, :2], fontsize=8)
        ax[r, 2].set_title(f"cam token PC1/PC2 (var {evr[0]:.2f}/{evr[1]:.2f})", fontsize=10)
        lo, hi = min(pos.min(), pred.min()), max(pos.max(), pred.max())
        for d, m in zip(range(3), ["o", "s", "^"]):
            ax[r, 3].scatter(pos[:, d], pred[:, d], c=col, cmap="viridis", s=26, marker=m, edgecolors="k", linewidths=.3, label="xyz"[d])
        ax[r, 3].plot([lo, hi], [lo, hi], "k--", lw=.8)
        ax[r, 3].set_title(f"top-6 fit: R² = {r2:.3f}", fontsize=10)
        ax[r, 3].set_xlabel("decoded centre coordinate"); ax[r, 3].set_ylabel("predicted from 6 PCs"); ax[r, 3].legend(fontsize=8)
        print(f"K={K} step {s}: R2(top3)={r3[steps.index(s)]:.3f} R2(top6)={r2:.3f}")
    cax = fig.add_axes([.92, .25, .012, .5]); fig.colorbar(sc, cax=cax, label="frame index")
    fig.suptitle("cam token PCA vs camera centre decoded by DVLT itself (s10, 27 frames; NOT GT, position only)", fontsize=12)
    out = f"{a.out}/pca_vs_camcentre.png"
    fig.savefig(out, dpi=115); print("saved", out)


if __name__ == "__main__":
    main()
