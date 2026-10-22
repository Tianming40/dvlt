"""
make_dashboard.py -- assemble the 20-frame probe experiments into ONE comparison figure.

Reads out/probe20_* produced by run_probe20.sh and draws a single dashboard:
  [traj: AbsRel vs GT per step]  [ksweep: final AbsRel vs GT per K]
  [perturb: GT AbsRel clean vs perturbed]  [photo: photometric residual along loop]

  python make_dashboard.py --out-root out --fig out/dashboard20.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def exists(p):
    return os.path.isfile(p)


def panel_traj(ax, root):
    """AbsRel vs GT, per step, one line per K (from traj.json gt fields)."""
    p = f"{root}/probe20_traj/traj.json"
    if not exists(p):
        ax.set_title("traj (missing)"); return
    d = load(p)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(d)))
    for (K, rows), c in zip(d.items(), cmap):
        steps = [r["step"] + 1 for r in rows]
        ab = [r.get("gt", {}).get("absrel", np.nan) if r.get("gt") else np.nan for r in rows]
        ax.plot(steps, ab, "-o", ms=3, color=c, label=K)
    ax.set_title("traj: depth AbsRel vs GT, per step")
    ax.set_xlabel("step"); ax.set_ylabel("AbsRel"); ax.set_yscale("log")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)


def panel_ksweep(ax, root):
    """Final AbsRel vs GT as a function of K (the real V-curve against ground truth)."""
    p = f"{root}/probe20_ksweep/ksweep.json"
    if not exists(p):
        ax.set_title("ksweep (missing)"); return
    rows = load(p)
    K = [r["K"] for r in rows]
    ab = [r.get("gt", {}).get("absrel", np.nan) if r.get("gt") else np.nan for r in rows]
    ax.plot(K, ab, "-o", color="#c0392b", label="AbsRel vs GT")
    ax.axvline(16, color="grey", ls="--", lw=1, alpha=0.6, label="trained K=16")
    ax.set_title("ksweep: final AbsRel vs GT per K")
    ax.set_xlabel("inference steps K"); ax.set_ylabel("AbsRel")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)


def panel_perturb(ax, root):
    """Clean vs perturbed GT AbsRel (final), grouped by perturbation config."""
    labels, clean, final = [], [], []
    for kind in ("gauss", "swap"):
        p = f"{root}/probe20_perturb_{kind}/perturb_{kind}.json"
        if not exists(p):
            continue
        for r in load(p):
            gc = (r.get("gt_clean") or {}).get("absrel", np.nan)
            gf = (r.get("gt_final") or {}).get("absrel", np.nan)
            labels.append(f"{kind}\n@{r['at']} e{r['eps']}")
            clean.append(gc); final.append(gf)
    if not labels:
        ax.set_title("perturb (missing)"); return
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, clean, w, label="clean final", color="#48c")
    ax.bar(x + w / 2, final, w, label="perturbed final", color="#c44")
    ax.set_title("perturb: GT AbsRel, clean vs perturbed")
    ax.set_ylabel("AbsRel"); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)


def panel_photo(ax, root):
    """Photometric residual + GT AbsRel along the standard loop."""
    p = f"{root}/probe20_photo/photo.json"
    if not exists(p):
        ax.set_title("photo (missing)"); return
    d = load(p)
    mon = d.get("monitor") or []
    if not mon:
        ax.set_title("photo (no monitor)"); return
    steps = [m[0] for m in mon]
    Lphoto = [m[1] for m in mon]
    ax.plot(steps, Lphoto, "-o", ms=3, color="#c0392b", label="L_photo")
    ax.set_xlabel("step"); ax.set_ylabel("photometric residual", color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b")
    # overlay GT AbsRel if present (4th tuple element)
    ab = [(m[3] or {}).get("absrel", np.nan) if len(m) > 3 and m[3] else np.nan for m in mon]
    if np.isfinite(ab).any():
        ax2 = ax.twinx()
        ax2.plot(steps, ab, "-s", ms=3, color="#2c6", label="AbsRel vs GT")
        ax2.set_ylabel("AbsRel vs GT", color="#2c6"); ax2.tick_params(axis="y", labelcolor="#2c6")
    ax.set_title("photo: residual along the loop")
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="out")
    ap.add_argument("--fig", default="out/dashboard20.png")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("DVLT probe @ 20 frames -- comparison dashboard (vs ground truth)",
                 fontsize=15, weight="bold")
    panel_traj(ax[0, 0], a.out_root)
    panel_ksweep(ax[0, 1], a.out_root)
    panel_perturb(ax[1, 0], a.out_root)
    panel_photo(ax[1, 1], a.out_root)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(a.fig, dpi=130)
    print(f"saved {a.fig}")


if __name__ == "__main__":
    main()
