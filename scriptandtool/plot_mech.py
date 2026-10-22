#!/usr/bin/env python3
"""Mechanism-intervention figure (renorm / ablate / photo) from saved grand data, with baseline + GT.

Writes out/full_grand/mech/mech.png: a 6-panel figure (gt_absrel, gt_ate, ray_to_std, depth_to_std,
tok_norm, tok_rel_to_clean) overlaying renorm, ablate:{first_half,second_half,all}, photo-guided
against the K=16 baseline (black dashed) and GT (=0 on error panels). No model/GPU; reads all.tsv.

  python scriptandtool/plot_mech.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PANELS = ["gt_absrel", "gt_ate", "ray_to_std", "depth_to_std", "tok_norm", "tok_rel_to_clean"]
GT_ZERO = {"gt_absrel", "gt_ate", "ray_to_std", "depth_to_std"}
VERSIONS = {
    "renorm":             "#e377c2",
    "ablate:first_half":  "#17becf",
    "ablate:second_half": "#bcbd22",
    "ablate:all":         "#7f7f7f",
    "photo-guided":       "#ff9900",
}


def main():
    df = pd.read_csv("out/full_grand/all.tsv", sep="\t")
    out = "out/full_grand/mech"; os.makedirs(out, exist_ok=True)
    base = df[df.version == "baseline"].sort_values("step")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.ravel(), PANELS):
        ax.plot(base.step, base[col], "k--", lw=2.8, zorder=20, label="baseline K=16")
        if col in GT_ZERO:
            ax.axhline(0.0, color="#888", lw=1.1, zorder=19)
        for v, c in VERSIONS.items():
            d = df[df.version == v].sort_values("step")
            if len(d) == 0:
                continue
            ax.plot(d.step, d[col], color=c, lw=2.0, alpha=0.95, marker="o", ms=3, label=v, zorder=6)
        ax.set_title(col); ax.set_xlabel("loop step"); ax.grid(alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("mechanism interventions: renorm / ablate(global attn) / photo  vs  baseline + GT", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(os.path.join(out, "mech.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}/mech.png")


if __name__ == "__main__":
    main()
