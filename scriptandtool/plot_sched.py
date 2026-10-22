#!/usr/bin/env python3
"""Time-schedule figures from saved grand data, overlaid with the two anchors (baseline + GT).

Writes to out/full_grand/sched/:
  sched_all.png   all schedules incl. frozen-256 (x to 256; shows the divergence)
  sched_le16.png  frozen excluded (x <= 16; the clean "local tweak harmless vs global reorder hurts")
Each is a 6-panel figure: gt_absrel, gt_ate, tok_norm, depth_to_std, eff_chan, cluster_cos.
No model/GPU; reads all.tsv.

  python scriptandtool/plot_sched.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

PANELS = ["gt_absrel", "gt_ate", "tok_norm", "depth_to_std", "eff_chan", "cluster_cos"]
GT_ZERO = {"gt_absrel", "gt_ate", "depth_to_std"}   # 0 = matches the K=16 standard / GT


def draw(df, base, scheds, out_path, title, logy_norm=False):
    cols = cm.tab10(np.linspace(0, 1, 10))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.ravel(), PANELS):
        ax.plot(base.step, base[col], "k--", lw=2.8, zorder=20, label="baseline K=16")
        if col in GT_ZERO:
            ax.axhline(0.0, color="#888", lw=1.1, zorder=19)
        for i, (v, d) in enumerate(scheds):
            ax.plot(d.step, d[col], color=cols[i % 10], lw=1.8, alpha=0.9,
                    label=v.replace("sched:", ""), zorder=6)
        ax.set_title(col); ax.set_xlabel("loop step"); ax.grid(alpha=0.3)
        if col == "tok_norm" and logy_norm:
            ax.set_yscale("log")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=9, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_path}")


def main():
    df = pd.read_csv("out/full_grand/all.tsv", sep="\t")
    out = "out/full_grand/sched"; os.makedirs(out, exist_ok=True)
    base = df[df.version == "baseline"].sort_values("step")
    sched_versions = sorted(df[df.family == "sched"].version.unique())

    all_s = [(v, df[df.version == v].sort_values("step")) for v in sched_versions]
    draw(df, base, all_s, os.path.join(out, "sched_all.png"),
         "time schedules (all, incl. frozen-256) vs baseline + GT   |  tok_norm log-scale",
         logy_norm=True)

    le16 = [(v, d) for v, d in all_s if d.step.max() <= 16]   # drops frozen (256)
    draw(df, base, le16, os.path.join(out, "sched_le16.png"),
         "time schedules (frozen excluded, steps<=16) vs baseline + GT   |  local tweaks harmless, global reorder hurts")
    print(f"done -> {out}/  (schedules: {[v.replace('sched:','') for v in sched_versions]})")


if __name__ == "__main__":
    main()
