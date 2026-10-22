#!/usr/bin/env python3
"""Per-perturbation-kind figures from saved grand data, each overlaid with the two anchors:
the K=16 BASELINE (black dashed) and GT (error = 0, grey line on GT panels).

For each kind (gauss/scale/dir/cam/register/swap) a 6-panel figure is written to
out/full_grand/perturb/<kind>.png:
  gt_absrel, gt_ate  (vs GT) | tok_rel_to_clean (recovery) | eff_chan, cluster_cos, tok_norm (state).
Perturbation lines are colored by eps and styled by injection step (at). No model/GPU; reads all.tsv.

  python scriptandtool/plot_perturb.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

PANELS = ["gt_absrel", "gt_ate", "tok_rel_to_clean", "eff_chan", "cluster_cos", "tok_norm"]
GT_ZERO = {"gt_absrel", "gt_ate"}          # panels where GT = perfect = 0
AT_STYLE = {1: "-", 4: "--", 12: ":"}       # injection-step -> line style
KINDS = ["gauss", "scale", "dir", "cam", "register", "swap"]


def main():
    df = pd.read_csv("out/full_grand/all.tsv", sep="\t")
    out = "out/full_grand/perturb"; os.makedirs(out, exist_ok=True)
    base = df[df.version == "baseline"].sort_values("step")

    for kind in KINDS:
        sub = df[(df.family == "perturb") & (df.kind == kind)]
        if len(sub) == 0:
            continue
        epsvals = sorted(sub.eps.dropna().unique())
        cmap = cm.viridis
        def ecolor(e):
            return cmap(0.1 + 0.85 * (epsvals.index(e) / max(len(epsvals) - 1, 1)))

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, col in zip(axes.ravel(), PANELS):
            # anchors
            ax.plot(base.step, base[col], "k--", lw=2.6, zorder=20, label="baseline K=16")
            if col in GT_ZERO:
                ax.axhline(0.0, color="#888", lw=1.2, ls="-", zorder=19, label="GT (perfect = 0)")
            # perturbation lines
            for v, d in sub.groupby("version"):
                e = d.eps.iloc[0]; at = int(d["at"].iloc[0])
                ax.plot(d.step, d[col], color=ecolor(e), ls=AT_STYLE.get(at, "-"),
                        lw=1.3, alpha=0.9, zorder=5)
            ax.set_title(col); ax.set_xlabel("loop step"); ax.grid(alpha=0.3)

        # legends: eps colorbar proxy + at styles + anchors
        eps_handles = [mlines.Line2D([], [], color=ecolor(e), lw=2, label=f"eps={e:g}") for e in epsvals]
        at_handles = [mlines.Line2D([], [], color="#444", ls=AT_STYLE.get(a, "-"), lw=2, label=f"inject @step {a}")
                      for a in sorted({int(x) for x in sub["at"].dropna().unique()})]
        anc = [mlines.Line2D([], [], color="k", ls="--", lw=2.6, label="baseline K=16"),
               mlines.Line2D([], [], color="#888", lw=1.2, label="GT (=0 on GT panels)")]
        fig.legend(handles=eps_handles + at_handles + anc, loc="lower center", ncol=8, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"perturbation = {kind}   (color = eps strength, line style = injection step; anchors: baseline + GT)",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0.04, 1, 0.98))
        fig.savefig(os.path.join(out, f"{kind}.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}/{kind}.png  ({sub.version.nunique()} versions, eps={epsvals})")
    print(f"done -> {out}/")


if __name__ == "__main__":
    main()
