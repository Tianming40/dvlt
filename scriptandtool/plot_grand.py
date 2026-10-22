#!/usr/bin/env python3
"""Re-plot grand's per-metric figures FROM THE SAVED DATA (no model, no GPU).

Reads out/full_grand/all.tsv and draws one figure per metric column, one line per version,
into a NEW output folder (never overwrites the original figs/).

  python scriptandtool/plot_grand.py                      # all versions
  python scriptandtool/plot_grand.py --max-step 64        # drop long-horizon versions (K=128/256, frozen-256)
  python scriptandtool/plot_grand.py --max-step 64 --out out/full_grand/figs_le64

Style mirrors the original (color by family; perturbations colored by kind, alpha grows with eps).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

ID_COLS = ["version", "family", "kind", "eps", "at", "step", "t"]
PALETTE = {"gauss": "#1f77b4", "scale": "#ff7f0e", "dir": "#2ca02c",
           "swap": "#d62728", "cam": "#9467bd", "register": "#8c564b"}


def build_styles(df):
    """One plot style per version, derived from the family/kind/eps columns in the data."""
    fam = df.groupby("version").agg(family=("family", "first"), kind=("kind", "first"),
                                    eps=("eps", "first")).to_dict("index")
    Ks = sorted({int(v.split("=")[1]) for v in fam if fam[v]["family"] == "K"})
    scheds = [v for v in fam if fam[v]["family"] == "sched"]
    ablates = [v for v in fam if fam[v]["family"] == "ablate"]
    epsvals = sorted({fam[v]["eps"] for v in fam if fam[v]["family"] == "perturb" and pd.notna(fam[v]["eps"])})
    schedcols = cm.tab10(np.linspace(0, 1, 10))
    abcols = {"ablate:first_half": "#17becf", "ablate:second_half": "#bcbd22", "ablate:all": "#7f7f7f"}

    def alpha_of(e):
        if len(epsvals) <= 1 or pd.isna(e):
            return 0.8
        return 0.28 + 0.62 * (epsvals.index(e) / (len(epsvals) - 1))

    styles = {}
    for v, info in fam.items():
        f = info["family"]
        if f == "base":
            styles[v] = dict(color="k", ls="--", lw=2.6, alpha=1.0, z=10)
        elif f == "K":
            k = int(v.split("=")[1]); c = cm.winter(Ks.index(k) / max(len(Ks) - 1, 1))
            styles[v] = dict(color=c, ls="-", lw=1.4, alpha=0.9, z=6)
        elif f == "sched":
            styles[v] = dict(color=schedcols[scheds.index(v) % 10], ls="-.", lw=1.6, alpha=0.95, z=6)
        elif f == "ablate":
            styles[v] = dict(color=abcols.get(v, "#555"), ls=":", lw=1.9, alpha=0.95, z=6)
        elif f == "renorm":
            styles[v] = dict(color="#e377c2", ls="-", lw=1.8, alpha=0.95, z=6)
        elif f == "photo":
            styles[v] = dict(color="#ff9900", ls=(0, (1, 1)), lw=2.3, alpha=1.0, z=7)
        else:  # perturb
            styles[v] = dict(color=PALETTE.get(info["kind"], "#333"), ls="-", lw=1.1,
                             alpha=alpha_of(info["eps"]), z=5)
    return styles, fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/full_grand/all.tsv")
    ap.add_argument("--out", default=None, help="output dir (default derived from --max-step; never the original figs/)")
    ap.add_argument("--max-step", type=int, default=None, help="drop versions whose max step exceeds this (e.g. 64 removes K=128/256 and frozen-256)")
    a = ap.parse_args()

    df = pd.read_csv(a.data, sep="\t")
    out = a.out or (f"out/full_grand/figs_le{a.max_step}" if a.max_step is not None else "out/full_grand/figs_all")
    assert os.path.abspath(out) != os.path.abspath("out/full_grand/figs"), "refusing to overwrite original figs/"
    os.makedirs(out, exist_ok=True)

    if a.max_step is not None:
        keep = df.groupby("version")["step"].max()
        keep = set(keep[keep <= a.max_step].index)
        dropped = sorted(set(df.version.unique()) - keep)
        df = df[df.version.isin(keep)]
        print(f"max-step {a.max_step}: kept {len(keep)} versions, dropped {len(dropped)}: {dropped}")

    styles, fam = build_styles(df)
    metrics = [c for c in df.columns if c not in ID_COLS]
    kinds_present = [k for k in PALETTE if any(fam[v]["family"] == "perturb" and fam[v]["kind"] == k for v in fam)]
    leg = []
    for v in df.version.unique():
        if fam[v]["family"] in ("base", "K", "sched", "ablate", "renorm", "photo"):
            s = styles[v]
            leg.append(mlines.Line2D([], [], color=s["color"], ls=s["ls"], lw=s["lw"], label=v))
    leg += [mlines.Line2D([], [], color=PALETTE[k], lw=1.4, label="perturb:" + k) for k in kinds_present]

    made = 0
    for col in metrics:
        if df[col].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for v, d in df.groupby("version"):
            s = styles[v]
            ax.plot(d.step, d[col], color=s["color"], ls=s["ls"], lw=s["lw"], alpha=s["alpha"], zorder=s["z"])
        ax.legend(handles=leg, fontsize=6, ncol=4, title="perturb: alpha ~ eps")
        ax.set_xlabel("loop step"); ax.set_ylabel(col); ax.grid(alpha=0.3)
        ax.set_title(f"{col}   ({df.version.nunique()} versions" + (f", max step <= {a.max_step}" if a.max_step else "") + ")")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"{col}.png"), dpi=120); plt.close(fig); made += 1
    print(f"wrote {made} figures to {out}/  (original out/full_grand/figs/ untouched)")


if __name__ == "__main__":
    main()
