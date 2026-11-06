#!/usr/bin/env python3
"""Plot the residual-injection diagnostic: L descent trajectory (L_hist) per eta, and the final GT AbsRel
(grad vs the no-inject baseline), for the photo (self-supervised) and gtdepth (oracle) residuals.

  PYTHONPATH=$(pwd) python shovel/plot_photodiag.py --base out/experiment1/photodiag
"""
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="out/experiment1/photodiag")
    a = ap.parse_args()

    # find per-eta dirs (eta<val>/), fall back to the base dir itself (single eta)
    etadirs = sorted(glob.glob(f"{a.base}/eta*"))
    if not etadirs:
        etadirs = [a.base]
    def eta_of(d):
        b = os.path.basename(d)
        return float(b[3:]) if b.startswith("eta") else None

    data = {}   # (res, eta) -> dict
    for d in etadirs:
        e = eta_of(d)
        for res in ("photo", "gtdepth"):
            fp = f"{d}/photo_{res}.json"
            if os.path.exists(fp):
                j = json.load(open(fp))
                inj = {r["mode"]: r for r in j["injection"]}
                data[(res, e)] = inj
    etas = sorted({e for (_, e) in data if e is not None})
    cmap = plt.cm.viridis
    ecol = {e: cmap(i / max(1, len(etas) - 1)) for i, e in enumerate(etas)}

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    # baseline (no-inject) AbsRel — same for all
    base_absrel = None
    for (res, e), inj in data.items():
        if "none" in inj and inj["none"].get("gt"):
            base_absrel = inj["none"]["gt"]["absrel"]; break

    for ci, res in enumerate(["gtdepth", "photo"]):
        for e in etas:
            inj = data.get((res, e))
            if not inj or "grad" not in inj:
                continue
            h = inj["grad"].get("L_hist", [])
            if h:
                ax[ci].plot(range(len(h)), h, "-o", ms=3, color=ecol[e], label=f"η={e}")
        ax[ci].set_title(f"{res}: residual L over inject iters\n(descend = good; oscillate = step too big)")
        ax[ci].set_xlabel("inject iter"); ax[ci].set_ylabel("residual L"); ax[ci].grid(alpha=.3); ax[ci].legend(fontsize=8)

    # final AbsRel (grad) vs baseline, per eta, both residuals
    for res, mk in [("gtdepth", "-o"), ("photo", "-s")]:
        ys = [data[(res, e)]["grad"]["gt"]["absrel"] for e in etas if (res, e) in data and data[(res, e)]["grad"].get("gt")]
        xs = [e for e in etas if (res, e) in data]
        if xs:
            ax[2].plot(xs, ys, mk, label=f"{res} (grad)")
    if base_absrel is not None:
        ax[2].axhline(base_absrel, color="k", ls="--", lw=1, label=f"no-inject baseline ({base_absrel:.4f})")
    ax[2].set_title("final GT AbsRel vs η\n(below dashed = injection HELPED)")
    ax[2].set_xlabel("η"); ax[2].set_ylabel("GT AbsRel"); ax[2].set_xscale("log"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)

    fig.suptitle("Residual-gradient injection diagnostic — photo (self-sup) vs gtdepth (oracle)")
    fig.tight_layout(); fig.savefig(f"{a.base}/diag.png", dpi=125)
    print(f"saved {a.base}/diag.png")
    # console summary
    print(f"\nbaseline (no inject) AbsRel = {base_absrel:.4f}")
    for res in ("gtdepth", "photo"):
        for e in etas:
            if (res, e) in data and data[(res, e)]["grad"].get("gt"):
                g = data[(res, e)]["grad"]; h = g.get("L_hist", [])
                trend = "DOWN" if len(h) > 1 and h[-1] < h[0] else "up/osc"
                print(f"  {res:>8} η={e:<6} final AbsRel={g['gt']['absrel']:.4f}  L {h[0]:.4f}->{h[-1]:.4f} ({trend})")


if __name__ == "__main__":
    main()
