#!/usr/bin/env python3
"""Report figures for the residual_opt / stationary-loop work (no new experiments, only existing logs).

  1. out/resopt/photo_ba/photo_ba_energy.png   photometric energy cannot tell GT from DVLT; optimiser goes below E(GT)
  2. out/resopt/refiner_curves.png             open- vs closed-loop flow-matching refiner over training (val + OOD)
  3. out/report/residual_summary.png           relative AbsRel change of every residual / refiner tried
  4. out/resopt/stationary_ksweep.png          option A variants vs K (re-run when new statA* runs finish)

  python residual_opt/plot_report.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/out"
# categorical slots (fixed order), light mode
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#8a8a84"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False})


def mean_metric(lst, k):
    return float(np.mean([m[k] for m in lst]))


# ----------------------------------------------------------------------------------------------- 1. photometric BA
def fig_photo_ba():
    d = json.load(open(f"{OUT}/resopt/photo_ba/photo_ba.json"))
    cfgs = ["cam+depth_lam1", "cam+depth_lam10", "cam_only"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    for r, sp in enumerate(["val", "ood_bf"]):
        R = d[sp]
        e = np.array([[x["dvlt"], x["gt"]] for x in R["energy"]])
        a = ax[r, 0]
        lo, hi = e.min() * 0.95, e.max() * 1.05
        a.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls="--", label="E(GT) = E(DVLT)")
        a.scatter(e[:, 0], e[:, 1], s=36, color=C[0], edgecolor="white", linewidth=1, zorder=3, label="one clip")
        n_lower = int((e[:, 1] < e[:, 0]).sum())
        a.set_title(f"[{sp}] energy per clip: GT lower on {n_lower}/{len(e)}\n"
                    f"mean E(GT) {e[:, 1].mean():.4f} vs E(DVLT) {e[:, 0].mean():.4f} (only {(1 - e[:, 1].mean() / e[:, 0].mean()) * 100:.1f}% lower)",
                    fontsize=10)
        a.set_xlabel("E(DVLT geometry)"); a.set_ylabel("E(GT geometry)"); a.legend(loc="upper left")
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        # optimisation paths: energy (sigma=1 phase only, same function as the energy check) vs error
        e_dvlt, e_gt = e[:, 0].mean(), e[:, 1].mean()
        for col, key, lab in [(1, "absrel", "depth AbsRel"), (2, "ate", "camera ATE (Sim3, rel.)")]:
            a = ax[r, col]
            base = mean_metric(R["dvlt"], key)
            for i, cfg in enumerate(cfgs):
                T = R["traj"][cfg]
                its = [p["it"] for p in T[0] if p["sigma"] == 1.0]
                Em = [np.mean([[p for p in t if p["it"] == it][0]["E"] for t in T]) for it in its]
                Mm = [np.mean([[p for p in t if p["it"] == it][0][key] for t in T]) for it in its]
                a.plot([e_dvlt] + Em, [base] + Mm, color=C[i + 1], marker="o", ms=5, label=cfg)
            a.scatter([e_dvlt], [base], s=90, color="black", zorder=4, label="DVLT (start)")
            a.axvline(e_gt, color=C[0], ls="--", lw=1.5, label="E(GT geometry)")
            a.set_xlabel("photometric energy (σ = 1)"); a.set_ylabel(lab)
            a.set_title(f"[{sp}] optimisation path: E falls below E(GT),\n{lab} gets worse  (σ=4, 2 phases not shown)", fontsize=10)
            if col == 1:
                a.legend(fontsize=8, loc="upper right")
    fig.suptitle("Direct photometric BA from DVLT init (no GT, no learning): the energy cannot discriminate at DVLT's ~1 px precision",
                 fontsize=12)
    fig.tight_layout()
    p = f"{OUT}/resopt/photo_ba/photo_ba_energy.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ----------------------------------------------------------------------------------------------- 2. refiners
def fig_refiners():
    runs = {"open-loop (ol_300)": f"{OUT}/resopt/ol_300/log.json", "closed-loop (cl_300)": f"{OUT}/resopt/cl_300/log.json"}
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for i, (name, f) in enumerate(runs.items()):
        d = json.load(open(f))
        steps = [e["step"] for e in d["log"]]
        for r, sp in enumerate(["val", "ood_bf"]):
            ev = [e["eval"] if sp == "val" else e["extra"]["ood_bf"] for e in d["log"]]
            base = d["baseline"] if sp == "val" else d["baseline_extra"]["ood_bf"]
            for c, key in enumerate(["absrel", "ate"]):
                a = ax[r, c]
                a.plot(steps, [e["4"][key] for e in ev], color=C[i], marker="o", ms=4, label=f"{name}, 4 Euler steps")
                if i == 0:
                    a.axhline(base[key], color="black", lw=1.5, ls="--", label="DVLT (input to the refiner)")
                a.set_title(f"[{sp}] {'depth AbsRel' if key == 'absrel' else 'camera ATE'}")
                if r == 1:
                    a.set_xlabel("training step")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Flow-matching refiners on frozen DVLT (300 ScanNet clips): memorise, never beat DVLT, worse out of distribution",
                 fontsize=12)
    fig.tight_layout()
    p = f"{OUT}/resopt/refiner_curves.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ----------------------------------------------------------------------------------------------- 3. summary
def fig_summary():
    def rel(base, new):            # + = better
        return (base - new) / base * 100
    pb = json.load(open(f"{OUT}/resopt/photo_ba/photo_ba.json"))["val"]
    pb_base = mean_metric(pb["dvlt"], "absrel")
    ol = json.load(open(f"{OUT}/resopt/ol_300/log.json")); cl = json.load(open(f"{OUT}/resopt/cl_300/log.json"))

    def refiner(d, sp):
        ev = [e["eval"] if sp == "val" else e["extra"]["ood_bf"] for e in d["log"]]
        base = d["baseline"] if sp == "val" else d["baseline_extra"]["ood_bf"]
        return rel(base["absrel"], ev[-1]["4"]["absrel"])
    # (label, value, group); values from logs / summaries listed in the caption
    rows = [
        ("GT depth residual, gradient into state (stage 2 oracle)", rel(0.0336, 0.0273), 0),
        ("GT depth force every step on gates-B+RMS loop (B4, @16)", rel(0.037, 0.026), 0),
        ("geometric consistency guidance (B5, best, @16)", rel(0.0370, 0.0363), 1),
        ("photometric guidance (B5, best, @16)", rel(0.0370, 0.0365), 1),
        ("feature guidance (B5, best, @16)", rel(0.0370, 0.0370), 1),
        ("whole-token latent consistency residual (K16)", rel(0.031, 0.048), 1),
        ("open-loop FM refiner, val (final)", refiner(ol, "val"), 2),
        ("open-loop FM refiner, OOD BundleFusion (final)", refiner(ol, "ood"), 2),
        ("closed-loop FM refiner, val (final)", refiner(cl, "val"), 2),
        ("closed-loop FM refiner, OOD BundleFusion (final)", refiner(cl, "ood"), 2),
        ("photometric BA, camera only (val; ATE -33%)", rel(pb_base, mean_metric(pb["cam_only"], "absrel")), 3),
        ("photometric BA, camera + depth, λ=10 (val)", rel(pb_base, mean_metric(pb["cam+depth_lam10"], "absrel")), 3),
        ("photometric BA, camera + depth, λ=1 (val)", rel(pb_base, mean_metric(pb["cam+depth_lam1"], "absrel")), 3),
    ]
    groups = ["uses GT (oracle)", "GT-free guidance in the loop", "learned refiner (trained with GT)", "GT-free photometric BA"]
    fig, a = plt.subplots(figsize=(12, 7))
    y = np.arange(len(rows))[::-1]
    for (lab, v, g), yy in zip(rows, y):
        a.barh(yy, v, height=0.62, color=C[g], edgecolor="white", linewidth=2)
        if v < -60:
            a.text(v + 2, yy, f"{v:+.1f}%", va="center", ha="left", fontsize=9, color="white", fontweight="bold")
        else:
            a.text(v + (1.5 if v >= 0 else -1.5), yy, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9, color="#333")
    a.set_yticks(y); a.set_yticklabels([r[0] for r in rows], fontsize=9)
    a.axvline(0, color="black", lw=1)
    a.set_xlabel("relative change of depth AbsRel vs. its own baseline  (+ = better)")
    a.set_xlim(-95, 45)
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color=C[g], label=n) for g, n in enumerate(groups)], loc="upper left", fontsize=9)
    a.set_title("Only GT-based residuals improve DVLT; every GT-free residual / refiner is neutral or harmful\n"
                "(baselines differ per row: s10 scene for oracle/B4/B5/latent rows, ScanNet val / BundleFusion for the rest)",
                fontsize=11)
    fig.tight_layout()
    os.makedirs(f"{OUT}/report", exist_ok=True)
    p = f"{OUT}/report/residual_summary.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ----------------------------------------------------------------------------------------------- 4. option A K-sweep
def fig_stationary():
    names = {"statA_gatesonly_bug": "constant gates only (trained)", "statA3": "gates + R",
             "statA3_rpull": "gates + R + learned pull (a -> ~0)", "statA4_af": "gates + R + fixed pull a=0.05",
             "statA4_af_fp1": "a=0.05, fixed-point weight 1.0", "statA4_af_bptt8": "a=0.05, BPTT window 8"}
    logs = {k: json.load(open(f"{OUT}/resopt/{k}/log.json")) for k in names if os.path.exists(f"{OUT}/resopt/{k}/log.json")}
    if not logs:
        return None
    any_log = next(iter(logs.values()))
    # finished runs: final checkpoint; runs still training: the best evaluation so far (lowest mean val AbsRel over K)
    pick, labels = {}, {}
    for k, d in logs.items():
        L = d["log"]
        if L[-1]["step"] >= d["args"]["steps"]:
            pick[k], labels[k] = L[-1], names[k]
        else:
            e = min(L, key=lambda e: np.mean([e["eval"]["val"][q]["absrel"] for q in e["eval"]["val"]]))
            pick[k] = e
            labels[k] = f"{names[k]}  [running: best @ {e['step']}/{d['args']['steps']}]"
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    for r, sp in enumerate(["val", "ood_bf"]):
        b = any_log["baseline"][sp]
        for c, key in enumerate(["absrel", "ate"]):
            a = ax[r, c]
            un = b["stationary_untrained"]
            Ks = sorted(int(k) for k in un)
            a.plot(Ks, [un[str(k)][key] for k in Ks], color=GREY, ls=":", marker="o", ms=4, label="constant gates, untrained")
            if "statA4_af" in logs:                    # same evaluation, constant gates + fixed pull a=0.05, R = 0
                ua = logs["statA4_af"]["baseline"][sp]["stationary_untrained"]
                a.plot(Ks, [ua[str(k)][key] for k in Ks], color=C[6], ls=":", marker="o", ms=4,
                       label="constant gates + fixed pull a=0.05, untrained")
            sf = f"{OUT}/resopt/solv3r_eval/solv3r_eval.json"
            if os.path.exists(sf):                    # released Solv3R, same clips and metrics (eval_solv3r.py)
                sv = json.load(open(sf))[sp]
                a.plot(Ks, [sv[str(k)][key] for k in Ks], color=C[7], marker="s", ms=6, lw=2.5,
                       label="Solv3R (released weights)")
            a.scatter([16], [b["dvlt_sched_K16"][key]], s=110, marker="*", color="black", zorder=5, label="DVLT schedule K16")
            a.scatter([64], [b["dvlt_sched_K64"][key]], s=70, marker="x", color="black", zorder=5, label="DVLT schedule K64")
            for i, (k, d) in enumerate(logs.items()):
                ev = pick[k]["eval"][sp]
                a.plot(Ks, [ev[str(K)][key] for K in Ks], color=C[i], marker="o", ms=5, label=labels[k])
            a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xticks(Ks); a.set_xticklabels(Ks)
            a.set_xlabel("number of calls K"); a.set_title(f"[{sp}] {'depth AbsRel' if key == 'absrel' else 'camera ATE'}")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Option A: stationary DVLT loop (frozen weights) — stability vs. number of calls\n(final checkpoint; runs still training: best evaluation so far)", fontsize=12)
    fig.tight_layout()
    p = f"{OUT}/resopt/stationary_ksweep.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


if __name__ == "__main__":
    for f in (fig_photo_ba, fig_refiners, fig_summary, fig_stationary):
        print(f.__name__, "->", f())


# ----------------------------------------------------------------------------------------------- 5. FM refiner summary
def fig_fm_summary():
    runs = [("smoke", "smoke\n1 clip, open", None), ("smoke_cl", "smoke_cl\n1 clip, closed", None),
            ("overfit4", "overfit4\n4 clips, open", None), ("small", "small\n80 clips, open\nval", "val"),
            ("ol_300", "ol_300\n300 clips, open\nval", "val"), ("ol_300", "ol_300\nOOD BundleFusion", "ood"),
            ("cl_300", "cl_300\n300 clips, closed\nval", "val"), ("cl_300", "cl_300\nOOD BundleFusion", "ood")]
    rows = []
    for run, lab, sp in runs:
        d = json.load(open(f"{OUT}/resopt/{run}/log.json"))
        L = d["log"]
        # best eval step = lowest val AbsRel over integration step counts n > 0 (as in train.py)
        bi = min(range(len(L)), key=lambda i: min(L[i]["eval"][n]["absrel"] for n in L[i]["eval"] if n != "0"))
        if sp == "ood":
            base = d["baseline_extra"]["ood_bf"]
            best, fin = L[bi]["extra"]["ood_bf"]["4"], L[-1]["extra"]["ood_bf"]["4"]
        else:
            base = d["baseline"]
            ev = L[bi]["eval"]; nb = min((n for n in ev if n != "0"), key=lambda n: ev[n]["absrel"])
            best, fin = ev[nb], L[-1]["eval"]["4"]
        rows.append((lab, base, best, fin))
    fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    x = np.arange(len(rows)); w = 0.26
    series = [("DVLT (input to the refiner)", 1, GREY), ("refiner, best checkpoint (picked on val)", 2, C[0]),
              ("refiner, final checkpoint", 3, C[1])]
    for a, key, name in [(ax[0], "absrel", "depth AbsRel"), (ax[1], "ate", "camera ATE (Sim3, rel.)")]:
        for j, (lab, idx, col) in enumerate(series):
            vals = [r[idx][key] for r in rows]
            a.bar(x + (j - 1) * w, vals, width=w, color=col, edgecolor="white", linewidth=2, label=lab)
            for xx, v in zip(x + (j - 1) * w, vals):
                a.text(xx, v * 1.04, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="#333", rotation=90)
        a.set_yscale("log"); a.set_ylabel(name)
        a.axvline(2.5, color="black", lw=1, ls="--")
        a.set_ylim(top=a.get_ylim()[1] * 2.2)
    ax[0].text(1, ax[0].get_ylim()[1] * 0.6, "fit check: evaluated on the TRAINING clips", ha="center", fontsize=10)
    ax[0].text(5, ax[0].get_ylim()[1] * 0.6, "generalisation: unseen ScanNet rooms (val) / BundleFusion (OOD)", ha="center", fontsize=10)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.955))
    ax[1].set_xticks(x); ax[1].set_xticklabels([r[0] for r in rows], fontsize=9)
    fig.suptitle("Flow-matching refiner on frozen DVLT: fits training clips easily, but never improves unseen scenes "
                 "(4 Euler steps; best = lowest val AbsRel over n)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = f"{OUT}/resopt/fm_refiner_summary.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


if __name__ == "__main__":
    print("fig_fm_summary ->", fig_fm_summary())
