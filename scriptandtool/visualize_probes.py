"""
visualize_probes.py -- turn the dvlt_probe JSON outputs under out/ into overview figures.

Scans every out/<run>/ folder, dispatches each JSON by type, and writes PNGs into
out/figures/. Only needs matplotlib + numpy (no torch / no GPU).

  python scriptandtool/visualize_probes.py                 # scan ./out, write ./out/figures
  python scriptandtool/visualize_probes.py --out-root out --fig-dir out/figures

Figure types produced:
  traj_<run>.png   -- per-trajectory panels: |x|, |dx|rel, cos(x,x'), dist-to-ref, gates, vs step
  ksweep.png       -- depth/ray distance to K_ref and |x|, vs inference step count K (the V-curve)
  perturb.png      -- injected vs final damage (token / depth / ray) across every perturb run
  photo_monitor.png, photo_injection.png -- photometric residual along the loop and under grad injection
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- discovery
def find_jsons(out_root):
    """Group every json under out_root by probe type (from filename)."""
    groups = {"traj": [], "ksweep": [], "perturb": [], "photo": []}
    for path in sorted(glob.glob(os.path.join(out_root, "*", "*.json"))):
        run = os.path.basename(os.path.dirname(path))
        base = os.path.basename(path)
        if base.startswith("traj"):
            groups["traj"].append((run, path))
        elif base.startswith("ksweep"):
            groups["ksweep"].append((run, path))
        elif base.startswith("perturb"):
            groups["perturb"].append((run, path))
        elif base.startswith("photo"):
            groups["photo"].append((run, path))
    return groups


def load(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- trajectory
def plot_traj(run, path, fig_dir):
    data = load(path)
    Ks = sorted(data.keys(), key=lambda k: int(k[1:]))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(Ks)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Trajectory inside the loop  --  {run}", fontsize=14, weight="bold")

    def steps(rows):
        return [r["step"] for r in rows]

    for Kname, c in zip(Ks, cmap):
        rows = data[Kname]
        s = steps(rows)
        axes[0, 0].plot(s, [r["tok_norm"] for r in rows], "-o", ms=3, color=c, label=Kname)
        axes[0, 1].plot(s, [r["upd_rel"] for r in rows], "-o", ms=3, color=c, label=Kname)
        axes[0, 2].plot(s, [r.get("tok_cos_step", np.nan) for r in rows], "-o", ms=3, color=c, label=Kname)
        axes[1, 0].plot(s, [r["dist_to_ref"]["depth"] for r in rows], "-o", ms=3, color=c, label=Kname)
        axes[1, 1].plot(s, [r["dist_to_ref"]["ray"] for r in rows], "-o", ms=3, color=c, label=Kname)
        # s_out gate of the frame sub-block (index 2 of the triple)
        axes[1, 2].plot(s, [r["gates"]["frame"][2] for r in rows], "-o", ms=3, color=c, label=Kname)

    titles = [
        ("|x|  (patch-token norm)", "step", "norm"),
        ("|dx|rel  (relative update)", "step", "ratio"),
        ("cos(x, x')  (step-to-step direction)", "step", "cosine"),
        ("depth dist -> reference", "step", "log-depth dist"),
        ("ray dist -> reference", "step", "degrees"),
        ("s_out gate (frame sub-block)", "step", "gate"),
    ]
    for ax, (t, xl, yl) in zip(axes.ravel(), titles):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(fig_dir, f"traj_{run}.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


# --------------------------------------------------------------------------- ksweep
def plot_ksweep(entries, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("K sweep: quality vs inference step count (overlay of runs)", fontsize=13, weight="bold")
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(entries), 1)))
    for (run, path), c in zip(entries, cmap):
        rows = load(path)
        K = [r["K"] for r in rows]
        axes[0].plot(K, [r["depth"] for r in rows], "-o", color=c, label=run)
        axes[1].plot(K, [r["ray"] for r in rows], "-o", color=c, label=run)
        axes[2].plot(K, [r["tok_norm"] for r in rows], "-o", color=c, label=run)
    for ax, t, yl in zip(axes, ["depth dist -> K_ref", "ray dist -> K_ref", "|x| vs K"],
                         ["log-depth dist", "degrees", "norm"]):
        ax.set_title(t, fontsize=10); ax.set_xlabel("inference steps K"); ax.set_ylabel(yl)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.axvline(16, color="grey", ls="--", lw=1, alpha=0.6)  # trained budget
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(fig_dir, "ksweep.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


# --------------------------------------------------------------------------- perturb
def plot_perturb(entries, fig_dir):
    # flatten every (run, config-row) into labelled bars
    labels, tok_inj, tok_fin, dep_inj, dep_fin, ray_inj, ray_fin = [], [], [], [], [], [], []
    for run, path in entries:
        for r in load(path):
            kind = r.get("kind", run)
            labels.append(f"{kind}\n@{r['at']} e{r['eps']}")
            tok_inj.append(r["tok_rel_inj"]); tok_fin.append(r["tok_rel_final"])
            dep_inj.append(r["depth_inj"]);  dep_fin.append(r["depth_final"])
            ray_inj.append(r["ray_inj"]);    ray_fin.append(r["ray_final"])
    if not labels:
        return None
    x = np.arange(len(labels)); w = 0.38
    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(labels) * 0.9), 11))
    fig.suptitle("Perturbation: damage right after injection vs after finishing the loop",
                 fontsize=13, weight="bold")
    for ax, inj, fin, t, yl in [
        (axes[0], tok_inj, tok_fin, "token-space relative damage  (final<inj => self-correct)", "rel norm"),
        (axes[1], dep_inj, dep_fin, "decoded depth distance", "log-depth dist"),
        (axes[2], ray_inj, ray_fin, "decoded ray angle", "degrees"),
    ]:
        ax.bar(x - w / 2, inj, w, label="at injection", color="#c44")
        ax.bar(x + w / 2, fin, w, label="after loop", color="#48c")
        ax.set_title(t, fontsize=10); ax.set_ylabel(yl)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(fig_dir, "perturb.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


# --------------------------------------------------------------------------- photo
def plot_photo(entries, fig_dir):
    outs = []
    # (A) monitor curves: identical across runs, so take the first that has it
    for run, path in entries:
        d = load(path)
        if d.get("monitor"):
            mon = np.array(d["monitor"])  # (steps, 3): step, L_photo, depth_dist
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(mon[:, 0], mon[:, 1], "-o", color="#c44", label="L_photo")
            ax1.set_xlabel("step"); ax1.set_ylabel("photometric residual", color="#c44")
            ax1.tick_params(axis="y", labelcolor="#c44")
            ax2 = ax1.twinx()
            ax2.plot(mon[:, 0], mon[:, 2], "-s", color="#48c", label="depth dist->final")
            ax2.set_ylabel("depth dist -> final", color="#48c")
            ax2.tick_params(axis="y", labelcolor="#48c")
            ax1.set_title(f"[A] photometric residual along the standard loop ({run})", fontsize=11)
            ax1.grid(alpha=0.3)
            fig.tight_layout()
            out = os.path.join(fig_dir, "photo_monitor.png")
            fig.savefig(out, dpi=120); plt.close(fig)
            outs.append(out)
            break

    # (B) injection L_hist across every photo run that actually optimized
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(entries), 1)))
    any_hist = False
    for (run, path), c in zip(entries, cmap):
        d = load(path)
        for inj in d.get("injection", []):
            hist = inj.get("L_hist") or []
            if inj.get("mode") == "grad" and len(hist) > 0:
                ax.plot(range(len(hist)), hist, "-o", ms=4, color=c, label=run)
                any_hist = True
    ax.set_title("[B] photometric loss during latent gradient injection", fontsize=11)
    ax.set_xlabel("optimizer iteration"); ax.set_ylabel("L_photo")
    ax.grid(alpha=0.3)
    if any_hist:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no grad-injection runs with history", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    out = os.path.join(fig_dir, "photo_injection.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    outs.append(out)
    return outs


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="out", help="folder holding the probe run subdirs")
    ap.add_argument("--fig-dir", default="out/figures", help="where to write PNG figures")
    a = ap.parse_args()

    os.makedirs(a.fig_dir, exist_ok=True)
    groups = find_jsons(a.out_root)
    written = []

    for run, path in groups["traj"]:
        written.append(plot_traj(run, path, a.fig_dir))
    if groups["ksweep"]:
        written.append(plot_ksweep(groups["ksweep"], a.fig_dir))
    if groups["perturb"]:
        w = plot_perturb(groups["perturb"], a.fig_dir)
        if w:
            written.append(w)
    if groups["photo"]:
        written.extend(plot_photo(groups["photo"], a.fig_dir))

    print(f"scanned {a.out_root}:")
    for k, v in groups.items():
        print(f"  {k:8s}: {len(v)} json")
    print(f"\nwrote {len([w for w in written if w])} figure(s) to {a.fig_dir}:")
    for w in written:
        if w:
            print(f"  {w}")


if __name__ == "__main__":
    main()
