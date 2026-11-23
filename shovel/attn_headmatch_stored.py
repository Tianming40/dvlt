#!/usr/bin/env python3
"""attn_headmatch_stored.py -- OFFLINE (no model, no GPU): which head / head-mean, at which step of which K,
best represents TRUE patch matching? Reads the Stage-2 stored attention in out/track_s10_K{K}/ (track_token.py):
  global_rows (T=8, Hh=12, S=27, P=869) per step  -- full global-attention rows of 8 frame-0 tokens
  meta.npz: tok_index (3 patches = near_tex, flat, far at tracked slots 5,6,7)

GT correspondence is RECOMPUTED here from gt.npz with an OCCLUSION test (the stored gt_target ignores
occlusion and includes frame 0 itself). A (query, target-frame j) PAIR is valid if j != 0, the query's 3D point
(GT depth) projects in front of camera j, inside the image, and |z_proj - z_gt(j)|/z_proj < occ.

For each pair the attention row is restricted to frame j's 864 patches and renormalised ("if it looks at
frame j, WHERE does it look"), then compared with the GT target patch g:
  pck0/pck1/pck2 : argmax within Chebyshev 0/1/2 patches of g      (chance ~ 1/864, 9/864, 25/864)
  rank           : rank of g among the 864 patches (1 = best)
  sa_err         : |soft-argmax position - g| in patches            (the "observed position" a residual would use)
  win1           : renormalised mass in the 3x3 window around g     (chance 9/864 = 0.010)
  fmass          : UN-renormalised mass the row gives frame j's patches at all
Head combos: each single head h0..h11, mean of all 12, {0,9}, {0,4,9}, and (IN-SAMPLE, overfit-prone) the best
head pair by pck1. Combining = averaging the heads' probability rows.

  PYTHONPATH=$(pwd) python shovel/attn_headmatch_stored.py --gt out/gt_00777c41d4_s10/gt.npz \
      --Ks 8 16 32 64 128 --out out/attnmatch_stored
"""
import argparse
import csv
import glob
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_gt  # noqa: E402

PATCH_SLOTS = [5, 6, 7]          # tracked-token slots of the 3 patches in global_rows


def gt_pairs(gd, c2w, K, rc_list, patch, occ):
    """(nP,S) target (row,col) + geometric-valid + occlusion-valid masks; frame 0 excluded from both."""
    S, H, W = gd.shape
    gh, gw = H // patch, W // patch
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    inv = np.linalg.inv(c2w)
    nP = len(rc_list)
    tgt = np.full((nP, S, 2), -1, np.int64)
    geo = np.zeros((nP, S), bool)
    occv = np.zeros((nP, S), bool)
    for pi, (r, c) in enumerate(rc_list):
        u, v = c * patch + patch // 2, r * patch + patch // 2
        d = float(gd[0, v, u])
        if d <= 1e-6:
            continue
        Xw = c2w[0] @ np.array([(u - cx) / fx * d, (v - cy) / fy * d, d, 1.0])
        for j in range(1, S):
            Pc = inv[j] @ Xw
            z = Pc[2]
            if z <= 1e-6:
                continue
            uj, vj = fx * Pc[0] / z + cx, fy * Pc[1] / z + cy
            if not (0 <= uj < W and 0 <= vj < H):
                continue
            tgt[pi, j] = (int(vj // patch), int(uj // patch))
            geo[pi, j] = True
            dj = float(gd[j, int(vj), int(uj)])
            occv[pi, j] = dj > 1e-6 and abs(z - dj) / z < occ
    return tgt, geo, occv


def pair_metrics(row, g, gh, gw):
    """row: (S,P) probs over all frames' tokens for ONE query, ONE combo; g=(j,r,c). Returns dict."""
    j, r, c = g
    pj = row[j, 5:].astype(np.float64)                       # 864 patch probs of frame j
    fmass = pj.sum()
    p = (pj / max(fmass, 1e-12)).reshape(gh, gw)
    am = int(np.argmax(p)); ar, ac = divmod(am, gw)
    cheb = max(abs(ar - r), abs(ac - c))
    rank = int((p > p[r, c]).sum()) + 1
    yy, xx = np.mgrid[0:gh, 0:gw]
    sy, sx = (p * yy).sum(), (p * xx).sum()
    win = p[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].sum()
    return dict(pck0=float(cheb == 0), pck1=float(cheb <= 1), pck2=float(cheb <= 2), rank=rank,
                sa_err=float(np.hypot(sy - r, sx - c)), win1=float(win), fmass=float(fmass))


METRICS = ["pck0", "pck1", "pck2", "rank", "sa_err", "win1", "fmass"]


def combos_fixed(Hh):
    c = {f"h{h}": [h] for h in range(Hh)}
    c["mean12"] = list(range(Hh))
    c["h0+9"] = [0, 9]
    c["h0+4+9"] = [0, 4, 9]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="out/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--track-root", default="out")
    ap.add_argument("--tag", default="s10")
    ap.add_argument("--Ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--occ", type=float, default=0.1)
    ap.add_argument("--out", default="out/attnmatch_stored")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    meta0 = np.load(f"{a.track_root}/track_{a.tag}_K{a.Ks[0]}/meta.npz")
    S, Hh, patch, pm = int(meta0["S"]), int(meta0["Hh"]), int(meta0["patch"]), int(meta0["patch_start_idx"])
    gh, gw = [int(v) for v in meta0["grid"]]
    names = [str(n) for n in meta0["patch_names"]]
    tok_index = meta0["tok_index"][PATCH_SLOTS]
    rc_list = [divmod(int(t) - pm, gw) for t in tok_index]
    gd, c2w, Kmat = load_gt(a.gt, 504, patch=patch, device="cpu")
    gd, c2w, Kmat = gd[:S].numpy(), c2w[:S].numpy(), Kmat.numpy()
    tgt, geo, occv = gt_pairs(gd, c2w, Kmat, rc_list, patch, a.occ)

    log = []
    def say(s=""):
        print(s); log.append(s)
    say(f"query patches (frame 0): " + ", ".join(f"{n}@(r{r},c{c})" for n, (r, c) in zip(names, rc_list)))
    say(f"stored gt_valid (incl. frame 0, no occlusion): {meta0['gt_valid'].sum(1).tolist()} = {int(meta0['gt_valid'].sum())}")
    say(f"geometric-only valid, frame 0 excluded        : {geo.sum(1).tolist()} = {int(geo.sum())}")
    say(f"OCCLUSION-checked valid (USED), occ={a.occ}   : {occv.sum(1).tolist()} = {int(occv.sum())}")
    pairs = [(pi, j) for pi in range(len(names)) for j in range(1, S) if occv[pi, j]]
    say(f"per-pair targets: " + "; ".join(f"{names[pi]}->f{j}(r{tgt[pi,j,0]},c{tgt[pi,j,1]})" for pi, j in pairs))
    say(f"chance: pck0={1/864:.4f} pck1~{9/864:.4f} pck2~{25/864:.4f} win1={9/864:.4f} rank~432 sa_err(uniform)~{np.hypot(gh/4, gw/4):.1f}")

    fixed = combos_fixed(Hh)
    pair_combos = {f"h{x}+{y}": [x, y] for x, y in itertools.combinations(range(Hh), 2)}
    allc = {**fixed, **pair_combos}

    summary_rows, raw_rows = [], []
    agg = {}                                                  # (K, combo) -> (steps, metric arrays)
    for K in a.Ks:
        d = f"{a.track_root}/track_{a.tag}_K{K}"
        files = sorted(glob.glob(f"{d}/step_*.npz"), key=lambda p: int(os.path.basename(p)[5:-4]))  # numeric, not lexicographic
        assert len(files) == K, f"{d}: expected {K} steps, found {len(files)}"
        for f in files:
            z = np.load(f)
            step, tn = int(z["step"]), float(z["t_now"])
            G = z["global_rows"][PATCH_SLOTS].astype(np.float32)          # (3,Hh,S,P)
            for cname, hs in allc.items():
                rowc = G[:, hs].mean(1)                                    # (3,S,P)
                ms = [pair_metrics(rowc[pi], (j, tgt[pi, j, 0], tgt[pi, j, 1]), gh, gw) for pi, j in pairs]
                mean = {m: float(np.mean([x[m] for x in ms])) for m in METRICS}
                mean["rank_med"] = float(np.median([x["rank"] for x in ms]))
                summary_rows.append(dict(K=K, step=step, t_now=round(tn, 4), combo=cname, **mean))
                agg.setdefault((K, cname), []).append((step, mean))
                if cname in fixed:
                    for (pi, j), x in zip(pairs, ms):
                        raw_rows.append(dict(K=K, step=step, combo=cname, patch=names[pi], frame=j, **x))

    def wcsv(path, rows):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    wcsv(f"{a.out}/summary_by_step.csv", summary_rows)
    wcsv(f"{a.out}/per_pair.csv", raw_rows)

    # ---------------- text report
    say("\n=== per K: matching quality over the loop (mean over pairs) ===")
    show = ["mean12", "h0+9", "h0+4+9"] + [f"h{h}" for h in range(Hh)]
    for K in a.Ks:
        say(f"\n--- K={K} ---   columns: pck1 / win1 / sa_err / rank_med / fmass")
        idx = sorted(set([0, 1, 2, 3] + [K // 4, K // 2, 3 * K // 4, K - 1]))
        idx = [i for i in idx if i < K]
        say(f"{'combo':>8} | " + " | ".join(f"step{i:>3}".rjust(27) for i in idx))
        for cn in show:
            st = dict(agg[(K, cn)])
            say(f"{cn:>8} | " + " | ".join(
                f"{st[i]['pck1']:.2f} {st[i]['win1']:.3f} {st[i]['sa_err']:5.1f} {st[i]['rank_med']:5.0f} {st[i]['fmass']:.3f}".rjust(27)
                for i in idx))
        best = []
        for cn in fixed:
            for s, m in agg[(K, cn)]:
                best.append((m["pck1"], m["win1"], cn, s))
        best.sort(reverse=True)
        say("  top-5 (combo, step) by pck1 then win1 [fixed combos]: " +
            ", ".join(f"{cn}@{s} pck1={p:.2f} win1={w:.3f}" for p, w, cn, s in best[:5]))
        bp = sorted(((m["pck1"], m["win1"], cn, s) for cn in pair_combos for s, m in agg[(K, cn)]), reverse=True)[:3]
        say("  best head PAIR (IN-SAMPLE, overfit-prone): " +
            ", ".join(f"{cn}@{s} pck1={p:.2f} win1={w:.3f}" for p, w, cn, s in bp))

    say("\n=== per patch (mean12 and h0+9, win1 at step 0 / best step) ===")
    for cn in ["mean12", "h0+9"]:
        for K in a.Ks:
            parts = []
            for pn in names:
                rr = [r for r in raw_rows if r["K"] == K and r["combo"] == cn and r["patch"] == pn]
                if not rr:
                    continue
                bystep = {}
                for r in rr:
                    bystep.setdefault(r["step"], []).append(r["win1"])
                s0 = np.mean(bystep[0]); sb, wb = max(((s, np.mean(v)) for s, v in bystep.items()), key=lambda x: x[1])
                parts.append(f"{pn}(n={len(bystep[0])}): s0={s0:.3f} best={wb:.3f}@{sb}")
            say(f"  {cn:>6} K={K:>3}: " + "  ".join(parts))

    with open(f"{a.out}/report.txt", "w") as fh:
        fh.write("\n".join(log) + "\n")

    # ---------------- plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        pm_list = [("pck1", "PCK@1 (argmax within 1 patch)", False), ("win1", "3x3 window mass (renorm. in frame j)", False),
                   ("sa_err", "soft-argmax error [patches]", False), ("rank_med", "median rank of GT patch", True),
                   ("fmass", "raw mass on target frame", False)]
        cmap = plt.get_cmap("tab20")
        for K in a.Ks:
            fig, ax = plt.subplots(1, len(pm_list), figsize=(5 * len(pm_list), 4))
            for c, (m, title, logy) in enumerate(pm_list):
                for h in range(Hh):
                    st = agg[(K, f"h{h}")]
                    ax[c].plot([s for s, _ in st], [v[m] for _, v in st], color=cmap(h), lw=1, alpha=.8, label=f"h{h}")
                for cn, sty in (("mean12", "k-"), ("h0+9", "r--")):
                    st = agg[(K, cn)]
                    ax[c].plot([s for s, _ in st], [v[m] for _, v in st], sty, lw=2.2, label=cn)
                if logy: ax[c].set_yscale("log")
                ax[c].set_title(title, fontsize=9); ax[c].set_xlabel("step"); ax[c].grid(alpha=.3)
            ax[0].legend(fontsize=6, ncol=2)
            fig.suptitle(f"K={K}: global-attention match quality vs GT ({len(pairs)} occlusion-checked pairs, 3 query patches)")
            fig.tight_layout(); fig.savefig(f"{a.out}/heads_K{K}.png", dpi=120); plt.close(fig)

        fig, ax = plt.subplots(1, len(a.Ks), figsize=(4.2 * len(a.Ks), 3.8), sharey=True)
        for c, K in enumerate(a.Ks):
            M = np.array([[dict(agg[(K, f'h{h}')])[s]["win1"] for s in range(K)] for h in range(Hh)])
            im = ax[c].imshow(M, aspect="auto", cmap="magma", vmin=0, interpolation="nearest")
            ax[c].set_title(f"K={K}  win1"); ax[c].set_xlabel("step"); ax[c].set_yticks(range(Hh))
            plt.colorbar(im, ax=ax[c])
        ax[0].set_ylabel("head")
        fig.tight_layout(); fig.savefig(f"{a.out}/head_step_heatmap_win1.png", dpi=120); plt.close(fig)

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for c, cn in enumerate(["mean12", "h0+9", "h9"]):
            for K in a.Ks:
                st = agg[(K, cn)]
                ax[c].plot([s / max(K - 1, 1) for s, _ in st], [v["win1"] for _, v in st], marker=".", ms=3, label=f"K={K}")
            ax[c].set_title(f"{cn}: win1 vs normalised step"); ax[c].set_xlabel("step / (K-1)"); ax[c].grid(alpha=.3)
        ax[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{a.out}/crossK_win1.png", dpi=120); plt.close(fig)
        print(f"plots saved in {a.out}/")
    except Exception as e:
        print("plot skipped:", e)
    print(f"report: {a.out}/report.txt")


if __name__ == "__main__":
    main()
