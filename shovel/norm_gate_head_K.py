#!/usr/bin/env python3
"""Across K in {8,16,32,64,128}, from the track_token dumps, analyze three channels of the collapse:
  (1) NORM      — per-token state norm ||z_out|| vs step (does the norm blow up, how fast per K).
  (2) HEADS     — per-head attention-output norm (anorm, 12 heads): which heads dominate and whether
                  head participation (effective #heads) shrinks with K.  [the real per-head view]
  (3) GATES     — the three gates (s_attn,s_mlp,s_out) for frame & global: mean value vs step, and
                  channel concentration (participation ratio of the 768 gate channels).  [per-CHANNEL,
                  not per-head: gates act after W_O which mixes heads, so channels != heads.]

  PYTHONPATH=$(pwd) python shovel/norm_gate_head_K.py \
      --dirs out/track_s10_K8 out/track_s10_K16 out/track_s10_K32 out/track_s10_K64 out/track_s10_K128 \
      --out out/track_s10_gatehead
"""
import argparse, glob, os, re as _re
def _nsort(ps):
    return sorted(ps, key=lambda p:int(_re.search(r'step_(\d+)',p).group(1)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pr(v, axis=None):
    """participation ratio of a nonneg vector: (sum)^2/sum(sq). 1=one component, n=uniform."""
    v = np.abs(np.asarray(v, np.float64))
    s1 = v.sum(axis=axis); s2 = (v * v).sum(axis=axis)
    return s1 * s1 / np.clip(s2, 1e-20, None)


def load_K(d):
    meta = np.load(f"{d}/meta.npz", allow_pickle=True)
    steps = _nsort(glob.glob(f"{d}/step_*.npz"))
    rows = []
    for sp in steps:
        z = np.load(sp)
        rows.append(dict(
            step=int(z["step"]), t=float(z["t_now"]),
            z_out=np.asarray(z["z_out"], np.float64),            # (8,768) state
            anorm_f=np.asarray(z["anorm_frame"], np.float64),    # (8,12) per-head attn-out norm (frame)
            anorm_g=np.asarray(z["anorm_global"], np.float64),   # (8,12) per-head (global)
            gf=np.asarray(z["gates_frame"], np.float64),         # (3,768) frame gates
            gg=np.asarray(z["gates_global"], np.float64)))       # (3,768) global gates
    return meta, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--tokens", default="cam,near_tex,flat,far")
    ap.add_argument("--out", default="out/track_gatehead")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data = {}
    for d in a.dirs:
        K = int(os.path.basename(d).split("_K")[-1])
        data[K] = load_K(d)
    Ks = sorted(data)
    names = list(data[Ks[0]][0]["tok_name"])
    want = [(nm, names.index(nm)) for nm in a.tokens.split(",") if nm in names]
    cmap = plt.cm.viridis
    kcol = {K: cmap(i / max(1, len(Ks) - 1)) for i, K in enumerate(Ks)}
    GN = ["s_attn", "s_mlp", "s_out"]
    print("Ks:", Ks, " tokens:", [n for n, _ in want])

    # ---------- Figure 1: NORM — ||z_out|| per token vs step, 5 K ----------
    fig, ax = plt.subplots(1, len(want), figsize=(4.3 * len(want), 3.6), squeeze=False)
    for ci, (nm, ti) in enumerate(want):
        for K in Ks:
            _, rows = data[K]
            st = [r["step"] for r in rows]; nr = [np.linalg.norm(r["z_out"][ti]) for r in rows]
            ax[0][ci].plot(st, nr, "-", color=kcol[K], label=f"K{K}", lw=1.5)
        ax[0][ci].set_title(f"{nm}: state norm |z_out|"); ax[0][ci].set_xlabel("step")
        ax[0][ci].set_yscale("log"); ax[0][ci].grid(alpha=.3); ax[0][ci].legend(fontsize=7)
    fig.suptitle("NORM — per-token state norm vs step (5 K)")
    fig.tight_layout(); fig.savefig(f"{a.out}/norm.png", dpi=125); plt.close(fig)
    print(f"saved {a.out}/norm.png")

    # ---------- Figure 2: HEADS — per-head anorm participation + endpoint head profile ----------
    # row0: head participation (eff #heads) vs step for global & frame, lines=K, for a patch token
    # row1: endpoint (t=1) 12-head anorm profile, bars grouped by K, global & frame
    tnm, tti = want[1] if len(want) > 1 else want[0]      # a patch token (near_tex)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for col, (lab, key) in enumerate([("GLOBAL", "anorm_g"), ("FRAME", "anorm_f")]):
        for K in Ks:
            _, rows = data[K]
            st = [r["step"] for r in rows]; hp = [pr(r[key][tti]) for r in rows]
            ax[0][col].plot(st, hp, "-", color=kcol[K], label=f"K{K}", lw=1.5)
        ax[0][col].set_title(f"{tnm}: {lab} head participation (eff #heads of anorm, 1..12)")
        ax[0][col].set_xlabel("step"); ax[0][col].axhline(12, color="k", ls=":", lw=.6)
        ax[0][col].grid(alpha=.3); ax[0][col].legend(fontsize=7)
        # endpoint per-head profile
        w = 0.8 / len(Ks)
        for i, K in enumerate(Ks):
            _, rows = data[K]
            prof = rows[-1][key][tti]; prof = prof / prof.sum()
            ax[1][col].bar(np.arange(12) + i * w, prof, w, color=kcol[K], label=f"K{K}")
        ax[1][col].set_title(f"{tnm}: {lab} per-head anorm share at t=1 (which heads dominate)")
        ax[1][col].set_xlabel("head"); ax[1][col].set_xticks(range(12)); ax[1][col].legend(fontsize=7)
    fig.suptitle(f"HEADS — per-head attention-output norm ({tnm})")
    fig.tight_layout(); fig.savefig(f"{a.out}/heads.png", dpi=125); plt.close(fig)
    print(f"saved {a.out}/heads.png")

    # ---------- Figure 3: GATES — mean value + channel concentration, frame & global, 3 gates ----------
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for g in range(3):
        for sub, key, ls in [("frame", "gf", "-"), ("global", "gg", "--")]:
            for K in Ks:
                _, rows = data[K]
                st = [r["step"] for r in rows]
                gm = [r[key][g].mean() for r in rows]
                gp = [pr(r[key][g]) / 768 for r in rows]       # normalized channel participation in [0,1]
                ax[0][g].plot(st, gm, ls, color=kcol[K], lw=1.3, label=f"{sub} K{K}" if g == 0 else None)
                ax[1][g].plot(st, gp, ls, color=kcol[K], lw=1.3)
        ax[0][g].set_title(f"{GN[g]} mean (— frame, -- global)"); ax[0][g].set_xlabel("step"); ax[0][g].grid(alpha=.3)
        ax[1][g].set_title(f"{GN[g]} channel participation (1=uniform over 768)"); ax[1][g].set_xlabel("step"); ax[1][g].grid(alpha=.3)
    ax[0][0].legend(fontsize=6, ncol=2)
    fig.suptitle("GATES — mean value (top) & channel concentration (bottom); — frame, -- global")
    fig.tight_layout(); fig.savefig(f"{a.out}/gates.png", dpi=125); plt.close(fig)
    print(f"saved {a.out}/gates.png")

    # ---------- console summary at endpoint ----------
    print("\n=== endpoint (last step) summary ===")
    for K in Ks:
        _, rows = data[K]; r = rows[-1]
        nt = np.linalg.norm(r["z_out"][tti])
        hg = pr(r["anorm_g"][tti]); hf = pr(r["anorm_f"][tti])
        print(f"K{K:<4} |z_out[{tnm}]|={nt:8.1f}  eff#heads global={hg:4.1f} frame={hf:4.1f}  "
              f"s_out frame_mean={r['gf'][2].mean():.3f} global_mean={r['gg'][2].mean():.3f}")


if __name__ == "__main__":
    main()
