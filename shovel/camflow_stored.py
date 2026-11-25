#!/usr/bin/env python3
"""camflow_stored.py -- OFFLINE (no model): does information flow cam -> patch or patch -> cam?
Reads Stage-2 track dumps out/stage1_2_collapse/track_s10_K{K}/ (8 tracked frame-0 tokens: cam(REF variant), reg1-4, near_tex, flat, far).

(1) patch as query  : attention mass a patch gives to cam / registers
      frame attn (own frame 0, RoPE: specials sit at position (0,0))  and  global attn (frame-0 cam vs other frames' cams)
(2) cam/reg as query: where the cam (and registers) look -- self / registers / patches; how concentrated on patches
      frame attn; global attn (own frame vs other frames; specials vs patches)
(3) state updates   : how much each token changes in the frame sub-block vs the global sub-block per step
      d_frame = |z_frame - z_in| / |z_in|,  d_global = |z_out - z_frame| / |z_frame|   (z_in = previous z_out; step 0: z0)
Uniform-attention reference: frame 1/869 per key (cam 0.0012, 4 regs 0.0046); global 27 cams 0.0012, 108 regs 0.0046.

  python shovel/camflow_stored.py --Ks 8 16 32 64 128 --out out/stage3_latent/camflow_stored
"""
import argparse
import glob
import os

import numpy as np

PM = 5                       # cam + 4 registers
NAMES = ["cam", "reg1", "reg2", "reg3", "reg4", "near_tex", "flat", "far"]
PATCH = [5, 6, 7]


def load(K, root):
    d = f"{root}/track_s10_K{K}"
    files = sorted(glob.glob(d + "/step_*.npz"), key=lambda p: int(os.path.basename(p)[5:-4]))
    meta = np.load(d + "/meta.npz")
    return [np.load(f) for f in files], meta


def ent_eff(p):
    """effective number of keys = exp(entropy) of a renormalised distribution (last axis)."""
    p = p / np.clip(p.sum(-1, keepdims=True), 1e-12, None)
    h = -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)
    return np.exp(h)


def analyse(K, root):
    steps, meta = load(K, root)
    z_prev = meta["z0"].astype(np.float64)
    rows = []
    for st in steps:
        fr = st["frame_rows"].astype(np.float64)            # (8,12,869)
        gr = st["global_rows"].astype(np.float64)           # (8,12,27,869)
        r = dict(step=int(st["step"]))
        # ---- (1) patch -> specials
        r["pf_cam"] = fr[PATCH, :, 0]                       # (3,12)
        r["pf_reg"] = fr[PATCH, :, 1:PM].sum(-1)
        r["pg_cam_own"] = gr[PATCH, :, 0, 0]
        r["pg_cam_oth"] = gr[PATCH, :, 1:, 0].sum(-1)
        r["pg_reg_all"] = gr[PATCH, :, :, 1:PM].sum((-1, -2))
        # ---- (2) cam / reg as query
        for q, nm in [(0, "c"), (1, "r1")]:
            r[f"{nm}f_self"] = fr[q, :, q]
            r[f"{nm}f_spec"] = fr[q, :, :PM].sum(-1)
            r[f"{nm}f_patch"] = fr[q, :, PM:].sum(-1)
            r[f"{nm}f_eff"] = ent_eff(fr[q, :, PM:])        # effective #patches it reads (out of 864)
            r[f"{nm}g_own"] = gr[q, :, 0, :].sum(-1)
            r[f"{nm}g_spec"] = gr[q, :, :, :PM].sum((-1, -2))
            r[f"{nm}g_patch"] = gr[q, :, :, PM:].sum((-1, -2))
            r[f"{nm}g_eff"] = ent_eff(gr[q, :, :, PM:].reshape(12, -1))
        # ---- (3) state updates
        zf, zo = st["z_frame"].astype(np.float64), st["z_out"].astype(np.float64)
        r["d_frame"] = np.linalg.norm(zf - z_prev, axis=-1) / np.linalg.norm(z_prev, axis=-1).clip(1e-9)
        r["d_global"] = np.linalg.norm(zo - zf, axis=-1) / np.linalg.norm(zf, axis=-1).clip(1e-9)
        r["norm"] = np.linalg.norm(zo, axis=-1)
        r["anorm_f"] = st["anorm_frame"].astype(np.float64)
        r["anorm_g"] = st["anorm_global"].astype(np.float64)
        rows.append(r)
        z_prev = zo
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--root", default="out")
    ap.add_argument("--out", default="out/stage3_latent/camflow_stored")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = []
    def say(s=""):
        print(s); log.append(s)

    R = {K: analyse(K, a.root) for K in a.Ks}
    say("uniform reference: frame cam 0.0012 / 4 regs 0.0046 ; global 27 cams 0.0012 / 108 regs 0.0046")
    for K, rows in R.items():
        idx = sorted(set([0, 1, 2, 3, len(rows) // 4, len(rows) // 2, len(rows) - 1]))
        say(f"\n================ K={K} ================")
        say("(1) PATCH as query -> specials   [mean over 12 heads; per patch near_tex/flat/far]")
        say(f"{'step':>5} | {'frame: ->cam':>24} | {'frame: ->4 reg':>24} | {'global: ->own cam':>24} | {'global: ->26 other cams':>26} | {'global: ->108 regs':>24}")
        for i in idx:
            r = rows[i]
            f = lambda k: " ".join(f"{v:.4f}" for v in r[k].mean(1))
            say(f"{r['step']:>5} | {f('pf_cam'):>24} | {f('pf_reg'):>24} | {f('pg_cam_own'):>24} | {f('pg_cam_oth'):>26} | {f('pg_reg_all'):>24}")
        say("    max over heads of frame ->cam (near_tex/flat/far) at last step: " +
            " ".join(f"{v:.3f}(h{int(h)})" for v, h in zip(rows[-1]['pf_cam'].max(1), rows[-1]['pf_cam'].argmax(1))))
        for nm, lab in [("c", "CAM"), ("r1", "REG1")]:
            say(f"(2) {lab} as query   [mean over heads]  frame: self / specials / patches / eff#patches(of 864)   global: own-frame / specials / patches / eff#patches(of 23328)")
            for i in idx:
                r = rows[i]
                say(f"{r['step']:>5} | frame {r[nm+'f_self'].mean():.3f} {r[nm+'f_spec'].mean():.3f} {r[nm+'f_patch'].mean():.3f} {r[nm+'f_eff'].mean():7.1f}"
                    f" | global {r[nm+'g_own'].mean():.3f} {r[nm+'g_spec'].mean():.3f} {r[nm+'g_patch'].mean():.3f} {r[nm+'g_eff'].mean():8.1f}")
        say("(3) relative state change per sub-block  d_frame / d_global   (cam, reg1, near_tex, flat, far) and |z_out|")
        for i in idx:
            r = rows[i]
            say(f"{r['step']:>5} | " + "  ".join(f"{NAMES[t]}:{r['d_frame'][t]:.3f}/{r['d_global'][t]:.3f}" for t in [0, 1, 5, 6, 7])
                + "  | |z| cam {:.1f} patch {:.1f}".format(r["norm"][0], r["norm"][PATCH].mean()))
    with open(f"{a.out}/report.txt", "w") as fh:
        fh.write("\n".join(log) + "\n")

    # ---------------- plots
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pc = ["tab:red", "tab:green", "tab:blue"]
    fig, ax = plt.subplots(len(a.Ks), 4, figsize=(20, 3.6 * len(a.Ks)), squeeze=False)
    for r_, K in enumerate(a.Ks):
        rows = R[K]; s = [r["step"] for r in rows]
        for j, (t, c) in enumerate(zip(["near_tex", "flat", "far"], pc)):
            ax[r_, 0].plot(s, [r["pf_cam"][j].mean() for r in rows], color=c, label=f"{t} → cam")
            ax[r_, 0].plot(s, [r["pf_reg"][j].mean() for r in rows], color=c, ls="--", label=f"{t} → 4 reg")
        ax[r_, 0].axhline(1 / 869, color="k", lw=.7, ls=":"); ax[r_, 0].set_yscale("log")
        ax[r_, 0].set_title(f"K={K} frame attn: patch → specials (head mean)", fontsize=9)
        for j, (t, c) in enumerate(zip(["near_tex", "flat", "far"], pc)):
            ax[r_, 1].plot(s, [r["pg_cam_own"][j].mean() for r in rows], color=c, label=f"{t} → own cam")
            ax[r_, 1].plot(s, [r["pg_cam_oth"][j].mean() for r in rows], color=c, ls="--", label=f"{t} → 26 other cams")
        ax[r_, 1].axhline(1 / 23463, color="k", lw=.7, ls=":"); ax[r_, 1].set_yscale("log")
        ax[r_, 1].set_title("global attn: patch → cam tokens", fontsize=9)
        for k_, lab, c in [("cf_self", "cam→self (frame)", "k"), ("cf_spec", "cam→specials (frame)", "tab:purple"),
                           ("cf_patch", "cam→patches (frame)", "tab:orange"), ("cg_own", "cam→own frame (global)", "tab:cyan"),
                           ("cg_patch", "cam→patches (global)", "tab:brown")]:
            ax[r_, 2].plot(s, [r[k_].mean() for r in rows], color=c, label=lab)
        ax[r_, 2].set_ylim(0, 1.02); ax[r_, 2].set_title("cam as query: where its attention goes", fontsize=9)
        for t, c in zip([0, 5, 6, 7], ["k", *pc]):
            ax[r_, 3].plot(s, [r["d_frame"][t] for r in rows], color=c, label=f"{NAMES[t]} frame")
            ax[r_, 3].plot(s, [r["d_global"][t] for r in rows], color=c, ls="--", label=f"{NAMES[t]} global")
        ax[r_, 3].set_yscale("log"); ax[r_, 3].set_title("relative state change per sub-block", fontsize=9)
        for c_ in range(4): ax[r_, c_].grid(alpha=.3); ax[r_, c_].set_xlabel("step")
        if r_ == 0:
            for c_ in range(4): ax[r_, c_].legend(fontsize=6.5, ncol=2)
    fig.tight_layout(); fig.savefig(f"{a.out}/camflow_overview.png", dpi=110); plt.close(fig)

    K = 16 if 16 in R else a.Ks[0]; rows = R[K]
    fig, ax = plt.subplots(1, 4, figsize=(20, 3.6))
    for j, t in enumerate(["near_tex", "flat", "far"]):
        M = np.array([r["pf_cam"][j] for r in rows]).T
        im = ax[j].imshow(M, aspect="auto", cmap="magma"); plt.colorbar(im, ax=ax[j])
        ax[j].set_title(f"K={K} frame attn: {t} → cam, per head", fontsize=9); ax[j].set_xlabel("step"); ax[j].set_ylabel("head")
    M = np.array([r["cf_patch"] for r in rows]).T
    im = ax[3].imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1); plt.colorbar(im, ax=ax[3])
    ax[3].set_title(f"K={K} frame attn: cam → all patches, per head", fontsize=9); ax[3].set_xlabel("step"); ax[3].set_ylabel("head")
    fig.tight_layout(); fig.savefig(f"{a.out}/perhead_K{K}.png", dpi=110); plt.close(fig)
    print(f"saved {a.out}/report.txt, camflow_overview.png, perhead_K{K}.png")


if __name__ == "__main__":
    main()
