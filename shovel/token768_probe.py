#!/usr/bin/env python3
"""token768_probe.py -- what is in DVLT's 768-d patch tokens? (frozen model, no training except linear probes)

--dump    : run DVLT once (s10, 27 frames, K=16), save the FULL state (27 x 869 x 768, fp16) at a few steps, plus per-patch
            targets decoded by DVLT itself at those steps (depth, ray direction, ray origin, world point), GT patch depth,
            patch mean RGB, and GT correspondences between frame pairs (for evaluation only).
--analyze : offline, five questions
  Q1/Q2  single-patch linear decodability: ridge probe  token(768) -> target, trained on even frames, tested on odd
         frames (held-out R^2). Two inputs: the patch token alone, and [patch token, frame-mean token] (collective info).
  Q3     subspaces: principal angles between the probe directions for depth / ray direction / ray origin / RGB.
  Q4     is the camera ADDITIVE? (a) variance of patch tokens: between-frame (frame-mean) vs within-frame part;
         (b) probe ray origin from raw tokens vs from tokens with their frame mean removed;
         (c) GT-corresponding patches across frames: cosine before / after removing each frame's mean (vs random pairs).
  Q5     massive channels: per-channel energy spectrum, top channels by token type, correlation with targets, spatial maps.
Targets are DVLT's OWN decoded quantities at the same step (what the token must encode for the heads); GT only for
depth (extra) and correspondences (evaluation).

  PYTHONPATH=$(pwd) python shovel/token768_probe.py --dump    --out out/stage3_latent/token768
  PYTHONPATH=$(pwd) python shovel/token768_probe.py --analyze --out out/stage3_latent/token768
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PM = 5
TARGETS = ["depth", "dir", "origin", "point", "rgb"]


# ============================================================================ dump
def dump(a):
    import torch
    from dvlt_probe import Probe, load_gt, gt_correspondence
    P = Probe("nvidia/dvlt", max_frames=a.max_frames)
    images = P.load(a.input)
    x0, rope, dims = P.init_state(images)
    B, S, H, W = dims
    ps = P.m.patch_size; gh, gw = H // ps, W // ps
    P.gt_depth, P.gt_c2w, P.gt_K = load_gt(a.gt, P.img_size, device=P.acc.device)
    P.gt_depth, P.gt_c2w = P.gt_depth[:S], P.gt_c2w[:S]

    def per_patch(t):                       # (S,H,W,C) -> (S, gh*gw, C), mean over each 14x14 patch
        S_, _, _, C = t.shape
        return t.reshape(S_, gh, ps, gw, ps, C).mean((2, 4)).reshape(S_, gh * gw, C)

    def targets(x):
        pr = P.decode(x, rope, dims)
        rays = pr["rays"][0].float(); dep = pr["depth"][0].float()          # (S,H,W,6), (S,H,W,1)
        d = rays[..., :3] / rays[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-9)
        pts = rays[..., 3:6] + rays[..., :3] * dep
        out = dict(depth=per_patch(dep.clamp_min(1e-6).log())[..., 0], dir=per_patch(d), origin=per_patch(rays[..., 3:6]),
                   point=per_patch(pts))
        out["dir"] = out["dir"] / out["dir"].norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}

    save = sorted(set(a.steps))
    states, tg = {}, {}
    with torch.no_grad():
        x = x0
        if 0 in save:
            states[0] = x.cpu().numpy().astype(np.float16); tg[0] = targets(x)
        for i in range(a.K):
            x = P.step(x, i, a.K, rope, B, S)
            if i + 1 in save:
                states[i + 1] = x.cpu().numpy().astype(np.float16); tg[i + 1] = targets(x)
                print(f"saved step {i + 1}")
    rgb = per_patch(images[0].permute(0, 2, 3, 1).float()).cpu().numpy()            # (S,864,3)
    gd = P.gt_depth.float()
    gdp = per_patch(torch.where(gd > 1e-6, gd.log(), torch.full_like(gd, float("nan")))[..., None])[..., 0].cpu().numpy()
    # GT correspondences for evaluation: adjacent pairs and some far pairs
    pairs = [(f, f + 1) for f in range(1, S - 1)] + [(f, f + 6) for f in range(1, S - 6, 3)]
    corr = []
    for (i, j) in pairs:
        valid, gyj, gxj = gt_correspondence(P, i, j, gh, gw, ps, H, W, occ_thresh=0.1)
        v = valid.flatten().cpu().numpy()
        src = np.nonzero(v)[0]
        dst = (gyj.flatten().cpu().numpy() * gw + gxj.flatten().cpu().numpy())[src]
        corr.append(np.stack([np.full_like(src, i), src, np.full_like(src, j), dst], 1))
    corr = np.concatenate(corr) if corr else np.zeros((0, 4), int)
    np.savez_compressed(f"{a.out}/states_K{a.K}.npz", steps=np.array(save),
                        **{f"x_{s}": states[s] for s in save})
    np.savez_compressed(f"{a.out}/targets_K{a.K}.npz", steps=np.array(save), rgb=rgb, gt_logdepth=gdp, corr=corr,
                        grid=np.array([gh, gw]), **{f"{k}_{s}": tg[s][k] for s in save for k in tg[s]})
    print(f"wrote {a.out}/states_K{a.K}.npz, targets_K{a.K}.npz ; {len(corr)} GT correspondences")


# ============================================================================ analyze helpers
def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    A = Xs.T @ Xs + lam * len(Xs) * np.eye(Xs.shape[1])
    Wt = np.linalg.solve(A, Xs.T @ (Y - ym))
    return Wt, mu, sd, ym


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1e-3):
    Wt, mu, sd, ym = ridge_fit(Xtr, Ytr, lam)
    pred = ((Xte - mu) / sd) @ Wt + ym
    return 1 - ((Yte - pred) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum(), Wt


def principal_cos(A, B):
    qa, _ = np.linalg.qr(A); qb, _ = np.linalg.qr(B)
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


# ============================================================================ analyze
def analyze(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    st = np.load(f"{a.out}/states_K{a.K}.npz"); tg = np.load(f"{a.out}/targets_K{a.K}.npz")
    steps = [int(s) for s in st["steps"]]
    S = st[f"x_{steps[0]}"].shape[0]
    tr = np.array([f for f in range(S) if f % 2 == 0]); te = np.array([f for f in range(S) if f % 2 == 1])
    log = []
    def say(s=""):
        print(s); log.append(s)

    def tgt(name, s):
        if name == "rgb":
            return tg["rgb"]
        if name == "gtdepth":
            return tg["gt_logdepth"][..., None]
        v = tg[f"{name}_{s}"]
        return v[..., None] if v.ndim == 2 else v

    # ---------------- Q1/Q2
    say("Q1/Q2  held-out-frame R^2 of a ridge probe  (train even frames, test odd frames)")
    names = ["depth", "gtdepth", "dir", "origin", "point", "rgb"]
    R2 = {m: np.zeros((len(names), len(steps))) for m in ["patch", "patch+framemean"]}
    probes = {}
    for si, s in enumerate(steps):
        X = st[f"x_{s}"].astype(np.float32)[:, PM:]                       # (S,864,768)
        fm = X.mean(1, keepdims=True).repeat(X.shape[1], 1)
        for ni, n in enumerate(names):
            Y = tgt(n, s).astype(np.float32)
            ok = np.isfinite(Y).all(-1)
            for m, XX in [("patch", X), ("patch+framemean", np.concatenate([X, fm], -1))]:
                Xtr, Ytr = XX[tr][ok[tr]], Y[tr][ok[tr]]
                Xte, Yte = XX[te][ok[te]], Y[te][ok[te]]
                r2, Wt = ridge_r2(Xtr, Ytr, Xte, Yte)
                R2[m][ni, si] = r2
                if m == "patch":
                    probes[(n, s)] = Wt
        say(f"  step {s:>2}: " + "  ".join(f"{n}={R2['patch'][ni, si]:.3f}/{R2['patch+framemean'][ni, si]:.3f}" for ni, n in enumerate(names))
            + "    (patch / patch+framemean)")

    # ---------------- Q3
    say("\nQ3  cosines of principal angles between probe directions (single patch probes; 1 = shared direction, 0 = orthogonal)")
    q3 = {}
    for s in steps:
        pairs = [("depth", "dir"), ("depth", "origin"), ("dir", "origin"), ("depth", "rgb"), ("origin", "rgb")]
        q3[s] = {f"{u}-{v}": principal_cos(probes[(u, s)], probes[(v, s)]) for u, v in pairs}
        say(f"  step {s:>2}: " + "  ".join(f"{k}=[{', '.join(f'{c:.2f}' for c in vals)}]" for k, vals in q3[s].items()))

    # ---------------- Q4
    say("\nQ4  additive camera?   (a) between-frame variance fraction   (b) origin probe R^2 raw vs frame-mean-removed"
        "   (c) cosine of GT-corresponding patches across frames, raw -> frame-mean-removed  (random pairs in brackets)")
    corr = tg["corr"]
    rng = np.random.default_rng(0)
    q4 = {"between": [], "origin_raw": [], "origin_rm": [], "corr_raw": [], "corr_rm": [], "rand_raw": [], "rand_rm": []}
    for s in steps:
        X = st[f"x_{s}"].astype(np.float32)[:, PM:]
        mu = X.reshape(-1, X.shape[-1]).mean(0)
        fm = X.mean(1, keepdims=True)
        tot = ((X - mu) ** 2).sum(); btw = (X.shape[1] * ((fm - mu) ** 2)).sum()
        Xr = X - fm
        Y = tgt("origin", s)
        r_raw, _ = ridge_r2(X[tr].reshape(-1, 768), Y[tr].reshape(-1, 3), X[te].reshape(-1, 768), Y[te].reshape(-1, 3))
        r_rm, _ = ridge_r2(Xr[tr].reshape(-1, 768), Y[tr].reshape(-1, 3), Xr[te].reshape(-1, 768), Y[te].reshape(-1, 3))
        cos = lambda U, V: ((U * V).sum(-1) / (np.linalg.norm(U, axis=-1) * np.linalg.norm(V, axis=-1) + 1e-9))
        fi, pi_, fj, pj = corr[:, 0], corr[:, 1], corr[:, 2], corr[:, 3]
        rj = rng.integers(0, X.shape[1], len(corr))
        q4["between"].append(btw / tot); q4["origin_raw"].append(r_raw); q4["origin_rm"].append(r_rm)
        q4["corr_raw"].append(cos(X[fi, pi_], X[fj, pj]).mean()); q4["corr_rm"].append(cos(Xr[fi, pi_], Xr[fj, pj]).mean())
        q4["rand_raw"].append(cos(X[fi, pi_], X[fj, rj]).mean()); q4["rand_rm"].append(cos(Xr[fi, pi_], Xr[fj, rj]).mean())
        say(f"  step {s:>2}: (a) between-frame {btw / tot:.3f}  (b) origin R^2 {r_raw:.3f} -> {r_rm:.3f}"
            f"  (c) corr {q4['corr_raw'][-1]:.3f} -> {q4['corr_rm'][-1]:.3f}  [random {q4['rand_raw'][-1]:.3f} -> {q4['rand_rm'][-1]:.3f}]")
    say(f"  ({len(corr)} GT correspondences: adjacent frame pairs + some 6-apart pairs)")

    # ---------------- Q5
    say("\nQ5  channel energy (patch tokens): share of the top channels, and what they correlate with")
    q5 = {}
    for s in steps:
        Xall = st[f"x_{s}"].astype(np.float32)
        Xp = Xall[:, PM:].reshape(-1, 768); Xs_ = Xall[:, :PM].reshape(-1, 768)
        e = (Xp ** 2).mean(0); order = np.argsort(-e); share = e[order] / e.sum()
        eff = e.sum() ** 2 / (e ** 2).sum()
        top = order[:5]
        dep = tgt("depth", s).reshape(-1); org = tgt("origin", s).reshape(-1, 3)
        cd = [np.corrcoef(Xp[:, c], dep)[0, 1] for c in top]
        q5[s] = dict(order=order, share=share, eff=eff, top=top)
        say(f"  step {s:>2}: eff#channels {eff:.1f}/768, top-5 share {share[:5].sum():.3f}, top channels {top.tolist()}"
            f" | mean value patch {Xp[:, top].mean(0).round(1).tolist()} special {Xs_[:, top].mean(0).round(1).tolist()}"
            f" | corr with depth {[round(float(c), 2) for c in cd]}")

    with open(f"{a.out}/report.txt", "w") as fh:
        fh.write("\n".join(log) + "\n")

    # ---------------- figures
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    for k, m in enumerate(["patch", "patch+framemean"]):
        im = ax[k].imshow(R2[m], vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax[k].set_xticks(range(len(steps))); ax[k].set_xticklabels(steps); ax[k].set_yticks(range(len(names))); ax[k].set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(steps)):
                ax[k].text(j, i, f"{R2[m][i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if R2[m][i, j] < .6 else "k")
        ax[k].set_title(f"Q1/Q2 held-out R²: input = {m}", fontsize=10); ax[k].set_xlabel("step")
    fig.colorbar(im, ax=ax, shrink=.8); fig.savefig(f"{a.out}/q12_probe_r2.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(1, len(q3[steps[0]]), figsize=(18, 3.2))
    for k, key in enumerate(q3[steps[0]]):
        for s in steps:
            ax[k].plot(range(1, len(q3[s][key]) + 1), q3[s][key], marker="o", label=f"step {s}")
        ax[k].set_ylim(0, 1); ax[k].set_title(f"Q3 {key}", fontsize=10); ax[k].set_xlabel("principal angle #"); ax[k].grid(alpha=.3)
    ax[0].set_ylabel("cos"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{a.out}/q3_subspace_angles.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    ax[0].plot(steps, q4["between"], "o-"); ax[0].set_title("Q4a between-frame share of patch-token variance", fontsize=10); ax[0].set_ylim(0, 1)
    ax[1].plot(steps, q4["origin_raw"], "o-", label="raw tokens"); ax[1].plot(steps, q4["origin_rm"], "s-", label="frame mean removed")
    ax[1].set_title("Q4b ray-origin probe R² (held-out frames)", fontsize=10); ax[1].legend(fontsize=8); ax[1].set_ylim(min(0, min(q4["origin_rm"])), 1)
    ax[2].plot(steps, q4["corr_raw"], "o-", c="tab:blue", label="GT-corresponding, raw")
    ax[2].plot(steps, q4["corr_rm"], "s-", c="tab:blue", ls="--", label="GT-corresponding, mean removed")
    ax[2].plot(steps, q4["rand_raw"], "o-", c="tab:gray", label="random, raw")
    ax[2].plot(steps, q4["rand_rm"], "s-", c="tab:gray", ls="--", label="random, mean removed")
    ax[2].set_title("Q4c cosine between patches of different frames", fontsize=10); ax[2].legend(fontsize=7)
    for x_ in ax: x_.set_xlabel("step"); x_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{a.out}/q4_additive_camera.png", dpi=120); plt.close(fig)

    gh, gw = [int(v) for v in tg["grid"]]
    fig, ax = plt.subplots(2, len(steps), figsize=(3.4 * len(steps), 6))
    for j, s in enumerate(steps):
        ax[0, j].semilogy(q5[s]["share"][:50], "o-", ms=3); ax[0, j].set_title(f"step {s}: eff {q5[s]['eff']:.1f}", fontsize=9)
        ax[0, j].set_xlabel("channel rank"); ax[0, j].grid(alpha=.3)
        c = q5[s]["top"][0]
        m = st[f"x_{s}"].astype(np.float32)[1, PM:, c].reshape(gh, gw)
        im = ax[1, j].imshow(m, cmap="coolwarm"); ax[1, j].set_title(f"frame 1, channel {c}", fontsize=9); ax[1, j].axis("off")
        plt.colorbar(im, ax=ax[1, j], shrink=.7)
    ax[0, 0].set_ylabel("energy share")
    fig.tight_layout(); fig.savefig(f"{a.out}/q5_channels.png", dpi=120); plt.close(fig)
    print(f"saved {a.out}/report.txt and q12/q3/q4/q5 figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--input", default="out/data/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--gt", default="out/data/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--out", default="out/stage3_latent/token768")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.dump:
        dump(a)
    if a.analyze:
        analyze(a)


if __name__ == "__main__":
    main()
