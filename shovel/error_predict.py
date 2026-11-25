#!/usr/bin/env python3
"""error_predict.py -- is DVLT's remaining error PREDICTABLE from its own tokens (i.e. is there a computable residual)?

Part 1  DVLT's own depth confidence (decoder depth_conf) vs the actual per-pixel depth error (median-scaled log error):
        Spearman(conf, |err|), median |err| of the most-confident 50% of pixels vs the rest.
Part 2  predict the SIGNED per-patch log-depth error  e = logdepth_pred + log(scale) - logdepth_gt  from tokens:
        inputs: token | frame-mean-removed token | token + mean of its 3x3 spatial neighbours (1536-d)
        models: ridge (lambda by 5-fold CV on the training frames) and a 2-layer MLP (256 hidden, early stopping)
        split: even frames -> odd frames and odd -> even; report the mean over both directions of
          R2 on the held-out frames, and AbsRel of the held-out frames before / after subtracting the predicted error.
        Also: share of the error variance that is a per-frame constant (cannot be learned per patch by construction is
        NOT excluded, it is just reported).
States: K16 steps 4, 8, 16 and K64 step 32 (collapsed), from out/stage3_latent/tokendump.
Data: out/stage3_latent/errpredict/errpredict.json, summary.txt, errpredict.png

  PYTHONPATH=$(pwd) python shovel/error_predict.py --out out/stage3_latent/errpredict
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlt_probe import Probe, load_gt  # noqa: E402

PM = 5


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + lam * len(Xs) * np.eye(X.shape[1]), Xs.T @ (y - y.mean()))
    return lambda Z: ((Z - mu) / sd) @ w + y.mean()


def ridge_cv(X, y, groups):
    best, bl = -1e9, None
    ug = np.unique(groups); folds = np.array_split(np.random.default_rng(0).permutation(ug), 5)
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        sc = []
        for f in folds:
            te = np.isin(groups, f)
            sc.append(r2(y[te], ridge_fit(X[~te], y[~te], lam)(X[te])))
        if np.mean(sc) > best:
            best, bl = np.mean(sc), lam
    return ridge_fit(X, y, bl), bl


def mlp_fit(X, y, dev):
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X)); nv = len(X) // 8
    va, tr = idx[:nv], idx[nv:]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.from_numpy((X - mu) / sd).float().to(dev); yt = torch.from_numpy(y - y.mean()).float().to(dev)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.GELU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(256, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, state, bad = 1e9, None, 0
    for ep in range(300):
        net.train()
        for b in np.array_split(rng.permutation(tr), max(len(tr) // 256, 1)):
            loss = (net(Xt[b])[:, 0] - yt[b]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float((net(Xt[va])[:, 0] - yt[va]).pow(2).mean())
        if vl < best - 1e-6:
            best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad > 20:
                break
    net.load_state_dict(state); net.eval()
    ym = y.mean()

    def pred(Z):
        with torch.no_grad():
            return net(torch.from_numpy((Z - mu) / sd).float().to(dev))[:, 0].cpu().numpy() + ym
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="out/stage3_latent/tokendump")
    ap.add_argument("--gt", default="out/data/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--states", nargs="+", default=["K16:4", "K16:8", "K16:16", "K64:32"])
    ap.add_argument("--out", default="out/stage3_latent/errpredict")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    P = Probe("nvidia/dvlt", max_frames=27)
    dev = P.acc.device
    meta = np.load(f"{a.dump}/K16/meta.npz")
    gh, gw = [int(v) for v in meta["grid"]]
    H, W = gh * 14, gw * 14
    S = int(meta["S"]); Np = gh * gw
    rope = torch.from_numpy(meta["rope"]).to(dev)
    gt_d, _, _ = load_gt(a.gt, max(H, W), device=dev); gt_d = gt_d[:S].float()
    gt_ld = meta["gt_logdepth"]                                                       # (S,Np) nan invalid
    res, lines = {}, []
    for spec in a.states:
        Kn, s = spec.split(":"); s = int(s)
        z = np.load(f"{a.dump}/{Kn}/step_{s:03d}.npy").astype(np.float32)
        ld = np.load(f"{a.dump}/{Kn}/targets_{s:03d}.npz")["logdepth"]                # (S,Np)
        R = res.setdefault(spec, {})
        # ---------------- part 1: depth confidence
        zt = torch.from_numpy(z).to(dev)
        ds, cs = [], []
        with torch.no_grad():
            for i in range(0, S, 4):
                pr = P.m._decode(zt[i:i + 4], H, W, 1, min(4, S - i), rope[i:i + 4])
                ds.append(pr["depth"][0, ..., 0].float()); cs.append(pr["depth_conf"][0].float())
        d, c = torch.cat(ds), torch.cat(cs)
        m = gt_d > 1e-6
        sc = (gt_d[m] / d[m]).median()
        err_px = ((d * sc).clamp_min(1e-6).log() - gt_d.clamp_min(1e-6).log())[m].cpu().numpy()
        conf = c[m].cpu().numpy()
        sub = np.random.default_rng(0).choice(len(conf), min(400000, len(conf)), replace=False)
        hi = conf >= np.median(conf)
        R["conf"] = dict(spearman_conf_abs_err=spearman(conf[sub], np.abs(err_px[sub])),
                         med_abs_err_conf_top50=float(np.median(np.abs(err_px[hi]))),
                         med_abs_err_conf_bottom50=float(np.median(np.abs(err_px[~hi]))))
        # ---------------- part 2: predict signed per-patch error
        valid = ~np.isnan(gt_ld)
        p_, g_ = np.exp(ld[valid]), np.exp(gt_ld[valid]); scale = np.median(g_ / p_)
        e = ld + np.log(scale) - gt_ld                                                  # (S,Np) signed log error
        frame_mean = np.array([np.nanmean(e[f]) for f in range(S)])
        var_frame = float(np.nanvar(np.repeat(frame_mean[:, None], Np, 1)[valid]) / np.nanvar(e[valid]))
        tok = z[:, PM:]                                                                  # (S,Np,768)
        tok_mr = tok - tok.mean(1, keepdims=True)
        g = tok.reshape(S, gh, gw, -1)
        pad = np.pad(g, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
        nb = sum(pad[:, 1 + dy:1 + dy + gh, 1 + dx:1 + dx + gw] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)) / 8
        tok_ctx = np.concatenate([tok, nb.reshape(S, Np, -1)], -1)
        inputs = {"token": tok, "token-framemean": tok_mr, "token+3x3ctx": tok_ctx}
        frames = np.arange(S)
        R["frame_constant_share_of_error_var"] = var_frame
        R["absrel_before"] = float(np.mean(np.abs(p_ * scale - g_) / g_))
        for iname, X in inputs.items():
            for model in ["ridge", "mlp"]:
                r2s, ab_b, ab_a = [], [], []
                for trset in (frames % 2 == 0, frames % 2 == 1):
                    tr_m = valid & trset[:, None]; te_m = valid & ~trset[:, None]
                    Xtr, ytr = X[tr_m], e[tr_m]
                    grp = np.repeat(frames[:, None], Np, 1)[tr_m]
                    f = ridge_cv(Xtr, ytr, grp)[0] if model == "ridge" else mlp_fit(Xtr, ytr, dev)
                    pred = f(X[te_m])
                    r2s.append(r2(e[te_m], pred))
                    pd_, gd_ = np.exp(ld[te_m]) * scale, np.exp(gt_ld[te_m])
                    ab_b.append(float(np.mean(np.abs(pd_ - gd_) / gd_)))
                    pc = pd_ * np.exp(-pred)
                    pc = pc * np.median(gd_ / pc)
                    ab_a.append(float(np.mean(np.abs(pc - gd_) / gd_)))
                R[f"{iname}|{model}"] = dict(r2=float(np.mean(r2s)), absrel_before=float(np.mean(ab_b)), absrel_after=float(np.mean(ab_a)))
        lines.append(f"\n===== {spec} =====  per-patch AbsRel {R['absrel_before']:.4f}; per-frame-constant share of error variance {var_frame:.3f}")
        cf = R["conf"]
        lines.append(f"  depth_conf: Spearman(conf, |err|) {cf['spearman_conf_abs_err']:+.3f}; median |log err| most-confident 50% "
                     f"{cf['med_abs_err_conf_top50']:.4f} vs rest {cf['med_abs_err_conf_bottom50']:.4f}")
        for k, v in R.items():
            if "|" in k:
                lines.append(f"  {k:>24}: held-out R2 {v['r2']:+.3f} | held-out AbsRel {v['absrel_before']:.4f} -> {v['absrel_after']:.4f} after subtracting the prediction")
        print("\n".join(lines[-9:]), flush=True)
    json.dump(res, open(f"{a.out}/errpredict.json", "w"), indent=1)
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = [k for k in res[a.states[0]] if "|" in k]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for j, k in enumerate(keys):
        ax[0].plot(a.states, [res[s][k]["r2"] for s in a.states], "-o", label=k)
        ax[1].plot(a.states, [res[s][k]["absrel_after"] / res[s][k]["absrel_before"] for s in a.states], "-o", label=k)
    ax[0].axhline(0, color="k", lw=.8); ax[0].set_title("held-out R2 of the predicted signed error"); ax[0].legend(fontsize=7)
    ax[1].axhline(1, color="k", lw=.8); ax[1].set_title("held-out AbsRel after / before subtracting the prediction")
    for x in ax: x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{a.out}/errpredict.png", dpi=120); plt.close(fig)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
