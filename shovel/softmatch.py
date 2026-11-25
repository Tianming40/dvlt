#!/usr/bin/env python3
"""softmatch.py -- how precise is ATTENTION-WEIGHTED (soft-argmax) matching with DVLT's own global attention?

For K16 steps --steps (attention of step s is computed from the stored state after s-1 steps, original schedule),
the global attention's Q, K are captured (attn_freeze.GlobalAttnControl). For a query patch of frame A, its attention
logits over the 864 patches of frame B (renormalised within B) give a distribution over B's patch centres.
Position estimates (pixels):
  argmax              centre of the most attended patch
  soft T              global soft-argmax  sum_i softmax(logits / T)_i * centre_i,  T in --temps
  win3 / win5         soft-argmax restricted to the 3x3 / 5x5 patches around the argmax (T = 1)
Heads: each of --heads, and the mean of their logits.
GT: patch centre of A + GT depth/pose/K -> continuous pixel in B (occlusion-checked); frame gaps --gaps.
Metrics: median pixel error, share < 3 / 7 / 14 px; confidence (peak prob, window spatial std) vs error: Spearman rho and
median error of the 50% most confident matches.
Data: out/stage3_latent/softmatch/softmatch.json, summary.txt, softmatch.png

  PYTHONPATH=$(pwd) python shovel/softmatch.py --out out/stage3_latent/softmatch
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvlt_probe import Probe, load_gt  # noqa: E402
from attn_freeze import GlobalAttnControl  # noqa: E402

PM = 5


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="out/stage3_latent/tokendump/K16")
    ap.add_argument("--gt", default="out/data/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--steps", type=int, nargs="+", default=[3, 4, 6])
    ap.add_argument("--heads", type=int, nargs="+", default=[0, 9])
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--temps", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    ap.add_argument("--occ", type=float, default=0.05)
    ap.add_argument("--out", default="out/stage3_latent/softmatch")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    P = Probe("nvidia/dvlt", max_frames=27)
    dev = P.acc.device
    meta = np.load(f"{a.dump}/meta.npz")
    gh, gw = [int(v) for v in meta["grid"]]
    H, W = gh * 14, gw * 14
    Np = gh * gw
    S = int(meta["S"]); K = int(meta["K"])
    rope = torch.from_numpy(meta["rope"]).to(dev)
    ctl = GlobalAttnControl(P.m)
    nh = P.m.recurrent_blocks[0].global_attn.attn.num_heads
    hd = P.m.recurrent_blocks[0].global_attn.attn.head_dim

    gt_d, c2w, Kin = load_gt(a.gt, max(H, W))
    gt_d, c2w = gt_d[:S].float(), c2w[:S].float()
    jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
    uc = torch.from_numpy(jj * 14 + 7.0).float().flatten(); vc = torch.from_numpy(ii * 14 + 7.0).float().flatten()
    centres = torch.stack([uc, vc], -1).to(dev)                                       # (Np,2) pixel centres
    Kinv = torch.linalg.inv(Kin)

    def gt_corr(fa, fb):
        d = gt_d[fa, vc.long(), uc.long()]
        Xc = (torch.stack([uc, vc, torch.ones_like(uc)], -1) @ Kinv.T) * d[:, None]
        Xw = Xc @ c2w[fa, :3, :3].T + c2w[fa, :3, 3]
        w2c = torch.linalg.inv(c2w[fb]); Xb = Xw @ w2c[:3, :3].T + w2c[:3, 3]
        zb = Xb[:, 2]; p = Xb @ Kin.T
        u, v = p[:, 0] / zb.clamp_min(1e-6), p[:, 1] / zb.clamp_min(1e-6)
        ok = (d > 1e-6) & (zb > 1e-3) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        db = gt_d[fb, v.clamp(0, H - 1).long(), u.clamp(0, W - 1).long()]
        ok = ok & (db > 1e-6) & ((zb / db.clamp_min(1e-6) - 1).abs() < a.occ)
        return ok, torch.stack([u, v], -1)

    ts = torch.linspace(0.0, 1.0, K).tolist()
    res, lines = {}, []
    variants = [f"h{h}" for h in a.heads] + ["mean(" + "+".join(map(str, a.heads)) + ")", "mean(all12)"]
    for s in a.steps:
        z = torch.from_numpy(np.load(f"{a.dump}/step_{s - 1:03d}.npy").astype(np.float32)).to(dev)
        ctl.mode = "capture"
        with torch.no_grad():
            P.m._interval_step(z, ts[s - 1], ts[s] if s < K else 1.0, rope, 1, S)
        ctl.mode = "normal"
        q, k = ctl.qk                                                                  # (1, heads, S*P, hd)
        q = q[0].float().reshape(nh, S, -1, hd)[:, :, PM:]; k = k[0].float().reshape(nh, S, -1, hd)[:, :, PM:]   # (h,S,Np,hd)
        acc = {v: {"err": {}, "conf_peak": [], "conf_std": [], "err_win3": []} for v in variants}
        store = {v: {m: [] for m in ["argmax"] + [f"soft{T:g}" for T in a.temps] + ["win3", "win5"]} for v in variants}
        gaps_all = []
        for fa in range(S):
            for g in a.gaps:
                for fb in (fa + g, fa - g):
                    if not 0 <= fb < S:
                        continue
                    ok, gtuv = gt_corr(fa, fb)
                    idx = torch.nonzero(ok).flatten()
                    if len(idx) == 0:
                        continue
                    gtuv = gtuv[idx].to(dev)
                    logit_h = torch.einsum("hqd,hkd->hqk", q[:, fa, idx.to(dev)], k[:, fb]) / hd ** 0.5   # (h,nq,Np)
                    for v in variants:
                        if v.startswith("h"):
                            L = logit_h[int(v[1:])]
                        elif v == "mean(all12)":
                            L = logit_h.mean(0)
                        else:
                            L = logit_h[a.heads].mean(0)
                        am = L.argmax(-1)
                        store[v]["argmax"].append((centres[am] - gtuv).norm(dim=-1))
                        for T in a.temps:
                            p = torch.softmax(L / T, -1)
                            store[v][f"soft{T:g}"].append((p @ centres - gtuv).norm(dim=-1))
                        p1 = torch.softmax(L, -1)
                        r0, c0 = am // gw, am % gw
                        rr = torch.arange(gh, device=dev)[None, :, None]; cc = torch.arange(gw, device=dev)[None, None, :]
                        for wname, rad in [("win3", 1), ("win5", 2)]:
                            msk = ((rr - r0[:, None, None]).abs() <= rad) & ((cc - c0[:, None, None]).abs() <= rad)
                            pw = p1 * msk.reshape(len(am), -1)
                            pw = pw / pw.sum(-1, keepdim=True).clamp_min(1e-12)
                            pos = pw @ centres
                            store[v][wname].append((pos - gtuv).norm(dim=-1))
                            if wname == "win3":
                                spread = ((centres[None] - pos[:, None]).pow(2).sum(-1) * pw).sum(-1).sqrt()
                                acc[v]["conf_std"].append(spread)
                        acc[v]["conf_peak"].append(p1.max(-1).values)
                    gaps_all.append(torch.full((len(idx),), g))
        gaps_all = torch.cat(gaps_all).numpy()
        R = res.setdefault(f"step{s}", {})
        lines.append(f"\n===== global attention of step {s} (K16), frame gaps {a.gaps} =====")
        lines.append("variant      method   | median px | <3px  <7px  <14px | by gap (median px)")
        for v in variants:
            R[v] = {}
            for mth, lst in store[v].items():
                e = torch.cat(lst).cpu().numpy()
                R[v][mth] = dict(median=float(np.median(e)), lt3=float((e < 3).mean()), lt7=float((e < 7).mean()),
                                 lt14=float((e < 14).mean()), by_gap={int(g): float(np.median(e[gaps_all == g])) for g in a.gaps})
                r = R[v][mth]
                lines.append(f"{v:>12} {mth:>8} | {r['median']:8.1f} | {r['lt3']:.3f} {r['lt7']:.3f} {r['lt14']:.3f} | "
                             + " ".join(f"g{g}:{x:.1f}" for g, x in r["by_gap"].items()))
            ew = torch.cat(store[v]["win3"]).cpu().numpy()
            cp = torch.cat(acc[v]["conf_peak"]).cpu().numpy(); cs = torch.cat(acc[v]["conf_std"]).cpu().numpy()
            top = cp >= np.median(cp)
            R[v]["confidence"] = dict(spearman_peak_vs_err=spearman(-cp, ew), spearman_std_vs_err=spearman(cs, ew),
                                      median_err_top50_peak=float(np.median(ew[top])), median_err_bottom50_peak=float(np.median(ew[~top])))
            c = R[v]["confidence"]
            lines.append(f"{'':>12} confidence (win3): Spearman(-peak, err) {c['spearman_peak_vs_err']:.2f}, Spearman(spread, err) "
                         f"{c['spearman_std_vs_err']:.2f}; median err most-confident 50% {c['median_err_top50_peak']:.1f}px vs rest "
                         f"{c['median_err_bottom50_peak']:.1f}px")
        print("\n".join(lines[-(len(variants) * 7 + 2):]), flush=True)
    ctl.restore()
    json.dump(res, open(f"{a.out}/softmatch.json", "w"), indent=1)
    open(f"{a.out}/summary.txt", "w").write("patch = 14 px; image 336x504\n" + "\n".join(lines) + "\n")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    methods = ["argmax"] + [f"soft{T:g}" for T in a.temps] + ["win3", "win5"]
    fig, ax = plt.subplots(1, len(a.steps), figsize=(6 * len(a.steps), 4.5), sharey=True)
    ax = np.atleast_1d(ax)
    for axx, s in zip(ax, a.steps):
        for j, v in enumerate(variants):
            axx.plot(methods, [res[f"step{s}"][v][m]["median"] for m in methods], "-o", color=f"C{j}", label=v)
        axx.axhline(14, color="gray", ls=":", lw=.8); axx.axhline(7, color="gray", ls="--", lw=.8)
        axx.set_title(f"step {s}: median pixel error (patch = 14 px)"); axx.grid(alpha=.3); axx.tick_params(axis="x", rotation=30)
    ax[0].legend(fontsize=8); ax[0].set_ylabel("median error (px)")
    fig.tight_layout(); fig.savefig(f"{a.out}/softmatch.png", dpi=120); plt.close(fig)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
