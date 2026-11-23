#!/usr/bin/env python3
"""Inspect what the restoring branch R of a trained StationaryLoop (train_stationary.py) learned.

  1. weights: gate change vs the (0.5,0.5) init; norms of R's layers; fc1 split into the h-part and the x0-part
  2. function along a trajectory (val clips, calls 1..Kmax): per-token norms of the global attention branch, the
     global MLP branch and R; cos(R, x0 - x) (does R pull toward the input?); cos(R, -(attn+mlp)) (does R cancel the
     block's update?); per-step relative change; channels where R writes most (vs the known massive channels)
  3. x0-dependence: ||R(h, x0) - R(h, x0 permuted across tokens of the same frame)|| / ||R(h, x0)||

  python residual_opt/inspect_stationary.py --run statA statA_rms
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dvlt_probe import Probe  # noqa: E402
from residual_opt.train_stationary import StationaryLoop, load_clip  # noqa: E402

MASSIVE = [25, 215, 480, 614, 701]


def cos(a, b):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-9)


@torch.no_grad()
def trajectory(loop, P, c, dev, Kmax, rec):
    img = torch.from_numpy(c["image"]).to(dev).float()[None] / 255.0
    x, rope, (B, S, H, W) = P.init_state(img)
    x = x.float(); x0 = x.clone()
    pm = P.m.patch_start_idx
    out = {}
    g = torch.Generator(device=dev).manual_seed(0)
    for k in range(1, Kmax + 1):
        _, Pn, C = x.shape
        xf = loop._sub(loop.fa, x.reshape(B * S, Pn, C), loop.g_frame, rope.reshape(B * S, Pn, 2))
        xg = xf.reshape(B, S * Pn, C); x0g = x0.reshape(B, S * Pn, C)
        b, gg = loop.ga, loop.g_global
        attn = gg[0] * b.ls1(b.attn(b.norm1(xg)))
        y = xg + attn
        h = b.norm2(y)
        mlp = gg[1] * b.ls2(b.mlp(h))
        r = loop.R(h, x0g)
        xn = gg[2] * (y + mlp + r)
        if loop.target_rms is not None:
            xn = xn * (loop.target_rms / xn.pow(2).mean((1, 2), keepdim=True).sqrt())
        xn = xn.reshape(B * S, Pn, C)
        if k in rec:
            sel = lambda t: t.reshape(B * S, Pn, C)[:, pm:].reshape(-1, C)     # patch tokens
            A, M, R, X, X0 = sel(attn), sel(mlp), sel(r), sel(y), sel(x0g)
            # x0 permuted within each frame
            perm = torch.cat([torch.randperm(Pn - pm, generator=g, device=dev) + f * (Pn - pm) for f in range(B * S)])
            r_perm = loop.R(h.reshape(B * S, Pn, C)[:, pm:].reshape(-1, C), X0[perm])
            ch = R.abs().mean(0)
            top = torch.topk(ch, 10).indices.tolist()
            out[k] = dict(
                norm_x=float(X.norm(dim=-1).mean()), norm_attn=float(A.norm(dim=-1).mean()),
                norm_mlp=float(M.norm(dim=-1).mean()), norm_R=float(R.norm(dim=-1).mean()),
                cos_R_to_x0=float(cos(R, X0 - X).mean()), cos_R_vs_update=float(cos(R, -(A + M)).mean()),
                cos_R_mlp=float(cos(R, M).mean()),
                x0_dependence=float((R - r_perm).norm(dim=-1).mean() / R.norm(dim=-1).mean()),
                rel_step=float(((xn - x)[:, pm:]).norm(dim=-1).mean() / x[:, pm:].norm(dim=-1).mean()),
                R_top_channels=top, R_massive_share=float(ch[MASSIVE].sum() / ch.sum()),
                massive_abs_x={c_: float(X[:, c_].abs().mean()) for c_ in MASSIVE})
        x = xn
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", default=["statA", "statA_rms"])
    ap.add_argument("--cache", default="/storage/user/zhtia/resopt_cache/small")
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--K", type=int, nargs="*", default=[1, 4, 8, 16, 32, 64, 128])
    a = ap.parse_args()
    P = Probe("nvidia/dvlt", max_frames=None)
    m = P.m; m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    dev = next(m.parameters()).device
    clips = [load_clip(f) for f in sorted(glob.glob(f"{a.cache}/val/*.npz"))[: a.clips]]
    res = {}
    for run in a.run:
        ck = torch.load(f"{ROOT}/out/resopt/{run}/best.pt", map_location=dev)
        loop = StationaryLoop(m, use_r=ck["R"] is not None)
        g0f, g0g = loop.g_frame.detach().clone(), loop.g_global.detach().clone()
        loop.g_frame.data.copy_(ck["g_frame"]); loop.g_global.data.copy_(ck["g_global"])
        loop.R.load_state_dict(ck["R"])
        if ck["args"].get("rms"):
            loop.target_rms = 0.831          # value printed by the training run
        R = loop.R
        W1 = R.fc1.weight
        wt = dict(
            gate_change_frame=[float((loop.g_frame[i] - g0f[i]).norm() / g0f[i].norm()) for i in range(3)],
            gate_change_global=[float((loop.g_global[i] - g0g[i]).norm() / g0g[i].norm()) for i in range(3)],
            fc1_norm_h=float(W1[:, :768].norm()), fc1_norm_x0=float(W1[:, 768:].norm()),
            fc2_norm=float(R.fc2.weight.norm()), fc2_bias_norm=float(R.fc2.bias.norm()),
            ln0_weight_mean=float(R.ln0.weight.mean()))
        print(f"\n===== {run} (best.pt from step-selection; rms={ck['args'].get('rms', False)})")
        print("weights:", json.dumps(wt))
        per = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for c in clips:
                for k, v in trajectory(loop, P, c, dev, max(a.K), set(a.K)).items():
                    per.setdefault(k, []).append(v)
        summ = {}
        print(f"{'K':>4} {'|x|':>6} {'|attn|':>7} {'|mlp|':>7} {'|R|':>7} {'cos(R,x0-x)':>11} {'cos(R,-upd)':>11} "
              f"{'cos(R,mlp)':>10} {'x0-dep':>7} {'relstep':>8} {'R@massive':>9}")
        for k in sorted(per):
            vs = per[k]
            s = {q: float(np.mean([v[q] for v in vs])) for q in vs[0] if isinstance(vs[0][q], float)}
            s["R_top_channels"] = vs[0]["R_top_channels"]
            s["massive_abs_x"] = {c_: float(np.mean([v["massive_abs_x"][c_] for v in vs])) for c_ in MASSIVE}
            summ[k] = s
            print(f"{k:>4} {s['norm_x']:6.2f} {s['norm_attn']:7.3f} {s['norm_mlp']:7.3f} {s['norm_R']:7.3f} "
                  f"{s['cos_R_to_x0']:11.3f} {s['cos_R_vs_update']:11.3f} {s['cos_R_mlp']:10.3f} "
                  f"{s['x0_dependence']:7.3f} {s['rel_step']:8.4f} {s['R_massive_share']:9.3f}")
        for k in (16, 128):
            if k in summ:
                print(f"  K{k}: R top channels {summ[k]['R_top_channels']} | |x| at massive channels "
                      + " ".join(f"{c_}:{v:.2f}" for c_, v in summ[k]['massive_abs_x'].items()))
        res[run] = dict(weights=wt, traj=summ)
    os.makedirs(f"{ROOT}/out/resopt/inspect", exist_ok=True)
    json.dump(res, open(f"{ROOT}/out/resopt/inspect/inspect_stationary.json", "w"), indent=1)
    print(f"\nsaved {ROOT}/out/resopt/inspect/inspect_stationary.json")


if __name__ == "__main__":
    main()
