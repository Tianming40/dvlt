#!/usr/bin/env python3
"""dec_diag.py -- isolate the role of attention INSIDE the decoder for token writes (follow-up of token_rw.py E2/E3).

Two decoders on the same stored tokens (out/stage3_latent/tokendump):
  normal : DVLT's decoder as is (2 frame-attention Blocks with RoPE in each of ray/depth decoder, then head)
  diag   : the same, but every decoder Block's attention is replaced by "attend only to yourself"
           (softmax = identity -> out_i = proj(v_i)); MLP, LayerNorm, heads unchanged.
           NOTE the depth head is a conv upsampler, so depth still mixes neighbours there; the ray head is per-patch linear,
           so under diag rays are exactly per-token.
For each decoder and quantity (logdepth, origin_x):
  fidelity : how far the diag decode is from the normal decode (the decoder was never trained without attention)
  own share: |d y_i / d token_i|^2 / |d y_i / d tokens of the frame|^2   (autograd, n-jac patches per frame)
  gain     : single-patch write along the Jacobian g_i (alpha*g/|g|^2) and along the covariance direction
             Cov(z,y)Var(y)^-1 (scaled so the probe predicts alpha) -> decoded change of patch i / alpha
  cov write on patch / 6x6 region / whole frame (as token_rw E2) -> gain

  PYTHONPATH=$(pwd) python shovel/dec_diag.py --steps K16:8 K16:16 --frames 3 12 20 --n-jac 24 --out out/stage3_latent/decdiag
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvlt_probe import Probe  # noqa: E402
from token_rw import Dec, fit_ridge, cov_write_mat, PM  # noqa: E402


def diag_forward(self, x, pos=None, block_mask=None, **kw):
    B, N, C = x.shape
    v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)[:, :, 2]      # (B,N,H,D)
    return self.proj_drop(self.proj(v.reshape(B, N, C)))


class DiagSwitch:
    def __init__(self, m):
        self.attns = [blk.attn for dec in (m.ray_decoder, m.depth_decoder) for blk in dec.blocks]
        self.orig = [a.forward for a in self.attns]

    def on(self):
        for a in self.attns:
            a.forward = diag_forward.__get__(a)

    def off(self):
        for a, f in zip(self.attns, self.orig):
            a.forward = f


def qty(out, name):
    return out["logdepth"] if name == "logdepth" else out["origin"][..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="out/stage3_latent/tokendump")
    ap.add_argument("--steps", nargs="+", default=["K16:8", "K16:16"])
    ap.add_argument("--frames", type=int, nargs="+", default=[3, 12, 20])
    ap.add_argument("--n-jac", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default="out/stage3_latent/decdiag")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = Probe("nvidia/dvlt", max_frames=27)
    dev = P.acc.device
    sw = DiagSwitch(P.m)
    log, res = [], {}

    def say(s=""):
        print(s, flush=True); log.append(s)

    for spec in a.steps:
        Kn, s = spec.split(":"); s = int(s)
        d = f"{a.dump}/{Kn}"
        meta = np.load(f"{d}/meta.npz")
        dec = Dec(P, meta)
        X = np.load(f"{d}/step_{s:03d}.npy").astype(np.float32)
        tg = np.load(f"{d}/targets_{s:03d}.npz")
        Np = X.shape[1] - PM; gh, gw = dec.gh, dec.gw
        Xp = X[:, PM:].reshape(-1, 768)
        Xt = torch.from_numpy(X).to(dev)
        fr = a.frames
        say(f"\n################ {spec} ################")
        R = res.setdefault(spec, {})
        for mode in ["normal", "diag"]:
            sw.on() if mode == "diag" else sw.off()
            b0 = dec(Xt[fr], fr)
            if mode == "normal":
                ref = b0
            fid = dict(logdepth_absdiff=float((b0["logdepth"] - ref["logdepth"]).abs().mean()),
                       dir_deg=float(torch.rad2deg(torch.arccos((b0["dir"] * ref["dir"]).sum(-1).clamp(-1, 1))).mean()),
                       origin_diff=float((b0["origin"] - ref["origin"]).norm(dim=-1).mean()))
            # probes are fit on THIS decoder's own outputs (all 27 frames)
            ball = dec(Xt, list(range(X.shape[0])))
            Rm = R.setdefault(mode, {"fidelity": fid})
            say(f"--- decoder = {mode}   fidelity vs normal: |Δlogdepth| {fid['logdepth_absdiff']:.3f}, Δdir {fid['dir_deg']:.2f}°, |Δorigin| {fid['origin_diff']:.3f}")
            for qn in ["logdepth", "origin_x"]:
                Y = qty(ball, qn).reshape(-1, 1).cpu().numpy()
                W, _ = fit_ridge(Xp, Y)
                wc = torch.from_numpy(cov_write_mat(Xp, Y)[0]).float().to(dev)
                wd = torch.from_numpy(W[0]).float().to(dev)
                base = qty(b0, qn)
                own, gj, gc = [], [], []
                rng = np.random.default_rng(0)
                for fi, f in enumerate(fr):
                    for pi in rng.choice(Np, a.n_jac, replace=False):
                        xf = Xt[f:f + 1].clone().requires_grad_(True)
                        qty(dec(xf, [f], grad=True), qn)[0, pi].backward()
                        g = xf.grad[0][PM:]                              # patch tokens only
                        own.append(float(g[pi].norm() ** 2 / (g.norm() ** 2 + 1e-12)))
                        for vec, den, store in [(g[pi], g[pi] @ g[pi], gj), (wc, wd @ wc, gc)]:
                            xm = Xt[f:f + 1].clone(); xm[0, PM + pi] += a.alpha * vec / den
                            store.append(float((qty(dec(xm, [f]), qn)[0, pi] - base[fi, pi]) / a.alpha))
                cy, cx = gh // 2, gw // 2
                masks = {"patch": [cy * gw + cx],
                         "region6x6": [(cy - 3 + i) * gw + (cx - 3 + j) for i in range(6) for j in range(6)],
                         "frame": list(range(Np))}
                cw = {}
                dz = a.alpha * wc / (wd @ wc)
                for mn, idx in masks.items():
                    xm = Xt[fr].clone(); xm[:, PM + np.array(idx)] += dz
                    cw[mn] = float((qty(dec(xm, fr), qn) - base)[:, idx].mean() / a.alpha)
                Rm[qn] = dict(own_share=float(np.mean(own)), gain_jac=float(np.mean(gj)), gain_cov_patch_rand=float(np.mean(gc)),
                              cov_write=cw)
                say(f"  {qn:>8}: own-token share {np.mean(own):.3f} | single-patch gain: Jacobian {np.mean(gj):.3f}, cov {np.mean(gc):.3f}"
                    f" | cov write gain patch {cw['patch']:.3f} / 6x6 {cw['region6x6']:.3f} / frame {cw['frame']:.3f}")
        sw.off()

    json.dump(res, open(f"{a.out}/decdiag.json", "w"), indent=1)
    with open(f"{a.out}/report.txt", "w") as fh:
        fh.write(" ".join(sys.argv) + "\n" + "\n".join(log) + "\n")
    print(f"saved {a.out}/report.txt")


if __name__ == "__main__":
    main()
