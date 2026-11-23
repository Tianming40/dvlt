#!/usr/bin/env python3
"""Option A: a STATIONARY DVLT loop with a learned, z0-anchored restoring term in the GLOBAL MLP.

Frozen DVLT (encoder, recurrent block weights, decoder heads). Changes:
  * all gates (frame + global sub-blocks, s_attn / s_mlp / s_out) become learnable CONSTANTS, initialised from DVLT's
    gate network at (t_now, t_next) = (0.5, 0.5)  -> the same map at every call (no time schedule);
  * the global sub-block's MLP gets a parallel, zero-initialised restoring branch that sees the token itself and its
    own initial value:   x <- x + s_mlp * LS2(MLP(LN(x))) + R([LN(x), LN(x0)])   (then x <- s_out * x)
Only R and the constant gates are trained.

Training (fixed-point shaping, Part II / Solv3R style): per clip draw K ~ log-uniform{kmin..kmax}, run K-b calls without
gradient and the last b calls with gradient (truncated BPTT), decode the final state, patch-level loss vs GT
(log-depth with per-clip median scale alignment, ray direction, ray origin / scene depth), plus a fixed-point
regulariser on the relative change of the last call.
Evaluation: K in {8,16,32,64,128} on val + ood_bf; baselines = DVLT original schedule K16 / K64, and the untrained
stationary loop (R = 0, constant gates) at the same K.

  python residual_opt/train_stationary.py --cache /storage/user/zhtia/resopt_cache/large --run statA
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dvlt_probe import Probe  # noqa: E402
from residual_opt.train import metrics  # noqa: E402

PS = 14


def per_patch(t, gh, gw):
    S = t.shape[0]
    return t.reshape(S, gh, PS, gw, PS, -1).mean((2, 4)).reshape(S, gh * gw, -1)


class Restorer(nn.Module):
    """gamma * R([LN(x), LN(x0)]) -> delta; zero-initialised fc2 and a LayerScale gamma (like DVLT's own branches)."""

    def __init__(self, d=768, hidden=1024, gamma_init=0.01, raw=False):
        super().__init__()
        self.raw = raw              # raw=True: inputs are the un-normalised stream y and the raw x0 (magnitudes kept)
        self.gamma = nn.Parameter(torch.full((d,), float(gamma_init)))
        self.ln0 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(2 * d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, h, x0):
        z0 = x0 if self.raw else self.ln0(x0)
        return self.gamma * self.fc2(F.gelu(self.fc1(torch.cat([h, z0], -1))))


class StationaryLoop(nn.Module):
    def __init__(self, m, use_r=True, t_init=(0.5, 0.5), gamma_init=0.01, pull=False, r_raw=False, alpha_fixed=None):
        super().__init__()
        self.m = m
        blk = m.recurrent_blocks[0]
        self.fa, self.ga = blk.frame_attn, blk.global_attn
        dev = next(m.parameters()).device
        tp = torch.tensor([list(t_init)], device=dev, dtype=torch.float32)
        with torch.no_grad():
            gf = self.fa.depth_scale(tp).float().reshape(3, -1)
            gg = self.ga.depth_scale(tp).float().reshape(3, -1)
        self.g_frame = nn.Parameter(gf.clone())   # rows: s_attn, s_mlp, s_out
        self.g_global = nn.Parameter(gg.clone())
        self.R = Restorer(gf.shape[1], gamma_init=gamma_init, raw=r_raw).to(dev) if use_r else None
        # optional Parcae-style linear pull toward the initial state: x <- x - alpha * (x - x0), alpha in [0,1], init 0
        self.alpha = nn.Parameter(torch.zeros(gf.shape[1], device=dev)) if pull else None
        self.alpha_fixed = alpha_fixed      # float -> constant (not learned) pull strength, like the training-free restart
        self.target_rms = None      # float -> after every call rescale the whole state to this RMS (Solv3R-style)

    def trainable(self):
        ps = [self.g_frame, self.g_global] + ([self.alpha] if self.alpha is not None else [])
        return ps + (list(self.R.parameters()) if self.R is not None else [])

    @staticmethod
    def _sub(b, x, g, pos, extra=None):
        x = x + g[0] * b.ls1(b.attn(b.norm1(x), pos=pos))
        h = b.norm2(x)
        y = x                                   # stream after attention, before the MLP (un-normalised)
        x = x + g[1] * b.ls2(b.mlp(h))
        if extra is not None:
            x = x + extra(h, y)
        return g[2] * x

    def call(self, x, x0, rope, B, S):
        _, P, C = x.shape
        x = self._sub(self.fa, x.reshape(B * S, P, C), self.g_frame, rope.reshape(B * S, P, 2))
        x = x.reshape(B, S * P, C)
        x0g = x0.reshape(B, S * P, C)
        extra = (lambda h, y: self.R(y if self.R.raw else h, x0g)) if self.R is not None else None
        x = self._sub(self.ga, x, self.g_global, None, extra)
        if self.alpha is not None:
            x = x - self.alpha.clamp(0.0, 1.0) * (x - x0g)
        if self.alpha_fixed is not None:
            x = x - self.alpha_fixed * (x - x0g)
        if self.target_rms is not None:
            rms = x.float().pow(2).mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
            x = x * (self.target_rms / rms).to(x.dtype)
        return x.reshape(B * S, P, C)


def geometry(m, x, H, W, rope, grad, chunk=2):
    """decode -> patch-level [logdepth, unit dir, origin] (S, Np, 7); chunks are checkpointed when grad is on."""
    S = x.shape[0]
    gh, gw = H // PS, W // PS

    def dec(xc, rc):
        pr = m._decode(xc, H, W, 1, xc.shape[0], rc)
        d = pr["depth"][0, ..., 0].float(); r = pr["rays"][0].float()
        dirs = r[..., :3] / r[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-9)
        pd = per_patch(dirs, gh, gw); pd = pd / pd.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return torch.cat([per_patch(d.clamp_min(1e-6).log()[..., None], gh, gw), pd, per_patch(r[..., 3:6], gh, gw)], -1)

    outs = []
    for j in range(0, S, chunk):
        xc, rc = x[j:j + chunk], rope[j:j + chunk]
        outs.append(checkpoint(dec, xc, rc, use_reentrant=False) if grad else dec(xc, rc))
    return torch.cat(outs)


def geo_loss(xp, xg, mask):
    """patch-level loss, per-clip median scale alignment (detached)."""
    s = (xg[..., 0][mask] - xp[..., 0][mask]).median().detach()
    l_d = (xp[..., 0] + s - xg[..., 0]).abs()[mask].mean()
    l_r = (xp[..., 1:4] - xg[..., 1:4]).norm(dim=-1).mean()
    scene = xg[..., 0][mask].exp().median()
    l_o = ((xp[..., 4:] * s.exp() - xg[..., 4:]).norm(dim=-1) / scene).mean()
    return l_d, l_r, l_o


def load_clip(f):
    z = np.load(f)
    return dict(image=z["image"], x_dvlt=z["x_dvlt"], x_gt=z["x_gt"], mask=z["mask"], name=os.path.basename(f)[:-4])


@torch.no_grad()
def run_stationary(loop, P, c, Ks, dev):
    """returns {K: geometry numpy} for all K in Ks (one trajectory, read out at each K)"""
    img = torch.from_numpy(c["image"]).to(dev).float()[None] / 255.0
    x, rope, (B, S, H, W) = P.init_state(img)
    x = x.float(); x0 = x.clone()
    out, Kmax = {}, max(Ks)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(1, Kmax + 1):
            x = loop.call(x, x0, rope, B, S).float()
            if k in Ks:
                out[k] = geometry(P.m, x, H, W, rope, grad=False).cpu().numpy()
    return out


@torch.no_grad()
def run_dvlt_sched(P, c, K, dev):
    img = torch.from_numpy(c["image"]).to(dev).float()[None] / 255.0
    x, rope, (B, S, H, W) = P.init_state(img)
    x = x.float()
    ts = torch.linspace(0.0, 1.0, K).tolist()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(K):
            x = P.m._interval_step(x, ts[i], ts[i + 1] if i + 1 < K else 1.0, rope, B, S).float()
        return geometry(P.m, x, H, W, rope, grad=False).cpu().numpy()


def mean_metrics(ms):
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


def evaluate(loop, P, clips, Ks, dev):
    loop.eval()
    res = {k: [] for k in Ks}
    for c in clips:
        g = run_stationary(loop, P, c, Ks, dev)
        for k in Ks:
            res[k].append(metrics(g[k], c["x_gt"], c["mask"]))
    loop.train()
    return {k: mean_metrics(v) for k, v in res.items()}


def fmt(ev):
    return " | ".join(f"K{k}: {m['absrel']:.4f} {m['dir_deg']:.2f}° {m['ate']:.4f}" for k, m in ev.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/storage/user/zhtia/resopt_cache/large")
    ap.add_argument("--run", default="statA")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kmin", type=int, default=8)
    ap.add_argument("--kmax", type=int, default=64)
    ap.add_argument("--bptt", type=int, default=4, help="calls with gradient at the end of each unroll")
    ap.add_argument("--w-dir", type=float, default=1.0)
    ap.add_argument("--w-org", type=float, default=1.0)
    ap.add_argument("--w-fp", type=float, default=0.1, help="fixed-point regulariser weight")
    ap.add_argument("--no-r", action="store_true", help="ablation: constant gates only, no restoring branch")
    ap.add_argument("--lr-r", type=float, default=None, help="learning rate of R (default: --lr)")
    ap.add_argument("--gamma-init", type=float, default=0.01, help="initial LayerScale of R's output")
    ap.add_argument("--r-raw", action="store_true", help="R sees the raw stream y and raw x0 instead of LN2(y), LN0(x0)")
    ap.add_argument("--alpha-fixed", type=float, default=None, help="constant linear pull x <- x - a*(x - x0), not learned")
    ap.add_argument("--pull", action="store_true", help="add the linear pull x <- x - alpha*(x - x0) (alpha learned, init 0)")
    ap.add_argument("--rms", action="store_true",
                    help="rescale the state to a fixed RMS after every call (target = mean RMS of DVLT's K16 state on 8 train clips)")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-K", type=int, nargs="*", default=[8, 16, 32, 64, 128])
    ap.add_argument("--extra-eval", nargs="*", default=["ood_bf"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = f"{ROOT}/out/resopt/{a.run}"
    os.makedirs(out, exist_ok=True)

    P = Probe("nvidia/dvlt", max_frames=None)
    m = P.m
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    dev = next(m.parameters()).device
    loop = StationaryLoop(m, use_r=not a.no_r, gamma_init=a.gamma_init, pull=a.pull, r_raw=a.r_raw, alpha_fixed=a.alpha_fixed)
    params = loop.trainable()
    print(f"trainable params {sum(p.numel() for p in params) / 1e6:.3f}M", flush=True)

    tr = [load_clip(f) for f in sorted(glob.glob(f"{a.cache}/train/*.npz"))[: a.max_train]]
    evs = {"val": [load_clip(f) for f in sorted(glob.glob(f"{a.cache}/val/*.npz"))[: a.max_eval]]}
    for sp in a.extra_eval:
        fs = sorted(glob.glob(f"{a.cache}/{sp}/*.npz"))[: a.max_eval]
        if fs:
            evs[sp] = [load_clip(f) for f in fs]
    print(f"train clips {len(tr)} | " + ", ".join(f"{k}: {len(v)}" for k, v in evs.items()), flush=True)
    if a.rms:
        rs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for c in tr[:8]:
                img = torch.from_numpy(c["image"]).to(dev).float()[None] / 255.0
                x, rope, (B, S, H, W) = P.init_state(img)
                x = x.float(); ts = torch.linspace(0.0, 1.0, 16).tolist()
                for i in range(16):
                    x = m._interval_step(x, ts[i], ts[i + 1] if i + 1 < 16 else 1.0, rope, B, S).float()
                rs.append(float(x.pow(2).mean().sqrt()))
        loop.target_rms = float(np.mean(rs))
        print(f"RMS normalisation on: target RMS {loop.target_rms:.3f} (DVLT K16 state; per-clip {np.round(rs, 3).tolist()})", flush=True)

    # baselines: DVLT original schedule (K16 = trained budget, K64 = past budget) and the untrained stationary loop
    base = {}
    for sp, cl in evs.items():
        base[sp] = {f"dvlt_sched_K{K}": mean_metrics([metrics(run_dvlt_sched(P, c, K, dev), c["x_gt"], c["mask"]) for c in cl])
                    for K in (16, 64)}
        base[sp]["stationary_untrained"] = evaluate(loop, P, cl, a.eval_K, dev)
        for K in (16, 64):
            b = base[sp][f"dvlt_sched_K{K}"]
            print(f"[{sp}] DVLT schedule K{K}: {b['absrel']:.4f} {b['dir_deg']:.2f}° {b['ate']:.4f}", flush=True)
        print(f"[{sp}] stationary untrained: " + fmt(base[sp]["stationary_untrained"]), flush=True)

    groups = [{"params": [loop.g_frame, loop.g_global] + ([loop.alpha] if loop.alpha is not None else []), "weight_decay": 0.0}]
    if loop.R is not None:
        groups.append({"params": list(loop.R.parameters()), "weight_decay": 0.01, "lr": a.lr_r or a.lr})
    opt = torch.optim.AdamW(groups, lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    rng = np.random.default_rng(a.seed)
    log, best, t0 = [], None, time.time()
    run = dict(l=0.0, d=0.0, r=0.0, o=0.0, fp=0.0, n=0)
    for step in range(1, a.steps + 1):
        c = tr[rng.integers(len(tr))]
        K = int(round(math.exp(rng.uniform(math.log(a.kmin), math.log(a.kmax)))))
        img = torch.from_numpy(c["image"]).to(dev).float()[None] / 255.0
        xg = torch.from_numpy(c["x_gt"]).to(dev); mask = torch.from_numpy(c["mask"]).to(dev)
        x, rope, (B, S, H, W) = P.init_state(img)
        x = x.float(); x0 = x.clone()
        # NOTE: the no-grad prefix and the gradient window must use SEPARATE autocast contexts. autocast caches the
        # bf16 copies of weights within one context; copies made under no_grad carry no autograd history, so reusing
        # them in the gradient window silently cuts the gradient to R (fc2 stayed exactly 0 in the first statA runs).
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(K - a.bptt):
                x = loop.call(x, x0, rope, B, S).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(a.bptt):
                xprev = x
                x = loop.call(x, x0, rope, B, S).float()
            xp = geometry(m, x, H, W, rope, grad=True)
        l_d, l_r, l_o = geo_loss(xp, xg, mask)
        pm = m.patch_start_idx
        l_fp = ((x[:, pm:] - xprev[:, pm:]).norm(dim=-1) / xprev[:, pm:].norm(dim=-1).clamp_min(1e-6)).mean()
        loss = l_d + a.w_dir * l_r + a.w_org * l_o + a.w_fp * l_fp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        for k, v in (("l", loss), ("d", l_d), ("r", l_r), ("o", l_o), ("fp", l_fp)):
            run[k] += float(v)
        run["n"] += 1
        if step == 2:
            print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if step % 50 == 0:
            n = run["n"]
            gs = loop.g_global.detach().mean(1).tolist()
            extra_log = ""
            if loop.R is not None:
                extra_log += f" |gammaR| {float(loop.R.gamma.abs().mean()):.4f} |fc2| {float(loop.R.fc2.weight.norm()):.3f}"
            if loop.alpha is not None:
                extra_log += f" alpha mean {float(loop.alpha.clamp(0, 1).mean()):.4f} max {float(loop.alpha.clamp(0, 1).max()):.4f}"
            print(f"step {step:>5} loss {run['l']/n:.4f} (depth {run['d']/n:.4f} dir {run['r']/n:.4f} org {run['o']/n:.4f} "
                  f"fp {run['fp']/n:.4f}) g_global {[round(g, 3) for g in gs]}{extra_log} lr {sched.get_last_lr()[0]:.1e} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            run = dict(l=0.0, d=0.0, r=0.0, o=0.0, fp=0.0, n=0)
        if step % a.eval_every == 0 or step == a.steps:
            ev = {sp: evaluate(loop, P, cl, a.eval_K, dev) for sp, cl in evs.items()}
            for sp, e in ev.items():
                print(f"  eval [{sp}] " + fmt(e), flush=True)
            log.append(dict(step=step, eval=ev))
            score = np.mean([ev["val"][k]["absrel"] for k in a.eval_K])     # mean over K: rewards stability too
            if best is None or score < best:
                best = score
                torch.save(dict(g_frame=loop.g_frame.detach().cpu(), g_global=loop.g_global.detach().cpu(),
                                alpha=loop.alpha.detach().cpu() if loop.alpha is not None else None,
                                R=loop.R.state_dict() if loop.R is not None else None, args=vars(a)), f"{out}/best.pt")
            json.dump(dict(args=vars(a), baseline=base, log=log), open(f"{out}/log.json", "w"), indent=1)
    print(f"done; outputs in {out}", flush=True)


if __name__ == "__main__":
    main()
