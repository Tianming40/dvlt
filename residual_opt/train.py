#!/usr/bin/env python3
"""Train / evaluate the flow-matching refiner (residual_opt/model.py) on cached DVLT clips (residual_opt/cache_dvlt.py).

Bridge flow matching from DVLT's per-patch geometry to GT:
    t ~ U(0,1),  x_t = (1-t) x_dvlt + t x_gt + sigma_b * sigma_v * sqrt(t(1-t)) * eps,   target v = x_gt - x_dvlt
Loss: MSE on v / sigma_v (per-channel velocity scale); log-depth channel only on patches with valid GT depth.
Inference: Euler integration from x_dvlt (t=0) to t=1 in n steps.
Evaluation per clip (patch level): depth AbsRel (median scale), ray-direction error (deg), camera-centre ATE (Sim3,
relative to trajectory extent) for DVLT (n=0) and the refiner with n = 1, 2, 4, 8 steps.

  overfit check : python residual_opt/train.py --cache <dir> --max-train 4 --eval-on-train --steps 1500 --run overfit4
  small run     : python residual_opt/train.py --cache <dir> --steps 6000 --run small
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from residual_opt.model import FlowRefiner, FlowRefinerCL, closed_loop_geometry  # noqa: E402
from dvlt_probe import sim3_ate  # noqa: E402

PM = 5


def load_clip(f):
    z = np.load(f)
    return dict(tokens=z["tokens"][:, PM:], image=z["image"], x_dvlt=z["x_dvlt"], x_gt=z["x_gt"], mask=z["mask"],
                name=os.path.basename(f)[:-4])


def to_dev(c, dev):
    return dict(tokens=torch.from_numpy(c["tokens"].astype(np.float32)).to(dev)[None],
                image=torch.from_numpy(c["image"]).to(dev).float()[None] / 255.0,
                xd=torch.from_numpy(c["x_dvlt"]).to(dev)[None], xg=torch.from_numpy(c["x_gt"]).to(dev)[None],
                mask=torch.from_numpy(c["mask"]).to(dev)[None])


def metrics(x, xg, mask):
    """x, xg: (S,Np,7) un-normalised numpy; mask (S,Np)"""
    p, g = np.exp(x[..., 0][mask]), np.exp(xg[..., 0][mask])
    s = np.median(g / p)
    absrel = float(np.mean(np.abs(p * s - g) / g))
    dp = x[..., 1:4] / np.linalg.norm(x[..., 1:4], axis=-1, keepdims=True)
    ang = float(np.degrees(np.arccos(np.clip((dp * xg[..., 1:4]).sum(-1), -1, 1))).mean())
    ate = sim3_ate(torch.from_numpy(x[..., 4:].mean(1)), torch.from_numpy(xg[..., 4:].mean(1)))
    return dict(absrel=absrel, dir_deg=ang, ate=float(ate))


def geo_of(model, x):
    """closed-loop reprojection of the current geometry x (1,S,Np,7); None for the open-loop model"""
    if not isinstance(model, FlowRefinerCL):
        return None
    return closed_loop_geometry(x[0], model.gh, model.gw)


@torch.no_grad()
def integrate(model, c, stats, n):
    mu, sx, sv = stats
    x = c["xd"].clone()
    xdn = (c["xd"] - mu) / sx
    for k in range(n):
        t = torch.full((1,), k / n, device=x.device)
        v = model((x - mu) / sx, xdn, c["tokens"], c["image"], t, geo=geo_of(model, x)) * sv
        x = x + v / n
    return x[0].cpu().numpy()


def evaluate(model, clips, stats, dev, ns=(0, 1, 2, 4, 8)):
    model.eval()
    out = {n: [] for n in ns}
    for c in clips:
        cd = to_dev(c, dev)
        for n in ns:
            x = c["x_dvlt"] if n == 0 else integrate(model, cd, stats, n)
            out[n].append(metrics(x, c["x_gt"], c["mask"]))
    model.train()
    return {n: {k: float(np.mean([m[k] for m in v])) for k in v[0]} for n, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/storage/user/zhtia/resopt_cache/small")
    ap.add_argument("--run", default="small")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigma-b", type=float, default=0.5)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--eval-on-train", action="store_true", help="evaluate on the training clips (overfit check)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--closed-loop", action="store_true", help="FlowRefinerCL: reproject with the current geometry at every step")
    ap.add_argument("--extra-eval", nargs="*", default=["ood_bf"], help="additional eval splits (sub-dirs of --cache)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    out = f"{ROOT}/out/resopt/{a.run}"
    os.makedirs(out, exist_ok=True)
    tr_files = sorted(glob.glob(f"{a.cache}/train/*.npz"))[: a.max_train]
    va_files = sorted(glob.glob(f"{a.cache}/val/*.npz"))
    train = [load_clip(f) for f in tr_files]
    val = train if a.eval_on_train else [load_clip(f) for f in va_files]
    extra = {sp: [load_clip(f) for f in sorted(glob.glob(f"{a.cache}/{sp}/*.npz"))] for sp in a.extra_eval}
    extra = {k: v for k, v in extra.items() if v}
    print(f"train clips {len(train)} | eval clips {len(val)} ({'train' if a.eval_on_train else 'val'}) | extra eval "
          + ", ".join(f"{k}: {len(v)}" for k, v in extra.items()), flush=True)

    # normalisation: inputs by DVLT-output stats, velocity by the per-channel std of (gt - dvlt)
    XD = np.concatenate([c["x_dvlt"].reshape(-1, 7) for c in train])
    V = np.concatenate([(c["x_gt"] - c["x_dvlt"]).reshape(-1, 7) for c in train])
    M = np.concatenate([c["mask"].reshape(-1) for c in train])
    sv = V.std(0); sv[0] = V[M, 0].std()
    stats = tuple(torch.tensor(s, dtype=torch.float32, device=dev) for s in (XD.mean(0), XD.std(0) + 1e-6, sv + 1e-6))
    print("velocity scale per channel:", np.round(sv, 4).tolist(), flush=True)
    gh, gw = train[0]["image"].shape[-2] // 14, train[0]["image"].shape[-1] // 14
    Model = FlowRefinerCL if a.closed_loop else FlowRefiner
    model = Model(grid=(gh, gw), d=a.d, depth=a.depth).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    mu, sx, svt = stats
    log = []
    base = evaluate(model, val, stats, dev, ns=(0,))[0]
    print(f"DVLT baseline on eval clips: AbsRel {base['absrel']:.4f} dir {base['dir_deg']:.3f}° ATE {base['ate']:.4f}", flush=True)
    base_extra = {k: evaluate(model, v, stats, dev, ns=(0,))[0] for k, v in extra.items()}
    for k, b in base_extra.items():
        print(f"DVLT baseline on {k}: AbsRel {b['absrel']:.4f} dir {b['dir_deg']:.3f}° ATE {b['ate']:.4f}", flush=True)
    best, t0 = None, time.time()
    rng = np.random.default_rng(a.seed)
    for step in range(1, a.steps + 1):
        lsum = 0.0
        for _ in range(a.accum):
            c = to_dev(train[rng.integers(len(train))], dev)
            t = torch.rand(1, device=dev)
            eps = torch.randn_like(c["xd"])
            xt = (1 - t) * c["xd"] + t * c["xg"] + a.sigma_b * svt * torch.sqrt(t * (1 - t)) * eps
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model((xt - mu) / sx, (c["xd"] - mu) / sx, c["tokens"], c["image"], t, geo=geo_of(model, xt)).float()
            tgt = (c["xg"] - c["xd"]) / svt
            err = (v - tgt) ** 2
            w = torch.ones_like(err); w[..., 0] = c["mask"].float()
            loss = (err * w).sum() / w.sum() / a.accum
            loss.backward(); lsum += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(); sched.step()
        if step % 100 == 0:
            print(f"step {step:>5} loss {lsum:.4f} lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(model, val, stats, dev)
            evx = {k: evaluate(model, v, stats, dev, ns=(0, 1, 4)) for k, v in extra.items()}
            log.append(dict(step=step, loss=lsum, eval=ev, extra=evx))
            print("  eval  " + " | ".join(f"n={n}: AbsRel {m['absrel']:.4f} dir {m['dir_deg']:.3f}° ATE {m['ate']:.4f}" for n, m in ev.items()), flush=True)
            for k, e in evx.items():
                print(f"  {k:>5} " + " | ".join(f"n={n}: AbsRel {m['absrel']:.4f} dir {m['dir_deg']:.3f}° ATE {m['ate']:.4f}" for n, m in e.items()), flush=True)
            if isinstance(model, FlowRefinerCL):
                print(f"  closed-loop gate {float(model.cl_gate):+.4f}", flush=True)
            score = min(ev[n]["absrel"] for n in ev if n > 0)
            if best is None or score < best:
                best = score
                torch.save(dict(model=model.state_dict(), stats=[s.cpu() for s in stats], args=vars(a), grid=(gh, gw)), f"{out}/best.pt")
            json.dump(dict(args=vars(a), baseline=base, baseline_extra=base_extra, log=log), open(f"{out}/log.json", "w"), indent=1)
    print(f"done; best eval AbsRel {best:.4f} (DVLT {base['absrel']:.4f}); outputs in {out}", flush=True)


if __name__ == "__main__":
    main()
