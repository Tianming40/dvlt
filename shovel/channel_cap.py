#!/usr/bin/env python3
"""channel_cap.py -- are the runaway MASSIVE channels a direct link in the collapse?  Cap channel magnitudes each step.

Reference = per-channel RMS over all tokens (all frames, cam + reg + patches) at step --ref (healthy, default 8).
From step ref+1 on, after every loop step, every capped channel c is rescaled  z[..., c] *= min(1, ref_rms[c] / rms[c])
(spatial pattern kept, only its amplitude limited).
Runs:
  sched K64 (original schedule; channels 480/614/701 grow during its collapse):
      base | target cap {480, 614, 701} | random cap (3 random ordinary channels, control) | all-channel cap (768)
  BR K128 (fixed gates B + RMS; there ch215 decays and ch25 takes over): base | all-channel cap
Per step: AbsRel, ATE (Sim3), and the RMS of channels 215, 480, 614, 701, 25.
Data: out/stage3_latent/channelcap/channelcap.json, summary.txt, channelcap.png

  PYTHONPATH=$(pwd) python shovel/channel_cap.py --out out/stage3_latent/channelcap
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvlt_probe import Probe, load_gt, depth_vs_gt, sim3_ate  # noqa: E402
from gate_fix import FixedGates, rms_normalize  # noqa: E402

WATCH = [215, 480, 614, 701, 25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/data/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--gt", default="out/data/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--ref", type=int, default=8)
    ap.add_argument("--target", type=int, nargs="+", default=[480, 614, 701])
    ap.add_argument("--out", default="out/stage3_latent/channelcap")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    P = Probe("nvidia/dvlt", max_frames=27)
    images = P.load(a.input)
    with torch.no_grad():
        x0, rope, dims = P.init_state(images)
    x0 = x0.float(); B, S, H, W = dims
    gt_d, gt_c2w, _ = load_gt(a.gt, P.img_size, device=x0.device)
    gt_d, gt_c = gt_d[:S].float(), gt_c2w[:S, :3, 3].float()
    fg = FixedGates(P.m, mode="B")
    rng = np.random.default_rng(0)
    pool = [c for c in range(768) if c not in set(WATCH) | {459, 118, 551, 104, 254, 6, 243, 250}]
    rand_ch = sorted(rng.choice(pool, len(a.target), replace=False).tolist())

    @torch.no_grad()
    def evaluate(z):
        ds, cs = [], []
        for i in range(0, S, 4):
            pr = P.m._decode(z[i:i + 4], H, W, 1, min(4, S - i), rope[i:i + 4])
            ds.append(pr["depth"][0, ..., 0].float()); cs.append(pr["rays"][0, ..., 3:6].float().mean((1, 2)))
        rms = z.reshape(-1, z.shape[-1]).pow(2).mean(0).sqrt()
        return dict(absrel=depth_vs_gt(torch.cat(ds), gt_d)["absrel"], ate=sim3_ate(torch.cat(cs), gt_c),
                    **{f"rms{c}": float(rms[c]) for c in WATCH}, tok_norm=float(z[:, 5:].norm(dim=-1).mean()))

    @torch.no_grad()
    def run(name, base, K, cap):
        t0 = time.time(); z = x0.clone(); rows = []; ref = None
        ts = torch.linspace(0.0, 1.0, K).tolist()
        if base == "BR":
            fg.enable()
        try:
            for i in range(K):
                if base == "sched":
                    z = P.m._interval_step(z, ts[i], ts[i + 1] if i + 1 < K else 1.0, rope, B, S).float()
                else:
                    z = rms_normalize(P.m._interval_step(z, 0.5, 0.5, rope, B, S).float())
                s = i + 1
                if s == a.ref:
                    ref = z.reshape(-1, z.shape[-1]).pow(2).mean(0).sqrt()
                elif s > a.ref and cap is not None:
                    rms = z.reshape(-1, z.shape[-1]).pow(2).mean(0).sqrt()
                    f = torch.ones_like(rms)
                    idx = torch.arange(768, device=z.device) if cap == "all" else torch.tensor(cap, device=z.device)
                    f[idx] = (ref[idx] / rms[idx].clamp_min(1e-12)).clamp(max=1.0)
                    z = z * f
                    if base == "BR":
                        z = rms_normalize(z)
                rows.append(dict(step=s, **evaluate(z)))
                r = rows[-1]
                if s in (8, 16, 24, 32, 48, 64, 96, 128):
                    print(f"  [{name}] step {s:>3}: AbsRel {r['absrel']:.4f} ATE {r['ate']:.4f} | rms "
                          + " ".join(f"c{c} {r[f'rms{c}']:.2f}" for c in WATCH), flush=True)
        finally:
            fg.restore()
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
        return dict(name=name, base=base, K=K, cap=cap, rows=rows)

    plan = [("K64_base", "sched", 64, None), ("K64_cap_target", "sched", 64, a.target),
            ("K64_cap_random", "sched", 64, rand_ch), ("K64_cap_all", "sched", 64, "all"),
            ("BR128_base", "BR", 128, None), ("BR128_cap_all", "BR", 128, "all")]
    res = json.load(open(f"{a.out}/channelcap.json")) if os.path.exists(f"{a.out}/channelcap.json") else {}
    for name, base, K, cap in plan:
        if name in res:
            continue
        res[name] = run(name, base, K, cap)
        json.dump(res, open(f"{a.out}/channelcap.json", "w"), indent=1)

    lines = [f"cap reference = per-channel RMS at step {a.ref}; target channels {a.target}; random control channels {rand_ch}"]
    for k, v in res.items():
        rw = {r["step"]: r for r in v["rows"]}
        marks = [s for s in (8, 16, 24, 32, 48, 64, 96, 128) if s <= v["K"]]
        b = min(v["rows"], key=lambda r: r["absrel"])
        lines.append(f"{k:>15}: AbsRel " + " ".join(f"{rw[s]['absrel']:.3f}" for s in marks) + " | ATE "
                     + " ".join(f"{rw[s]['ate']:.3f}" for s in marks) + f" | best {b['absrel']:.4f} ({b['step']})  [steps {marks}]")
    lines.append("reference: DVLT K16 AbsRel 0.0309 ATE 0.0083")
    print("\n" + "\n".join(lines))
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(22, 8))
    for row_i, pref in enumerate(["K64", "BR128"]):
        runs = {k: v for k, v in res.items() if k.startswith(pref)}
        for c, (key, t) in enumerate([("absrel", "AbsRel"), ("ate", "ATE"), ("rms480", "RMS ch480"), ("rms215", "RMS ch215")]):
            for j, (nm, v) in enumerate(runs.items()):
                ax[row_i, c].plot([r["step"] for r in v["rows"]], [r[key] for r in v["rows"]], color=f"C{j}", label=nm)
            ax[row_i, c].axvline(a.ref, color="k", ls=":", lw=.8); ax[row_i, c].set_title(f"{pref}: {t}"); ax[row_i, c].grid(alpha=.3)
            if key in ("absrel", "ate"): ax[row_i, c].set_yscale("log")
        ax[row_i, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.out}/channelcap.png", dpi=120); plt.close(fig)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
