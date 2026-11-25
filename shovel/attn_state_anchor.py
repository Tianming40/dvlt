#!/usr/bin/env python3
"""attn_state_anchor.py -- combine ATTENTION anchoring (keeps the camera) with STATE anchoring (should keep the depth).

Base: frozen DVLT with fixed gates B + RMS state norm (gate_fix), K steps (default 128), fp32. Training-free.
  attention anchor (at step --anchor, i.e. before the drift; global attention only, see attn_freeze.GlobalAttnControl):
      freeze : from step anchor+1 use the stored Q_a, K_a with the current V          (A = A_a)
      soft   : softmax((Q K^T + beta * Q_a K_a^T) / sqrt(d))                         (log-prior anchoring to A_a)
  state anchor (restart / APPNP-style re-injection of the encoder state):
      z <- RMS( (1 - alpha) * F(z) + alpha * x0_hat ),   x0_hat = x0 rescaled to the same RMS
Runs: base, freeze, soft b1, soft b4, restart a.02, restart a.05, freeze + a.02, soft b1 + a.02.

  PYTHONPATH=$(pwd) python shovel/attn_state_anchor.py --out out/stage3_latent/attnstate
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvlt_probe import Probe, load_gt, depth_vs_gt, sim3_ate  # noqa: E402
from solv3r_attractor import _keff_and_norm  # noqa: E402
from attn_freeze import GlobalAttnControl  # noqa: E402
from gate_fix import FixedGates, rms_normalize  # noqa: E402

PM = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/data/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--gt", default="out/data/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--anchor", type=int, default=8)
    ap.add_argument("--out", default="out/stage3_latent/attnstate")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    P = Probe("nvidia/dvlt", max_frames=a.max_frames)
    images = P.load(a.input)
    with torch.no_grad():
        x0, rope, dims = P.init_state(images)
    x0 = x0.float()
    B, S, H, W = dims
    gt_d, gt_c2w, _ = load_gt(a.gt, P.img_size, device=x0.device)
    gt_d, gt_c = gt_d[:S].float(), gt_c2w[:S, :3, 3].float()
    ctl = GlobalAttnControl(P.m)
    fg = FixedGates(P.m, mode="B")
    x0_hat = rms_normalize(x0)

    @torch.no_grad()
    def evaluate(z):
        ds, cs = [], []
        for i in range(0, S, 4):
            pr = P.m._decode(z[i:i + 4], H, W, 1, min(4, S - i), rope[i:i + 4])
            ds.append(pr["depth"][0, ..., 0].float()); cs.append(pr["rays"][0, ..., 3:6].float().mean((1, 2)))
        d, c = torch.cat(ds), torch.cat(cs)
        nrm, keff = _keff_and_norm(z, PM)
        return dict(absrel=depth_vs_gt(d, gt_d)["absrel"], ate=sim3_ate(c, gt_c), tok_norm=float(nrm), k_eff=float(keff))

    @torch.no_grad()
    def run(name, attn=None, beta=None, alpha=0.0):
        t0 = time.time()
        z = x0.clone(); rows = []
        fg.enable()
        try:
            for i in range(a.K):
                if attn is not None and i + 1 == a.anchor:
                    ctl.mode = "capture"
                elif attn is not None and i + 1 > a.anchor:
                    ctl.mode = attn; ctl.beta = beta or 0.0
                else:
                    ctl.mode = "normal"
                z = rms_normalize(P.m._interval_step(z, 0.5, 0.5, rope, B, S).float())
                if alpha > 0:
                    z = rms_normalize((1 - alpha) * z + alpha * x0_hat)
                rows.append(dict(step=i + 1, **evaluate(z)))
                r = rows[-1]
                if (i + 1) in (4, 8, 16, 32, 64, 96, 128):
                    print(f"  [{name}] step {i + 1:>3}: AbsRel {r['absrel']:.4f} ATE {r['ate']:.4f} k_eff {r['k_eff']:.2f}", flush=True)
        finally:
            fg.restore(); ctl.mode, ctl.qk = "normal", None
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
        return dict(name=name, attn=attn, beta=beta, alpha=alpha, rows=rows)

    plan = [("base", None, None, 0.0), (f"freeze@{a.anchor}", "frozen", None, 0.0),
            (f"soft@{a.anchor}_b1", "soft", 1.0, 0.0), (f"soft@{a.anchor}_b4", "soft", 4.0, 0.0),
            ("restart_a0.02", None, None, 0.02), ("restart_a0.05", None, None, 0.05),
            (f"freeze@{a.anchor}+a0.02", "frozen", None, 0.02), (f"soft@{a.anchor}_b1+a0.02", "soft", 1.0, 0.02)]
    res = {}
    if os.path.exists(f"{a.out}/attnstate.json"):
        res = json.load(open(f"{a.out}/attnstate.json"))
    for name, attn, beta, alpha in plan:
        if name in res:
            continue
        res[name] = run(name, attn, beta, alpha)
        json.dump(res, open(f"{a.out}/attnstate.json", "w"), indent=1)
    ctl.restore()

    marks = [s for s in (8, 16, 32, 64, 96, 128) if s <= a.K]
    lines = ["run | AbsRel @" + "/".join(map(str, marks)) + " | ATE @" + "/".join(map(str, marks)) + " | best AbsRel (step) | k_eff @end"]
    for k, v in res.items():
        rw = {r["step"]: r for r in v["rows"]}
        b = min(v["rows"], key=lambda r: r["absrel"])
        lines.append(f"{k:>18}: " + " ".join(f"{rw[s]['absrel']:.3f}" for s in marks) + " | "
                     + " ".join(f"{rw[s]['ate']:.3f}" for s in marks) + f" | {b['absrel']:.4f} ({b['step']}) | {v['rows'][-1]['k_eff']:.2f}")
    lines.append("reference: DVLT K16 (in budget) AbsRel 0.0309 ATE 0.0083 | B4 oracle-guided B+RMS 0.026@16 0.027@32")
    print("\n" + "\n".join(lines))
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
    for c, (key, t) in enumerate([("absrel", "AbsRel"), ("ate", "ATE (Sim3, rel.)"), ("k_eff", "k_eff")]):
        for j, (nm, v) in enumerate(res.items()):
            ax[c].plot([r["step"] for r in v["rows"]], [r[key] for r in v["rows"]], lw=1.1, color=f"C{j}", label=nm,
                       ls="-" if v["alpha"] == 0 else "--")
        ax[c].axvline(a.anchor, color="k", ls=":", lw=.8); ax[c].set_title(t); ax[c].set_xlabel("step (fixed gates B + RMS)"); ax[c].grid(alpha=.3)
        if key != "k_eff": ax[c].set_yscale("log")
    ax[0].axhline(0.0309, color="k", lw=.6); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{a.out}/attnstate.png", dpi=130); plt.close(fig)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
