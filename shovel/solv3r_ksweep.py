#!/usr/bin/env python3
"""solv3r_ksweep.py -- ORIGINAL Solv3R (released weights, unmodified), run SEPARATELY for each K (number of solver calls),
recording metrics after every call; plotted against normalised progress call/K in [0, 1] (one line per K).

Per call (call 0 = start state = encodings):
  absrel, d1  : depth vs GT (median-scale aligned)            -- dvlt_probe.depth_vs_gt (same as all DVLT runs)
  ate_rel     : camera centres (mean ray origin) vs GT, Sim(3) -- dvlt_probe.sim3_ate   (same as all DVLT runs)
  tok_norm    : mean patch-token |z|;  k_eff : Gram participation ratio (1024 patch subsample)
  rel_change  : |z_c - z_{c-1}| / |z_{c-1}|  (Solv3R's own stopping quantity)
Images / GT loaded exactly as in the DVLT experiments (Probe.load + load_gt), so numbers are directly comparable.

  PYTHONPATH=$(pwd):$(pwd)/third_party/solv3r python shovel/solv3r_ksweep.py \
      --input out/data/gt_00777c41d4_s10/probe_input --gt out/data/gt_00777c41d4_s10/gt.npz --max-frames 27 \
      --Ks 8 16 32 64 128 256 --out out/solv3r/solv3rexperiment/ksweep
Does NOT modify src/dvlt, dvlt_probe.py or third_party/solv3r.
"""
import argparse
import csv
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "solv3r"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvlt_probe import Probe, load_gt, depth_vs_gt, sim3_ate  # noqa: E402
from solv3r_attractor import _keff_and_norm  # noqa: E402


def metrics(model, z, ctx, gt_depth, gt_c2w, pm):
    pred = model.decode(z, ctx)
    r = depth_vs_gt(pred["depth"][..., 0].float(), gt_depth)
    centres = pred["rays"][..., 3:6].float().mean(dim=(1, 2))           # (S,3) mean ray origin per view
    r["ate_rel"] = sim3_ate(centres, gt_c2w[:, :3, 3])
    r["tok_norm"], r["k_eff"] = _keff_and_norm(z, pm)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--Ks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--out", default="out/solv3r/solv3rexperiment/ksweep")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from solv3r import Solv3R
    from solv3r.model import autocast

    Pload = Probe("nvidia/dvlt", max_frames=a.max_frames)
    images = Pload.load(a.input)                                          # (1,S,3,H,W) in [0,1]
    S = images.shape[1]
    gt_depth, gt_c2w, _ = load_gt(a.gt, Pload.img_size, device="cuda")
    gt_depth, gt_c2w = gt_depth[:S].float(), gt_c2w[:S].float()
    del Pload
    torch.cuda.empty_cache()

    model = Solv3R.from_pretrained()
    ctx = model.encode(images[0].to(model.device))
    pm = 5                                                                # cam + 4 registers, then patches

    runs = {}
    for K in a.Ks:                                                        # each K = a SEPARATE run from the start state
        z = model.start(ctx)
        rows = [dict(K=K, call=0, progress=0.0, rel_change=float("nan"), **metrics(model, z, ctx, gt_depth, gt_c2w, pm))]
        print(f"\n[K={K}]  call   0  absrel={rows[0]['absrel']:.4f} ate={rows[0]['ate_rel']:.4f} "
              f"|z|={rows[0]['tok_norm']:.1f} k_eff={rows[0]['k_eff']:.2f}")
        for c in range(1, K + 1):
            with torch.no_grad(), autocast():
                zn = model.step(z, ctx)
            rc = ((zn.float() - z.float()).norm() / z.float().norm().clamp_min(1e-9)).item()
            z = zn
            r = dict(K=K, call=c, progress=c / K, rel_change=rc, **metrics(model, z, ctx, gt_depth, gt_c2w, pm))
            rows.append(r)
            if c in (1, 2, 4) or c % max(K // 8, 1) == 0:
                print(f"         call {c:>3}  absrel={r['absrel']:.4f} d1={r['d1']:.3f} ate={r['ate_rel']:.4f} "
                      f"|z|={r['tok_norm']:.1f} k_eff={r['k_eff']:.2f} dz/z={rc:.2e}")
        runs[K] = rows
        with open(f"{a.out}/solv3r_K{K}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    json.dump({str(k): v for k, v in runs.items()}, open(f"{a.out}/solv3r_ksweep.json", "w"), indent=1)
    lines = ["FINAL CALL per K: absrel / d1 / ate_rel / |z| / k_eff / rel_change"]
    for K, rows in runs.items():
        f = rows[-1]
        lines.append(f"  K={K:>4}: absrel={f['absrel']:.4f} d1={f['d1']:.3f} ate={f['ate_rel']:.4f} "
                     f"|z|={f['tok_norm']:.1f} k_eff={f['k_eff']:.2f} dz/z={f['rel_change']:.2e}")
    print("\n" + "\n".join(lines))
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        keys = [("absrel", "depth AbsRel vs GT", True), ("ate_rel", "camera ATE (rel., Sim3)", True),
                ("tok_norm", "patch-token |z|", False), ("k_eff", "k_eff (low = clustered)", False),
                ("rel_change", "|dz|/|z| per call", True)]
        fig, ax = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
        for i, (k, title, logy) in enumerate(keys):
            for K, rows in runs.items():
                pts = [(r["progress"], r[k]) for r in rows if r[k] == r[k]]
                ax[i].plot([p for p, _ in pts], [v for _, v in pts], marker=".", ms=3, lw=1.2, label=f"K={K}")
            if logy: ax[i].set_yscale("log")
            ax[i].set_title(title); ax[i].set_xlabel("progress  call / K"); ax[i].set_xlim(0, 1); ax[i].grid(alpha=.3)
        ax[0].legend(fontsize=8)
        fig.suptitle(f"original Solv3R, separate runs per K ({S} frames)")
        fig.tight_layout(); fig.savefig(f"{a.out}/solv3r_ksweep.png", dpi=130); plt.close(fig)
        print(f"saved {a.out}/solv3r_ksweep.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
