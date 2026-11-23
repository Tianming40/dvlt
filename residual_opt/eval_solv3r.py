#!/usr/bin/env python3
"""Evaluate the released Solv3R on the SAME cached clips and with the SAME patch-level metrics as train_stationary.py
(inference only).  One trajectory per clip from the encodings, read out at K = 8, 16, 32, 64, 128 calls.

  python residual_opt/eval_solv3r.py            -> out/resopt/solv3r_eval/{summary.txt, solv3r_eval.json}
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
sys.path.insert(0, f"{ROOT}/third_party/solv3r")
from solv3r import Solv3R  # noqa: E402
from solv3r.model import autocast  # noqa: E402
from residual_opt.train import metrics  # noqa: E402

PS = 14


def per_patch(t, gh, gw):
    S = t.shape[0]
    return t.reshape(S, gh, PS, gw, PS, -1).mean((2, 4)).reshape(S, gh * gw, -1)


def geometry(out, H, W):
    """Solv3R decode -> patch-level [log depth, unit ray dir, ray origin] (S, Np, 7), as in cache_dvlt.py"""
    gh, gw = H // PS, W // PS
    d = out["depth"][..., 0].float(); r = out["rays"].float()
    dirs = r[..., :3] / r[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-9)
    pd = per_patch(dirs, gh, gw); pd = pd / pd.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.cat([per_patch(d.clamp_min(1e-6).log()[..., None], gh, gw), pd, per_patch(r[..., 3:6], gh, gw)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/storage/user/zhtia/resopt_cache/small")
    ap.add_argument("--splits", nargs="+", default=["val", "ood_bf"])
    ap.add_argument("--K", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--out", default=f"{ROOT}/out/resopt/solv3r_eval")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = Solv3R.from_pretrained()
    res, lines = {}, []
    for sp in a.splits:
        files = sorted(glob.glob(f"{a.cache}/{sp}/*.npz"))
        per = {k: [] for k in a.K}
        for f in files:
            z = np.load(f)
            img = torch.from_numpy(z["image"]).float() / 255.0                    # (S,3,378,504), DVLT preprocessing
            ctx = model.encode(img)
            st = model.start(ctx)
            with autocast():
                for k in range(1, max(a.K) + 1):
                    st = model.step(st, ctx)
                    if k in a.K:
                        out = model.decode(st, ctx)
                        x = geometry(out, ctx.H, ctx.W).cpu().numpy()
                        per[k].append(metrics(x, z["x_gt"], z["mask"]))
        res[sp] = {k: {m: float(np.mean([v[m] for v in vs])) for m in vs[0]} for k, vs in per.items()}
        line = f"[{sp}] {len(files)} clips | " + " | ".join(
            f"K{k}: {m['absrel']:.4f} {m['dir_deg']:.2f}° {m['ate']:.4f}" for k, m in res[sp].items())
        print(line, flush=True); lines.append(line)
    open(f"{a.out}/summary.txt", "w").write("Solv3R (released weights), patch-level AbsRel / ray-dir deg / ATE\n" + "\n".join(lines) + "\n")
    json.dump(res, open(f"{a.out}/solv3r_eval.json", "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
