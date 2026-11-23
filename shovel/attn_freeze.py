#!/usr/bin/env python3
"""attn_freeze.py -- is attention DRIFT the cause of the collapse?  Freeze the GLOBAL attention from step k on.

Global attention A = softmax(Q K^T / sqrt(d)) depends only on (Q, K). At step k we run normally and store the global
attention's Q, K (after q_norm / k_norm; global attn has no RoPE); from step k+1 on every global attention uses these
frozen Q, K with the CURRENT V  ==  the attention map is frozen at A_k (no 23463^2 matrix is stored).
Frame attention is untouched. Base: original DVLT time schedule, K=64 (collapses past the K<=16 training budget), fp32.

Optional soft anchoring (--beta, not run by default): softmax(Q K^T + beta * Q_k K_k^T) / sqrt(d) == attention with the
log-prior beta*log A_k (row constants cancel), computed exactly by concatenating [Q, sqrt(beta) Q_k] and [K, sqrt(beta) K_k]
along the head dimension.

  PYTHONPATH=$(pwd) python shovel/attn_freeze.py --freeze-at 4 8 16 --out out/attnfreeze
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvlt_probe import Probe, load_gt, depth_vs_gt, sim3_ate  # noqa: E402
from solv3r_attractor import _keff_and_norm  # noqa: E402

PM = 5


class GlobalAttnControl:
    """Wraps recurrent_blocks[0].global_attn.attn.forward: mode 'normal' | 'capture' | 'frozen' | 'soft'."""

    def __init__(self, m):
        self.attn = m.recurrent_blocks[0].global_attn.attn
        self.orig = self.attn.forward
        self.mode, self.qk, self.beta = "normal", None, 0.0
        self.attn.forward = self.forward

    def forward(self, x, pos=None, block_mask=None):
        a = self.attn
        if self.mode == "normal":
            return self.orig(x, pos=pos, block_mask=block_mask)
        B, N, C = x.shape
        qkv = a.qkv(x).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = a.q_norm(q), a.k_norm(k)
        if a.rope is not None and pos is not None:
            q, k = a.rope(q, pos), a.rope(k, pos)
        q, k = q.to(v.dtype), k.to(v.dtype)
        if self.mode == "capture":
            self.qk = (q.detach().clone(), k.detach().clone())
        elif self.mode == "frozen":
            q, k = self.qk
        elif self.mode == "soft":
            sb = self.beta ** 0.5
            q = torch.cat([q, sb * self.qk[0]], -1)
            k = torch.cat([k, sb * self.qk[1]], -1)
        o = Fn.scaled_dot_product_attention(q, k, v, scale=a.head_dim ** -0.5)
        o = o.transpose(1, 2).reshape(B, N, C)
        return a.proj_drop(a.proj(o))

    def restore(self):
        self.attn.forward = self.orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--gt", default="out/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--freeze-at", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--beta", type=float, nargs="*", default=[], help="soft anchoring strengths (anchor at the first --freeze-at)")
    ap.add_argument("--out", default="out/attnfreeze")
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
    ts = torch.linspace(0.0, 1.0, a.K).tolist()
    ctl = GlobalAttnControl(P.m)

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
    def run(name, freeze_at=None, beta=None):
        t0 = time.time()
        z = x0.clone(); rows = []
        for i in range(a.K):
            if freeze_at is not None and i + 1 == freeze_at:
                ctl.mode = "capture"
            elif freeze_at is not None and i + 1 > freeze_at:
                ctl.mode = "frozen" if beta is None else "soft"; ctl.beta = beta or 0.0
            else:
                ctl.mode = "normal"
            z = P.m._interval_step(z, ts[i], ts[i + 1] if i + 1 < a.K else 1.0, rope, B, S).float()
            rows.append(dict(step=i + 1, **evaluate(z)))
            r = rows[-1]
            if (i + 1) % 8 == 0 or i + 1 in (1, 2, 4) or i + 1 == freeze_at:
                print(f"  [{name}] step {i + 1:>3}: AbsRel {r['absrel']:.4f} ATE {r['ate']:.4f} |z| {r['tok_norm']:.1f} k_eff {r['k_eff']:.2f}"
                      + ("   <- attention frozen from here on" if i + 1 == freeze_at else ""), flush=True)
        ctl.mode, ctl.qk = "normal", None
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
        return dict(name=name, freeze_at=freeze_at, beta=beta, rows=rows)

    res = {"base": run("base")}
    for k in a.freeze_at:
        res[f"freeze@{k}"] = run(f"freeze@{k}", k)
    for b in a.beta:
        res[f"soft@{a.freeze_at[0]}_b{b:g}"] = run(f"soft@{a.freeze_at[0]}_b{b:g}", a.freeze_at[0], b)
    ctl.restore()
    json.dump(res, open(f"{a.out}/attnfreeze.json", "w"), indent=1)

    marks = [s for s in (4, 8, 16, 24, 32, 48, 64) if s <= a.K]
    lines = ["run | AbsRel @" + "/".join(map(str, marks)) + " | ATE @" + "/".join(map(str, marks)) + " | k_eff @" + "/".join(map(str, marks))]
    for k, v in res.items():
        rw = {r["step"]: r for r in v["rows"]}
        lines.append(f"{k:>12}: " + " ".join(f"{rw[s]['absrel']:.3f}" for s in marks) + " | "
                     + " ".join(f"{rw[s]['ate']:.3f}" for s in marks) + " | " + " ".join(f"{rw[s]['k_eff']:.2f}" for s in marks))
    lines.append("reference: DVLT K16 (in budget) AbsRel 0.0309 ATE 0.0083")
    print("\n" + "\n".join(lines))
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    for c, (key, t, logy) in enumerate([("absrel", "AbsRel", True), ("ate", "ATE (Sim3, rel.)", True), ("k_eff", "k_eff (low = clustered)", False), ("tok_norm", "|z|", False)]):
        for j, (nm, v) in enumerate(res.items()):
            ax[c].plot([r["step"] for r in v["rows"]], [r[key] for r in v["rows"]], lw=1.2, color=f"C{j}", label=nm)
            if v["freeze_at"]:
                ax[c].axvline(v["freeze_at"], color=f"C{j}", ls=":", lw=.8)
        ax[c].set_title(t); ax[c].set_xlabel(f"step (K={a.K} schedule)"); ax[c].grid(alpha=.3)
        if logy: ax[c].set_yscale("log")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.out}/attnfreeze.png", dpi=130); plt.close(fig)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
