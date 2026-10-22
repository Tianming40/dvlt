#!/usr/bin/env python3
"""Matrix-level visualization of attention (a)symmetry.

Runs the baseline loop to a chosen step, grabs the global-attention logit matrix L on a small token
subsample, and draws: L, its transpose Lᵀ, the symmetric part S=½(L+Lᵀ), the antisymmetric part
A=½(L−Lᵀ), and a scatter of L_ij vs L_ji (points on the diagonal = symmetric). Prints the asym value.

  python scriptandtool/viz_asym.py --step 8 --ntok 96 --frames 20
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dvlt_probe import Probe, _patch_global_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/gt_00777c41d4/probe_input")
    ap.add_argument("--ckpt", default="nvidia/dvlt")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--step", type=int, default=8, help="loop step to visualize")
    ap.add_argument("--ntok", type=int, default=96, help="#tokens in the matrix (small = readable)")
    ap.add_argument("--out", default="out/full_grand/asym_matrix.png")
    a = ap.parse_args()

    P = Probe(a.ckpt, max_frames=a.frames)
    images = P.load(a.input)
    g_attn = P.m.recurrent_blocks[0].global_attn.attn
    cap = {}
    hk = g_attn.register_forward_pre_hook(lambda m, args, kw=None: cap.__setitem__("x", args[0].detach()), with_kwargs=True)

    with torch.no_grad(), P.acc.autocast():
        x, rope, dims = P.init_state(images)
        B, S, H, W = dims
        P_full = x.shape[1]
        for i in range(a.step + 1):
            x = P.step(x, i, max(a.step + 1, 16), rope, B, S)
        xg = cap["x"][0]                                  # (SP, C) global-attn input at this step
    hk.remove()

    pidx = _patch_global_idx(P, S, P_full).to(xg.device)
    g2 = torch.Generator(device="cpu").manual_seed(0)
    sub = pidx[torch.randperm(pidx.numel(), generator=g2)[:a.ntok]].sort().values.to(xg.device)

    with torch.no_grad(), torch.autocast(device_type=P.acc.device.type, enabled=False):
        qkv = g_attn.qkv(xg).reshape(xg.shape[0], 3, g_attn.num_heads, g_attn.head_dim).permute(1, 2, 0, 3)
        q, k, _ = qkv.unbind(0)
        q, k = g_attn.q_norm(q).float(), g_attn.k_norm(k).float()
        scale = g_attn.head_dim ** -0.5
        qs, ks = q[:, sub], k[:, sub]
        L = ((qs * scale) @ ks.transpose(-1, -2)).mean(0)         # (n,n) head-mean logit matrix
        Lt = L.t()
        Sym = 0.5 * (L + Lt)
        Asym = 0.5 * (L - Lt)
        asym = (Asym.norm() / L.norm().clamp_min(1e-9)).item()
    L, Lt, Sym, Asym = (t.cpu().numpy() for t in (L, Lt, Sym, Asym))
    print(f"step {a.step}, {a.ntok} tokens: asym = ||A||/||L|| = {asym:.3f}  (0=symmetric, 1=antisymmetric)")

    fig = plt.figure(figsize=(16, 9))
    vmax = np.percentile(np.abs(L), 99)
    amax = np.percentile(np.abs(Asym), 99)
    specs = [("L  (L_ij = token i's attention affinity for j)", L, "viridis", None),
             ("Lᵀ  (transpose)", Lt, "viridis", None),
             ("symmetric part  S = ½(L+Lᵀ)  (mutual)", Sym, "viridis", None),
             ("antisymmetric part  A = ½(L−Lᵀ)  (one-way)", Asym, "RdBu_r", amax)]
    for n, (title, M, cmap, am) in enumerate(specs):
        ax = fig.add_subplot(2, 3, n + 1)
        if am is None:
            im = ax.imshow(M, cmap=cmap, vmin=np.percentile(M, 1), vmax=np.percentile(M, 99))
        else:
            im = ax.imshow(M, cmap=cmap, vmin=-am, vmax=am)
        ax.set_title(title, fontsize=10); ax.set_xlabel("token j"); ax.set_ylabel("token i")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # scatter L_ij vs L_ji (off-diagonal); on the diagonal line y=x => symmetric
    ax = fig.add_subplot(2, 3, 5)
    iu = np.triu_indices_from(L, k=1)
    ax.scatter(L[iu], L.T[iu], s=4, alpha=0.3, color="#1f77b4")
    lo, hi = min(L[iu].min(), L.T[iu].min()), max(L[iu].max(), L.T[iu].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="L_ij = L_ji (symmetric)")
    ax.set_xlabel("L_ij"); ax.set_ylabel("L_ji"); ax.set_title("L_ij vs L_ji  (tight on red line = symmetric)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_aspect("equal")

    ax = fig.add_subplot(2, 3, 6); ax.axis("off")
    ax.text(0.05, 0.6, f"asym = ‖A‖ / ‖L‖ = {asym:.3f}\n\n0 = perfectly symmetric\n     (mutual, gradient-flow-like)\n1 = purely one-way\n\nstep {a.step}, {a.ntok} tokens,\nhead-mean global-attn logits",
            fontsize=13, va="center")
    fig.suptitle(f"Global-attention interaction matrix: symmetric vs antisymmetric  (asym={asym:.3f})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
