#!/usr/bin/env python3
"""corr_residual.py -- internal correspondence residual on frozen DVLT (no training, no external matcher, no readout).

1. Run the loop normally for `record_step` steps. During the global attention of step `record_step`, take heads 0 and 9
   (the correspondence heads), and for every patch query i (frame a) and every other frame b compute:
     loc[i,b]  sub-patch landing position in frame b (3x3 soft-argmax around the in-frame peak)
     mass[i,b] attention mass the query gives frame b's patches (head mean)
     peak[i,b] largest in-frame-renormalised probability
   Edge (i in a -> j in b), j = patch nearest to loc[i,b], is kept if
     mass[i,b] > mass_min * uniform_frame_share                (frame-level visibility)
     |loc[j,a] - pos_i| <= cycle_max  patches                  (round trip i -> j -> back lands near i)
     cos(z_i - m_a, z_j - m_b) >= cos_min                      (similar after removing frame means)
   weight w_ij = peak[i,b]. Edges are FIXED after recording.
2. From step `record_step` on, after every loop step:  r_ij = (z_i - m_a) - (z_j - m_b),
     z_i -= eta * mean_e(w r),  z_j += eta * mean_e(w r)      (per-token weighted mean over its edges; patch tokens only)
Conditions: eta sweep x gate configs  {sched: DVLT time schedule (runs of K=16 and K=64),
                                       A: fixed gates f(0.5) (one run of K_fixed steps),
                                       Bnorm: fixed gates mode B + Solv3R RMS norm (one run of K_fixed steps)}.
Logged every `every` steps: AbsRel, d1, ATE (GT), |x|, k_eff, mean |r| over edges, mean cosine of random patch pairs
(mean-removed; rising = global flattening).

  PYTHONPATH=$(pwd) python shovel/corr_residual.py --max-frames 27 --etas 0 0.01 0.03 0.1 0.3 \
      --gates sched A Bnorm --sched-Ks 16 64 --K-fixed 64 --out out/corrres
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvlt_probe import Probe, load_gt, gt_metrics, _dyn_sub, _dyn_row  # noqa: E402
from gate_fix import FixedGates, rms_normalize  # noqa: E402

HEADS = [0, 9]


class QKCapture:
    """Swap the global-attention forward to grab normalised q,k of the chosen heads once (math identical otherwise)."""

    def __init__(self, P):
        self.g = P.m.recurrent_blocks[0].global_attn.attn
        self.orig = self.g.forward
        self.want, self.qk = False, None

    def __enter__(self):
        g = self

        def fwd(x, pos=None, block_mask=None):
            if g.want:
                a = g.g
                B, N, C = x.shape
                qkv = a.qkv(x).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                q, k = a.q_norm(q), a.k_norm(k)
                g.qk = (q[0, HEADS].float().clone(), k[0, HEADS].float().clone(), a.scale)
                g.want = False
            return g.orig(x, pos=pos, block_mask=block_mask)
        self.g.forward = fwd
        return self

    def __exit__(self, *e):
        self.g.forward = self.orig


@torch.no_grad()
def extract_edges(qk, x, S, Pf, pm, gh, gw, a):
    """returns edges (E,4) [frame_a, patch_i, frame_b, patch_j] (patch index within frame) and weights (E,)."""
    q, k, scale = qk                                                   # (2,N,64)
    dev = q.device
    Np = gh * gw
    pidx = torch.cat([torch.arange(f * Pf + pm, (f + 1) * Pf, device=dev) for f in range(S)])   # global ids of patches
    yy, xx = torch.meshgrid(torch.arange(gh, device=dev), torch.arange(gw, device=dev), indexing="ij")
    pos = torch.stack([yy.flatten(), xx.flatten()], -1).float()        # (Np,2)
    loc = torch.zeros(S * Np, S, 2, device=dev); mass = torch.zeros(S * Np, S, device=dev); peak = torch.zeros(S * Np, S, device=dev)
    for c0 in range(0, S * Np, a.chunk):
        qi = pidx[c0:c0 + a.chunk]
        p = ((q[:, qi] * scale) @ k.transpose(-1, -2)).softmax(-1).mean(0)          # (c, N) head mean
        pp = p[:, pidx].reshape(len(qi), S, Np)                                   # (c,S,Np)
        m = pp.sum(-1); r = pp / m.clamp_min(1e-12)[..., None]
        pk, am = r.max(-1)                                                         # (c,S)
        ay, ax = am // gw, am % gw
        num = torch.zeros(len(qi), S, 2, device=dev); den = torch.zeros(len(qi), S, device=dev)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y2, x2 = ay + dy, ax + dx
                ok = (y2 >= 0) & (y2 < gh) & (x2 >= 0) & (x2 < gw)
                w = r.gather(-1, (y2.clamp(0, gh - 1) * gw + x2.clamp(0, gw - 1))[..., None])[..., 0] * ok
                num += w[..., None] * torch.stack([y2, x2], -1).float(); den += w
        loc[c0:c0 + len(qi)] = num / den.clamp_min(1e-12)[..., None]
        mass[c0:c0 + len(qi)] = m; peak[c0:c0 + len(qi)] = pk
    # candidate edges: every patch i (frame fa) -> every other frame fb
    fa = (torch.arange(S * Np, device=dev) // Np)[:, None].expand(-1, S)
    fb = torch.arange(S, device=dev)[None, :].expand(S * Np, -1)
    ii = (torch.arange(S * Np, device=dev) % Np)[:, None].expand(-1, S)
    jj = (loc[..., 0].round().clamp(0, gh - 1) * gw + loc[..., 1].round().clamp(0, gw - 1)).long()
    keep = fa != fb
    uni = Np / (S * (Pf))
    keep &= mass > a.mass_min * uni
    back = loc[fb * Np + jj, fa]                                                    # (S*Np,S,2) j's landing back in frame a
    cyc = (back - pos[ii]).norm(dim=-1)
    keep &= cyc <= a.cycle_max
    X = x[:, pm:].float()                                                           # (S,Np,C)
    Xr = X - X.mean(1, keepdim=True)
    sel = keep.nonzero(as_tuple=True)
    A_ = Xr[fa[sel], ii[sel]]; B_ = Xr[fb[sel], jj[sel]]
    cs = F.cosine_similarity(A_, B_, dim=-1)
    ok = cs >= a.cos_min
    edges = torch.stack([fa[sel][ok], ii[sel][ok], fb[sel][ok], jj[sel][ok]], -1)
    w = peak[sel][ok]
    stats = dict(candidates=int((fa != fb).sum()), after_mass=int(((fa != fb) & (mass > a.mass_min * uni)).sum()),
                 after_cycle=int(keep.sum()), kept=int(ok.sum()), mean_w=float(w.mean()) if len(w) else 0.0,
                 frac_patches_with_edge=float(torch.zeros(S * Np, device=dev).index_fill_(0, (edges[:, 0] * Np + edges[:, 1]), 1).mean()))
    return edges, w, stats


@torch.no_grad()
def correct(x, edges, w, eta, pm):
    """one correction on patch tokens: pull corresponding tokens together in the frame-mean-removed space."""
    X = x[:, pm:].float()
    S, Np, C = X.shape
    m = X.mean(1, keepdim=True)
    fa, ii, fb, jj = edges.unbind(-1)
    r = (X[fa, ii] - m[fa, 0]) - (X[fb, jj] - m[fb, 0])                             # (E,C)
    wr = w[:, None] * r
    acc = torch.zeros(S * Np, C, device=x.device); cnt = torch.zeros(S * Np, device=x.device)
    acc.index_add_(0, fa * Np + ii, -wr); cnt.index_add_(0, fa * Np + ii, w)
    acc.index_add_(0, fb * Np + jj, wr); cnt.index_add_(0, fb * Np + jj, w)
    delta = (acc / cnt.clamp_min(1e-6)[:, None]).reshape(S, Np, C)
    x = x.clone(); x[:, pm:] = (X + eta * delta).to(x.dtype)
    return x, float(r.norm(dim=-1).mean())


def run_condition(P, images, a, gate, K, eta, sub, rng_pairs):
    B_, S, H, W = None, None, None, None
    x, rope, dims = P.init_state(images)
    B_, S, H, W = dims
    pm, Pf = P.m.patch_start_idx, x.shape[1]
    gh, gw = H // P.m.patch_size, W // P.m.patch_size
    fg = FixedGates(P.m, mode={"sched": "off", "A": "A", "Bnorm": "B"}[gate]).enable()
    rows, edges, w, est = [], None, None, {}
    try:
        with QKCapture(P) as cap, torch.no_grad():
            for i in range(K):
                cap.want = (i == a.record_step)
                x = P.step(x, i, K, rope, B_, S)
                if gate == "Bnorm":
                    x = rms_normalize(x)
                if i == a.record_step:
                    edges, w, est = extract_edges(cap.qk, x, S, Pf, pm, gh, gw, a)
                rmean = float("nan")
                if edges is not None and eta > 0 and len(edges):
                    x, rmean = correct(x, edges, w, eta, pm)
                if (i + 1) % a.every == 0 or i == K - 1:
                    r = _dyn_row(P, x, rope, dims, sub, i + 1, 0.0, 0.0, decode=False)
                    r.update(gt_metrics(P, P.decode(x, rope, dims)))
                    Xr = x[:, pm:].float(); Xr = Xr - Xr.mean(1, keepdim=True)
                    fa, ia, fb, ib = rng_pairs
                    r["rand_cos_rm"] = float(F.cosine_similarity(Xr[fa, ia], Xr[fb, ib], dim=-1).mean())
                    r["res_mean"] = rmean
                    rows.append(r)
    finally:
        fg.restore()
    return rows, est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--gt", default="out/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--etas", type=float, nargs="+", default=[0, 0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--gates", nargs="+", default=["sched", "A", "Bnorm"])
    ap.add_argument("--sched-Ks", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--K-fixed", type=int, default=64)
    ap.add_argument("--record-step", type=int, default=3, help="loop step index whose global attention defines the edges")
    ap.add_argument("--mass-min", type=float, default=1.0, help="frame mass threshold, x uniform share")
    ap.add_argument("--cycle-max", type=float, default=1.5, help="round-trip error threshold (patches)")
    ap.add_argument("--cos-min", type=float, default=0.2, help="cosine threshold after frame-mean removal")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", default="out/corrres")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    P = Probe("nvidia/dvlt", max_frames=a.max_frames)
    images = P.load(a.input)
    S = images.shape[1]
    P.gt_depth, P.gt_c2w, P.gt_K = load_gt(a.gt, P.img_size, device=P.acc.device)
    P.gt_depth, P.gt_c2w = P.gt_depth[:S], P.gt_c2w[:S]
    x0, _, dims = P.init_state(images)
    sub = _dyn_sub(P, x0, S)
    Np = x0.shape[1] - P.m.patch_start_idx
    g = torch.Generator().manual_seed(0)
    n = 4000
    fa = torch.randint(0, S, (n,), generator=g); fb = (fa + torch.randint(1, S, (n,), generator=g)) % S
    rng_pairs = tuple(t.to(x0.device) for t in (fa, torch.randint(0, Np, (n,), generator=g), fb, torch.randint(0, Np, (n,), generator=g)))

    plan = []
    for gate in a.gates:
        for K in (a.sched_Ks if gate == "sched" else [a.K_fixed]):
            plan.append((gate, K))
    results, log = {}, []
    for gate, K in plan:
        for eta in a.etas:
            rows, est = run_condition(P, images, a, gate, K, eta, sub, rng_pairs)
            key = f"{gate}_K{K}_eta{eta:g}"
            results[key] = dict(gate=gate, K=K, eta=eta, rows=rows, edges=est)
            last = rows[-1]
            line = (f"[{key:>22}] edges kept {est.get('kept', 0)} (patches with edge {est.get('frac_patches_with_edge', 0):.2f}) | "
                    + " ".join(f"s{r['step']}:{r['absrel']:.3f}" for r in rows if r['step'] in (4, 8, 16, 32, 64, K))
                    + f" | final AbsRel {last['absrel']:.4f} ATE {last.get('ate_rel', float('nan')):.4f} k_eff {last['k_eff']:.2f}"
                      f" |x| {last['tok_norm']:.1f} randcos {last['rand_cos_rm']:.3f} |r| {last['res_mean']:.2f}")
            print(line, flush=True); log.append(line)
            if eta == a.etas[0]:
                log.append(f"    edge stats: {est}")
    json.dump(dict(args=vars(a), results=results), open(f"{a.out}/corrres.json", "w"), indent=1)
    with open(f"{a.out}/summary.txt", "w") as fh:
        fh.write(" ".join(sys.argv) + "\n" + "\n".join(log) + "\n")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    metrics = [("absrel", "AbsRel vs GT", True), ("ate_rel", "ATE (rel.)", True), ("k_eff", "k_eff", False),
               ("tok_norm", "|x|", True), ("rand_cos_rm", "random-pair cos (mean removed)", False)]
    fig, ax = plt.subplots(len(plan), len(metrics), figsize=(4.2 * len(metrics), 3.3 * len(plan)), squeeze=False)
    for r_, (gate, K) in enumerate(plan):
        for eta in a.etas:
            rows = results[f"{gate}_K{K}_eta{eta:g}"]["rows"]
            for c_, (m, t, lg) in enumerate(metrics):
                ax[r_, c_].plot([r["step"] for r in rows], [r.get(m, np.nan) for r in rows], marker=".", label=f"η={eta:g}")
                if lg: ax[r_, c_].set_yscale("log")
                ax[r_, c_].set_title(f"{gate} K={K}: {t}", fontsize=9); ax[r_, c_].grid(alpha=.3)
                ax[r_, c_].axvline(a.record_step + 1, color="k", ls=":", lw=.7)
        ax[r_, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{a.out}/corrres.png", dpi=110); plt.close(fig)
    print(f"saved {a.out}/corrres.json, summary.txt, corrres.png")


if __name__ == "__main__":
    main()
