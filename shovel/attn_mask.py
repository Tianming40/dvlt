"""
attn_mask.py -- ORACLE visibility masks on DVLT's GLOBAL (cross-view) attention.

Question: global attention is pure content matching (DVLT's global_attn has rope=None). A patch in
frame A attends to ALL patches of ALL frames, even frames where its 3D point is not visible.
  (1) how much attention mass is "wasted" on frames where the query patch is NOT visible?  [offline]
  (2) if we mask those keys out (-inf, softmax renormalizes over the rest), does
      collapse get delayed (k_eff / |x|) and does accuracy improve (AbsRel / ATE)?          [run]

Mask modes (True = key kept). Special tokens (cam + 4 registers) are ALWAYS kept as keys (they are the
learned "dustbin" -- softmax must put unmatched mass somewhere), special-token QUERIES are unmasked, and a
patch always sees its OWN frame.
  base    : no mask (original DVLT)
  vis     : L1, frame-level visibility. Other-frame patches kept only if the query's 3D point (GT depth)
            reprojects into that frame, in front, inside, and not occluded (GT depth test)
  winR    : L2, window. As vis, but only patches within R patches (Chebyshev) of the GT reprojection
            (win1 = 3x3, win2 = 5x5, ...). Upper bound for "know WHERE in the other frame".

Visibility/correspondence = dvlt_probe.gt_correspondence (same as cmd_attn). Metrics = dvlt_probe
_dyn_row (|x|, k_eff) + gt_metrics (AbsRel, d1, ATE). Mask applied at EVERY step of the loop.

  python shovel/attn_mask.py --exp all --input out/gt_00777c41d4_s10/probe_input \
      --gt out/gt_00777c41d4_s10/gt.npz --max-frames 27 --Ks 8 16 32 64 128 --out out/attentionmask

Does NOT modify src/dvlt or dvlt_probe.py (the global attention forward is swapped on the instance
and restored).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> dvlt_probe
from dvlt_probe import (Probe, load_gt, gt_correspondence, gt_metrics,  # noqa: E402
                        _dyn_sub, _dyn_row, _dyn_plot)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_fix import FixedGates, rms_normalize  # noqa: E402


# ----------------------------------------------------------------------------- GT visibility tables
@torch.no_grad()
def build_tables(P, S, H, W):
    """Per-token tables over the GLOBAL layout (S*Pf tokens, token t of frame s at s*Pf+t).

    vis[n, j]  : query token n's 3D point is visible in frame j (own frame always True; special rows True)
    py, px     : GT reprojected patch coords of token n in frame j (valid only where vis & j != own)
    covis[i,j] : fraction of frame-i patches (with valid GT depth) visible in frame j
    """
    patch, pm = P.m.patch_size, P.m.patch_start_idx
    ph, pw = H // patch, W // patch
    Np, Pf = ph * pw, pm + ph * pw
    N, dev = S * Pf, P.gt_depth.device
    vis = torch.zeros(N, S, dtype=torch.bool, device=dev)
    py = torch.zeros(N, S, dtype=torch.long, device=dev)
    px = torch.zeros(N, S, dtype=torch.long, device=dev)
    covis = np.zeros((S, S), dtype=np.float32)
    for i in range(S):
        rows = slice(i * Pf + pm, (i + 1) * Pf)
        dvalid = (P.gt_depth[i][(torch.arange(ph, device=dev) * patch + patch // 2)[:, None],
                                (torch.arange(pw, device=dev) * patch + patch // 2)[None, :]] > 1e-6).flatten()
        for j in range(S):
            if j == i:
                vis[rows, j] = True
                covis[i, j] = 1.0
                continue
            valid, gyj, gxj = gt_correspondence(P, i, j, ph, pw, patch, H, W, occ_thresh=P.occ_thresh)
            vis[rows, j] = valid.flatten()
            py[rows, j] = gyj.flatten()
            px[rows, j] = gxj.flatten()
            covis[i, j] = (valid.flatten() & dvalid).float().sum().item() / max(dvalid.sum().item(), 1)
        vis[i * Pf:i * Pf + pm, :] = True                      # special-token queries: unmasked
    qframe = torch.arange(N, device=dev) // Pf
    tok = torch.arange(N, device=dev) % Pf
    is_special = tok < pm
    kgy = torch.where(is_special, torch.zeros_like(tok), (tok - pm) // pw)
    kgx = torch.where(is_special, torch.zeros_like(tok), (tok - pm) % pw)
    return dict(vis=vis, py=py, px=px, covis=covis, qframe=qframe, is_special=is_special,
                kgy=kgy, kgx=kgx, ph=ph, pw=pw, Np=Np, Pf=Pf, pm=pm, N=N, S=S)


@torch.no_grad()
def build_mask(T, mode):
    """(N,N) bool, True = key kept. None for 'base'.
    mode = vis | winR, optionally with suffix '_noself' (also drop the query's OWN-frame patch keys, so global
    attention only reads OTHER frames; own-frame info then comes only from frame attention). Special tokens
    (cam + registers of all frames, incl. own frame) are always kept as keys; special-token queries unmasked."""
    if mode == "base":
        return None
    noself = mode.endswith("_noself")
    core = mode[:-len("_noself")] if noself else mode
    r = None if core == "vis" else int(core[3:])
    S, Pf, pm, N = T["S"], T["Pf"], T["pm"], T["N"]
    mask = torch.ones(N, N, dtype=torch.bool, device=T["vis"].device)   # specials kept as keys by default
    for j in range(S):
        cols = slice(j * Pf + pm, (j + 1) * Pf)
        m = T["vis"][:, j:j + 1].expand(N, Pf - pm).clone()             # (N, Np)
        if r is not None:
            m &= (T["kgy"][cols][None] - T["py"][:, j:j + 1]).abs() <= r
            m &= (T["kgx"][cols][None] - T["px"][:, j:j + 1]).abs() <= r
        m[T["qframe"] == j] = not noself                                 # own frame: kept, or dropped (_noself)
        mask[:, cols] = m
    mask[T["is_special"]] = True                                         # special queries unmasked
    return mask


# ----------------------------------------------------------------------------- attention-mass breakdown
@torch.no_grad()
def mass_stats(q, k, scale, mask, T, qidx, chunk=256):
    """Where does each PATCH query's global-attention mass go? (per head, averaged over queries qidx)

      special : cam+register keys of all frames (the dustbin)
      own     : patches of the query's own frame
      vis     : patches of OTHER frames where the query IS visible
      invis   : patches of OTHER frames where the query is NOT visible  (= the "waste")
      corr1   : 3x3 window around the GT reprojection, in visible other frames (subset of vis)
    Probabilities are those actually used by the run (mask applied if any). Also returns the uniform-
    attention share of invisible keys (chance level for 'invis').
    q,k: (H, N, d) fp32.
    """
    S, Pf, pm, Np, pw, ph = T["S"], T["Pf"], T["pm"], T["Np"], T["pw"], T["ph"]
    dev = q.device
    patch_cols = torch.cat([torch.arange(j * Pf + pm, (j + 1) * Pf, device=dev) for j in range(S)])
    spec_cols = torch.cat([torch.arange(j * Pf, j * Pf + pm, device=dev) for j in range(S)])
    Hh = q.shape[0]
    acc = {k_: torch.zeros(Hh, device=dev, dtype=torch.float64) for k_ in ("special", "own", "vis", "invis", "corr1")}
    n_noother = 0
    uni_invis = 0.0
    for c0 in range(0, qidx.numel(), chunk):
        qi = qidx[c0:c0 + chunk]
        c = qi.numel()
        logit = (q[:, qi] * scale) @ k.transpose(-1, -2)                  # (H, c, N)
        if mask is not None:
            logit.masked_fill_(~mask[qi][None], float("-inf"))
        p = logit.softmax(dim=-1)
        acc["special"] += p[..., spec_cols].sum(-1).sum(-1).double()
        pp = p[..., patch_cols].reshape(Hh, c, S, Np)
        fm = pp.sum(-1)                                                   # (H, c, S) mass per key frame
        own = F.one_hot(T["qframe"][qi], S).bool()                        # (c, S)
        visq = T["vis"][qi]
        acc["own"] += (fm * own[None]).sum((-1, -2)).double()
        acc["vis"] += (fm * (visq & ~own)[None]).sum((-1, -2)).double()
        acc["invis"] += (fm * (~visq & ~own)[None]).sum((-1, -2)).double()
        okf = visq & ~own
        n_noother += (okf.sum(-1) == 0).sum().item()
        uni_invis += ((~visq & ~own).sum(-1).float() * Np / T["N"]).sum().item()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y, x = T["py"][qi] + dy, T["px"][qi] + dx
                ok = okf & (y >= 0) & (y < ph) & (x >= 0) & (x < pw)
                idx = (y.clamp(0, ph - 1) * pw + x.clamp(0, pw - 1))      # (c, S)
                g = pp.gather(-1, idx[None, :, :, None].expand(Hh, c, S, 1))[..., 0]
                acc["corr1"] += (g * ok[None]).sum((-1, -2)).double()
    n = qidx.numel()
    per_head = {k_: (v / n).tolist() for k_, v in acc.items()}
    mean = {k_: float(np.mean(v)) for k_, v in per_head.items()}
    return dict(mean=mean, per_head=per_head, n_queries=n,
                frac_no_other_visible=n_noother / n, uniform_invis_share=uni_invis / n)


# ----------------------------------------------------------------------------- global-attention override
class MaskedGlobal:
    """Swaps DVLT's GLOBAL attention forward on the instance. Same math as Attention.forward's fused
    path (global has no RoPE), plus an optional bool attn_mask and an optional one-shot mass breakdown."""

    def __init__(self, P, T, stat_q):
        self.g = P.m.recurrent_blocks[0].global_attn.attn
        self.orig = self.g.forward
        self.T, self.stat_q = T, stat_q
        self.mask = None
        self.want = False
        self.stats = None

    def set(self, mask):
        self.mask = mask
        self.g.forward = self._fwd

    def restore(self):
        self.g.forward = self.orig

    def _fwd(self, x, pos=None, block_mask=None):
        g = self.g
        B, N, C = x.shape
        qkv = g.qkv(x).reshape(B, N, 3, g.num_heads, g.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = g.q_norm(q), g.k_norm(k)
        q, k = q.to(v.dtype), k.to(v.dtype)
        if self.want:
            with torch.autocast(device_type=x.device.type, enabled=False):
                self.stats = mass_stats(q[0].float(), k[0].float(), g.scale, self.mask, self.T, self.stat_q)
            self.want = False
        am = None if self.mask is None else self.mask[None, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return g.proj(out.transpose(1, 2).reshape(B, N, C))


# ----------------------------------------------------------------------------- experiments
def setup(a):
    P = Probe(a.ckpt, max_frames=a.max_frames)
    images = P.load(a.input)
    B, S, _, H, W = images.shape
    P.gt_depth, P.gt_c2w, P.gt_K = load_gt(a.gt, P.img_size, device=P.acc.device)
    P.gt_depth, P.gt_c2w = P.gt_depth[:S], P.gt_c2w[:S]
    P.occ_thresh = a.occ
    P.fixed_gates = FixedGates(P.m, mode=a.gates, t0=a.gate_t, sout_frame=a.sout_frame,
                               sout_global=a.sout_global).enable()          # 'off' = untouched DVLT schedule
    T = build_tables(P, S, H, W)
    g = torch.Generator(device="cpu").manual_seed(0)
    pidx = torch.nonzero(~T["is_special"]).flatten().cpu()
    stat_q = pidx[torch.randperm(pidx.numel(), generator=g)[:min(a.stat_queries, pidx.numel())]].sort().values
    return P, images, T, stat_q.to(P.acc.device)


def exp_covis(P, T, a):
    cv = T["covis"]
    S = T["S"]
    off = cv[~np.eye(S, dtype=bool)]
    nvis = T["vis"][~T["is_special"]].sum(-1) - 1                         # other frames visible per patch
    lines = [f"frames={S}  occ_thresh={a.occ}",
             f"frame-pair covisibility (off-diag): mean={off.mean():.3f} median={np.median(off):.3f} "
             f"frac pairs <5%={np.mean(off < 0.05):.3f}  >50%={np.mean(off > 0.5):.3f}",
             f"#other frames a patch is visible in: mean={nvis.float().mean():.2f} of {S-1}  "
             f"zero={(nvis == 0).float().mean():.3f}"]
    print("\n".join(lines))
    np.save(f"{a.out}/covis.npy", cv)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        im = ax[0].imshow(cv, vmin=0, vmax=1, cmap="viridis"); plt.colorbar(im, ax=ax[0])
        ax[0].set_title("GT covisibility: frac of frame-i patches visible in j"); ax[0].set_xlabel("j"); ax[0].set_ylabel("i")
        ax[1].hist(nvis.cpu().numpy(), bins=np.arange(-0.5, S + 0.5, 1)); ax[1].set_title("# other frames each patch is visible in")
        fig.tight_layout(); fig.savefig(f"{a.out}/covis.png", dpi=130); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)
    return lines


@torch.no_grad()
def exp_offline(P, images, T, stat_q, a):
    """A1 = the first global-attention call (step 0) of the unmasked loop. Mass breakdown per head."""
    x0, rope, dims = P.init_state(images)
    B, S, H, W = dims
    mg = MaskedGlobal(P, T, stat_q)
    out = {}
    try:
        for mode in a.modes:
            mg.set(build_mask(T, mode))
            mg.want = True
            _ = P.step(x0.clone(), 0, a.offline_K, rope, B, S)
            out[mode] = mg.stats
            m = mg.stats["mean"]
            print(f"[A1 {mode:>5}] special={m['special']:.3f} own={m['own']:.3f} vis={m['vis']:.3f} "
                  f"invis={m['invis']:.3f} corr3x3={m['corr1']:.3f}   (uniform invis share="
                  f"{mg.stats['uniform_invis_share']:.3f})")
    finally:
        mg.restore()
    ph = out["base"]["per_head"]
    print("base A1 per head:   head  special   own    vis   invis  corr3x3")
    for h in range(len(ph["own"])):
        print(f"                   {h:>4}  {ph['special'][h]:.3f}  {ph['own'][h]:.3f}  {ph['vis'][h]:.3f}  "
              f"{ph['invis'][h]:.3f}  {ph['corr1'][h]:.3f}")
    json.dump(dict(K=a.offline_K, stats=out), open(f"{a.out}/offline_A1.json", "w"), indent=1)
    return out


@torch.no_grad()
def exp_run(P, images, T, stat_q, a):
    x0, rope, dims = P.init_state(images)
    B, S, H, W = dims
    sub = _dyn_sub(P, x0, S)
    mg = MaskedGlobal(P, T, stat_q)
    summary = {}
    try:
        for K in a.Ks:
            ts = torch.linspace(0.0, 1.0, K).tolist()
            runs = {}
            for mode in a.modes:
                mask = build_mask(T, mode)                       # built per mode, freed after (memory)
                if mask is not None:
                    print(f"mask {mode:>11}: keeps {mask.float().mean().item():.4f} of the {T['N']}^2 global pairs")
                mg.set(mask)
                x = x0.detach().clone()
                rows = [_dyn_row(P, x, rope, dims, sub, 0, 0.0, 0.0, decode=False)]
                print(f"\n[K={K} {mode}]")
                for i in range(K):
                    mg.want = (i % a.stat_every == 0) or (i == K - 1)
                    x = P.step(x, i, K, rope, B, S)          # one loop step (masked global attention if mode != base)
                    if a.norm:                               # then Solv3R-style RMS normalization of the whole state
                        x = rms_normalize(x, a.rms_target)
                    r = _dyn_row(P, x, rope, dims, sub, i + 1, ts[i], ts[i + 1] if i + 1 < K else 1.0, decode=False)
                    if mg.stats is not None:
                        r["mass"] = mg.stats["mean"]
                        mg.stats = None
                    if (i + 1) % a.every == 0 or i == K - 1:
                        r.update(gt_metrics(P, P.decode(x, rope, dims)))
                        ms = r.get("mass", {})
                        print(f"  step {i+1:>3} |x|={r['tok_norm']:7.1f} k_eff={r['k_eff']:5.2f} "
                              f"absrel={r['absrel']:.4f} ate={r.get('ate_rel', float('nan')):.4f}"
                              + (f"  invis={ms['invis']:.3f} spec={ms['special']:.3f} corr={ms['corr1']:.3f}" if ms else ""))
                    rows.append(r)
                runs[mode] = rows
                mg.mask = None; del mask; torch.cuda.empty_cache()
                fin = rows[-1]
                summary.setdefault(K, {})[mode] = dict(absrel=fin["absrel"], d1=fin["d1"],
                                                       ate=fin.get("ate_rel"), k_eff=fin["k_eff"],
                                                       tok_norm=fin["tok_norm"])
            json.dump(dict(K=K, runs=runs), open(f"{a.out}/run_K{K}.json", "w"), indent=1)
            _dyn_plot(a.out, f"run_K{K}", runs)
            plot_mass(a.out, K, runs)
    finally:
        mg.restore()
    lines = ["\nFINAL-STEP SUMMARY (AbsRel / ATE / k_eff / |x|)"]
    for K, d in summary.items():
        for mode, v in d.items():
            lines.append(f"  K={K:>4} {mode:>11}: absrel={v['absrel']:.4f} d1={v['d1']:.3f} ate={v['ate'] or float('nan'):.4f} "
                         f"k_eff={v['k_eff']:.2f} |x|={v['tok_norm']:.1f}")
    print("\n".join(lines))
    json.dump(summary, open(f"{a.out}/summary_runs.json", "w"), indent=1)
    return lines


def plot_mass(out, K, runs):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        keys = ["invis", "vis", "own", "special", "corr1"]
        fig, ax = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.6))
        for c, kk in enumerate(keys):
            for rn, rows in runs.items():
                xs = [r["step"] for r in rows if "mass" in r]
                ys = [r["mass"][kk] for r in rows if "mass" in r]
                if xs: ax[c].plot(xs, ys, marker="o", ms=3, label=rn)
            ax[c].set_title(f"global-attn mass: {kk}"); ax[c].set_xlabel("step"); ax[c].grid(alpha=.3)
        ax[0].legend(fontsize=7)
        fig.suptitle(f"K={K}  where patch queries put their global attention"); fig.tight_layout()
        fig.savefig(f"{out}/mass_K{K}.png", dpi=130); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["covis", "offline", "run", "all"], default="all")
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ckpt", default="nvidia/dvlt")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--Ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--modes", nargs="+", default=["base", "vis", "win1", "win2", "win4"])
    ap.add_argument("--occ", type=float, default=0.1, help="occlusion test: |z_reproj - z_gt|/z < occ")
    ap.add_argument("--offline-K", type=int, default=16, help="K whose step-0 interval conditions A1")
    ap.add_argument("--stat-queries", type=int, default=2048, help="patch queries sampled for the mass breakdown")
    ap.add_argument("--stat-every", type=int, default=4)
    ap.add_argument("--every", type=int, default=4, help="decode (AbsRel/ATE) every N steps")
    ap.add_argument("--gates", choices=["off", "A", "B"], default="off",
                    help="off: DVLT time-conditioned gates. A: all gates = DVLT f(t0,t0). B: A + s_out rescaled to "
                         "mean sout_frame/sout_global (Solv3R level). With A/B, K is just the number of steps.")
    ap.add_argument("--gate-t", type=float, default=0.5)
    ap.add_argument("--sout-frame", type=float, default=0.739)
    ap.add_argument("--sout-global", type=float, default=0.780)
    ap.add_argument("--norm", action="store_true",
                    help="after every step, rescale the whole state to a fixed RMS (Solv3R SolverBlock normalization)")
    ap.add_argument("--rms-target", type=float, default=0.852, help="target state RMS for --norm (Solv3R: 0.852)")
    ap.add_argument("--out", default="out/attentionmask")
    a = ap.parse_args()
    assert a.modes[0] == "base", "first mode must be 'base' (reference)"
    os.makedirs(a.out, exist_ok=True)

    P, images, T, stat_q = setup(a)
    print(f"frames={T['S']} tokens/frame={T['Pf']} global N={T['N']} grid={T['ph']}x{T['pw']}  modes={a.modes}")
    lines = [" ".join(sys.argv)]
    if a.gates != "off":                                   # build the constants now (one warm-up step) and log them
        x0, rope, dims = P.init_state(images)
        with torch.no_grad():
            _ = P.step(x0, 0, 2, rope, dims[0], dims[1])
    print(P.fixed_gates.summary()); lines.append(P.fixed_gates.summary())
    nl = f"norm: {'Solv3R RMS rescale to ' + str(a.rms_target) + ' after every step' if a.norm else 'off'}"
    print(nl); lines.append(nl)
    if a.exp in ("covis", "all"):
        lines += exp_covis(P, T, a)
    if a.exp in ("offline", "all"):
        exp_offline(P, images, T, stat_q, a)
    if a.exp in ("run", "all"):
        lines += exp_run(P, images, T, stat_q, a)
    with open(f"{a.out}/summary_{a.exp}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\noutputs in {a.out}/")


if __name__ == "__main__":
    main()
