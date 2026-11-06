#!/usr/bin/env python3
"""Compare the STORED attention of the tracked tokens across K (16/64/128), from the track_token dumps.

The collapse story says "global attention diffuses to uniform past the budget" (= directional collapse).
This makes that visible: per-head attention rows -> scalar summaries (entropy / self-frame mass / top-k
share) plotted vs the schedule position t (so K16/64/128 overlay despite different step counts), plus a
side-by-side heatmap of the global attention onto the token's OWN frame at the endpoint t=1.

  PYTHONPATH=$(pwd) python shovel/attn_compare_K.py \
      --dirs out/track_s10_K16 out/track_s10_K64 out/track_s10_K128 \
      --tokens near_tex,flat,far --out out/attn_compare_s10
"""
import argparse, glob, os, re as _re
def _nsort(ps):
    return sorted(ps, key=lambda p:int(_re.search(r'step_(\d+)',p).group(1)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def entropy(p, axis=None):
    """Shannon entropy normalized to [0,1] by log(n) of the distribution's support."""
    p = np.asarray(p, np.float64)                 # stored rows are fp16: 1e-12 underflows to 0 there
    p = np.clip(p, 1e-12, None)
    p = p / p.sum(axis=axis, keepdims=True)
    n = p.shape[axis] if axis is not None else p.size
    return float(-(p * np.log(p)).sum(axis=axis) / np.log(n)) if axis is None else \
        -(p * np.log(p)).sum(axis=axis) / np.log(n)


def load_K(d):
    """Return (K, meta, list-of-step-dicts) for one track dump."""
    meta = np.load(f"{d}/meta.npz", allow_pickle=True)
    steps = _nsort(glob.glob(f"{d}/step_*.npz"))
    rows = []
    for sp in steps:
        z = np.load(sp)
        rows.append(dict(step=int(z["step"]), t=float(z["t_now"]),
                         frame=z["frame_rows"], glob=z["global_rows"]))   # (8,12,869) , (8,12,27,869)
    return len(steps), meta, rows


def token_scalars(rows, ti, ps, own_frame=0):
    """Per-step scalar summaries for tracked-token index ti. Returns dict of t-indexed arrays."""
    t = np.array([r["t"] for r in rows])
    g_ent, g_self, g_top8, f_ent = [], [], [], []
    for r in rows:
        g = np.asarray(r["glob"][ti], np.float64)   # (12,27,869) per-head global attention (fp16 -> f64)
        gf = g.reshape(12, -1)                    # (12, 27*869) full global distribution per head
        gf = gf / gf.sum(1, keepdims=True)
        g_ent.append(entropy(gf, axis=1).mean())                      # mean per-head global entropy
        self_mass = g[:, own_frame, :].sum(1) / g.reshape(12, -1).sum(1)   # mass on own frame per head
        g_self.append(self_mass.mean())
        srt = np.sort(gf, 1)[:, ::-1]
        g_top8.append(srt[:, :8].sum(1).mean())                       # top-8 share per head, mean
        f = np.asarray(r["frame"][ti], np.float64)   # (12,869) frame attention (fp16 -> f64)
        f = f / f.sum(1, keepdims=True)
        f_ent.append(entropy(f, axis=1).mean())
    return dict(t=t, g_ent=np.array(g_ent), g_self=np.array(g_self),
                g_top8=np.array(g_top8), f_ent=np.array(f_ent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="track_<tag>_K<K> dumps (any order)")
    ap.add_argument("--tokens", default="near_tex,flat,far")
    ap.add_argument("--out", default="out/attn_compare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data = {}
    for d in a.dirs:
        K, meta, rows = load_K(d)
        data[K] = dict(meta=meta, rows=rows)
    Ks = sorted(data)
    meta0 = data[Ks[0]]["meta"]
    names = list(meta0["tok_name"]); ps = int(meta0["patch_start_idx"])
    gh, gw = [int(x) for x in meta0["grid"]]
    want = a.tokens.split(",")
    tok_idx = {nm: names.index(nm) for nm in want if nm in names}
    colors = {16: "#2ca02c", 64: "#ff7f0e", 128: "#d62728"}
    print("Ks:", Ks, " tokens:", tok_idx, " grid:", (gh, gw))

    # ---- Figure 1: scalar trends vs t (rows=tokens, cols=metrics) ----
    metrics = [("g_ent", "GLOBAL attn entropy (1=uniform)"), ("g_self", "mass on OWN frame"),
               ("g_top8", "GLOBAL top-8 share (1=peaked)"), ("f_ent", "FRAME attn entropy")]
    fig, ax = plt.subplots(len(tok_idx), len(metrics), figsize=(4.2 * len(metrics), 3.3 * len(tok_idx)),
                           squeeze=False)
    for ri, (nm, ti) in enumerate(tok_idx.items()):
        for K in Ks:
            s = token_scalars(data[K]["rows"], ti, ps)
            for ci, (mk, _) in enumerate(metrics):
                ax[ri][ci].plot(s["t"], s[mk], "-o", ms=2.5, color=colors.get(K, None), label=f"K{K}", alpha=.85)
        for ci, (mk, title) in enumerate(metrics):
            ax[ri][ci].set_title(f"{nm}: {title}", fontsize=9); ax[ri][ci].grid(alpha=.3)
            ax[ri][ci].set_xlabel("t (schedule pos)"); ax[ri][ci].legend(fontsize=7)
    fig.suptitle("Attention vs K (aligned by schedule position t) — tracked frame-0 tokens", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{a.out}/trends.png", dpi=125); plt.close(fig)
    print(f"saved {a.out}/trends.png")

    # ---- Figure 2: endpoint (t=1) global-attn-onto-own-frame heatmap, head-mean, K16|K64|K128 ----
    fig, ax = plt.subplots(len(tok_idx), len(Ks), figsize=(3.6 * len(Ks), 3.0 * len(tok_idx)), squeeze=False)
    for ri, (nm, ti) in enumerate(tok_idx.items()):
        for ci, K in enumerate(Ks):
            last = data[K]["rows"][-1]                         # t=1 endpoint
            g = last["glob"][ti]                               # (12,27,869)
            patch = g[:, 0, ps:ps + gh * gw].mean(0).reshape(gh, gw)   # head-mean, own-frame patches
            ent = entropy(g.reshape(12, -1) / g.reshape(12, -1).sum(1, keepdims=True), axis=1).mean()
            im = ax[ri][ci].imshow(patch, cmap="hot")
            ax[ri][ci].set_title(f"{nm}  K{K}  (t=1)\nglobal-attn entropy={ent:.3f}", fontsize=9)
            ax[ri][ci].set_xticks([]); ax[ri][ci].set_yticks([])
            fig.colorbar(im, ax=ax[ri][ci], fraction=.046)
    fig.suptitle("Endpoint global attention onto OWN frame (head-mean) — sharp (K16) -> diffuse (K128)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{a.out}/endpoint_heatmap.png", dpi=125); plt.close(fig)
    print(f"saved {a.out}/endpoint_heatmap.png")

    # ---- console summary table at t=1 ----
    print("\n=== endpoint (t=1) global-attn entropy / self-frame mass / top8 ===")
    print(f"{'token':<10}" + "".join(f"K{K:<14}" for K in Ks))
    for nm, ti in tok_idx.items():
        cells = []
        for K in Ks:
            s = token_scalars(data[K]["rows"][-1:], ti, ps)
            cells.append(f"ent{s['g_ent'][0]:.3f} self{s['g_self'][0]:.2f} t8{s['g_top8'][0]:.3f}")
        print(f"{nm:<10}" + "  ".join(cells))


if __name__ == "__main__":
    main()
