#!/usr/bin/env python3
"""What do the gates DO, directionally? Across K in {8,16,32,64,128}, from track_token gates.
The gate s = 1 + delta (delta = MLP(gamma(t)), zero-init). All the gate's action is in delta.
Track per step: ||delta|| (how far from identity), cos(s_k, s_{k+1}) (step-to-step rotation),
cos(delta_k, delta_0) (cumulative rotation). Reveals if the channel-selection pattern rotates.

  PYTHONPATH=$(pwd) python shovel/gate_dir_K.py \
    --dirs out/track_s10_K8 out/track_s10_K16 out/track_s10_K32 out/track_s10_K64 out/track_s10_K128 \
    --out out/track_s10_gatedir
"""
import argparse, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _nsort(paths):
    import re
    return sorted(paths, key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))


def load_gates(d):
    steps = _nsort(glob.glob(f"{d}/step_*.npz"))   # NUMERIC sort (step_100 must follow step_99, not step_10)
    gf, gg, ts = [], [], []
    for sp in steps:
        z = np.load(sp)
        gf.append(np.asarray(z["gates_frame"], np.float64))   # (3,768)
        gg.append(np.asarray(z["gates_global"], np.float64))
        ts.append(float(z["t_now"]))
    return np.array(gf), np.array(gg), np.array(ts)           # (T,3,768),(T,3,768),(T,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", default="out/track_gatedir")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data = {int(os.path.basename(d).split("_K")[-1]): load_gates(d) for d in a.dirs}
    Ks = sorted(data)
    cmap = plt.cm.viridis; kcol = {K: cmap(i/max(1, len(Ks)-1)) for i, K in enumerate(Ks)}
    GN = ["s_attn", "s_mlp", "s_out"]

    # 3 metrics (rows) x 3 gates (cols); frame solid, global dashed
    fig, ax = plt.subplots(3, 3, figsize=(15, 11))
    for gi in range(3):
        for sub, arr_idx, ls in [("frame", 0, "-"), ("global", 1, "--")]:
            for K in Ks:
                G = data[K][arr_idx][:, gi, :]               # (T,768) this gate over steps
                st = np.arange(len(G))
                dn = np.linalg.norm(G - 1.0, axis=1)          # ||delta||
                cs = [cos(G[k], G[k+1]) for k in range(len(G)-1)]      # step-to-step
                c0 = [cos(G[k]-1.0, G[0]-1.0) for k in range(len(G))]  # delta vs delta_0
                ax[0][gi].plot(st, dn, ls, color=kcol[K], lw=1.3, label=f"{sub} K{K}" if gi == 0 else None)
                ax[1][gi].plot(st[:-1], cs, ls, color=kcol[K], lw=1.3)
                ax[2][gi].plot(st, c0, ls, color=kcol[K], lw=1.3)
        ax[0][gi].set_title(f"{GN[gi]}: ||s-1||  (distance from identity)")
        ax[1][gi].set_title(f"{GN[gi]}: cos(s_k, s_(k+1))  (step rotation)")
        ax[2][gi].set_title(f"{GN[gi]}: cos(delta_k, delta_0)  (cumulative rotation)")
        for r in range(3):
            ax[r][gi].set_xlabel("step"); ax[r][gi].grid(alpha=.3)
    ax[0][0].legend(fontsize=6, ncol=2)
    fig.suptitle("What the gates DO, directionally — (- frame, -- global), across 5 K", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{a.out}/gate_direction.png", dpi=125); plt.close(fig)
    print(f"saved {a.out}/gate_direction.png")

    # console: endpoint summary
    print("\n=== endpoint (t=1) per gate: ||s-1||  cos(delta_K, delta_0) ===  [frame / global]")
    for K in Ks:
        gf, gg, _ = data[K]
        row = []
        for gi in range(3):
            for arr in (gf, gg):
                G = arr[:, gi, :]; dn = np.linalg.norm(G[-1]-1); c0 = cos(G[-1]-1, G[0]-1)
                row.append(f"{dn:5.2f}/{c0:+.2f}")
        print(f"K{K:<4} " + "  ".join(f"{GN[i]} {row[2*i]}|{row[2*i+1]}" for i in range(3)))


if __name__ == "__main__":
    main()
