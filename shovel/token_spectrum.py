#!/usr/bin/env python3
"""token_spectrum.py -- SPATIAL spectrum of the token field: is attention a low-pass filter that kills the high
frequencies (geometry) and grows the DC (a component shared by all patches), and does that track the collapse?

For every state, every frame and every channel, the 24x36 patch values form a small image -> 2D FFT -> power spectrum.
Runs:
  K16, K64           original DVLT schedule, stored states (out/tokendump)
  BR_base, BR_restart fixed gates B + RMS, K=128, without / with state restart z <- RMS((1-a) F(z) + a RMS(x0)), a=0.05
                     (run live; spectrum computed on the fly; full states kept only at --keep steps)
Per state:
  frac[DC, low, mid, high]   share of the total spatial energy (all channels, all frames) in radial frequency bands
                             DC = (0,0); low r <= 0.1; mid 0.1 < r <= 0.25; high r > 0.25 (r in cycles / patch)
  frac_nomass[...]           the same without the 4 highest-energy channels
  radial[nb]                 radially averaged power spectrum (log-spaced bins, DC excluded), all channels
  ch_dc[768], ch_energy[768] per-channel DC share of that channel's energy, and the channel's share of the total energy
  gdc                        share of the energy in the component shared by ALL patches of ALL frames (global mean)
  absrel                     decoded depth AbsRel (per-patch for stored runs, per-pixel for live runs)
Data: out/tokenspectrum/spectrum.npz (+ summary.json/txt, figures, states_<run>_step<s>.npy).

  PYTHONPATH=$(pwd) python shovel/token_spectrum.py --out out/tokenspectrum
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

PM = 5
NB = 12


def spectrum_stats(z, gh, gw, edges, massive=None):
    """z: (S, P, C) torch tensor (cuda). returns dict of numpy stats."""
    S, P, C = z.shape
    f = z[:, PM:].float().reshape(S, gh, gw, C).permute(0, 3, 1, 2)                 # (S,C,gh,gw)
    Pw = torch.fft.fft2(f, norm="ortho").abs() ** 2                                 # energy-preserving
    ky = torch.fft.fftfreq(gh, device=z.device)[:, None]; kx = torch.fft.fftfreq(gw, device=z.device)[None, :]
    r = torch.sqrt(ky ** 2 + kx ** 2)                                               # cycles / patch
    tot_c = Pw.sum((0, 2, 3))                                                       # (C,)
    tot = tot_c.sum()
    dc = Pw[:, :, 0, 0].sum(0)
    bands = [r == 0, (r > 0) & (r <= 0.1), (r > 0.1) & (r <= 0.25), r > 0.25]
    eb = torch.stack([Pw[:, :, b].sum((0, 2)) for b in bands])                      # (4, C)
    frac = (eb.sum(1) / tot).cpu().numpy()
    if massive is None:
        massive = torch.argsort(tot_c, descending=True)[:4]
    keep = torch.ones(C, dtype=torch.bool, device=z.device); keep[massive] = False
    frac_nomass = (eb[:, keep].sum(1) / tot_c[keep].sum()).cpu().numpy()
    rad = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r > lo) & (r <= hi)
        rad.append(float(Pw[:, :, m].sum() / max(int(m.sum()), 1) / (S * C)) if m.any() else np.nan)
    gmean = z[:, PM:].float().reshape(-1, C).mean(0)
    gdc = float((gmean ** 2).sum() * S * gh * gw / tot)
    return dict(frac=frac, frac_nomass=frac_nomass, radial=np.array(rad), ch_dc=(dc / tot_c.clamp_min(1e-12)).cpu().numpy(),
                ch_energy=(tot_c / tot).cpu().numpy(), gdc=gdc, massive=massive.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="out/tokendump")
    ap.add_argument("--input", default="out/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--gt", default="out/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--K-br", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--keep", type=int, nargs="+", default=[0, 8, 16, 32, 64, 128])
    ap.add_argument("--out", default="out/tokenspectrum")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    meta = np.load(f"{a.dump}/K16/meta.npz")
    gh, gw = [int(v) for v in meta["grid"]]
    rmax = float(np.sqrt(0.5 ** 2 + 0.5 ** 2))
    edges = np.concatenate([[0.0], np.geomspace(1.0 / max(gh, gw), rmax + 1e-6, NB)])
    gt_ld = meta["gt_logdepth"]; m = ~np.isnan(gt_ld)

    def absrel_patch(ld):
        p, g = np.exp(ld[m]), np.exp(gt_ld[m]); s = np.median(g / p)
        return float(np.mean(np.abs(p * s - g) / g))

    res, massive_ref = {}, None
    # ---- stored runs
    for K in (16, 64):
        rows = []
        for s in range(K + 1):
            z = torch.from_numpy(np.load(f"{a.dump}/K{K}/step_{s:03d}.npy").astype(np.float32)).to(dev)
            st = spectrum_stats(z, gh, gw, edges)
            st["absrel"] = absrel_patch(np.load(f"{a.dump}/K{K}/targets_{s:03d}.npz")["logdepth"])
            rows.append(st)
        res[f"K{K}"] = rows
        print(f"K{K} done: DC {rows[0]['frac'][0]:.3f} -> {rows[K // 4]['frac'][0]:.3f} -> {rows[-1]['frac'][0]:.3f};"
              f" high {rows[0]['frac'][3]:.3f} -> {rows[-1]['frac'][3]:.3f}", flush=True)

    # ---- live runs (fixed gates B + RMS, with / without restart)
    from dvlt_probe import Probe, load_gt, depth_vs_gt
    from gate_fix import FixedGates, rms_normalize
    torch.backends.cuda.matmul.allow_tf32 = False
    P = Probe("nvidia/dvlt", max_frames=27)
    images = P.load(a.input)
    with torch.no_grad():
        x0, rope, dims = P.init_state(images)
    x0 = x0.float(); B, S, H, W = dims
    gt_d, _, _ = load_gt(a.gt, P.img_size, device=x0.device); gt_d = gt_d[:S].float()
    fg = FixedGates(P.m, mode="B"); x0_hat = rms_normalize(x0)

    @torch.no_grad()
    def absrel_live(z):
        ds = []
        for i in range(0, S, 4):
            ds.append(P.m._decode(z[i:i + 4], H, W, 1, min(4, S - i), rope[i:i + 4])["depth"][0, ..., 0].float())
        return depth_vs_gt(torch.cat(ds), gt_d)["absrel"]

    for name, alpha in [("BR_base", 0.0), (f"BR_restart{a.alpha:g}", a.alpha)]:
        t0 = time.time(); rows = []
        fg.enable()
        try:
            with torch.no_grad():
                z = x0.clone()
                for s in range(a.K_br + 1):
                    if s > 0:
                        z = rms_normalize(P.m._interval_step(z, 0.5, 0.5, rope, B, S).float())
                        if alpha > 0:
                            z = rms_normalize((1 - alpha) * z + alpha * x0_hat)
                    st = spectrum_stats(z, gh, gw, edges); st["absrel"] = absrel_live(z); rows.append(st)
                    if s in a.keep:
                        np.save(f"{a.out}/states_{name}_step{s:03d}.npy", z.cpu().numpy().astype(np.float16))
        finally:
            fg.restore()
        res[name] = rows
        print(f"{name} done in {time.time() - t0:.0f}s: AbsRel {rows[16]['absrel']:.3f}@16 {rows[-1]['absrel']:.3f}@{a.K_br};"
              f" DC {rows[16]['frac'][0]:.3f} -> {rows[-1]['frac'][0]:.3f}; high {rows[16]['frac'][3]:.3f} -> {rows[-1]['frac'][3]:.3f}", flush=True)

    # ---- save data
    arrs = {}
    for run, rows in res.items():
        for key in ("frac", "frac_nomass", "radial", "ch_dc", "ch_energy", "massive"):
            arrs[f"{run}__{key}"] = np.stack([r[key] for r in rows])
        arrs[f"{run}__gdc"] = np.array([r["gdc"] for r in rows]); arrs[f"{run}__absrel"] = np.array([r["absrel"] for r in rows])
    arrs["radial_edges"] = edges; arrs["bands"] = np.array(["DC", "low<=0.1", "mid<=0.25", "high>0.25"])
    np.savez(f"{a.out}/spectrum.npz", **arrs)
    named = [215, 480, 614, 701]
    summ, lines = {}, []
    for run, rows in res.items():
        K = len(rows) - 1
        marks = [s for s in (0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128) if s <= K]
        summ[run] = {s: dict(absrel=rows[s]["absrel"], frac=rows[s]["frac"].tolist(), frac_nomass=rows[s]["frac_nomass"].tolist(),
                             gdc=rows[s]["gdc"], massive=rows[s]["massive"].tolist(),
                             named={c: dict(dc=float(rows[s]["ch_dc"][c]), energy=float(rows[s]["ch_energy"][c])) for c in named})
                     for s in marks}
        lines.append(f"\n[{run}]  step | AbsRel | energy share DC / low / mid / high | same w/o top-4 channels | global-shared share |"
                     f" top-4 channels | ch215/480/614/701: energy share (DC share)")
        for s in marks:
            r = rows[s]
            lines.append(f"  {s:>3} | {r['absrel']:.3f} | " + " / ".join(f"{v:.3f}" for v in r["frac"]) + " | "
                         + " / ".join(f"{v:.3f}" for v in r["frac_nomass"]) + f" | {r['gdc']:.3f} | {r['massive'].tolist()} | "
                         + " ".join(f"{r['ch_energy'][c]:.3f}({r['ch_dc'][c]:.2f})" for c in named))
    json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1)
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(18, 8.5))
    titles = ["DC (shared by the whole frame)", "high frequency (r > 0.25)", "DC without the top-4 channels",
              "high freq. without the top-4 channels", "depth AbsRel", "energy share of ch 215+480+614+701"]
    for j, (run, rows) in enumerate(res.items()):
        K = len(rows) - 1; xs = np.arange(K + 1) / K
        ys = [[r["frac"][0] for r in rows], [r["frac"][3] for r in rows], [r["frac_nomass"][0] for r in rows],
              [r["frac_nomass"][3] for r in rows], [r["absrel"] for r in rows], [sum(r["ch_energy"][c] for c in named) for r in rows]]
        for i, y in enumerate(ys):
            ax.flat[i].plot(xs, y, color=f"C{j}", label=run)
    for i, t in enumerate(titles):
        ax.flat[i].set_title(t, fontsize=10); ax.flat[i].set_xlabel("progress  step / K"); ax.flat[i].grid(alpha=.3)
    ax.flat[4].set_yscale("log"); ax.flat[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.out}/bands_vs_step.png", dpi=120); plt.close(fig)

    centers = np.sqrt(edges[1:] * np.maximum(edges[:-1], edges[1] / 2))
    fig, ax = plt.subplots(1, len(res), figsize=(5 * len(res), 4), sharey=True)
    for axx, (run, rows) in zip(ax, res.items()):
        K = len(rows) - 1
        sel = [s for s in (0, 2, 8, 16, 32, 64, 128) if s <= K]
        for s in sel:
            axx.loglog(centers, rows[s]["radial"], "-o", ms=3, color=plt.cm.viridis(s / K), label=f"step {s}")
        axx.set_title(run); axx.set_xlabel("spatial frequency (cycles / patch)"); axx.grid(alpha=.3, which="both"); axx.legend(fontsize=7)
    ax[0].set_ylabel("mean power per frequency (DC excluded)")
    fig.tight_layout(); fig.savefig(f"{a.out}/radial_spectrum.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(18, 5))
    for axx, run in zip(ax, ["K64", f"BR_restart{a.alpha:g}"]):
        M = np.stack([r["ch_energy"] * r["ch_dc"] for r in res[run]])               # DC energy per channel (share of total)
        im = axx.imshow(np.log10(M + 1e-6), aspect="auto", cmap="magma", vmin=-5, vmax=0)
        axx.set_title(f"{run}: log10 DC energy share per channel"); axx.set_xlabel("channel"); axx.set_ylabel("step")
        fig.colorbar(im, ax=axx, fraction=.03)
    fig.tight_layout(); fig.savefig(f"{a.out}/channel_dc.png", dpi=110); plt.close(fig)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
