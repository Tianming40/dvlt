#!/usr/bin/env python3
"""token_rw.py -- READ / WRITE experiments on stored DVLT tokens (out/stage3_latent/tokendump), decode immediately, no further loop.

Probes are ridge regressions  token(768) -> DVLT's OWN decoded per-patch quantity (= a linear emulator of the decoder):
  logdepth (1), dir (3, unit ray direction), origin (3, ray origin = camera centre), point (3), rgb (3, from the image).
Per step s:
  E1 READ      held-out-frame R^2 (train even frames / test odd), and stability of probe directions across steps.
  E2 WRITE     probe direction: shift log-depth by alpha on (a) one patch, (b) a 6x6 region, (c) the whole frame; decode.
               gain = decoded change / alpha on edited patches; leakage = ray-direction / origin change; locality =
               |change| on NON-edited patches of the frame relative to alpha.
  E3 JACOBIAN  decoder's true sensitivity: g_i = d(decoded logdepth of patch i)/d(token i) by autograd (first-order
               interventional emulator). cos(g_i, probe w); share of |d y_i/d tokens| on token i itself (locality);
               write along g_i (alpha * g/|g|^2) vs along w -> gains.
  E4 CAMERA    shift the whole frame's camera centre by dc (origin probe, pseudo-inverse) -> decoded origin shift, depth.
  E5 SWAP      replace frame A's patch components inside the origin (and origin+direction) probe subspace by frame B's
               (same pixel) -> does A's decoded camera move to B's? (centre_toB, rays vs B)
  E6 ABLATE    remove the within-frame variation of the depth subspace (set to frame mean) / set origin subspace to the
               all-frame mean -> decoded depth std / camera centres.
  E7 CHANNEL   massive channel (largest energy, e.g. 215): zero it / set to its mean -> decoded changes.

  PYTHONPATH=$(pwd) python shovel/token_rw.py --dump out/stage3_latent/tokendump --steps K16:4 K16:8 K16:16 K64:64 --frames 3 12 20 \
      --pairs 4:5 12:13 3:20 --out out/stage3_latent/tokenrw
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlt_probe import Probe  # noqa: E402

PM = 5
QTY = {"logdepth": 1, "dir": 3, "origin": 3, "point": 3, "rgb": 3}


# ----------------------------------------------------------------------------- decoding (selected frames only)
class Dec:
    def __init__(self, P, meta):
        self.m = P.m
        self.gh, self.gw = [int(v) for v in meta["grid"]]
        self.ps = P.m.patch_size
        self.H, self.W = self.gh * self.ps, self.gw * self.ps
        self.rope = torch.from_numpy(meta["rope"]).to(P.acc.device)
        self.dev = P.acc.device

    def per_patch(self, t):
        n = t.shape[0]
        return t.reshape(n, self.gh, self.ps, self.gw, self.ps, t.shape[-1]).mean((2, 4)).reshape(n, self.gh * self.gw, t.shape[-1])

    def __call__(self, x, frames, grad=False):
        """x: (n,P,C) tokens of the given frames -> dict of per-patch tensors (n,Np,...)."""
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            pr = self.m._decode(x, self.H, self.W, 1, x.shape[0], self.rope[frames])
            rays = pr["rays"][0].float(); dep = pr["depth"][0].float()
            d = rays[..., :3] / rays[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-9)
            dd = self.per_patch(d); dd = dd / dd.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            return dict(logdepth=self.per_patch(dep.clamp_min(1e-6).log())[..., 0], dir=dd,
                        origin=self.per_patch(rays[..., 3:6]), pose_enc=pr["pose_enc"][0].float())


# ----------------------------------------------------------------------------- probes
def fit_ridge(X, Y, lam=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    Wt = np.linalg.solve(Xs.T @ Xs + lam * len(Xs) * np.eye(X.shape[1]), Xs.T @ (Y - ym))
    Wraw = (Wt / sd[:, None]).T                                   # (k,768) raw-space weights: y = Wraw z + b
    b = ym - Wraw @ mu
    return Wraw, b


def r2(Y, P_):
    return float(1 - ((Y - P_) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def pinv_write(W, dy):
    """minimal-norm dz with W dz = dy.  W (k,768), dy (...,k) -> (...,768)"""
    G = np.linalg.inv(W @ W.T)
    return dy @ (G @ W)


def cov_write_mat(X, Y):
    """observational write: E[dz | dy] = Cov(z,y) Var(y)^-1 dy.  returns M (k,768): dz = dy @ M"""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Cz = Xc.T @ Yc / len(X); Cy = Yc.T @ Yc / len(X)
    return np.linalg.solve(Cy, Cz.T)


def projector(W):
    Q, _ = np.linalg.qr(W.T)                                      # (768,k)
    return Q @ Q.T


def ang_deg(u, v):
    return torch.rad2deg(torch.arccos((u * v).sum(-1).clamp(-1, 1)))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="out/stage3_latent/tokendump")
    ap.add_argument("--steps", nargs="+", default=["K16:4", "K16:8", "K16:16", "K64:64"])
    ap.add_argument("--frames", type=int, nargs="+", default=[3, 12, 20])
    ap.add_argument("--pairs", nargs="+", default=["4:5", "12:13", "3:20"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--n-jac", type=int, default=24, help="patches per frame for the autograd Jacobian")
    ap.add_argument("--out", default="out/stage3_latent/tokenrw")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pairs = [tuple(int(v) for v in p.split(":")) for p in a.pairs]
    P = Probe("nvidia/dvlt", max_frames=27)
    dev = P.acc.device
    log, res = [], {}
    def say(s=""):
        print(s, flush=True); log.append(s)
    probe_dirs = {}

    for spec in a.steps:
        Kn, s = spec.split(":"); s = int(s)
        d = f"{a.dump}/{Kn}"
        meta = np.load(f"{d}/meta.npz")
        dec = Dec(P, meta)
        X = np.load(f"{d}/step_{s:03d}.npy").astype(np.float32)               # (S,P,C)
        tg = np.load(f"{d}/targets_{s:03d}.npz")
        S, Np = X.shape[0], X.shape[1] - PM
        gh, gw = dec.gh, dec.gw
        Xp = X[:, PM:]
        T = {k: (tg[k][..., None] if k == "logdepth" else tg[k]) for k in ["logdepth", "dir", "origin", "point"]}
        T["rgb"] = meta["rgb"]
        R = res.setdefault(spec, {})
        say(f"\n################ {spec} ################   mean |patch token| = {np.linalg.norm(Xp, axis=-1).mean():.2f}")

        # ---------------- E1 READ
        tr = np.arange(S) % 2 == 0; te = ~tr
        W, Bb = {}, {}
        say("E1 READ  held-out R^2: " + "  ".join(
            f"{k}={r2(T[k][te].reshape(-1, QTY[k]), (Xp[te].reshape(-1, 768) @ fit_ridge(Xp[tr].reshape(-1, 768), T[k][tr].reshape(-1, QTY[k]))[0].T) + fit_ridge(Xp[tr].reshape(-1, 768), T[k][tr].reshape(-1, QTY[k]))[1]):.3f}"
            for k in QTY))
        for k in QTY:
            W[k], Bb[k] = fit_ridge(Xp.reshape(-1, 768), T[k].reshape(-1, QTY[k]))      # all frames, for writing
        probe_dirs[spec] = {k: W[k] for k in QTY}
        CW = {k: cov_write_mat(Xp.reshape(-1, 768), T[k].reshape(-1, QTY[k])) for k in ["logdepth", "origin"]}
        Xt = torch.from_numpy(X).to(dev)

        def base(fr):
            return dec(Xt[fr], fr)
        frames = a.frames

        # ---------------- E2 WRITE along probe direction (log-depth)
        cy, cx = gh // 2, gw // 2
        masks = {"patch": [cy * gw + cx],
                 "region6x6": [(cy - 3 + i) * gw + (cx - 3 + j) for i in range(6) for j in range(6)],
                 "frame": list(range(Np))}
        e2 = []
        b0 = base(frames)
        for alpha, wk in [(al, wk) for al in a.alphas for wk in ["pinv", "cov"]]:
            dzn = pinv_write(W["logdepth"], np.array([alpha])) if wk == "pinv" else np.array([alpha]) @ CW["logdepth"]
            dz = torch.from_numpy(dzn).float().to(dev)   # (768,)
            for mname, idx in masks.items():
                Xm = Xt[frames].clone(); Xm[:, PM + np.array(idx)] += dz
                out = dec(Xm, frames)
                dl = out["logdepth"] - b0["logdepth"]                                  # (n,Np)
                edit = torch.zeros(Np, dtype=torch.bool, device=dev); edit[idx] = True
                gain = float(dl[:, edit].mean() / alpha)
                leak_dir = float(ang_deg(out["dir"][:, edit], b0["dir"][:, edit]).mean())
                leak_org = float((out["origin"][:, edit] - b0["origin"][:, edit]).norm(dim=-1).mean())
                other = float(dl[:, ~edit].abs().mean() / alpha) if (~edit).any() else float("nan")
                e2.append(dict(alpha=alpha, write=wk, dz_norm=float(np.linalg.norm(dzn)), mask=mname, gain=gain, leak_dir_deg=leak_dir, leak_origin=leak_org, other_rel=other))
                say(f"E2 WRITE[{wk:>4}] alpha={alpha:g} |dz|={np.linalg.norm(dzn):.3f} {mname:>9}: gain {gain:.3f}  leak dir {leak_dir:.3f}°  leak origin {leak_org:.4f}"
                    f"  non-edited |Δ|/alpha {other:.3f}")
        R["E2"] = e2

        # ---------------- E3 JACOBIAN of the decoder (autograd) vs probe direction
        rng = np.random.default_rng(0)
        cosines, own_share, gains_jac, gains_prb = [], [], [], []
        wd = torch.from_numpy(W["logdepth"][0]).float().to(dev)
        wc = torch.from_numpy(CW["logdepth"][0]).float().to(dev)
        cos_cov, gains_cov = [], []
        for f in frames:
            pids = rng.choice(Np, a.n_jac, replace=False)
            for pi in pids:
                xf = Xt[f:f + 1].clone().requires_grad_(True)
                out = dec(xf, [f], grad=True)
                out["logdepth"][0, pi].backward()
                g = xf.grad[0]                                                   # (P,C)
                gi = g[PM + pi]
                own_share.append(float(gi.norm() ** 2 / (g.norm() ** 2 + 1e-12)))
                cosines.append(float(torch.nn.functional.cosine_similarity(gi, wd, dim=0)))
                cos_cov.append(float(torch.nn.functional.cosine_similarity(gi, wc, dim=0)))
                for vec, store in [(gi, gains_jac), (wd, gains_prb), (wc, gains_cov)]:
                    xm = Xt[f:f + 1].clone(); xm[0, PM + pi] += 0.1 * vec / ((wd if vec is wc else vec) @ vec)   # probe-predicted Δ = 0.1
                    o2 = dec(xm, [f])
                    store.append(float((o2["logdepth"][0, pi] - b0["logdepth"][frames.index(f), pi]) / 0.1))
        R["E3"] = dict(cos_mean=float(np.mean(cosines)), cos_abs_mean=float(np.mean(np.abs(cosines))),
                       own_share=float(np.mean(own_share)), gain_jac=float(np.mean(gains_jac)), gain_probe=float(np.mean(gains_prb)),
                       cos_cov=float(np.mean(cos_cov)), gain_cov=float(np.mean(gains_cov)))
        say(f"E3 JACOBIAN ({len(cosines)} patches): cos(dDecoded/dToken, probe w) mean {np.mean(cosines):.3f} (|cos| {np.mean(np.abs(cosines)):.3f})"
            f" | share of sensitivity on the patch's own token {np.mean(own_share):.3f}"
            f" | single-patch write gain: along Jacobian {np.mean(gains_jac):.3f}, along probe(pinv) {np.mean(gains_prb):.3f},"
            f" along cov {np.mean(gains_cov):.3f} (cos(J,cov) {np.mean(cos_cov):.3f})")

        # ---------------- E4 CAMERA write (whole frame, origin probe)
        c_all = tg["origin"].mean(1)                                                     # (S,3) per-frame centres
        scale = float(np.linalg.norm(c_all - c_all.mean(0), axis=1).mean())
        e4 = []
        for wk, axis in [(wk, ax) for wk in ["pinv", "cov"] for ax in range(3)]:
            dc = np.zeros(3); dc[axis] = 0.2 * scale
            dz = torch.from_numpy(pinv_write(W["origin"], dc) if wk == "pinv" else dc @ CW["origin"]).float().to(dev)
            Xm = Xt[frames].clone(); Xm[:, PM:] += dz
            out = dec(Xm, frames)
            got = (out["origin"].mean(1) - b0["origin"].mean(1)).cpu().numpy()          # (n,3)
            e4.append(dict(write=wk, axis=axis, wanted=dc.tolist(), got=got.mean(0).tolist(),
                           gain_on_axis=float(got[:, axis].mean() / dc[axis]),
                           off_axis=float(np.abs(np.delete(got, axis, 1)).mean() / dc[axis]),
                           depth_change=float((out["logdepth"] - b0["logdepth"]).abs().mean()),
                           dir_change_deg=float(ang_deg(out["dir"], b0["dir"]).mean())))
            say(f"E4 CAMERA[{wk:>4}] shift axis {axis} by {dc[axis]:.3f}: gain on axis {e4[-1]['gain_on_axis']:.3f}, off-axis {e4[-1]['off_axis']:.3f},"
                f" |Δlogdepth| {e4[-1]['depth_change']:.4f}, Δdir {e4[-1]['dir_change_deg']:.3f}°")
        R["E4"] = e4

        # ---------------- E5 SWAP inside probe subspaces
        PO = torch.from_numpy(projector(W["origin"])).float().to(dev)
        POD = torch.from_numpy(projector(np.concatenate([W["origin"], W["dir"]]))).float().to(dev)
        e5 = []
        for (A, Bf) in pairs:
            bb = dec(Xt[[A, Bf]], [A, Bf])
            cA, cB = bb["origin"][0].mean(0), bb["origin"][1].mean(0)
            dAB = float(ang_deg(bb["dir"][0], bb["dir"][1]).mean())
            for nm, Pr in [("origin", PO), ("origin+dir", POD)]:
                xa = Xt[A:A + 1].clone(); xa[0, PM:] += (Xt[Bf, PM:] - Xt[A, PM:]) @ Pr
                o = dec(xa, [A])
                c2 = o["origin"][0].mean(0)
                r_ = dict(A=A, B=Bf, subspace=nm, centre_toB=float((c2 - cB).norm() / (cA - cB).norm()),
                          dir_toB_deg=float(ang_deg(o["dir"][0], bb["dir"][1]).mean()), dir_AB_deg=dAB,
                          depth_change=float((o["logdepth"][0] - bb["logdepth"][0]).abs().mean()))
                e5.append(r_)
                say(f"E5 SWAP {A}->{Bf} [{nm:>10}]: centre_toB {r_['centre_toB']:.3f} (0 = moved to B), dir_toB {r_['dir_toB_deg']:.2f}°"
                    f" (A-B {dAB:.2f}°), |Δlogdepth| {r_['depth_change']:.4f}")
        R["E5"] = e5

        # ---------------- E6 ABLATE subspaces
        PD = torch.from_numpy(projector(W["logdepth"])).float().to(dev)
        xa = Xt[frames].clone()
        fm = xa[:, PM:].mean(1, keepdim=True)
        xa[:, PM:] = xa[:, PM:] - (xa[:, PM:] - fm) @ PD
        o = dec(xa, frames)
        std0, std1 = float(b0["logdepth"].std(1).mean()), float(o["logdepth"].std(1).mean())
        allmean = Xt[:, PM:].reshape(-1, 768).mean(0)
        xb = Xt[frames].clone(); xb[:, PM:] = xb[:, PM:] - (xb[:, PM:] - allmean) @ PO
        o2 = dec(xb, frames)
        cen0, cen1 = b0["origin"].mean(1), o2["origin"].mean(1)
        spread0 = float((cen0 - cen0.mean(0)).norm(dim=-1).mean()); spread1 = float((cen1 - cen1.mean(0)).norm(dim=-1).mean())
        R["E6"] = dict(depth_std_before=std0, depth_std_after=std1, centre_spread_before=spread0, centre_spread_after=spread1,
                       origin_ablate_depth_change=float((o2["logdepth"] - b0["logdepth"]).abs().mean()))
        say(f"E6 ABLATE depth subspace -> within-frame log-depth std {std0:.3f} -> {std1:.3f};"
            f"  origin subspace -> centre spread across the {len(frames)} frames {spread0:.3f} -> {spread1:.3f}"
            f" (|Δlogdepth| {R['E6']['origin_ablate_depth_change']:.4f})")

        # ---------------- E7 massive channel
        e = (Xp.reshape(-1, 768) ** 2).mean(0); ch = int(np.argmax(e))
        e7 = {}
        for nm, val in [("zero", 0.0), ("mean", float(Xp[..., ch].mean()))]:
            xc = Xt[frames].clone(); xc[:, :, ch] = val
            o = dec(xc, frames)
            e7[nm] = dict(depth=float((o["logdepth"] - b0["logdepth"]).abs().mean()),
                          dir_deg=float(ang_deg(o["dir"], b0["dir"]).mean()),
                          origin=float((o["origin"] - b0["origin"]).norm(dim=-1).mean()))
        R["E7"] = dict(channel=ch, energy_share=float(e[ch] / e.sum()), **e7)
        say(f"E7 CHANNEL {ch} (energy share {e[ch] / e.sum():.2f}): zero -> |Δlogdepth| {e7['zero']['depth']:.4f}, Δdir {e7['zero']['dir_deg']:.3f}°,"
            f" |Δorigin| {e7['zero']['origin']:.4f} ;  set to mean -> |Δlogdepth| {e7['mean']['depth']:.4f}, Δdir {e7['mean']['dir_deg']:.3f}°")

    # ---------------- probe-direction stability across steps
    say("\nProbe direction stability (|cos| between raw probe weight vectors of different steps; logdepth / origin row 0):")
    specs = list(probe_dirs)
    for k in ["logdepth", "origin"]:
        M = np.array([[abs(float(np.dot(probe_dirs[p][k][0], probe_dirs[q][k][0]) /
                                 (np.linalg.norm(probe_dirs[p][k][0]) * np.linalg.norm(probe_dirs[q][k][0])))) for q in specs] for p in specs])
        res[f"stability_{k}"] = M.tolist()
        say(f"  {k}: " + " | ".join(f"{p}: " + " ".join(f"{v:.2f}" for v in row) for p, row in zip(specs, M)))

    json.dump(res, open(f"{a.out}/tokenrw.json", "w"), indent=1)
    with open(f"{a.out}/report.txt", "w") as fh:
        fh.write(" ".join(sys.argv) + "\n" + "\n".join(log) + "\n")
    print(f"saved {a.out}/report.txt, tokenrw.json")


if __name__ == "__main__":
    main()
