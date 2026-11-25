#!/usr/bin/env python3
"""latent_dyn.py -- A-series diagnostics of DVLT's 768-d token space and loop dynamics (frozen model, stored tokens).

All loop/decoder evaluations in fp32 (no autocast): finite differences are linear to ~3e-4 in fp32 (bf16: 0.34).
Stored states: out/stage3_latent/tokendump/K{16,64}/step_XXX.npy (state after s steps), meta.npz (rope, ...).

A1 OBSERVABLE vs UNOBSERVABLE split of the loop update dz_k = z_{k+1} - z_k   (per frame f in --frames)
   decoder map D_f: tokens of frame f (869x768) -> per-patch [logdepth, dir(3), origin(3)] standardized by global stds.
   Row space of J = dD_f/dz sketched with r random cotangents (Y = J^T U, r backward passes); W = left sing. vectors of Y.
   report: frac of |dz_f|^2 in top-64 / top-256 observable directions (random-direction baseline = k/dim),
           decoder gain |J dz|/|dz| divided by the mean gain of random directions (=1 -> as observable as noise),
           |dz_obs|, |dz_unobs| (top-256), sketch spectrum decay.
A2 LOCAL LYAPUNOV along the stored trajectory: m perturbations propagated by J_k (central differences of one loop step,
   fp32), re-orthonormalized by QR each step -> growth factors g_1..g_m per step; relative = g / (|z_{k+1}|/|z_k|).
A3 DOMINANT MODES of J_k at selected (K, step): block power iteration (m vectors) + Rayleigh-Ritz -> eigenvalues and
   Ritz vectors; each mode characterized: special-token share, frame-mean share, global-mean share, cos with z (radial),
   share on the massive channel, decoder-observability ratio (A1 gain), decoded change for a 1%-|z| push.
A4 LIE GENERATORS: one rank-r matrix G (768x768, shared by ALL tokens of ALL frames) per world motion
   (rot x/y/z by eps rad, trans x/y/z by eps*scene scale, scale by eps) so that D(z + G z) - D(z) = target motion of the
   decoded geometry (dorigin, ddir, dlogdepth). Train on even frames, test on odd frames. Controls: "inconsistent"
   (origin rotated about x, rays about y: not a rigid motion) and "noise" (i.i.d. per-patch targets).
   Commutator check: [G_rx, G_ry] z should decode like a rotation about z by eps^2.

  PYTHONPATH=$(pwd) python shovel/latent_dyn.py --parts a1 a2 a3 a4 --out out/stage3_latent/latentdyn
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
from dvlt_probe import Probe  # noqa: E402
from token_rw import Dec, PM  # noqa: E402

DEV = "cuda"


def load_state(dump, K, s):
    return torch.from_numpy(np.load(f"{dump}/K{K}/step_{s:03d}.npy").astype(np.float32)).to(DEV)


class Loop:
    def __init__(self, P, dump, K):
        self.P, self.K = P, K
        meta = np.load(f"{dump}/K{K}/meta.npz")
        self.rope = torch.from_numpy(meta["rope"]).to(DEV)
        self.ts = torch.linspace(0.0, 1.0, K).tolist()

    @torch.no_grad()
    def F(self, z, k):
        t_now = self.ts[k]; t_next = self.ts[k + 1] if k + 1 < self.K else 1.0
        S = z.shape[0]
        return self.P.m._interval_step(z, t_now, t_next, self.rope, 1, S).float()

    def jvp(self, z, k, v, rel=1e-2):
        eps = rel * z.norm() / v.norm().clamp_min(1e-12)
        return (self.F(z + eps * v, k) - self.F(z - eps * v, k)) / (2 * eps)


def std_out(o, sd):
    """per-patch decoded quantities -> standardized flat vector (n, Np*7)"""
    return torch.cat([o["logdepth"][..., None] / sd[0], o["dir"] / sd[1], o["origin"] / sd[2]], -1).flatten(1)


# ----------------------------------------------------------------------------- characterisation helpers
def obs_gain(dec, sd, z, frames, v, n_rand=4, rel=1e-2, seed=0):
    """decoder gain |J v_f|/|v_f| relative to random directions, averaged over frames"""
    g = torch.Generator(device=DEV).manual_seed(seed)
    ratios = []
    for f in frames:
        zf, vf = z[f:f + 1], v[f:f + 1]
        def gain(u):
            e = rel * zf.norm() / u.norm()
            d = std_out(dec(zf + e * u, [f]), sd) - std_out(dec(zf - e * u, [f]), sd)
            return float(d.norm() / (2 * e * u.norm()))
        gr = np.mean([gain(torch.randn(vf.shape, generator=g, device=DEV)) for _ in range(n_rand)])
        ratios.append(gain(vf) / gr)
    return float(np.mean(ratios))


def characterise(v, z, dec, sd, frames, ch):
    v = v / v.norm()
    vp = v[:, PM:]
    Np = vp.shape[1]
    r = dict(spec_share=float(v[:, :PM].pow(2).sum()),
             frame_mean_share=float((vp.mean(1).pow(2).sum() * Np) / vp.pow(2).sum()),
             global_mean_share=float((vp.reshape(-1, 768).mean(0).pow(2).sum() * vp.shape[0] * Np) / vp.pow(2).sum()),
             cos_with_z=float((v * z).sum() / z.norm()),
             ch_share=float(v[..., ch].pow(2).sum()),
             obs_ratio=obs_gain(dec, sd, z, frames, v))
    eps = 1e-2 * z.norm()
    b, o = dec(z[frames], frames), dec(z[frames] + eps * v[frames], frames)
    r.update(d_logdepth=float((o["logdepth"] - b["logdepth"]).abs().mean()),
             d_dir_deg=float(torch.rad2deg(torch.arccos((o["dir"] * b["dir"]).sum(-1).clamp(-1, 1))).mean()),
             d_origin=float((o["origin"] - b["origin"]).norm(dim=-1).mean()))
    return r


# ----------------------------------------------------------------------------- A1
def run_a1(P, a, say):
    out = {}
    for K, steps in [(16, list(range(16))), (64, [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 63])]:
        meta = np.load(f"{a.dump}/K{K}/meta.npz"); dec = Dec(P, meta)
        rows = []
        for k in steps:
            z0, z1 = load_state(a.dump, K, k), load_state(a.dump, K, k + 1)
            tg = np.load(f"{a.dump}/K{K}/targets_{k:03d}.npz")
            sd = torch.tensor([tg["logdepth"].std(), tg["dir"].std(), tg["origin"].std()], device=DEV)
            dz = z1 - z0
            fr_rows = []
            for f in a.frames:
                zf = z0[f:f + 1].clone().requires_grad_(True)
                y = std_out(dec(zf, [f], grad=True), sd)[0]
                g = torch.Generator(device=DEV).manual_seed(f)
                Y = []
                for i in range(a.rank):
                    u = torch.randn(y.shape, generator=g, device=DEV)
                    gr, = torch.autograd.grad(y, zf, grad_outputs=u, retain_graph=i < a.rank - 1)
                    Y.append(gr.flatten())
                Y = torch.stack(Y, 1)                                         # (dim, r)
                W, Sv, _ = torch.linalg.svd(Y, full_matrices=False)
                d = dz[f].flatten()
                c = W.T @ d
                e = d.pow(2).sum()
                fr_rows.append(dict(frac64=float(c[:64].pow(2).sum() / e), frac256=float(c.pow(2).sum() / e),
                                    dz_obs=float(c.norm()), dz_unobs=float((e - c.pow(2).sum()).clamp_min(0).sqrt()),
                                    spec_decay=float(Sv[-1] / Sv[0]), spec_decay64=float(Sv[min(63, len(Sv) - 1)] / Sv[0])))
                del Y, W
            gain = obs_gain(dec, sd, z0, a.frames, dz)
            dim = 869 * 768
            row = dict(K=K, step=k, gain_ratio=gain, rand64=64 / dim, rand256=a.rank / dim,
                       dz_norm=float(dz.norm()), z_norm=float(z0.norm()),
                       **{key: float(np.mean([r_[key] for r_ in fr_rows])) for key in fr_rows[0]})
            rows.append(row)
            say(f"A1 K{K} step {k:>2}: |dz|/|z| {row['dz_norm'] / row['z_norm']:.3f} | frac in top-64 obs {row['frac64']:.4f}"
                f" top-256 {row['frac256']:.4f} (random {row['rand256']:.1e}) | |dz_obs| {row['dz_obs']:.2f} |dz_unobs| {row['dz_unobs']:.2f}"
                f" | decoder gain vs random {gain:.2f} | sketch σ256/σ1 {row['spec_decay']:.3f}")
        out[f"K{K}"] = rows
    return out


# ----------------------------------------------------------------------------- A2
def run_a2(P, a, say):
    out = {}
    for K in [16, 64]:
        L = Loop(P, a.dump, K)
        z = load_state(a.dump, K, 0)
        g = torch.Generator(device=DEV).manual_seed(0)
        Q = torch.randn((a.m2,) + tuple(z.shape), generator=g, device=DEV)
        Q = torch.linalg.qr(Q.flatten(1).T)[0].T.reshape(Q.shape)
        rows, logsum = [], np.zeros(a.m2)
        for k in range(K):
            zk, zk1 = load_state(a.dump, K, k), load_state(a.dump, K, k + 1)
            V = torch.stack([L.jvp(zk, k, Q[i]) for i in range(a.m2)])
            Qn, R = torch.linalg.qr(V.flatten(1).T)
            sgn = torch.sign(torch.diag(R))
            Q = (Qn * sgn).T.reshape(V.shape); gk = torch.diag(R).abs().cpu().numpy()
            logsum += np.log(gk)
            nr = float(zk1.norm() / zk.norm())
            rows.append(dict(step=k, g=gk.tolist(), norm_growth=nr, rel_g1=float(gk[0] / nr)))
            say(f"A2 K{K} step {k:>2}: growth g1..g{a.m2} = " + " ".join(f"{x:.3f}" for x in gk) +
                f" | |z| growth {nr:.3f} | g1/|z|growth {gk[0] / nr:.3f}")
        say(f"A2 K{K}: mean log-growth per step (Lyapunov) = " + " ".join(f"{x / K:+.3f}" for x in logsum))
        out[f"K{K}"] = dict(rows=rows, lyap=(logsum / K).tolist())
        if K == 64:
            meta = np.load(f"{a.dump}/K{K}/meta.npz"); dec = Dec(P, meta)
            tg = np.load(f"{a.dump}/K{K}/targets_{K:03d}.npz")
            sd = torch.tensor([tg["logdepth"].std(), tg["dir"].std(), tg["origin"].std()], device=DEV)
            zf = load_state(a.dump, K, K)
            ch = int(zf[:, PM:].pow(2).mean((0, 1)).argmax())
            out[f"K{K}"]["final_lyap_vector"] = characterise(Q[0], zf, dec, sd, a.frames, ch)
            say(f"A2 K64 final top Lyapunov vector: {out[f'K{K}']['final_lyap_vector']}")
    return out


# ----------------------------------------------------------------------------- A3
def run_a3(P, a, say):
    out = {}
    for spec in a.a3_points:
        K, k = [int(v) for v in spec.split(":")]
        L = Loop(P, a.dump, K)
        meta = np.load(f"{a.dump}/K{K}/meta.npz"); dec = Dec(P, meta)
        tg = np.load(f"{a.dump}/K{K}/targets_{k:03d}.npz")
        sd = torch.tensor([tg["logdepth"].std(), tg["dir"].std(), tg["origin"].std()], device=DEV)
        z = load_state(a.dump, K, k)
        ch = int(z[:, PM:].pow(2).mean((0, 1)).argmax())
        g = torch.Generator(device=DEV).manual_seed(1)
        m = a.m3
        Q = torch.linalg.qr(torch.randn(z.numel(), m, generator=g, device=DEV))[0]
        t0 = time.time()
        for it in range(a.iters):
            V = torch.stack([L.jvp(z, k, Q[:, i].reshape(z.shape)).flatten() for i in range(m)], 1)
            Q = torch.linalg.qr(V)[0]
        V = torch.stack([L.jvp(z, k, Q[:, i].reshape(z.shape)).flatten() for i in range(m)], 1)
        H = (Q.T @ V).double().cpu().numpy()
        lam, Yv = np.linalg.eig(H)
        order = np.argsort(-np.abs(lam)); lam, Yv = lam[order], Yv[:, order]
        resid = []
        modes = []
        for j in range(m):
            y = torch.from_numpy(np.real(Yv[:, j])).float().to(DEV)
            x = Q @ y; x = x / x.norm()
            Jx = L.jvp(z, k, x.reshape(z.shape)).flatten()
            resid.append(float((Jx - float(np.real(lam[j])) * x).norm() / Jx.norm()) if abs(np.imag(lam[j])) < 1e-6 else float("nan"))
            c = characterise(x.reshape(z.shape), z, dec, sd, a.frames, ch)
            c.update(lam_abs=float(abs(lam[j])), lam_re=float(np.real(lam[j])), lam_im=float(np.imag(lam[j])), ritz_resid=resid[-1])
            modes.append(c)
        # reference: same characterisation of a random direction and of the radial direction z itself
        ref = dict(random=characterise(torch.randn(z.shape, generator=g, device=DEV), z, dec, sd, a.frames, ch),
                   radial=characterise(z.clone(), z, dec, sd, a.frames, ch),
                   radial_gain=float(L.jvp(z, k, z / z.norm()).norm()))
        out[spec] = dict(modes=modes, ref=ref, z_norm_growth=float(load_state(a.dump, K, k + 1).norm() / z.norm()))
        say(f"\nA3 K{K} step {k} ({time.time() - t0:.0f}s): |z| growth {out[spec]['z_norm_growth']:.3f}, gain along z itself {ref['radial_gain']:.3f}")
        for j, c in enumerate(modes):
            say(f"  mode {j}: λ={c['lam_re']:+.3f}{c['lam_im']:+.3f}i |λ|={c['lam_abs']:.3f} (ritz resid {c['ritz_resid']:.2f}) | spec {c['spec_share']:.3f}"
                f" frame-mean {c['frame_mean_share']:.3f} global-mean {c['global_mean_share']:.3f} cos(z) {c['cos_with_z']:+.3f}"
                f" ch{ch} {c['ch_share']:.3f} | obs ratio {c['obs_ratio']:.2f} | push 1%|z|: Δlogd {c['d_logdepth']:.4f} Δdir {c['d_dir_deg']:.3f}° Δorg {c['d_origin']:.4f}")
        for nm in ["random", "radial"]:
            c = ref[nm]
            say(f"  [{nm:>6}]: spec {c['spec_share']:.3f} frame-mean {c['frame_mean_share']:.3f} global-mean {c['global_mean_share']:.3f}"
                f" cos(z) {c['cos_with_z']:+.3f} ch {c['ch_share']:.3f} | obs ratio {c['obs_ratio']:.2f} | Δlogd {c['d_logdepth']:.4f}"
                f" Δdir {c['d_dir_deg']:.3f}° Δorg {c['d_origin']:.4f}")
    return out


# ----------------------------------------------------------------------------- A4
def skew(w):
    return torch.tensor([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=torch.float32, device=DEV)


def run_a4(P, a, say):
    K, s = [int(v) for v in a.a4_state.split(":")]
    meta = np.load(f"{a.dump}/K{K}/meta.npz"); dec = Dec(P, meta)
    z = load_state(a.dump, K, s)
    S = z.shape[0]
    base = {k: torch.cat([dec(z[i:i + 4], list(range(i, min(i + 4, S))))[k] for i in range(0, S, 4)]) for k in ["origin", "dir", "logdepth"]}
    c_all = base["origin"].mean(1)
    scale = float((c_all - c_all.mean(0)).norm(dim=-1).mean())
    eps = a.eps
    tr, te = list(range(0, S, 2)), list(range(1, S, 2))
    gN = torch.Generator(device=DEV).manual_seed(7)
    noise = {k: torch.randn(base[k].shape, generator=gN, device=DEV) for k in ["origin", "dir", "logdepth"]}

    def target(name, fr):
        o, d = base["origin"][fr], base["dir"][fr]
        zero_ld = torch.zeros_like(base["logdepth"][fr])
        e = torch.eye(3, device=DEV)
        if name.startswith("rot"):
            W = skew(eps * e["xyz".index(name[-1])])
            return o @ W.T, d @ W.T, zero_ld
        if name.startswith("trans"):
            t = eps * scale * e["xyz".index(name[-1])]
            return torch.zeros_like(o) + t, torch.zeros_like(d), zero_ld
        if name == "scale":
            return eps * o, torch.zeros_like(d), zero_ld + eps
        if name == "inconsistent":
            return o @ skew(eps * e[0]).T, d @ skew(eps * e[1]).T, zero_ld
        if name == "noise":
            return (eps * scale * noise["origin"][fr], eps * noise["dir"][fr], eps * noise["logdepth"][fr])
        raise ValueError(name)

    def pred(U, V, fr, grad=True):
        outs = []
        for i in range(0, len(fr), 4):
            f4 = fr[i:i + 4]; zf = z[f4]
            o = dec(zf + (zf @ V) @ U.T, f4, grad=grad)
            outs.append((o["origin"] - base["origin"][f4], o["dir"] - base["dir"][f4], o["logdepth"] - base["logdepth"][f4]))
        return tuple(torch.cat([q[j] for q in outs]) for j in range(3))

    def norm3(t):                                           # put the three components on a common dimensionless scale
        return torch.cat([t[0].flatten(1) / scale, t[1].flatten(1), t[2].flatten(1)], 1)

    def score(U, V, name, fr):
        with torch.no_grad():
            p = pred(U, V, fr, grad=False); t = target(name, fr)
            P_, T_ = norm3(p), norm3(t)
            comp = {}
            for i, nm in enumerate(["origin", "dir", "logdepth"]):
                pi_, ti_ = p[i].flatten() / (scale if i == 0 else 1), t[i].flatten() / (scale if i == 0 else 1)
                comp[nm] = dict(target_rms=float(ti_.pow(2).mean().sqrt()), err_rms=float((pi_ - ti_).pow(2).mean().sqrt()))
            return float(1 - (P_ - T_).pow(2).sum() / T_.pow(2).sum()), comp

    names = ["rotx", "roty", "rotz", "transx", "transy", "transz", "scale", "inconsistent", "noise"]
    G, res = {}, {}
    for name in names:
        g = torch.Generator(device=DEV).manual_seed(11)
        U = (1e-3 * torch.randn(768, a.g_rank, generator=g, device=DEV)).requires_grad_(True)
        V = (torch.randn(768, a.g_rank, generator=g, device=DEV) / 768 ** 0.5 / 25).requires_grad_(True)
        opt = torch.optim.Adam([U, V], lr=a.lr)
        rng = np.random.default_rng(0)
        t0 = time.time()
        for it in range(a.g_iters):
            fr = sorted(rng.choice(tr, 4, replace=False).tolist())
            p, t = pred(U, V, fr), target(name, fr)
            loss = (norm3(p) - norm3(t)).pow(2).mean() / norm3(t).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        rtr, _ = score(U, V, name, tr); rte, comp = score(U, V, name, te)
        G[name] = (U.detach(), V.detach())
        res[name] = dict(R_train=rtr, R_test=rte, comp_test=comp)
        say(f"A4 {name:>12}: explained (1-err/target) train {rtr:.3f}  test {rte:.3f}  ({time.time() - t0:.0f}s) | test err/target rms: "
            + " ".join(f"{k} {v['err_rms']:.4f}/{v['target_rms']:.4f}" for k, v in comp.items()))
    # commutator [Gx, Gy] z  vs  rotation about z by eps^2
    (Ux, Vx), (Uy, Vy) = G["rotx"], G["roty"]
    def Gz(U, V, x):
        return (x @ V) @ U.T
    with torch.no_grad():
        zf = z[te]
        cz = Gz(Ux, Vx, Gz(Uy, Vy, zf)) - Gz(Uy, Vy, Gz(Ux, Vx, zf))
        o = {k: torch.cat([dec((zf + cz)[i:i + 4], te[i:i + 4])[k] for i in range(0, len(te), 4)]) for k in ["origin", "dir", "logdepth"]}
        pc = (o["origin"] - base["origin"][te], o["dir"] - base["dir"][te], o["logdepth"] - base["logdepth"][te])
        W = skew(torch.tensor([0., 0., eps ** 2]))
        tc = (base["origin"][te] @ W.T, base["dir"][te] @ W.T, torch.zeros_like(base["logdepth"][te]))
        Pc, Tc = norm3(pc).flatten(), norm3(tc).flatten()
        cos = float((Pc @ Tc) / (Pc.norm() * Tc.norm() + 1e-12)); ratio = float(Pc.norm() / Tc.norm())
        # compare with the other generators' directions too
        cos_other = {}
        for nm in ["rotx", "roty", "transz", "scale"]:
            t_ = target(nm, te); T_ = norm3(t_).flatten()
            cos_other[nm] = float((Pc @ T_) / (Pc.norm() * T_.norm() + 1e-12))
    res["commutator"] = dict(cos_with_rotz=cos, magnitude_ratio=ratio, cos_other=cos_other)
    say(f"A4 commutator [G_rx, G_ry] z on test frames: cos with rot_z(eps²) target {cos:+.3f}, |decoded|/|target| {ratio:.2f};"
        f" cos with " + ", ".join(f"{k} {v:+.3f}" for k, v in cos_other.items()))
    # pairwise overlap of generators (are rotations/translations different directions in token space?)
    ov = {}
    for i, n1 in enumerate(names[:7]):
        for n2 in names[i + 1:7]:
            A = (G[n1][0] @ G[n1][1].T).flatten(); B = (G[n2][0] @ G[n2][1].T).flatten()
            ov[f"{n1}-{n2}"] = float((A @ B) / (A.norm() * B.norm()))
    res["generator_cos"] = ov
    say("A4 |cos| between generator matrices: " + " ".join(f"{k} {abs(v):.2f}" for k, v in ov.items()))
    res["scale_scene"] = scale
    return res


# ----------------------------------------------------------------------------- plots
def plots(R, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if "a1" in R:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        for K, rows in R["a1"].items():
            st = [r["step"] for r in rows]
            ax[0].plot(st, [r["frac256"] for r in rows], "-o", ms=3, label=f"{K} top-256")
            ax[0].plot(st, [r["frac64"] for r in rows], "--", ms=3, label=f"{K} top-64")
            ax[1].plot(st, [r["dz_obs"] for r in rows], "-o", ms=3, label=f"{K} |dz_obs|")
            ax[1].plot(st, [r["dz_unobs"] for r in rows], "--", ms=3, label=f"{K} |dz_unobs|")
            ax[2].plot(st, [r["gain_ratio"] for r in rows], "-o", ms=3, label=K)
        ax[0].set_yscale("log"); ax[0].set_title("share of |dz|² in observable subspace"); ax[0].axhline(R["a1"]["K16"][0]["rand256"], color="gray", ls=":")
        ax[1].set_title("|dz| observable vs unobservable (per frame)"); ax[1].set_yscale("log")
        ax[2].set_title("decoder gain of dz / random direction"); ax[2].axhline(1, color="gray", ls=":")
        for x in ax: x.set_xlabel("step"); x.grid(alpha=.3); x.legend(fontsize=7); x.axvline(16, color="k", alpha=.3)
        fig.tight_layout(); fig.savefig(f"{out}/a1_observability.png", dpi=130); plt.close(fig)
    if "a2" in R:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for K, d in R["a2"].items():
            st = [r["step"] for r in d["rows"]]
            ax[0].plot(st, [r["g"][0] for r in d["rows"]], "-o", ms=3, label=f"{K} g1")
            ax[0].plot(st, [r["norm_growth"] for r in d["rows"]], ":", label=f"{K} |z| growth")
            ax[1].plot(st, [r["rel_g1"] for r in d["rows"]], "-o", ms=3, label=K)
        ax[0].set_title("top local growth factor per step"); ax[1].set_title("g1 / |z| growth  (>1: perturbations grow faster than the state)")
        for x in ax: x.axhline(1, color="gray", ls="--"); x.set_xlabel("step"); x.grid(alpha=.3); x.legend(fontsize=8); x.axvline(16, color="k", alpha=.3)
        fig.tight_layout(); fig.savefig(f"{out}/a2_lyapunov.png", dpi=130); plt.close(fig)
    if "a3" in R:
        pts = list(R["a3"]); keys = ["lam_abs", "frame_mean_share", "global_mean_share", "cos_with_z", "obs_ratio", "spec_share"]
        fig, ax = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.6))
        for i, k in enumerate(keys):
            for j, p in enumerate(pts):
                vals = [mo[k] for mo in R["a3"][p]["modes"]]
                ax[i].plot(range(len(vals)), vals, "-o", ms=4, label=p)
                if k != "lam_abs":
                    ax[i].axhline(R["a3"][p]["ref"]["random"][k], color=f"C{j}", ls=":", alpha=.6)
            ax[i].set_title(k + ("  (dotted = random dir)" if k != "lam_abs" else "")); ax[i].set_xlabel("mode"); ax[i].grid(alpha=.3)
        ax[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{out}/a3_modes.png", dpi=130); plt.close(fig)
    if "a4" in R:
        names = [n for n in R["a4"] if isinstance(R["a4"][n], dict) and "R_test" in R["a4"][n]]
        fig, ax = plt.subplots(figsize=(9, 3.8))
        x = np.arange(len(names))
        ax.bar(x - .2, [R["a4"][n]["R_train"] for n in names], .4, label="train frames")
        ax.bar(x + .2, [R["a4"][n]["R_test"] for n in names], .4, label="held-out frames")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30); ax.axhline(0, color="k", lw=.8)
        ax.set_title("A4: one shared low-rank G per world motion — fraction of target motion explained"); ax.legend(); ax.grid(alpha=.3, axis="y")
        fig.tight_layout(); fig.savefig(f"{out}/a4_generators.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="out/stage3_latent/tokendump")
    ap.add_argument("--parts", nargs="+", default=["a1", "a2", "a3", "a4"])
    ap.add_argument("--frames", type=int, nargs="+", default=[3, 12, 20])
    ap.add_argument("--rank", type=int, default=256, help="A1 sketch size")
    ap.add_argument("--m2", type=int, default=4)
    ap.add_argument("--m3", type=int, default=8)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--a3-points", nargs="+", default=["16:8", "16:15", "64:32"])
    ap.add_argument("--a4-state", default="16:16")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--g-rank", type=int, default=32)
    ap.add_argument("--g-iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="out/stage3_latent/latentdyn")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = Probe("nvidia/dvlt", max_frames=27)
    log, R = [], {}
    if os.path.exists(f"{a.out}/latentdyn.json"):
        R = json.load(open(f"{a.out}/latentdyn.json"))

    def say(s=""):
        print(s, flush=True); log.append(s)
    torch.backends.cuda.matmul.allow_tf32 = False
    for part in a.parts:
        say(f"\n==================== {part.upper()} ====================")
        R[part] = {"a1": run_a1, "a2": run_a2, "a3": run_a3, "a4": run_a4}[part](P, a, say)
        json.dump(R, open(f"{a.out}/latentdyn.json", "w"), indent=1)
        with open(f"{a.out}/report_{part}.txt", "w") as fh:
            fh.write(" ".join(sys.argv) + "\n" + "\n".join(log) + "\n")
        try:
            plots(R, a.out)
        except Exception as e:  # noqa: BLE001
            say(f"plot failed: {e}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
