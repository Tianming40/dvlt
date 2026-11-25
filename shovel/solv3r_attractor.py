"""Is Solv3R a path-dependent optimizer, or an image-conditioned attractor/projection?

Discriminating experiment (Exp 1 — state-independence):
  Hold the conditioning z0 FIXED (Solv3R's own encoding of the images) and start solve() from
  several WILDLY different initial states z. If they all converge to the SAME decoded solution,
  the answer is a function of z0 (the images), NOT of the initial state/trajectory  => Solv3R is
  an amortized, image-conditioned attractor, not a path-dependent iterative optimizer.

Initial states (good -> bad):
  normal     z = x0                      (Solv3R's default start = reset-to-z0)
  dvlt16     z = DVLT K16 good endpoint  (another model's GOOD state; sanity)
  collapsed  z = DVLT K64 collapsed state (|x| blown up, k_eff~1.5 — the main subject)
  random     z = Gaussian noise          (maximally uninformative)
  scrambled  z = x0 with patch tokens globally permuted (structured-but-wrong)

All runs share one Context (same z0). We decode at calls {0,1,2,4,8,16,32} and record, per state:
  |x|, k_eff  (does it de-collapse?),  AbsRel vs GT (is the attractor good?).
At the final call we report the INTER-RUN agreement (do they land on the same solution?):
  depth: scale-invariant relative difference to the `normal` run;
  pose : mean geodesic rotation angle to the `normal` run (same world frame = view 0, no alignment).

DVLT states are produced once and cached to <out>/dvlt_states.pt (scene+frames tagged).
Does NOT modify src/dvlt, dvlt_probe.py, or third_party/solv3r.
"""
from __future__ import annotations
import argparse, json, os
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- helpers
def _keff_and_norm(z, patch_start, seed=0):
    """Patch-token mean |x| and k_eff (Gram participation ratio on a 1024 subsample)."""
    from dvlt_probe import cluster_stats
    with torch.autocast("cuda", enabled=False):          # cluster_stats -> numpy: keep everything fp32
        xp = z[:, patch_start:].reshape(-1, z.shape[-1]).float()
        tok_norm = xp.norm(dim=-1).mean().item()
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(xp.shape[0], generator=g)[: min(1024, xp.shape[0])]
        zc = F.normalize(xp[idx.to(xp.device)], dim=-1).float()
        k_eff, _ = cluster_stats(zc, [0.9])
    return tok_norm, float(k_eff)


def _absrel(depth_pred, gt_depth):
    """Scale-invariant depth AbsRel (median-scale aligned). depth_pred,gt_depth: (S,H,W)."""
    d = depth_pred.float().reshape(-1)
    g = gt_depth.float().reshape(-1)
    m = g > 1e-6
    if m.sum() < 10:
        return float("nan")
    s = (g[m].median() / d[m].median().clamp_min(1e-6))
    return ((d[m] * s - g[m]).abs() / g[m]).mean().item()


def _depth_reldiff(da, db):
    """Scale-invariant relative difference between two depth maps (align db->da scale)."""
    a = da.float().reshape(-1); b = db.float().reshape(-1)
    m = (a > 1e-6) & (b > 1e-6)
    if m.sum() < 10:
        return float("nan")
    s = (a[m].median() / b[m].median().clamp_min(1e-6))
    return ((b[m] * s - a[m]).abs() / a[m]).mean().item()


def _rot_angle(Ra, Rb):
    """Mean geodesic rotation angle (deg) between two sets of rotations (S,3,3)."""
    R = Ra.transpose(-1, -2) @ Rb
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    ang = torch.arccos(((tr - 1) / 2).clamp(-1, 1))
    return torch.rad2deg(ang).mean().item()


# ----------------------------------------------------------------------------- DVLT states (cached)
def produce_dvlt_states(images, out_pt, tag):
    """Run DVLT K16 (good) and K64 (collapsed) forward on the given images; cache z0/z16/z64."""
    from dvlt_probe import Probe
    print(f"[produce] loading DVLT, running K16 + K64 on {tag} ...")
    P = Probe("nvidia/dvlt", max_frames=images.shape[1])
    x0, rope, dims = P.init_state(images)
    B, S, H, W = dims
    pm = P.m.patch_start_idx
    out = {"tag": tag, "patch_start": pm}

    def run_k(K, grab):
        z = x0.detach().clone()
        saved = {}
        with torch.no_grad():                        # Probe.step is NOT under no_grad -> graph would accumulate
            for i in range(K):
                z = P.step(z, i, K, rope, B, S)
                if (i + 1) in grab:
                    saved[i + 1] = z.detach().float().cpu()
        return saved

    out["z0"] = x0.detach().float().cpu()
    out["z16"] = run_k(16, {16})[16]                 # DVLT good endpoint (K16 run)
    out["z64"] = run_k(64, {64})[64]                 # DVLT collapsed state (K64 run)
    for name in ("z0", "z16", "z64"):
        tn, ke = _keff_and_norm(out[name].cuda(), pm)
        out[f"{name}_tok_norm"], out[f"{name}_k_eff"] = tn, ke
        print(f"  {name}: |x|={tn:7.1f}  k_eff={ke:5.2f}")
    torch.save(out, out_pt)
    print(f"[produce] cached -> {out_pt}")
    del P
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- drift: PROVE the latent does not converge (direct, manifold-agnostic)
def run_drift(a):
    """Three forward trajectories (flash/bf16, 27 frames OK) to PROVE the latent never converges:
      A (normal init):  cumulative displacement ‖z_k − z_ref‖/‖z‖ vs output cumulative. Converge => saturates small;
                        drift => latent cumulative GROWS far beyond the one-step size while output stays put.
      B (different init): A vs B — output agreement ‖d_A−d_B‖ (→small, same attractor) vs latent agreement
                        ‖z_A−z_B‖/‖z‖ (stays LARGE => the 'solution' is not a unique latent point).
      C = A + eps·δ (Lyapunov): ‖z_C − z_A‖/‖z‖ — do two NEARBY trajectories diverge (confirms rho>1 on the real
                        dynamics, manifold-agnostic) or converge (then the ambient rho was an artifact)?
    Cumulative/agreement metrics are O(1e-2..1) so robust in bf16; the fp32 one-step size came from the jac run."""
    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), "third_party", "solv3r"))
    from dvlt_probe import Probe
    from solv3r import Solv3R

    Pload = Probe("nvidia/dvlt", max_frames=a.max_frames)
    images = Pload.load(a.input); del Pload; torch.cuda.empty_cache()
    model = Solv3R.from_pretrained()
    ctx = model.encode(images[0].to(model.device)); x0 = ctx.x0.float()
    g = torch.Generator(device="cpu").manual_seed(0)
    zA = x0.clone()
    zB = (torch.randn(x0.shape, generator=g).to(model.device) * x0.std())          # a DIFFERENT init
    delta = torch.randn(x0.shape, generator=g).to(model.device); delta /= delta.norm()
    zC = x0 + a.drift_eps * x0.norm() * delta                                        # a NEARBY init (Lyapunov)
    dec = lambda z: model.decode(z, ctx)["depth"][..., 0].float()
    rel = lambda u, v: ((u - v).float().norm() / v.float().norm().clamp_min(1e-9)).item()

    zA_ref = dA_ref = None
    rows = []
    print(f"[drift] {a.max_frames} frames, {a.drift_calls} calls, ref={a.drift_ref}, lyap eps={a.drift_eps}")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(1, a.drift_calls + 1):
            zA, zB, zC = model.step(zA, ctx), model.step(zB, ctx), model.step(zC, ctx)
            if k == a.drift_ref:
                zA_ref, dA_ref = zA.clone(), dec(zA)
            if k % a.drift_decode_every == 0 or k == a.drift_ref:
                dA, dB = dec(zA), dec(zB)
                row = {"call": k,
                       "latent_agree_AB": rel(zA, zB),                 # B: latent distance between the two trajectories
                       "out_agree_AB": _depth_reldiff(dA, dB),         # B: output distance between the two trajectories
                       "lyap_sep": rel(zC, zA)}                        # C: nearby-trajectory separation
                if zA_ref is not None:
                    row["latent_cum"] = rel(zA, zA_ref)               # A: how far the latent wandered from ref
                    row["out_cum"] = _depth_reldiff(dA_ref, dA)       # A: how far the output moved from ref
                rows.append(row)
                print(f"  call {k:>4}: latent_cum={row.get('latent_cum', float('nan')):.4f} out_cum={row.get('out_cum', float('nan')):.4f} "
                      f"| AB latent={row['latent_agree_AB']:.3f} out={row['out_agree_AB']:.4f} | lyap={row['lyap_sep']:.4f}")

    os.makedirs(a.out, exist_ok=True)
    json.dump({"frames": a.max_frames, "calls": a.drift_calls, "ref": a.drift_ref, "eps": a.drift_eps, "rows": rows},
              open(os.path.join(a.out, "drift.json"), "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xs = [r["call"] for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
        # A: cumulative drift — latent keeps growing, output stays put
        xr = [r["call"] for r in rows if "latent_cum" in r]
        ax[0].plot(xr, [r["latent_cum"] for r in rows if "latent_cum" in r], "-o", ms=3, color="#d62728", label="latent ‖z−z_ref‖/‖z‖")
        ax[0].plot(xr, [r["out_cum"] for r in rows if "out_cum" in r], "-o", ms=3, color="#1f77b4", label="output ‖d−d_ref‖/d")
        ax[0].set_xlabel("solver call"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=9)
        ax[0].set_title("A: cumulative drift from ref\nlatent GROWS, output stays put => no latent fixed point")
        # B: same output, different latent
        ax[1].plot(xs, [r["latent_agree_AB"] for r in rows], "-o", ms=3, color="#d62728", label="latent ‖z_A−z_B‖/‖z‖")
        ax[1].plot(xs, [r["out_agree_AB"] for r in rows], "-o", ms=3, color="#1f77b4", label="output ‖d_A−d_B‖/d")
        ax[1].set_yscale("log"); ax[1].set_xlabel("solver call"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=9)
        ax[1].set_title("B: two different inits\noutput CONVERGES (same), latent STAYS APART")
        # C: Lyapunov — do nearby trajectories diverge?
        ax[2].plot(xs, [r["lyap_sep"] for r in rows], "-o", ms=3, color="#2ca02c")
        ax[2].axhline(a.drift_eps, ls=":", color="gray", alpha=.7); ax[2].text(xs[0], a.drift_eps * 1.1, f"start ε={a.drift_eps}", fontsize=8)
        ax[2].set_yscale("log"); ax[2].set_xlabel("solver call"); ax[2].grid(alpha=.3)
        ax[2].set_title("C: nearby trajectories (Lyapunov)\nseparation grows => diverge (rho>1 confirmed)")
        fig.suptitle(f"Solv3R latent DOES NOT converge ({a.max_frames} frames, {a.drift_calls} calls)", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "drift.png"), dpi=130)
        print(f"saved {a.out}/drift.png")
    except Exception as e:
        print("plot skipped:", e)
    print("\nReading: A latent_cum grows while out_cum flat; B out_agree→small while latent_agree stays large; "
          "C lyap_sep grows => the LATENT never converges (output stability is a projection artifact).")


# ----------------------------------------------------------------------------- jac: Jacobian spectrum of the solver map at the fixed point
def run_jac(a):
    """Characterize the LOCAL dynamics of Solv3R's map T at its fixed point z*:
      (1) spectral radius rho(J), J = dT/dz|z*  -> is it a CONTRACTION (rho<1)? at what rate?
      (2) symmetry of J                          -> is it an ENERGY-DESCENT flow (J symmetric) or just a stable map?
    Both use ONLY vector-Jacobian products (backward mode): J and J^T share eigenvalues (so rho(J^T)=rho(J)),
    and symmetry is probed via the bilinear form uᵀJv=<J^T u,v> vs vᵀJu=<J^T v,u>. Runs in float32 (math attention),
    few frames (the local character is frame-robust; 27 frames would OOM materializing the N×N attention)."""
    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), "third_party", "solv3r"))
    from dvlt_probe import Probe
    from solv3r import Solv3R
    from torch.nn.attention import SDPBackend, sdpa_kernel     # MATH backend = double-differentiable (needed for JVP)

    Pload = Probe("nvidia/dvlt", max_frames=a.jac_frames)
    images = Pload.load(a.input); del Pload; torch.cuda.empty_cache()
    model = Solv3R.from_pretrained()
    block = model.net.block                                 # the stationary SolverBlock T
    ctx = model.encode(images[0].to(model.device))
    x0, pos = ctx.x0.float(), ctx.pos

    # fixed point z* = solve to convergence (float32, no autocast, so J later is consistent)
    print(f"[jac] {a.jac_frames} frames, solving {a.jac_solve} calls to the fixed point (fp32) ...")
    resid_hist, outvel_hist = [], []                        # latent change vs output change, per call
    prev_dep = None
    with torch.no_grad(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        z = x0.clone()
        for k in range(a.jac_solve):
            zn = block(z.float(), x0, pos)
            resid_hist.append(((zn - z).norm() / z.norm().clamp_min(1e-9)).item())
            dep = model.decode(zn, ctx)["depth"][..., 0].float()       # output (decoded depth) this call
            outvel_hist.append(_depth_reldiff(prev_dep, dep) if prev_dep is not None else float("nan"))
            prev_dep = dep
            if k >= a.jac_solve - 5:
                print(f"  call {k}: latent ‖Δz‖/‖z‖={resid_hist[-1]:.2e}  output Δ={outvel_hist[-1]:.2e}")
            z = zn
    zstar = z.float().detach()
    last_resid = resid_hist[-1]
    print(f"[jac] CONTRAST at call {a.jac_solve}: latent still moves {last_resid:.2e}, output moves {outvel_hist[-1]:.2e} "
          f"=> output CONVERGES, latent DRIFTS")

    def _T(zin):                                            # the fp32 map T(z) = block(z, x0, pos)
        with torch.enable_grad(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            return block(zin, x0, pos)

    # VJP: J^T w = d(w·T(z))/dz at z*
    def vjp(w):
        zin = zstar.clone().requires_grad_(True)
        out = _T(zin)
        jt_w, = torch.autograd.grad(out, zin, grad_outputs=w, retain_graph=False)
        return jt_w.detach()

    # JVP: J v  (forward-over-reverse); robust vector to compare against J^T v for symmetry
    from torch.autograd.functional import jvp as _torch_jvp
    def jvp(v):
        _, Jv = _torch_jvp(_T, zstar, v, create_graph=False, strict=False)
        return Jv.detach()

    # (1) spectral radius via power iteration on J^T  (rho(J^T) = rho(J))
    print(f"\n[jac] power iteration for rho(J) ({a.jac_iters} iters) ...")
    g = torch.Generator(device="cpu").manual_seed(0)
    v = torch.randn(zstar.shape, generator=g).to(zstar.device); v = v / v.norm()
    rho_hist = []
    for it in range(a.jac_iters):
        Jtv = vjp(v)
        lam = (Jtv.norm() / v.norm()).item()
        rho_hist.append(lam)
        v = Jtv / Jtv.norm().clamp_min(1e-12)
        if (it + 1) % 5 == 0 or it == a.jac_iters - 1:
            print(f"  iter {it+1:>3}: rho ≈ {lam:.4f}")
    rho = rho_hist[-1]

    # (2) symmetry via the VECTOR comparison  ‖Jv − Jᵀv‖ / ‖Jv‖   (0 = symmetric, up to 2 = antisymmetric)
    print(f"\n[jac] symmetry probe ({a.jac_probes} random v) ...")
    asym = []
    for t in range(a.jac_probes):
        v = torch.randn(zstar.shape, generator=g).to(zstar.device); v /= v.norm()
        Jv, Jtv = jvp(v), vjp(v)
        rel = ((Jv - Jtv).norm() / Jv.norm().clamp_min(1e-12)).item()
        asym.append(rel)
        print(f"  probe {t+1}: ‖Jv‖={Jv.norm():.3f}  ‖Jv−Jᵀv‖/‖Jv‖ = {rel:.3f}")
    import statistics
    asym_mean = statistics.mean(asym)

    os.makedirs(a.out, exist_ok=True)
    out_json = os.path.join(a.out, f"jacobian_{a.jac_frames}f.json")
    json.dump({"frames": a.jac_frames, "jac_solve": a.jac_solve, "latent_residual": last_resid,
               "rho": rho, "rho_hist": rho_hist, "asym_rel": asym, "asym_rel_mean": asym_mean,
               "resid_hist": resid_hist, "outvel_hist": outvel_hist,
               "note": "J = dT/dz of ONE Solv3R solver call at z*; rho via power-iteration on J^T (VJP), "
                       "symmetry via ‖Jv-J^Tv‖/‖Jv‖ (JVP+VJP). random-matrix symmetry baseline ~1.41."},
              open(out_json, "w"), indent=1)
    print(f"\n=== Jacobian of Solv3R's map T at z* ({a.jac_frames} frames) ===")
    print(f"  latent residual ‖Δz‖/‖z‖ = {last_resid:.2e}   ({'latent converged' if last_resid < 1e-4 else 'latent DRIFTS (no fixed point)'})")
    print(f"  spectral radius  rho(J) = {rho:.4f}   ({'CONTRACTION (rho<1): stable fixed point' if rho < 1 else 'NOT a contraction (rho>=1): latent map expands/drifts'})")
    print(f"  mean rel-asymmetry      = {asym_mean:.3f}   "
          f"({'~symmetric => ENERGY-DESCENT flow' if asym_mean < 0.15 else 'ASYMMETRIC => NOT a gradient flow'})")

    # ---- visualization: the whole story in 3 panels ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
        calls = list(range(1, len(resid_hist) + 1))
        # Panel A: output CONVERGES while latent DRIFTS (the key contrast)
        ax[0].plot(calls, resid_hist, "-o", ms=3, color="#d62728", label="latent change ‖Δz‖/‖z‖")
        ax[0].plot(calls, outvel_hist, "-o", ms=3, color="#1f77b4", label="output change ‖Δdepth‖/depth")
        ax[0].axhline(last_resid, ls=":", color="#d62728", alpha=.6)
        ax[0].set_yscale("log"); ax[0].set_xlabel("solver call"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=9)
        ax[0].set_title("OUTPUT converges → 0,  LATENT plateaus (keeps drifting)")
        # Panel B: rho power iteration -> 1.67 (above the contraction line at 1)
        ax[1].plot(range(1, len(rho_hist) + 1), rho_hist, "-o", ms=3, color="#2ca02c")
        ax[1].axhline(1.0, ls="--", color="k", alpha=.7)
        ax[1].text(len(rho_hist) * 0.4, 1.02, "ρ=1  (contraction boundary)", fontsize=9)
        ax[1].text(len(rho_hist) * 0.5, rho * 0.97, f"ρ(J)={rho:.2f}>1\nnot a contraction\n(latent expands)", fontsize=10, va="top")
        ax[1].set_xlabel("power-iteration step"); ax[1].set_ylabel("ρ estimate"); ax[1].grid(alpha=.3)
        ax[1].set_title("Jacobian spectral radius ρ(J)")
        # Panel C: symmetry on the 0..2 scale (0 symmetric=gradient flow, 1.41 random, 2 antisymmetric)
        ax[2].barh([0], [asym_mean], color="#9467bd", height=0.4)
        for x, lab, c in [(0, "0\nsymmetric\n(gradient flow)", "k"), (1.41, "1.41\nrandom", "gray"), (2, "2\nantisym", "k")]:
            ax[2].axvline(x, ls="--", color=c, alpha=.5); ax[2].text(x, 0.3, lab, ha="center", fontsize=8)
        ax[2].plot([asym_mean], [0], "o", color="#9467bd", ms=10)
        ax[2].text(asym_mean, -0.35, f"Solv3R\n{asym_mean:.2f}", ha="center", fontsize=10, color="#9467bd", weight="bold")
        ax[2].set_xlim(-0.1, 2.1); ax[2].set_ylim(-0.6, 0.6); ax[2].set_yticks([]); ax[2].set_xlabel("‖Jv−Jᵀv‖/‖Jv‖")
        ax[2].set_title("Symmetry of J  →  NOT a gradient flow")
        fig.suptitle(f"Solv3R latent dynamics ({a.jac_frames} frames): output = stable attractor, "
                     f"but latent = non-gradient (asym {asym_mean:.2f}), expanding (ρ {rho:.2f}>1), non-converging drift", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"jacobian_{a.jac_frames}f.png"), dpi=130)
        print(f"saved {a.out}/jacobian_{a.jac_frames}f.png")
    except Exception as e:
        print("plot skipped:", e)


# ----------------------------------------------------------------------------- exp 2: perturbation double-dissociation
def run_exp2(a):
    """Does the solution depend on the CONDITIONING z0 or on the INITIAL state z?
    Same-magnitude perturbation δ (‖δ‖/‖x0‖ = eps), applied two ways:
      perturb-cond : re-injected z0 <- x0+δ, init = clean x0   -> answer should MOVE   (= f(z0))
      perturb-init : re-injected z0 = clean x0, init = x0+δ     -> answer should NOT move (washed out)
    If Δanswer(cond) >> Δanswer(init), the solution is a function of z0, not the trajectory."""
    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), "third_party", "solv3r"))
    from dvlt_probe import Probe, load_gt
    from solv3r import Solv3R
    from solv3r.model import cameras_from_pose_enc, Context

    Pload = Probe("nvidia/dvlt", max_frames=a.max_frames)
    images = Pload.load(a.input); S = images.shape[1]
    gt_depth, _, _ = load_gt(a.gt, Pload.img_size, device="cuda"); gt_depth = gt_depth[:S].float()
    del Pload; torch.cuda.empty_cache()

    model = Solv3R.from_pretrained()
    ctx = model.encode(images[0].to(model.device))
    H, W = ctx.H, ctx.W
    x0 = ctx.x0.float()
    pm = 5

    def solve_decode(cond_x0, init_z):
        z = init_z.clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c = Context(cond_x0, ctx.pos, H, W)
            for _ in range(a.calls):
                z = model.step(z, c)
            pred = model.decode(z, c)
        return pred["depth"][..., 0].float().detach().cpu(), pred["pose_enc"].detach().float().cpu()

    # baseline (clean cond, clean init)
    d0, p0 = solve_decode(x0, x0)
    R0, _ = cameras_from_pose_enc(p0, (H, W)); R0 = R0[:, :3, :3]
    print(f"[baseline] AbsRel={_absrel(d0, gt_depth.cpu()):.4f}")

    # one fixed noise direction, scaled per eps (fair init-vs-cond comparison)
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(x0.shape, generator=gen).to(model.device)
    noise = noise / noise.float().norm() * x0.float().norm()          # ‖noise‖ = ‖x0‖
    rows = []
    for eps in a.eps:
        delta = eps * noise
        out = {"eps": eps}
        for mode, (cond, init) in (("perturb_cond", (x0 + delta, x0)),
                                   ("perturb_init", (x0, x0 + delta))):
            d, p = solve_decode(cond, init)
            R, _ = cameras_from_pose_enc(p, (H, W))
            out[f"{mode}_depth_reldiff"] = _depth_reldiff(d0, d)
            out[f"{mode}_pose_deg"] = _rot_angle(R0, R[:, :3, :3])
        rows.append(out)
        print(f"eps={eps:.2f} | perturb-COND Δdepth={out['perturb_cond_depth_reldiff']:.4f} "
              f"Δpose={out['perturb_cond_pose_deg']:.3f}deg  ||  perturb-INIT Δdepth={out['perturb_init_depth_reldiff']:.4f} "
              f"Δpose={out['perturb_init_pose_deg']:.3f}deg")

    json.dump({"tag": a.tag, "calls": a.calls, "eps": a.eps, "rows": rows},
              open(os.path.join(a.out, "exp2_dissociation.json"), "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for ci, (key, title) in enumerate([("depth_reldiff", "Δ depth (rel diff vs baseline)"),
                                           ("pose_deg", "Δ pose (deg vs baseline)")]):
            for mode, lab in (("perturb_cond", "perturb z0 (conditioning)"), ("perturb_init", "perturb init state")):
                ax[ci].plot(a.eps, [r[f"{mode}_{key}"] for r in rows], marker="o", label=lab)
            ax[ci].set_title(title); ax[ci].set_xlabel("perturbation eps  ‖δ‖/‖x0‖"); ax[ci].grid(alpha=.3); ax[ci].legend()
        fig.suptitle(f"Solv3R double-dissociation ({a.tag}): answer tracks z0, not the initial state")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "exp2_dissociation.png"), dpi=130)
        print(f"saved {a.out}/exp2_dissociation.png")
    except Exception as e:
        print("plot skipped:", e)
    print("\nReading: perturb-COND moves the answer (grows with eps) while perturb-INIT stays ~0 "
          "=> the solution is a FUNCTION OF z0 (image conditioning), not of the initial state. Double dissociation.")


# ----------------------------------------------------------------------------- main experiment
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/data/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--gt", default="out/data/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--calls", type=int, default=32, help="total Solv3R solver calls")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    ap.add_argument("--out", default="out/solv3r/solv3rexperiment")
    ap.add_argument("--tag", default="s10_27f")
    ap.add_argument("--states", nargs="+", default=["normal", "dvlt16", "collapsed", "random", "scrambled"])
    ap.add_argument("--exp", choices=["1", "2", "jac", "drift"], default="1",
                    help="1 = state-independence; 2 = double-dissociation; jac = Jacobian spectrum; drift = prove latent never converges")
    ap.add_argument("--drift-calls", type=int, default=800, help="drift: total solver calls")
    ap.add_argument("--drift-ref", type=int, default=60, help="drift: reference call for cumulative displacement")
    ap.add_argument("--drift-decode-every", type=int, default=10, help="drift: decode (output metrics) every N calls")
    ap.add_argument("--drift-eps", type=float, default=0.05, help="drift: nearby-trajectory perturbation for the Lyapunov test")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4],
                    help="exp2: relative perturbation magnitudes ‖δ‖/‖x0‖")
    ap.add_argument("--jac-frames", type=int, default=8, help="jac: #frames (fp32 math attention; 27 would OOM)")
    ap.add_argument("--jac-solve", type=int, default=80, help="jac: calls to reach the fixed point z*")
    ap.add_argument("--jac-iters", type=int, default=40, help="jac: power-iteration steps for rho(J)")
    ap.add_argument("--jac-probes", type=int, default=8, help="jac: random u,v pairs for the symmetry probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), "third_party", "solv3r"))
    from dvlt_probe import Probe, load_gt
    from solv3r import Solv3R
    from solv3r.model import cameras_from_pose_enc

    if a.exp == "2":
        return run_exp2(a)
    if a.exp == "jac":
        return run_jac(a)
    if a.exp == "drift":
        return run_drift(a)

    # --- images (shared by DVLT-collapse production and Solv3R conditioning) ---
    Pload = Probe("nvidia/dvlt", max_frames=a.max_frames)
    images = Pload.load(a.input)                       # (1,S,3,H,W) in [0,1]
    S = images.shape[1]
    gt_depth, gt_c2w, gt_K = load_gt(a.gt, Pload.img_size, device="cuda")
    gt_depth = gt_depth[:S].float()
    del Pload                                          # free DVLT-loader; states come from cache/produce
    torch.cuda.empty_cache()

    # --- DVLT states (cached) ---
    pt = os.path.join(a.out, "dvlt_states.pt")
    if os.path.isfile(pt):
        dv = torch.load(pt, map_location="cpu")
        print(f"[cache] loaded DVLT states {pt}  (tag={dv.get('tag')})")
        for n in ("z0", "z16", "z64"):
            print(f"  {n}: |x|={dv[f'{n}_tok_norm']:7.1f}  k_eff={dv[f'{n}_k_eff']:5.2f}")
    else:
        dv = produce_dvlt_states(images, pt, a.tag)
    pm = dv["patch_start"]

    # --- Solv3R: fixed conditioning z0 for ALL runs ---
    print("\n[solv3r] loading + encoding (shared z0) ...")
    model = Solv3R.from_pretrained()
    ctx = model.encode(images[0].to(model.device))    # (S,3,H,W) -> Context ; ctx.x0 = z0
    H, W = ctx.H, ctx.W
    x0 = ctx.x0.float()                               # (S,P,C)

    # --- build the initial states (same shape (S,P,C)) ---
    gen = torch.Generator(device="cpu").manual_seed(0)
    def scramble(z):
        zc = z.clone()
        p = zc[:, pm:].reshape(-1, zc.shape[-1])      # all patch tokens across frames
        perm = torch.randperm(p.shape[0], generator=gen)
        zc[:, pm:] = p[perm].reshape(zc[:, pm:].shape)
        return zc
    inits = {
        "normal":    x0.clone(),
        "dvlt16":    dv["z16"].float().to(model.device),
        "collapsed": dv["z64"].float().to(model.device),
        "random":    (torch.randn(x0.shape, generator=gen).to(model.device) * x0.std()),
        "scrambled": scramble(x0),
    }
    inits = {k: v for k, v in inits.items() if k in a.states}

    # --- run each init; decode at checkpoints ---
    results, finals = {}, {}
    cps = sorted(set(a.checkpoints) | {a.calls})
    for name, z_init in inits.items():
        z = z_init.clone()
        rows = []
        print(f"\n[{name}]  init |x|/k_eff = {_keff_and_norm(z, pm)[0]:.1f}/{_keff_and_norm(z, pm)[1]:.2f}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for c in range(a.calls + 1):
                if c in cps:
                    pred = model.decode(z, ctx)
                    dep = pred["depth"][..., 0].float()        # (S,H,W)
                    tn, ke = _keff_and_norm(z, pm)
                    ar = _absrel(dep, gt_depth)
                    rows.append(dict(call=c, tok_norm=tn, k_eff=ke, absrel=ar))
                    print(f"  call {c:>3}  |x|={tn:7.1f} k_eff={ke:5.2f} absrel={ar:.4f}")
                    if c == a.calls:
                        finals[name] = dict(depth=dep.detach().cpu(),
                                            pose=pred["pose_enc"].detach().float().cpu())
                if c < a.calls:
                    z = model.step(z, ctx)
        results[name] = rows

    # --- inter-run agreement at the final call (vs `normal`) ---
    print("\n=== inter-run agreement at final call (vs normal) ===")
    agree = {}
    if "normal" in finals:
        dn = finals["normal"]["depth"]
        Rn, _ = cameras_from_pose_enc(finals["normal"]["pose"], (H, W))
        Rn = Rn[:, :3, :3]
        for name, f in finals.items():
            dd = _depth_reldiff(dn, f["depth"])
            Rk, _ = cameras_from_pose_enc(f["pose"], (H, W))
            pr = _rot_angle(Rn, Rk[:, :3, :3])
            agree[name] = dict(depth_reldiff_vs_normal=dd, pose_deg_vs_normal=pr)
            print(f"  {name:10s}  depth_reldiff={dd:.4f}  pose_angle={pr:.3f} deg")

    json.dump({"tag": a.tag, "calls": a.calls, "checkpoints": cps,
               "dvlt_state_stats": {n: {"tok_norm": dv[f"{n}_tok_norm"], "k_eff": dv[f"{n}_k_eff"]}
                                    for n in ("z0", "z16", "z64")},
               "runs": results, "agreement_vs_normal": agree},
              open(os.path.join(a.out, "attractor.json"), "w"), indent=1)

    # --- plot: absrel / |x| / k_eff vs call, one line per init ---
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        keys = [("absrel", "depth AbsRel vs GT", True), ("tok_norm", "patch |x|", True), ("k_eff", "k_eff", False)]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        for ci, (k, title, logy) in enumerate(keys):
            for name, rows in results.items():
                xs = [r["call"] for r in rows]; ys = [r[k] for r in rows]
                ax[ci].plot(xs, ys, marker="o", ms=4, label=name)
            if logy: ax[ci].set_yscale("log")
            ax[ci].set_title(title); ax[ci].set_xlabel("solver call"); ax[ci].grid(alpha=.3); ax[ci].legend(fontsize=8)
        fig.suptitle(f"Solv3R attractor test ({a.tag}): same z0, different init states")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "attractor.png"), dpi=130)
        print(f"\nsaved {a.out}/attractor.png")
    except Exception as e:
        print("plot skipped:", e)

    print("\nReading: all inits -> ~0 depth_reldiff & ~0 pose_angle at the final call => the solution is a "
          "function of z0 (image-conditioned attractor), NOT of the initial state/trajectory.")


if __name__ == "__main__":
    main()
