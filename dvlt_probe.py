"""
dvlt_probe.py -- look inside Deja View's loop, then inject residuals.

Run from a GPU machine with dvlt installed (`pip install -e .[all]` in nv-tlabs/dvlt).

  python dvlt_probe.py traj    --input scene_dir --Ks 8 12 16 24 32
  python dvlt_probe.py ksweep  --input scene_dir
  python dvlt_probe.py perturb --input scene_dir --K 16 --at 2 8 14 --eps 0.05 0.2 --kind gauss
  python dvlt_probe.py kgt     --input scene_dir --gt gt.npz --Ks 4 8 12 16 20 24 32 64 --every 1
  python dvlt_probe.py sched   --input scene_dir --gt gt.npz --schedules baseline half reverse shuffle frozen double
  python dvlt_probe.py ablate  --input scene_dir --gt gt.npz --ablate none first_half second_half all
  python dvlt_probe.py attn    --input scene_dir --gt gt.npz --max-frames 6 --K 16
  python dvlt_probe.py flowdump --input scene_dir --K 16 --out probe_out   # -> probe_out/flow.json for dvlt_flow_viewer.html
  python dvlt_probe.py photo   --input scene_dir --K 16 --start 8 --eta 0.02 --iters 5

Pass --gt gt.npz (depth (S,h,w) [+ optional c2w (S,4,4)]) to any command to also report
depth AbsRel/d1 and camera-centre ATE against ground truth; without it the GT columns read "n/a".

Everything drives the loop by hand (same math as DVLTModel._solve_inference_linspace_k):
    ts = linspace(0,1,K); x = _interval_step(x, ts[i], ts[i+1] or 1.0, ...)
so we can read / modify x between steps.  NOT tested in the authoring sandbox (no GPU/torch).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from dvlt.common.constants import DataField
from dvlt.model.dvlt.model import DVLT, _slice_expand_flatten
from dvlt.model_components.pose_encoding import pose_enc_to_extri_intri
from dvlt.util.preprocess import load_sequence, preprocess_images


# ----------------------------------------------------------------------------- setup
class Probe:
    def __init__(self, ckpt, img_size=504, precision="bf16", max_frames=None,
            renorm=False, renorm_target=None, renorm_from=0):
        self.acc = Accelerator(mixed_precision=precision)
        self.dvlt = DVLT(img_size=img_size)
        self.dvlt.load_pretrained(ckpt, strict=True)
        self.dvlt.setup_test(self.acc)
        self.m = self.dvlt.model  # DVLTModel
        self.img_size = img_size
        self.max_frames = max_frames
        self.gates = {}
        # --- ground truth (set by main when --gt is given; None => GT columns read n/a) ---
        self.gt_depth = None
        self.gt_c2w = None
        self.gt_K = None
        # --- norm control (for the "fixed norm" experiment) ---
        self.renorm = renorm                # bool: enable per-step renorm of patch tokens
        self.renorm_target = renorm_target  # None => use |x| at step `renorm_from`; else fixed float
        self.renorm_from = renorm_from      # only renorm from this step index onward
        self._renorm_frozen = None          # captured target norm on first renorm step
        self._install_gate_hooks()


    # record mean of (s_attn, s_mlp, s_out) per sub-block on every step
    def _install_gate_hooks(self):
        blk = self.m.recurrent_blocks[0]
        for name, sub in (("frame", blk.frame_attn), ("global", blk.global_attn)):
            def hook(_mod, _inp, out, name=name):
                s_attn, s_mlp, s_out = out.float().chunk(3, dim=-1)
                self.gates[name] = (s_attn.mean().item(), s_mlp.mean().item(), s_out.mean().item())
            sub.depth_scale.register_forward_hook(hook)

    def load(self, path):
        _, frames = load_sequence(path)
        if self.max_frames:
            frames = frames[: self.max_frames]
        batch = preprocess_images(frames, img_size=self.img_size, patch_size=14, device=self.acc.device)
        return batch[DataField.IMAGES]  # (1,S,3,H,W) in [0,1]

    @torch.no_grad()
    def init_state(self, images):
        m = self.m
        B, S, _, H, W = images.shape
        z0 = m._encode_images(images)
        reg = m.register_token.expand(B, S, -1, -1).reshape(B * S, m.num_register_tokens, -1)
        cam = _slice_expand_flatten(m.camera_token, B, S)
        x = torch.cat([cam, reg, z0], dim=1)
        rope = m._get_rope_positions(B * S, H, W, images.device)
        return x, rope, (B, S, H, W)

    def step(self, x, i, K, rope, B, S):
        ts = torch.linspace(0.0, 1.0, K).tolist()
        t_now = ts[i]
        t_next = ts[i + 1] if i + 1 < K else 1.0
        x_new = self.m._interval_step(x, t_now, t_next, rope, B, S)

        # --- optional: renormalize patch tokens to a fixed norm ---
        # This freezes the "norm gauge" that otherwise grows monotonically with steps.
        if self.renorm and i >= self.renorm_from:
            pm = self.m.patch_start_idx
            patch = x_new[:, pm:].float()
            cur = patch.norm(dim=-1, keepdim=True)               # (BS, Npatch, 1)

            # decide target norm: capture once on first renorm step, or use fixed value
            if self.renorm_target is not None:
                target = self.renorm_target
            else:
                if self._renorm_frozen is None:
                    self._renorm_frozen = cur.mean().item()      # scalar, from the first renorm step
                target = self._renorm_frozen

            patch = patch / cur.clamp_min(1e-6) * target
            x_new = x_new.clone()
            x_new[:, pm:] = patch.to(x_new.dtype)
        return x_new

    def step_t(self, x, t_now, t_next, rope, B, S):
        """One step at an EXPLICIT (t_now, t_next) interval, bypassing the linspace
        schedule (and renorm). Used by cmd_sched to feed arbitrary time tables, so we
        can test whether the loop is a time-conditioned flow or a fixed-point iteration.
        """
        return self.m._interval_step(x, float(t_now), float(t_next), rope, B, S)

    def step_ablate(self, x, i, K, rope, B, S, ablate_global=False):
        """One normal step, but optionally with the GLOBAL (cross-view) attention disabled
        for this step -- the global sub-block is temporarily replaced by identity, so only
        the within-view frame attention runs. Lets cmd_ablate ask when cross-view
        information is written into the state. dvlt source is untouched (we patch the
        instance's bound forward and restore it).
        """
        if not ablate_global:
            return self.step(x, i, K, rope, B, S)
        g = self.m.recurrent_blocks[0].global_attn
        orig = g.forward
        g.forward = lambda x, *args, **kwargs: x   # identity: cross-view comm removed
        try:
            return self.step(x, i, K, rope, B, S)
        finally:
            g.forward = orig

    def decode(self, x, rope, dims):
        B, S, H, W = dims
        return self.m._decode(x, H, W, B, S, rope)


# ----------------------------------------------------------------------------- metrics
def log_depth_dist(d, d_ref):
    """median |log d - log d_ref| after median-scale alignment (scale is a free gauge)."""
    l = torch.log(d.clamp_min(1e-6)).flatten()
    r = torch.log(d_ref.clamp_min(1e-6)).flatten()
    l = l - (l - r).median()
    return (l - r).abs().median().item()


def ray_angle_dist(r, r_ref):
    """mean angle (deg) between predicted ray directions."""
    a = F.normalize(r[..., :3].float(), dim=-1)
    b = F.normalize(r_ref[..., :3].float(), dim=-1)
    return torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1, 1))).mean().item()


def out_dist(p, p_ref):
    return dict(depth=log_depth_dist(p["depth"], p_ref["depth"]), ray=ray_angle_dist(p["rays"], p_ref["rays"]))


def patch_part(x, m):
    return x[:, m.patch_start_idx:]


def gt_metrics(P, pred):
    """vs ground truth (only if --gt given): depth AbsRel/d1 after median-scale alignment, camera-centre ATE.

    Returns None when no GT is loaded, so every command stays backward compatible without --gt.
    """
    if getattr(P, "gt_depth", None) is None:
        return None
    d = pred["depth"].to(P.gt_depth.device)
    S = d.shape[1]
    r = depth_vs_gt(d, P.gt_depth[:S])
    if P.gt_c2w is not None:
        o = pred["rays"].to(P.gt_depth.device)[0, :, :, :, 3:6].mean(dim=(1, 2))
        r["ate_rel"] = sim3_ate(o, P.gt_c2w[:S, :3, 3])
    return r


def fmt_gt(g):
    return "  GT: n/a" if g is None else f"  GT AbsRel={g['absrel']:.4f} d1={g['d1']:.3f}" + (f" ATE={g['ate_rel']:.4f}" if "ate_rel" in g else "")


# ----------------------------------------------------------------------------- exp 1: trajectory
@torch.no_grad()
def run_traj(P, images, K, decode=True):
    x, rope, dims = P.init_state(images)
    B, S, H, W = dims
    rec = []
    prev_delta = None
    for i in range(K):
        x_new = P.step(x, i, K, rope, B, S)
        delta = (x_new - x).float()
        pn = patch_part(x_new, P.m).float().norm(dim=-1)            # (BS, Npatch)
        dn = patch_part(delta, P.m).norm(dim=-1)                     # absolute update norm per token
        r = dict(
            step=i,
            tok_norm=pn.mean().item(),
            cam_norm=x_new[:, 0].float().norm(dim=-1).mean().item(),
            upd_abs=dn.mean().item(),
            upd_rel=(dn / pn.clamp_min(1e-6)).mean().item(),
            gates={k: v for k, v in P.gates.items()},
        )
        if prev_delta is not None:
            r["upd_cos_prev"] = F.cosine_similarity(delta.flatten(1), prev_delta.flatten(1), dim=-1).mean().item()
        # direction vs norm split: how much of the change is rotation vs scaling
        xo, xn = patch_part(x, P.m).float(), patch_part(x_new, P.m).float()
        r["tok_cos_step"] = F.cosine_similarity(xo, xn, dim=-1).mean().item()
        # channel concentration (how many channels carry the energy -- low = collapsed)
        e = xn.pow(2).mean(dim=(0, 1))
        tot = e.sum().clamp_min(1e-9)
        r["top1_share"] = (e.max() / tot).item()
        r["top8_share"] = (e.topk(8).values.sum() / tot).item()
        r["eff_chan"] = ((e.sum() ** 2) / e.pow(2).sum().clamp_min(1e-12)).item()
        # inter-token clustering: mean pairwise direction cosine over a token subsample (Geshkovski clustering)
        u = _unit(xn).reshape(-1, xn.shape[-1])
        sub = u[torch.randperm(u.shape[0], device=u.device)[:512]]
        npair = sub.shape[0] * (sub.shape[0] - 1) / 2
        r["cluster_cos"] = ((sub @ sub.t()).triu(1).sum() / max(npair, 1)).item()
        r["_upd_map"] = dn.reshape(B * S, -1).cpu().numpy()          # for heatmaps
        if decode:
            r["_pred"] = {k: v.float().cpu() for k, v in P.decode(x_new, rope, dims).items() if k in ("depth", "rays")}
        rec.append(r)
        prev_delta, x = delta, x_new
    return rec, x, rope, dims


def cmd_traj(P, a):
    images = P.load(a.input)
    ref_rec, _, _, _ = run_traj(P, images, a.ref_K)
    ref = {k: v.to(P.acc.device) for k, v in ref_rec[-1]["_pred"].items()}
    out, maps = {}, {}
    for K in a.Ks:
        rec, *_ = run_traj(P, images, K)
        final = rec[-1]["_pred"]
        rows = []
        for r in rec:
            p = r.pop("_pred")
            maps[f"K{K}_s{r['step']}"] = r.pop("_upd_map")
            r["dist_to_own_final"] = out_dist(p, final)
            r["dist_to_ref"] = out_dist(p, {k: v.cpu() for k, v in ref.items()})
            r["gt"] = gt_metrics(P, p)
            rows.append(r)
        out[f"K{K}"] = rows
        print(f"\nK={K}")
        print("step  |x|    |dx|abs |dx|rel cos(dx,dx_prev) cos(x,x')  s_out(f/g)    d->final(depth)  d->ref(depth)")
        for r in rows:
            g = r["gates"]
            print(f"{r['step']:>3}  {r['tok_norm']:6.1f} {r['upd_abs']:7.2f} {r['upd_rel']:7.3f} "
                  f"{r.get('upd_cos_prev', float('nan')):12.3f} {r['tok_cos_step']:10.4f}  "
                  f"{g['frame'][2]:.3f}/{g['global'][2]:.3f}   {r['dist_to_own_final']['depth']:.4f}      {r['dist_to_ref']['depth']:.4f}" + fmt_gt(r["gt"]))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/traj.json", "w"), indent=1)
    np.savez_compressed(f"{a.out}/upd_maps.npz", **maps)


# ----------------------------------------------------------------------------- exp 2: K sweep (incl. beyond 16)
def cmd_ksweep(P, a):
    images = P.load(a.input)
    ref_rec, *_ = run_traj(P, images, a.ref_K)
    ref = ref_rec[-1]["_pred"]
    rows = []
    for K in a.Ks:
        rec, *_ = run_traj(P, images, K, decode=False)
        # decode only the final state
        x, rope, dims = P.init_state(images)
        B, S, H, W = dims
        with torch.no_grad():
            for i in range(K):
                x = P.step(x, i, K, rope, B, S)
            p = {k: v.float().cpu() for k, v in P.decode(x, rope, dims).items() if k in ("depth", "rays")}
        d = out_dist(p, ref)
        gm = gt_metrics(P, p)
        rows.append(dict(K=K, tok_norm=rec[-1]["tok_norm"], gt=gm, **d))
        print(f"K={K:>3}  |x|={rec[-1]['tok_norm']:7.1f}  depth_dist_to_K{a.ref_K}={d['depth']:.4f}  ray_deg={d['ray']:.3f}" + fmt_gt(gm))
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(f"{a.out}/ksweep.json", "w"), indent=1)


# ----------------------------------------------------------------------------- exp 3: perturb mid-loop
def make_perturbation(x, kind, eps, P, gen):
    pm = P.m.patch_start_idx
    xp = x.clone()
    patch = xp[:, pm:]
    if kind == "gauss":        # isotropic noise, relative to per-token norm
        n = torch.randn(patch.shape, generator=gen, device=x.device, dtype=torch.float32).to(x.dtype)
        n = F.normalize(n, dim=-1) * patch.float().norm(dim=-1, keepdim=True).to(x.dtype)
        xp[:, pm:] = patch + eps * n
    elif kind == "scale":      # pure norm change (direction untouched): tests the norm-gauge hypothesis
        xp[:, pm:] = patch * (1.0 + eps)
    elif kind == "dir":        # pure direction change at fixed norm
        n = torch.randn(patch.shape, generator=gen, device=x.device, dtype=torch.float32).to(x.dtype)
        n = F.normalize(n, dim=-1) * patch.float().norm(dim=-1, keepdim=True).to(x.dtype)
        v = F.normalize((patch + eps * n).float(), dim=-1) * patch.float().norm(dim=-1, keepdim=True)
        xp[:, pm:] = v.to(x.dtype)
    elif kind == "swap":       # replace frame 0's tokens by frame 1's: structured, wrong-view error
        xp[0, pm:] = x[1, pm:]
    elif kind == "cam":        # gauss noise on ONLY the camera token (idx 0); patches untouched
        tok = xp[:, 0:1]
        n = torch.randn(tok.shape, generator=gen, device=x.device, dtype=torch.float32).to(x.dtype)
        n = F.normalize(n, dim=-1) * tok.float().norm(dim=-1, keepdim=True).to(x.dtype)
        xp[:, 0:1] = tok + eps * n
    elif kind == "register":   # gauss noise on ONLY the register tokens (idx 1:pm); patches untouched
        tok = xp[:, 1:pm]
        n = torch.randn(tok.shape, generator=gen, device=x.device, dtype=torch.float32).to(x.dtype)
        n = F.normalize(n, dim=-1) * tok.float().norm(dim=-1, keepdim=True).to(x.dtype)
        xp[:, 1:pm] = tok + eps * n
    else:
        raise ValueError(kind)
    return xp


@torch.no_grad()
def cmd_perturb(P, a):
    images = P.load(a.input)
    gen = torch.Generator(device=images.device).manual_seed(0)
    K = a.K
    x0, rope, dims = P.init_state(images)
    B, S, H, W = dims
    # clean trajectory states
    clean = [x0]
    for i in range(K):
        clean.append(P.step(clean[-1], i, K, rope, B, S))
    clean_pred = {k: v.float().cpu() for k, v in P.decode(clean[-1], rope, dims).items() if k in ("depth", "rays")}
    gt_clean = gt_metrics(P, clean_pred)
    print("clean final:" + fmt_gt(gt_clean))
    rows = []
    for j in a.at:
        for eps in a.eps:
            xp = make_perturbation(clean[j], a.kind, eps, P, gen)
            # damage right after injection, measured in decoded space and token space
            p_inj = {k: v.float().cpu() for k, v in P.decode(xp, rope, dims).items() if k in ("depth", "rays")}
            p_clean_j = {k: v.float().cpu() for k, v in P.decode(clean[j], rope, dims).items() if k in ("depth", "rays")}
            dmg0 = out_dist(p_inj, p_clean_j)
            tok0 = (patch_part(xp - clean[j], P.m).float().norm(dim=-1) / patch_part(clean[j], P.m).float().norm(dim=-1)).mean().item()
            x = xp
            for i in range(j, K):
                x = P.step(x, i, K, rope, B, S)
            pK = {k: v.float().cpu() for k, v in P.decode(x, rope, dims).items() if k in ("depth", "rays")}
            dmgK = out_dist(pK, clean_pred)
            gtK = gt_metrics(P, pK)
            tokK = (patch_part(x - clean[K], P.m).float().norm(dim=-1) / patch_part(clean[K], P.m).float().norm(dim=-1)).mean().item()
            rows.append(dict(at=j, eps=eps, kind=a.kind, tok_rel_inj=tok0, tok_rel_final=tokK,
                             depth_inj=dmg0["depth"], depth_final=dmgK["depth"],
                             ray_inj=dmg0["ray"], ray_final=dmgK["ray"],
                             recovery_tok=tokK / max(tok0, 1e-9), gt_final=gtK, gt_clean=gt_clean))
            print(f"inject@{j:>2} {a.kind} eps={eps:<5} token-rel {tok0:.3f}->{tokK:.3f}  "
                  f"depth {dmg0['depth']:.4f}->{dmgK['depth']:.4f}  ray {dmg0['ray']:.3f}->{dmgK['ray']:.3f}" + fmt_gt(gtK))
    print("\nReading: ratio << 1 => loop self-corrects (contraction); ~1 => perturbation is just carried; >1 => amplified.")
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(f"{a.out}/perturb_{a.kind}.json", "w"), indent=1)


# ----------------------------------------------------------------------------- exp 4: residual injection (photometric)
def camera_from_pred(pred, dims, convention):
    B, S, H, W = dims
    E, Kmat = pose_enc_to_extri_intri(pred["pose_enc"].float(), (H, W))
    R, t = E[..., :3], E[..., 3]
    return R, t, Kmat  # (B,S,3,3),(B,S,3),(B,S,3,3)


def detect_convention(pred, dims):
    """c2w if camera translation == ray origin of that view, else w2c."""
    R, t, _ = camera_from_pred(pred, dims, None)
    o = pred["rays"][..., 3:6].float().mean(dim=(2, 3))  # (B,S,3)
    err_c2w = (t - o).norm(dim=-1).mean()
    err_w2c = (-(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1) - o).norm(dim=-1).mean()
    return "c2w" if err_c2w <= err_w2c else "w2c"


def photometric_residual(pred, images, dims, convention, pairs=None, stride=2):
    """
    r_ij(u) = I_i(u) - I_j(pi_j(P_i(u))),   P_i = o_i + d_i * depth_i   (all differentiable in x).
    Returns scalar robust loss and the per-pixel residual maps (for use as innovation / gating).
    """
    B, S, H, W = dims
    I = images.float()[0]                                         # (S,3,H,W)
    rays, depth = pred["rays"].float()[0], pred["depth"].float()[0]  # (S,H,W,6),(S,H,W,1)
    Pw = rays[..., 3:6] + rays[..., :3] * depth                   # (S,H,W,3) world points
    R, t, Kmat = camera_from_pred(pred, dims, convention)
    R, t, Kmat = R[0], t[0], Kmat[0]
    if pairs is None:
        pairs = [(i, (i + 1) % S) for i in range(S)] + [(i, (i - 1) % S) for i in range(S)]
    losses, maps = [], {}
    for i, j in pairs:
        pts = Pw[i, ::stride, ::stride].reshape(-1, 3)
        if convention == "c2w":
            Xc = (pts - t[j]) @ R[j]                              # R^T (P - t)
        else:
            Xc = pts @ R[j].T + t[j]
        z = Xc[:, 2:3]
        uv = (Xc @ Kmat[j].T)[:, :2] / z.clamp_min(1e-3)
        gx = uv[:, 0] / (W - 1) * 2 - 1
        gy = uv[:, 1] / (H - 1) * 2 - 1
        grid = torch.stack([gx, gy], -1).reshape(1, H // stride + (H % stride > 0), W // stride + (W % stride > 0), 2)
        warped = F.grid_sample(I[j:j + 1], grid, align_corners=True, padding_mode="zeros")
        ref = I[i:i + 1, :, ::stride, ::stride]
        valid = ((grid.abs() <= 1).all(-1) & (z.reshape(grid.shape[:3]) > 1e-3)).float().unsqueeze(1)
        res = (warped - ref)
        charb = torch.sqrt(res ** 2 + 1e-4) * valid
        losses.append(charb.sum() / valid.sum().clamp_min(1.0) / 3)
        maps[(i, j)] = (res.detach(), valid.detach())
    return torch.stack(losses).mean(), maps


def _pred_f(P, x, rope, dims):
    p = P.decode(x, rope, dims)
    return {k: (v.float() if torch.is_tensor(v) else v) for k, v in p.items()}


def cmd_photo(P, a):
    images = P.load(a.input)
    K = a.K
    x, rope, dims = P.init_state(images)
    B, S, H, W = dims

    # reference: standard K-step output (what the released model gives)
    with torch.no_grad():
        xs = [x]
        for i in range(K):
            xs.append(P.step(xs[-1], i, K, rope, B, S))
        ref_pred = {k: v.float().cpu() for k, v in _pred_f(P, xs[-1], rope, dims).items() if k in ("depth", "rays")}
        conv = detect_convention(_pred_f(P, xs[-1], rope, dims), dims)
    print(f"camera pose_enc convention detected: {conv}")

    # (A) monitor only: does the photometric residual track distance-to-endpoint along the trajectory?
    print("\n[A] monitor: photo residual along the standard loop")
    with torch.no_grad():
        monitor = []
        for i, xi in enumerate(xs):
            if i == 0:
                continue
            pr = _pred_f(P, xi, rope, dims)
            L, _ = photometric_residual(pr, images, dims, conv, stride=a.stride)
            d = out_dist({k: v.cpu() for k, v in pr.items() if k in ("depth", "rays")}, ref_pred)["depth"]
            gm = gt_metrics(P, {k: v for k, v in pr.items() if k in ("depth", "rays")})
            monitor.append((i, L.item(), d, gm))
            print(f"  step {i:>2}  L_photo={L.item():.5f}  depth_dist_to_final={d:.4f}" + fmt_gt(gm))

    # (B) gradient injection: after the standard loop up to `start`, take eta-sized steps on x down grad L_photo
    #     (a "latent Gauss-Newton-lite"): x <- x - eta*|x| * g/|g|, g = dL_photo/dx through decoders.
    print(f"\n[B] gradient injection from step {a.start}: eta={a.eta}, iters={a.iters}, "
          f"{'finish loop afterwards' if a.finish else 'no further loop steps'}")
    rows = []
    for mode in ("none", "grad"):
        xcur = xs[a.start].detach().clone()
        hist = []
        if mode == "grad":
            for it in range(a.iters):
                xcur.requires_grad_(True)
                with torch.enable_grad(), P.acc.autocast():
                    pr = _pred_f(P, xcur, rope, dims)
                    L, _ = photometric_residual(pr, images, dims, conv, stride=a.stride)
                    g, = torch.autograd.grad(L, xcur)
                with torch.no_grad():
                    gn = g.float().norm().clamp_min(1e-12)
                    step = a.eta * xcur.float().norm() * (g.float() / gn)
                    xcur = (xcur.float() - step).to(xs[a.start].dtype).detach()
                hist.append(L.item())
        # optionally run the remaining standard steps
        if a.finish:
            with torch.no_grad():
                for i in range(a.start, K):
                    xcur = P.step(xcur, i, K, rope, B, S)
        with torch.no_grad():
            pr = _pred_f(P, xcur, rope, dims)
            L, _ = photometric_residual(pr, images, dims, conv, stride=a.stride)
            d = out_dist({k: v.cpu() for k, v in pr.items() if k in ("depth", "rays")}, ref_pred)
            tn = patch_part(xcur, P.m).float().norm(dim=-1).mean().item()
        gm = gt_metrics(P, {k: v for k, v in pr.items() if k in ("depth", "rays")})
        rows.append(dict(mode=mode, gt=gm, L_photo=L.item(), depth_dist=d["depth"], ray_dist=d["ray"], tok_norm=tn, L_hist=hist))
        print(f"  {mode:>5}: L_photo={L.item():.5f}  depth_dist_to_ref={d['depth']:.4f}  ray_deg={d['ray']:.3f}  |x|={tn:.1f}  L_hist={['%.5f'%h for h in hist]}" + fmt_gt(gm))
    print("\nReading: L_photo down AND depth/ray moving toward ref => residual gradient is a useful correction direction;"
          "\n         L_photo down but geometry drifting / |x| inflating => the network lets you fit photometry in a degenerate way"
          "\n         (norm gauge or view-dependent appearance) -> need norm control / robust kernel / pose-consistent residual.")
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(monitor=monitor, injection=rows, convention=conv), open(f"{a.out}/photo.json", "w"), indent=1)


# ----------------------------------------------------------------------------- exp 5: K sweep vs ground truth, every step
def load_gt(path, img_size, patch=14, device="cpu"):
    """
    GT npz with  depth:      (S,h,w) metric/any-scale depth, 0 or nan = invalid  [required]
                 c2w:        (S,4,4) camera-to-world                             [optional -> ATE]
                 intrinsics: (3,3) pinhole K at (h,w)                            [optional -> attn corr]
    Depth is pushed through the SAME pipeline as preprocess_images (longest side -> img_size,
    centre crop to patch multiple), NEAREST. Intrinsics are rescaled/cropped to match.
    Returns (depth, c2w, K) all at the model input resolution. Different aspect ratios across
    frames are not handled.
    """
    g = np.load(path)
    d = torch.from_numpy(np.nan_to_num(g["depth"].astype(np.float32), nan=0.0))   # (S,h,w)
    S, h, w = d.shape
    sc = img_size / max(h, w)
    nh, nw = int(round(h * sc)), int(round(w * sc))
    d = F.interpolate(d[:, None], size=(nh, nw), mode="nearest")[:, 0]
    ch, cw = nh // patch * patch, nw // patch * patch
    top, left = (nh - ch) // 2, (nw - cw) // 2
    d = d[:, top:top + ch, left:left + cw]
    c2w = torch.from_numpy(g["c2w"].astype(np.float32)) if "c2w" in g.files else None
    K = None
    if "intrinsics" in g.files:
        K = torch.from_numpy(g["intrinsics"].astype(np.float32)).clone()  # at (h,w)
        K[0, 0] *= sc; K[1, 1] *= sc                       # fx, fy
        K[0, 2] = K[0, 2] * sc - left                      # cx (resize then crop)
        K[1, 2] = K[1, 2] * sc - top                       # cy
    return (d.to(device),
            c2w.to(device) if c2w is not None else None,
            K.to(device) if K is not None else None)


def depth_vs_gt(pred_depth, gt):
    """AbsRel / delta<1.25 after per-sequence median-scale alignment (scale is a free gauge)."""
    p = pred_depth.float().reshape(gt.shape)
    m = gt > 1e-6
    if m.sum() < 10:
        return dict(absrel=float("nan"), d1=float("nan"), scale=float("nan"))
    sc = (gt[m] / p[m].clamp_min(1e-6)).median()
    q = (p * sc)[m]
    g = gt[m]
    absrel = ((q - g).abs() / g).mean().item()
    d1 = (torch.maximum(q / g, g / q) < 1.25).float().mean().item()
    return dict(absrel=absrel, d1=d1, scale=sc.item())


def sim3_ate(pred_c, gt_c):
    """RMSE of camera centres after Umeyama Sim(3) alignment. (S,3) each; needs S>=3."""
    p, g = pred_c.double().cpu().numpy(), gt_c.double().cpu().numpy()
    if len(p) < 3:
        return float("nan")
    mp, mg = p.mean(0), g.mean(0)
    P_, G_ = p - mp, g - mg
    U, D, Vt = np.linalg.svd(G_.T @ P_ / len(p))
    Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sg[2, 2] = -1
    R = U @ Sg @ Vt
    var = (P_ ** 2).sum() / len(p)
    s = (D * np.diag(Sg)).sum() / max(var, 1e-12)
    t = mg - s * R @ mp
    err = g - (s * (R @ p.T).T + t)
    return float(np.sqrt((err ** 2).sum(1).mean()) / max(np.linalg.norm(G_, axis=1).mean(), 1e-9))  # relative to trajectory extent


@torch.no_grad()
def cmd_kgt(P, a):
    assert a.gt, "kgt requires --gt gt.npz (depth (S,h,w) [+ optional c2w (S,4,4)])"
    images = P.load(a.input)
    B, S, _, H, W = images.shape
    gt_depth, gt_c2w, _ = load_gt(a.gt, P.img_size, device=images.device)
    gt_depth = gt_depth[:S]
    assert gt_depth.shape[-2:] == (H, W), f"GT depth {tuple(gt_depth.shape[-2:])} != model input {(H, W)} (aspect mismatch?)"

    # reference: the K=ref_K endpoint (self-reference), so every step is compared BOTH to
    # ground truth (depth_vs_gt) AND to the model's own K=ref_K final output (out_dist).
    xr, rope, dims = P.init_state(images)
    for i in range(a.ref_K):
        xr = P.step(xr, i, a.ref_K, rope, B, S)
    ref = {k: v.float() for k, v in P.decode(xr, rope, dims).items() if k in ("depth", "rays")}
    print(f"reference = K={a.ref_K} endpoint (self-reference, alongside GT)")

    rows = []
    for K in a.Ks:
        x, rope, dims = P.init_state(images)
        prev = None
        for i in range(K):
            x = P.step(x, i, K, rope, B, S)
            last = i == K - 1
            if not (last or (i + 1) % a.every == 0 or i == 0):
                continue
            pr = {k: v.float() for k, v in P.decode(x, rope, dims).items() if k in ("depth", "rays")}
            r = dict(K=K, step=i + 1, t=(i + 1) / K, tok_norm=patch_part(x, P.m).float().norm(dim=-1).mean().item())
            # (a) vs ground truth
            r.update(depth_vs_gt(pr["depth"], gt_depth))
            if gt_c2w is not None:
                r["ate_rel"] = sim3_ate(pr["rays"][0, :, :, :, 3:6].mean(dim=(1, 2)), gt_c2w[:S, :3, 3])
            # (b) vs the model's own K=ref_K endpoint (self-reference)
            dref = out_dist(pr, ref)
            r["to_ref_depth"], r["to_ref_ray"] = dref["depth"], dref["ray"]
            if prev is not None:
                r["depth_change"] = log_depth_dist(pr["depth"], prev)
            prev = pr["depth"]
            rows.append(r)
            print(f"K={K:>3} step {i+1:>3}/{K:<3} |x|={r['tok_norm']:7.1f}  "
                  f"GT: AbsRel={r['absrel']:.4f} d1={r['d1']:.3f} ATE={r.get('ate_rel', float('nan')):.4f}  "
                  f"| K{a.ref_K}: depth={r['to_ref_depth']:.4f} ray={r['to_ref_ray']:.3f}")
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(f"{a.out}/kgt.json", "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(21, 4.2))
        for K in a.Ks:
            rr = [r for r in rows if r["K"] == K]
            ax[0].plot([r["t"] for r in rr], [r["absrel"] for r in rr], marker="o", ms=3, label=f"K={K}")
            ax[1].plot([r["t"] for r in rr], [r["tok_norm"] for r in rr], marker="o", ms=3, label=f"K={K}")
            if gt_c2w is not None:
                ax[2].plot([r["t"] for r in rr], [r["ate_rel"] for r in rr], marker="o", ms=3, label=f"K={K}")
            ax[3].plot([r["t"] for r in rr], [r["to_ref_depth"] for r in rr], marker="o", ms=3, label=f"K={K}")
        ax[0].set_title("depth AbsRel vs GT (intermediate x)"); ax[1].set_title("mean patch-token norm")
        ax[2].set_title("camera-centre ATE / extent (Sim3)")
        ax[3].set_title(f"depth dist -> K={a.ref_K} endpoint (self-ref)")
        for x_ in ax: x_.set_xlabel("t = step / K"); x_.legend(fontsize=7)
        ax[0].set_yscale("log"); plt.tight_layout(); plt.savefig(f"{a.out}/kgt.png", dpi=130)
        print(f"saved {a.out}/kgt.png")
    except Exception as e:
        print("plot skipped:", e)
    print("\nReading: AbsRel(step) per K. If curves for different K lie on top of each other in t=step/K -> one fixed trajectory, K only sets"
          "\n         resolution. If the final AbsRel is U-shaped in K (best near 16, worse for 24/32/64) -> out-of-range steps hurt (norm/step-size shift)."
          "\n         Intermediate points decode x through heads trained only on the final state, so early AbsRel is expected to be large.")


# ----------------------------------------------------------------------------- exp 6: time-schedule tricks (transport vs optimization)
def build_schedule(name, K, frozen_t, steps, seed=0, mod_at=8, mod_n=2):
    """Return a list of (t_now, t_next) pairs for a named time schedule.

    baseline  standard linspace(0,1,K) grid -- what the model normally runs.
    half      only the first half of the K-step table (stop around t=0.5).
    reverse   the K pairs in reverse order (late-stage t first).
    shuffle   the K pairs in random order (seeded).
    frozen    the SAME (frozen_t, frozen_t) pair, repeated `steps` times -> fixed-point test.
    double    claim each step is twice as large (t_next jumps 2*dt) -> step-size mismatch.
    skip      baseline with the step at index `mod_at` dropped -> is each step necessary?
    repeat    baseline with the step at index `mod_at` applied `mod_n` times -> does an
              extra application of one step help or hurt (Euler-style over-stepping)?
    """
    ts = torch.linspace(0.0, 1.0, K).tolist()
    base = [(ts[i], ts[i + 1] if i + 1 < K else 1.0) for i in range(K)]
    if name == "baseline":
        return base
    if name == "half":
        return base[: max(1, K // 2)]
    if name == "reverse":
        return base[::-1]
    if name == "shuffle":
        import random
        s = list(base)
        random.Random(seed).shuffle(s)
        return s
    if name == "frozen":
        return [(frozen_t, frozen_t)] * steps
    if name == "double":
        dt = 1.0 / (K - 1) if K > 1 else 1.0
        return [(ts[i], min(1.0, ts[i] + 2 * dt)) for i in range(K)]
    if name == "skip":
        return [p for i, p in enumerate(base) if i != mod_at]
    if name == "repeat":
        out = []
        for i, p in enumerate(base):
            out.append(p)
            if i == mod_at:
                out.extend([p] * (mod_n - 1))
        return out
    raise ValueError(f"unknown schedule '{name}'")


@torch.no_grad()
def cmd_sched(P, a):
    """Feed hand-made (t_now, t_next) schedules through the loop and watch, per step,
    how depth/pose track ground truth. Tests whether t is a genuine 'where am I on the
    trajectory' signal (reverse/shuffle should break) or an ignorable bias, and whether
    a frozen t converges to a fixed point (iteration) or keeps drifting (flow).
    """
    images = P.load(a.input)
    B, S, _, H, W = images.shape
    has_gt = P.gt_depth is not None
    out = {}
    for name in a.schedules:
        sched = build_schedule(name, a.sched_K, a.frozen_t, a.sched_steps, seed=0,
                               mod_at=a.mod_at, mod_n=a.mod_n)
        x, rope, dims = P.init_state(images)
        rows, prev = [], None
        tag = (f"  (frozen_t={a.frozen_t})" if name == "frozen"
               else f"  (@{a.mod_at}" + (f" x{a.mod_n}" if name == "repeat" else "") + ")"
               if name in ("skip", "repeat") else "")
        print(f"\n[{name}]  {len(sched)} steps{tag}")
        for i, (t_now, t_next) in enumerate(sched):
            x = P.step_t(x, t_now, t_next, rope, B, S)
            last = i == len(sched) - 1
            if not (last or (i + 1) % a.every == 0 or i == 0):
                continue
            pr = {k: v.float() for k, v in P.decode(x, rope, dims).items() if k in ("depth", "rays")}
            r = dict(step=i + 1, t_now=round(t_now, 4), t_next=round(t_next, 4),
                     tok_norm=patch_part(x, P.m).float().norm(dim=-1).mean().item())
            if has_gt:
                r.update(depth_vs_gt(pr["depth"], P.gt_depth[:S]))
                if P.gt_c2w is not None:
                    r["ate_rel"] = sim3_ate(pr["rays"][0, :, :, :, 3:6].mean(dim=(1, 2)), P.gt_c2w[:S, :3, 3])
            if prev is not None:
                r["depth_change"] = log_depth_dist(pr["depth"], prev)
            prev = pr["depth"]
            rows.append(r)
            gt_str = (f"AbsRel={r['absrel']:.4f} d1={r['d1']:.3f} "
                      f"ATE={r.get('ate_rel', float('nan')):.4f}") if has_gt else "GT: n/a"
            print(f"  step {i+1:>3}/{len(sched):<3} t=({r['t_now']:.3f},{r['t_next']:.3f}) "
                  f"|x|={r['tok_norm']:7.1f}  {gt_str}  d(depth)/prev={r.get('depth_change', float('nan')):.4f}")
        out[name] = rows
    # final-state comparison table
    if has_gt:
        print("\n=== final state per schedule (vs GT) ===")
        print("schedule   steps   |x|     AbsRel   d1      ATE")
        for name, rows in out.items():
            if rows:
                r = rows[-1]
                print(f"{name:<10} {r['step']:>5}  {r['tok_norm']:7.1f}  {r['absrel']:.4f}  {r['d1']:.3f}  "
                      f"{r.get('ate_rel', float('nan')):.4f}")
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/sched.json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_ax = 2 + (1 if has_gt and any("ate_rel" in r for rows in out.values() for r in rows) else 0)
        fig, ax = plt.subplots(1, n_ax, figsize=(5.3 * n_ax, 4.2))
        for name, rows in out.items():
            xs = [r["step"] for r in rows]
            if has_gt:
                ax[0].plot(xs, [r["absrel"] for r in rows], marker="o", ms=3, label=name)
            ax[1].plot(xs, [r["tok_norm"] for r in rows], marker="o", ms=3, label=name)
            if n_ax == 3:
                ax[2].plot(xs, [r.get("ate_rel", float("nan")) for r in rows], marker="o", ms=3, label=name)
        ax[0].set_title("depth AbsRel vs GT" if has_gt else "(no GT)")
        if has_gt:
            ax[0].set_yscale("log")
        ax[1].set_title("mean patch-token norm |x|")
        if n_ax == 3:
            ax[2].set_title("camera-centre ATE / extent")
        for x_ in ax:
            x_.set_xlabel("step"); x_.legend(fontsize=7); x_.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{a.out}/sched.png", dpi=130)
        print(f"saved {a.out}/sched.png")
    except Exception as e:
        print("plot skipped:", e)
    print("\nReading: reverse/shuffle collapsing => t carries real 'stage' info (not an ignorable bias)."
          "\n         frozen t: |x| and AbsRel settling => fixed-point iteration; drifting => time-dependent flow."
          "\n         half still accurate at its end => model learned a direction field to the endpoint, not a fixed timetable.")


# ----------------------------------------------------------------------------- exp 7: global-attention ablation (where is cross-view info written?)
@torch.no_grad()
def cmd_ablate(P, a):
    """Disable the GLOBAL (cross-view) attention at chosen steps and watch, per step,
    how depth and camera track GT. Frame attention is always kept, so this isolates the
    one channel that carries multi-view information. Hypothesis: pose/camera depends on
    EARLY global attention (views must agree on a frame first); depth refines later and
    tolerates late ablation better. Reveals when cross-view info is written in.
    """
    images = P.load(a.input)
    B, S, _, H, W = images.shape
    K = a.K
    has_gt = P.gt_depth is not None
    # named conditions -> which step indices have global attention OFF
    def steps_for(name):
        if name == "none":        return set()
        if name == "all":         return set(range(K))
        if name == "first_half":  return set(range(K // 2))
        if name == "second_half": return set(range(K // 2, K))
        raise ValueError(f"unknown ablate condition '{name}'")

    out = {}
    for name in a.ablate:
        ab = steps_for(name)
        x, rope, dims = P.init_state(images)
        rows, prev = [], None
        print(f"\n[{name}]  global-attn OFF at {len(ab)}/{K} steps")
        for i in range(K):
            x = P.step_ablate(x, i, K, rope, B, S, ablate_global=(i in ab))
            last = i == K - 1
            if not (last or (i + 1) % a.every == 0 or i == 0):
                continue
            pr = {k: v.float() for k, v in P.decode(x, rope, dims).items() if k in ("depth", "rays")}
            r = dict(step=i + 1, tok_norm=patch_part(x, P.m).float().norm(dim=-1).mean().item())
            if has_gt:
                r.update(depth_vs_gt(pr["depth"], P.gt_depth[:S]))
                if P.gt_c2w is not None:
                    r["ate_rel"] = sim3_ate(pr["rays"][0, :, :, :, 3:6].mean(dim=(1, 2)), P.gt_c2w[:S, :3, 3])
            if prev is not None:
                r["depth_change"] = log_depth_dist(pr["depth"], prev)
            prev = pr["depth"]
            rows.append(r)
            gt_str = (f"AbsRel={r['absrel']:.4f} d1={r['d1']:.3f} ATE={r.get('ate_rel', float('nan')):.4f}"
                      if has_gt else "GT: n/a")
            print(f"  step {i+1:>3}/{K:<3} |x|={r['tok_norm']:7.1f}  {gt_str}")
        out[name] = rows
    if has_gt:
        print("\n=== final state per condition (vs GT) ===")
        print("condition     |x|     AbsRel   d1      ATE")
        for name, rows in out.items():
            if rows:
                r = rows[-1]
                print(f"{name:<12} {r['tok_norm']:7.1f}  {r['absrel']:.4f}  {r['d1']:.3f}  {r.get('ate_rel', float('nan')):.4f}")
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/ablate.json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_ax = 2 if has_gt else 1
        fig, ax = plt.subplots(1, n_ax, figsize=(6 * n_ax, 4.5), squeeze=False)
        ax = ax[0]
        for name, rows in out.items():
            xs = [r["step"] for r in rows]
            if has_gt:
                ax[0].plot(xs, [r["absrel"] for r in rows], "-o", ms=3, label=name)
                ax[1].plot(xs, [r.get("ate_rel", float("nan")) for r in rows], "-o", ms=3, label=name)
        if has_gt:
            ax[0].set_title("depth AbsRel vs GT"); ax[0].set_yscale("log")
            ax[1].set_title("camera ATE vs GT")
            for x_ in ax:
                x_.set_xlabel("step"); x_.legend(fontsize=8); x_.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{a.out}/ablate.png", dpi=130)
        print(f"saved {a.out}/ablate.png")
    except Exception as e:
        print("plot skipped:", e)
    print("\nReading: compare final depth AbsRel vs camera ATE across conditions."
          "\n         first_half hurts camera most => cross-view info (pose) is written EARLY."
          "\n         second_half hurts depth most => depth refinement happens LATE."
          "\n         'all' = never any cross-view => the per-view baseline the loop builds on.")


# ----------------------------------------------------------------------------- exp 8: global-attention maps vs GT correspondences
def gt_correspondence(P, i, j, ph, pw, patch, H, W, occ_thresh=0.1):
    """For every patch (gy,gx) in view i, find the GT-corresponding patch (gy',gx') in view j.

    Unproject the patch-centre pixel via GT depth + intrinsics, transform to view j with GT poses,
    reproject. A correspondence is valid if it lands in front of camera j, inside the image, and
    is not occluded (reprojected depth ~ GT depth at the landing pixel). Returns boolean valid
    mask (ph,pw) and target patch indices gyj,gxj (ph,pw) long.
    """
    dev = P.gt_depth.device
    K = P.gt_K
    Kinv = torch.inverse(K)
    c2w_i = P.gt_c2w[i]
    w2c_j = torch.inverse(P.gt_c2w[j])
    d_i, d_j = P.gt_depth[i], P.gt_depth[j]  # (H,W)

    gy, gx = torch.meshgrid(torch.arange(ph, device=dev), torch.arange(pw, device=dev), indexing="ij")
    u = (gx * patch + patch // 2).clamp(0, W - 1)
    v = (gy * patch + patch // 2).clamp(0, H - 1)
    depth = d_i[v, u]                                  # (ph,pw)
    valid = depth > 1e-6

    pix = torch.stack([u.float(), v.float(), torch.ones_like(u.float())], dim=-1)  # (ph,pw,3)
    Xc = depth[..., None] * (pix @ Kinv.T)             # cam i
    Xw = Xc @ c2w_i[:3, :3].T + c2w_i[:3, 3]           # world
    Xj = Xw @ w2c_j[:3, :3].T + w2c_j[:3, 3]           # cam j
    zj = Xj[..., 2]
    valid = valid & (zj > 1e-6)
    uvj = Xj @ K.T
    uj = uvj[..., 0] / zj.clamp(min=1e-6)
    vj = uvj[..., 1] / zj.clamp(min=1e-6)
    valid = valid & (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)
    # occlusion: compare reprojected depth zj to GT depth of view j at the landing pixel
    uj_c = uj.clamp(0, W - 1).long(); vj_c = vj.clamp(0, H - 1).long()
    dj_at = d_j[vj_c, uj_c]
    valid = valid & (dj_at > 1e-6) & ((zj - dj_at).abs() / zj.clamp(min=1e-6) < occ_thresh)

    gyj = (vj / patch).long().clamp(0, ph - 1)
    gxj = (uj / patch).long().clamp(0, pw - 1)
    return valid, gyj, gxj


@torch.no_grad()
def cmd_attn(P, a):
    """Export the GLOBAL (cross-view) attention and score it against GT correspondences.

    For query patches in view i, does the attention concentrate on the patch in view j that is
    the SAME 3D point (from GT depth+pose)? Directly tests the 'global attention = feature
    matching' claim. Needs --gt with intrinsics (re-run make_gt if missing).
    """
    assert P.gt_depth is not None and P.gt_K is not None, \
        "attn needs --gt with intrinsics. Re-run make_gt (it now saves intrinsics)."
    images = P.load(a.input)
    B, S, _, H, W = images.shape
    assert B == 1 and S >= 2, "attn needs one scene with >=2 frames"
    patch = P.m.patch_size
    ph, pw = H // patch, W // patch
    K_steps = a.K

    # capture the input to the global attention via a forward-pre-hook (no source change)
    g_attn = P.m.recurrent_blocks[0].global_attn.attn
    cap = {}
    h = g_attn.register_forward_pre_hook(lambda m, args, kw=None: cap.__setitem__("x", args[0].detach()),
                                         with_kwargs=True)

    # precompute GT correspondences for adjacent pairs (i, i+1), subsampled queries
    pairs = [(i, (i + 1) % S) for i in range(S)]
    corr = {}
    for (i, j) in pairs:
        valid, gyj, gxj = gt_correspondence(P, i, j, ph, pw, patch, H, W)
        # subsample query patches on a stride grid, keep only valid ones
        qy, qx = torch.meshgrid(torch.arange(0, ph, a.attn_stride), torch.arange(0, pw, a.attn_stride), indexing="ij")
        qy, qx = qy.flatten().to(valid.device), qx.flatten().to(valid.device)
        keep = valid[qy, qx]
        corr[(i, j)] = (qy[keep], qx[keep], gyj[qy[keep], qx[keep]], gxj[qy[keep], qx[keep]])

    heatmap = {}  # a sample (attn over view j, GT target) captured at the last analysed step
    try:
        x_state, rope, dims = P.init_state(images)
        rows = []
        for step in range(K_steps):
            x_state = P.step(x_state, step, K_steps, rope, dims[0], dims[1])
            if not ((step + 1) in a.attn_steps or step == K_steps - 1):
                continue
            xg = cap["x"]                                    # (1, S*Nf, C) global-attn input
            SN = xg.shape[1]; Nf = SN // S
            qkv = g_attn.qkv(xg).reshape(1, SN, 3, g_attn.num_heads, g_attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            q, k = g_attn.q_norm(q), g_attn.k_norm(k)        # (1,heads,SN,hd)
            q, k = q[0], k[0]                                # (heads,SN,hd)
            scale = g_attn.head_dim ** -0.5

            # accumulate matching stats over all pairs
            dists, hits, n = [], [], 0
            perhead_d = torch.zeros(g_attn.num_heads, device=q.device)
            for (i, j) in pairs:
                qy, qx, tyj, txj = corr[(i, j)]
                if len(qy) == 0:
                    continue
                qglob = i * Nf + P.m.patch_start_idx + qy * pw + qx    # global query indices
                jcol0 = j * Nf + P.m.patch_start_idx
                jcols = jcol0 + torch.arange(ph * pw, device=q.device)  # view-j patch columns
                qsel = q[:, qglob, :]                        # (heads, nq, hd)
                scores = (qsel * scale) @ k.transpose(-1, -2)  # (heads, nq, SN)
                aw = scores.softmax(dim=-1)[:, :, jcols].reshape(g_attn.num_heads, len(qy), ph, pw)  # over view j
                aw = aw / aw.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-9)   # renormalise over view j
                gyg, gxg = torch.meshgrid(torch.arange(ph, device=q.device).float(),
                                          torch.arange(pw, device=q.device).float(), indexing="ij")
                # attention-weighted distance to GT target (patch units), per head per query
                dy = (aw * gyg).sum(dim=(-1, -2)) - tyj.float()   # (heads,nq)
                dx = (aw * gxg).sum(dim=(-1, -2)) - txj.float()
                wdist = torch.sqrt(dy ** 2 + dx ** 2)             # (heads,nq)
                # top-1 hit: argmax patch within 1 patch of target
                flat = aw.reshape(g_attn.num_heads, len(qy), -1).argmax(-1)
                ay, ax = flat // pw, flat % pw
                hit = ((ay - tyj).abs() <= 1) & ((ax - txj).abs() <= 1)  # (heads,nq)
                dists.append(wdist.mean(0)); hits.append(hit.float().mean(0))   # head-avg per query
                perhead_d += wdist.mean(1) * len(qy)
                n += len(qy)
                if i == 0:  # stash a sample heatmap for pair (0,1): the query nearest grid centre
                    qc = ((qy.float() - ph / 2) ** 2 + (qx.float() - pw / 2) ** 2).argmin().item()
                    heatmap = dict(step=step + 1, i=i, j=j, qy=int(qy[qc]), qx=int(qx[qc]),
                                   tyj=int(tyj[qc]), txj=int(txj[qc]),
                                   aw=aw[:, qc].detach().float().cpu().numpy())  # (heads,ph,pw)
            if n == 0:
                continue
            mean_d = torch.cat(dists).mean().item()
            hit_r = torch.cat(hits).mean().item()
            perhead_d = (perhead_d / n)
            chance = float(np.mean([np.hypot(ph, pw) / 3]))  # rough uniform-attn baseline
            best_head = int(perhead_d.argmin()); best_d = float(perhead_d.min())
            rows.append(dict(step=step + 1, mean_dist=mean_d, top1_hit=hit_r, n=n,
                             best_head=best_head, best_head_dist=best_d,
                             perhead_dist=[round(float(x), 3) for x in perhead_d]))
            print(f"step {step+1:>3}/{K_steps}  n={n:>4}  mean_dist={mean_d:5.2f} patch  "
                  f"top1_hit={hit_r:5.1%}  best_head={best_head}(d={best_d:.2f})")
    finally:
        h.remove()

    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(f"{a.out}/attn.json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        st = [r["step"] for r in rows]
        ax[0].plot(st, [r["mean_dist"] for r in rows], "-o", label="attn-weighted dist to GT corr")
        ax[0].plot(st, [r["best_head_dist"] for r in rows], "-o", label="best head")
        ax[0].axhline(np.hypot(ph, pw) / 3, ls="--", color="grey", label="~chance")
        ax[0].set_title("matching error vs step (lower=better)"); ax[0].set_ylabel("patch distance")
        ax[1].plot(st, [r["top1_hit"] for r in rows], "-o", color="#2c6")
        ax[1].set_title("top-1 hit rate (argmax within 1 patch of GT)"); ax[1].set_ylabel("hit rate")
        for x_ in ax:
            x_.set_xlabel("step"); x_.grid(alpha=0.3); x_.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(f"{a.out}/attn.png", dpi=130)
        print(f"saved {a.out}/attn.png")

        # heatmap: query patch in view i (left) + its attention over view j with GT target (right)
        if heatmap and rows:
            bh = rows[-1]["best_head"]
            hm = heatmap["aw"][bh]                            # (ph,pw) best-head attention over view j
            Ii = images[0, heatmap["i"]].float().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            Ij = images[0, heatmap["j"]].float().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5))
            ax2[0].imshow(Ii)
            ax2[0].add_patch(plt.Rectangle((heatmap["qx"] * patch, heatmap["qy"] * patch), patch, patch,
                                           fill=False, ec="red", lw=2))
            ax2[0].set_title(f"view {heatmap['i']}: query patch (red)")
            ax2[1].imshow(Ij)
            ax2[1].imshow(np.kron(hm, np.ones((patch, patch))), cmap="jet", alpha=0.5,
                          extent=(0, Ij.shape[1], Ij.shape[0], 0))
            ax2[1].scatter([(heatmap["txj"] + 0.5) * patch], [(heatmap["tyj"] + 0.5) * patch],
                           c="lime", s=200, marker="*", edgecolors="k", label="GT correspondence")
            ax2[1].set_title(f"view {heatmap['j']}: attention (head {bh}) + GT target"); ax2[1].legend()
            for x_ in ax2:
                x_.axis("off")
            fig2.suptitle(f"global attention as matching -- step {heatmap['step']}, best head {bh}", weight="bold")
            plt.tight_layout(); plt.savefig(f"{a.out}/attn_heatmap.png", dpi=130)
            print(f"saved {a.out}/attn_heatmap.png")
    except Exception as e:
        print("plot skipped:", e)
    print("\nReading: if mean_dist drops well below chance and top1_hit rises over steps,"
          "\n         the global attention IS doing (emergent) feature matching on GT correspondences."
          "\n         best_head shows matching is often concentrated in a few heads.")


# ----------------------------------------------------------------------------- exp 9: dump the flow for the viewer
def _unit(xp):
    """per-token direction as the decoder sees it: remove the channel mean, then normalise (LayerNorm without affine)."""
    return F.normalize(xp - xp.mean(-1, keepdim=True), dim=-1)


@torch.no_grad()
def cmd_flowdump(P, a):
    """
    Run one K-step pass and write flow.json for dvlt_flow_viewer.html:
      traj      PCA(3) positions of a random subset of patch tokens at every state 0..K  (tokens on the unit sphere)
      ang_map   per-token angle (deg) between consecutive states, whole patch grid, uint8
      depth     decoded depth (log, per-frame 2-98% normalised) at every state, patch-grid resolution, uint8
      curves    mean |x|, relative update, angular speed, cos to final state, share of |x|^2 in the 8 biggest channels
      tgrid     update direction / size when the SAME state is pushed with other time pairs  (how much does t matter?)
    """
    import base64
    import io
    from PIL import Image
    images = P.load(a.input)
    K = a.K
    x, rope, dims = P.init_state(images)
    B, S, H, W = dims
    gh, gw = H // 14, W // 14
    m = P.m
    states = [x]
    for i in range(K):
        states.append(P.step(states[-1], i, K, rope, B, S))
    Uv = [_unit(patch_part(s_, m).float()) for s_ in states]            # (BS, Np, C)
    BS, Np, C = Uv[0].shape

    # ---- sampled tokens + PCA (one basis for all steps)
    g = torch.Generator().manual_seed(0)
    n_all = BS * Np
    idx = torch.randperm(n_all, generator=g)[: min(a.ntok, n_all)].to(Uv[0].device)
    Z = torch.stack([u.reshape(-1, C)[idx] for u in Uv])                # (K+1, ntok, C)
    M_ = Z.reshape(-1, C).float()
    mu = M_.mean(0, keepdim=True)
    with torch.autocast("cuda", enabled=False):   # QR in pca_lowrank is not implemented for bf16
        _, Sv, V = torch.pca_lowrank(M_, q=3, center=True)
    proj = (Z - mu) @ V[:, :3]                                           # (K+1, ntok, 3)
    scale = proj.abs().max().clamp_min(1e-9)
    proj = (proj / scale).cpu()
    tot_var = ((M_ - mu) ** 2).sum() / (M_.shape[0] - 1)
    expl = ((Sv ** 2) / (M_.shape[0] - 1) / tot_var).cpu().tolist()
    fr, rest = idx // Np, idx % Np
    tok_idx = torch.stack([fr, rest // gw, rest % gw], 1).cpu().tolist()

    # ---- angular speed maps + curves
    ang = []
    for k in range(K):
        c = (Uv[k] * Uv[k + 1]).sum(-1).clamp(-1, 1)
        ang.append(torch.rad2deg(torch.acos(c)))                         # (BS, Np)
    ang_t = torch.stack(ang)                                             # (K, BS, Np)
    ang_max = float(torch.quantile(ang_t.flatten()[:: max(1, ang_t.numel() // 1_000_000)], 0.99).clamp_min(1e-6))
    ang_u8 = (ang_t / ang_max).clamp(0, 1).mul(255).round().to(torch.uint8).reshape(K, S, gh, gw).cpu()
    raw = [patch_part(s_, m).float() for s_ in states]
    curves = dict(
        norm=[r.norm(dim=-1).mean().item() for r in raw],
        rel_upd=[((raw[k + 1] - raw[k]).norm(dim=-1) / raw[k].norm(dim=-1).clamp_min(1e-6)).mean().item() for k in range(K)],
        ang_speed=[ang_t[k].mean().item() for k in range(K)],
        cos_final=[(Uv[k] * Uv[K]).sum(-1).mean().item() for k in range(K + 1)],
        top8_share=[(r.pow(2).topk(8, dim=-1).values.sum(-1) / r.pow(2).sum(-1).clamp_min(1e-9)).mean().item() for r in raw],
    )

    # ---- decoded depth at every state
    depth_u8 = None
    if not a.no_depth:
        maps = []
        for s_ in states:
            d = P.decode(s_, rope, dims)["depth"].float()[0, ..., 0]      # (S,H,W)
            maps.append(F.adaptive_avg_pool2d(torch.log(d.clamp_min(1e-6))[:, None], (gh, gw))[:, 0])
        fin = maps[-1].reshape(S, -1)
        lo, hi = torch.quantile(fin, 0.02, dim=1), torch.quantile(fin, 0.98, dim=1)
        depth_u8 = torch.stack([((mp - lo[:, None, None]) / (hi - lo).clamp_min(1e-6)[:, None, None]).clamp(0, 1) for mp in maps])
        depth_u8 = depth_u8.mul(255).round().to(torch.uint8).cpu()       # (K+1,S,gh,gw)

    # ---- how much does the time pair matter?  push the same state with other (t, t+dt)
    dt = 1.0 / (K - 1)
    tg = torch.linspace(0.0, 1.0 - dt, 9).tolist()
    ks = sorted(set([0, K // 4, K // 2, (3 * K) // 4]))
    ts_true = torch.linspace(0.0, 1.0, K).tolist()
    cos_m, ratio_m = [], []
    for k in ks:
        xk = states[k]
        d_true = patch_part(states[k + 1] - xk, m).float().flatten()
        rc, rr = [], []
        for tn in tg:
            xn = m._interval_step(xk, tn, tn + dt, rope, B, S)
            d = patch_part(xn - xk, m).float().flatten()
            rc.append(F.cosine_similarity(d, d_true, dim=0).item())
            rr.append((d.norm() / d_true.norm().clamp_min(1e-9)).item())
        cos_m.append(rc)
        ratio_m.append(rr)

    # ---- thumbnails
    thumbs = []
    for j in range(S):
        im = F.interpolate(images[0, j:j + 1].float(), size=(max(8, round(96 * H / W)), 96), mode="bilinear", antialias=True)[0]
        pil = Image.fromarray((im.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        thumbs.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())

    out = dict(
        format="dvlt-flow-v1", K=K, S=S, gh=gh, gw=gw, frames=thumbs, pca_var=expl,
        tok_idx=tok_idx, traj=[[[round(v, 4) for v in p3] for p3 in st] for st in proj.tolist()],
        ang_max=ang_max, ang_map=ang_u8.flatten().tolist(),
        depth=None if depth_u8 is None else depth_u8.flatten().tolist(),
        curves=curves,
        tgrid=dict(t=tg, ks=ks, t_true=[ts_true[k] for k in ks], cos=cos_m, ratio=ratio_m),
    )
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/flow.json"
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(f"wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)  K={K} S={S} grid={gh}x{gw} tokens={len(tok_idx)}")
    print("PCA explained variance (3 comps):", [round(e, 3) for e in expl])
    print("angular speed per step (deg):", [round(v, 2) for v in curves["ang_speed"]])
    print("cos to final per state:", [round(v, 3) for v in curves["cos_final"]])


# ----------------------------------------------------------------------------- exp 10: channel concentration (why it diverges past K_max)
@torch.no_grad()
def cmd_channels(P, a):
    """Run a long K pass and track, per step, how concentrated the patch-token energy is in a few
    channels, alongside GT quality. Tests the paper's claim that beyond K_max 'a handful of channels
    grow without bound' -> low-rank collapse. If concentration rises in lock-step with quality
    dropping past ~K_max, the channel-blowup mechanism is confirmed.
    """
    images = P.load(a.input)
    B, S, _, H, W = images.shape
    K = a.K
    has_gt = P.gt_depth is not None
    m = P.m
    x, rope, dims = P.init_state(images)

    # track per-channel energy across the loop (mean over tokens of x_c^2)
    def chan_stats(xp):
        p = patch_part(xp, m).float()                 # (BS, Np, C)
        e = p.pow(2).mean(dim=(0, 1))                 # (C,) per-channel energy
        tot = e.sum().clamp_min(1e-9)
        top1 = e.max() / tot
        top8 = e.topk(8).values.sum() / tot
        # participation ratio = (sum e)^2 / sum e^2  (effective number of active channels; low => concentrated)
        pr = (e.sum() ** 2) / (e.pow(2).sum().clamp_min(1e-12))
        # cluster_cos = mean pairwise direction cosine over a token subsample (-> 1 = all tokens
        # collapsed to ONE direction on the sphere = the over-smoothing attractor / "one point")
        u = _unit(p).reshape(-1, p.shape[-1])
        sub = u[torch.randperm(u.shape[0], device=u.device)[:512]]
        npair = sub.shape[0] * (sub.shape[0] - 1) / 2
        cluster = ((sub @ sub.t()).triu(1).sum() / max(npair, 1)).item()
        return top1.item(), top8.item(), pr.item(), cluster, e

    frozen = getattr(a, "frozen", False)
    mode = f"FROZEN t={a.frozen_t} (iterate one map -> attractor)" if frozen else "linspace (finer integration of [0,1])"
    print(f"channels mode: {mode}, {K} steps\n")
    rows = []
    e_hist = []
    for i in range(K + 1):
        top1, top8, pr, cluster, e = chan_stats(x)
        e_hist.append(e.cpu())
        r = dict(step=i, tok_norm=patch_part(x, m).float().norm(dim=-1).mean().item(),
                 top1_share=top1, top8_share=top8, participation_ratio=pr, cluster_cos=cluster)
        if has_gt:
            pr_dec = {k: v.float() for k, v in P.decode(x, rope, dims).items() if k in ("depth", "rays")}
            r.update(depth_vs_gt(pr_dec["depth"], P.gt_depth[:S]))
            if P.gt_c2w is not None:
                r["ate_rel"] = sim3_ate(pr_dec["rays"][0, :, :, :, 3:6].mean(dim=(1, 2)), P.gt_c2w[:S, :3, 3])
        rows.append(r)
        gt_str = (f"AbsRel={r['absrel']:.4f} ATE={r.get('ate_rel', float('nan')):.4f}" if has_gt else "GT:n/a")
        print(f"step {i:>3}/{K}  |x|={r['tok_norm']:7.1f}  top1={top1:.3f}  cluster={cluster:.3f}  "
              f"eff_chan={pr:6.1f}/{e.numel()}  {gt_str}")
        if i < K:
            x = (P.step_t(x, a.frozen_t, a.frozen_t, rope, dims[0], dims[1]) if frozen
                 else P.step(x, i, K, rope, dims[0], dims[1]))

    # which channels dominate at the end? (consistent small set?)
    e_fin = e_hist[-1]
    top_ch = e_fin.topk(8).indices.tolist()
    print(f"\ntop-8 channels at step {K}: {top_ch}")
    print(f"their energy share: {(e_fin[top_ch].sum() / e_fin.sum()).item():.3f}")

    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(rows=rows, top_channels=top_ch), open(f"{a.out}/channels.json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        st = [r["step"] for r in rows]
        n_ax = 2
        fig, ax = plt.subplots(1, n_ax, figsize=(6 * n_ax, 4.5))
        ax[0].plot(st, [r["top1_share"] for r in rows], "-o", ms=3, label="top-1 share")
        ax[0].plot(st, [r["top8_share"] for r in rows], "-o", ms=3, label="top-8 share")
        ax[0].plot(st, [r["cluster_cos"] for r in rows], "-^", ms=3, label="cluster_cos (->1=one point)")
        ax[0].axvline(16, ls="--", color="grey", alpha=0.6, label="K=16")
        ax[0].set_title("channel concentration (higher=more collapsed)"); ax[0].set_xlabel("step"); ax[0].set_ylabel("energy share")
        ax0b = ax[0].twinx()
        ax0b.plot(st, [r["participation_ratio"] for r in rows], "-s", ms=3, color="#c44", label="eff. #channels")
        ax0b.set_ylabel("effective #channels", color="#c44"); ax0b.tick_params(axis="y", labelcolor="#c44")
        if has_gt:
            ax[1].plot(st, [r.get("absrel", float("nan")) for r in rows], "-o", ms=3, label="depth AbsRel vs GT")
            ax[1].plot(st, [r.get("ate_rel", float("nan")) for r in rows], "-o", ms=3, label="camera ATE vs GT")
            ax[1].axvline(16, ls="--", color="grey", alpha=0.6, label="K=16")
            ax[1].set_title("GT quality (overlay to compare with concentration)"); ax[1].set_yscale("log")
        for x_ in ax:
            x_.set_xlabel("step"); x_.legend(fontsize=8); x_.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{a.out}/channels.png", dpi=130)
        print(f"saved {a.out}/channels.png")
    except Exception as e:
        print("plot skipped:", e)
    print("\nReading: if concentration (top-k share up / eff-channels down) rises past ~K=16 in lock-step with"
          "\n         AbsRel/ATE worsening, the collapse IS driven by a few channels growing unbounded (low-rank collapse).")


# ----------------------------------------------------------------------------- exp 11: field smoothness in x (local Lipschitz)
@torch.no_grad()
def cmd_smoothx(P, a):
    """Is the velocity field v(x)=step(x)-x Lipschitz in x?  At a few steps, perturb the state by eps
    and measure how much the ONE-STEP update changes: ||v(x+d)-v(x)|| / ||d||.  For a smooth field
    this ratio converges to a finite local Lipschitz constant L as eps->0 (||dv|| linear in ||dx||).
    L<1 => the step contracts perturbations (explains gauss self-correction); L>1 directions amplify.
    """
    images = P.load(a.input)
    B, S, _, H, W = images.shape
    K = a.K
    gen = torch.Generator(device=images.device).manual_seed(0)
    x, rope, dims = P.init_state(images)
    clean = [x]
    for i in range(K):
        clean.append(P.step(clean[-1], i, K, rope, B, S))

    def v_of(xp, i):
        return patch_part(P.step(xp, i, K, rope, B, S) - xp, P.m).float()

    rows = []
    for i in a.at:
        xk = clean[i]
        v0 = v_of(xk, i)
        for eps in a.eps:
            ratios, dxs, dvs = [], [], []
            for _ in range(a.reps):
                xp = make_perturbation(xk, "gauss", eps, P, gen)   # patch-token gauss, rel to per-token norm
                dx = patch_part(xp - xk, P.m).float()
                dv = v_of(xp, i) - v0
                dxn, dvn = dx.norm().item(), dv.norm().item()
                ratios.append(dvn / max(dxn, 1e-9)); dxs.append(dxn); dvs.append(dvn)
            L = float(np.mean(ratios))
            rows.append(dict(step=i, eps=eps, L=L, dx=float(np.mean(dxs)), dv=float(np.mean(dvs))))
            print(f"step {i:>3}  eps={eps:<6} ||dx||={np.mean(dxs):8.2f}  ||dv||={np.mean(dvs):8.2f}  L=||dv||/||dx||={L:.3f}")
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(f"{a.out}/smoothx.json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for i in a.at:
            rr = [r for r in rows if r["step"] == i]
            ax[0].plot([r["dx"] for r in rr], [r["dv"] for r in rr], "-o", ms=4, label=f"step {i}")
            ax[1].plot([r["eps"] for r in rr], [r["L"] for r in rr], "-o", ms=4, label=f"step {i}")
        ax[0].set_title("||dv|| vs ||dx||  (straight line through 0 => smooth)")
        ax[0].set_xlabel("||dx|| (perturb size)"); ax[0].set_ylabel("||dv|| (update change)")
        ax[1].axhline(1.0, ls="--", color="grey", label="L=1 (contract/expand boundary)")
        ax[1].set_title("local Lipschitz L vs eps  (flat as eps->0 => smooth)")
        ax[1].set_xlabel("eps"); ax[1].set_ylabel("L = ||dv||/||dx||"); ax[1].set_xscale("log")
        for x_ in ax:
            x_.legend(fontsize=8); x_.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{a.out}/smoothx.png", dpi=130)
        print(f"saved {a.out}/smoothx.png")
    except Exception as e:
        print("plot skipped:", e)
    print("\nReading: ||dv|| linear in ||dx|| and L converging to a finite value as eps->0 => field is locally"
          "\n         Lipschitz (well-posed flow). L<1 => step contracts perturbations (explains self-correction).")


# ----------------------------------------------------------------------------- exp 12: field smoothness in the input
@torch.no_grad()
def cmd_smoothinput(P, a):
    """Is the flow smooth in the INPUT?  Add small pixel noise to the images, re-run the whole K-step
    loop, and measure how far the output / trajectory moves vs the input change.  Bounded, linear
    response => 'input determines endpoint' is a continuous (stable) map, not chaotically sensitive.
    """
    images = P.load(a.input)
    B, S, _, H, W = images.shape
    K = a.K
    gen = torch.Generator(device=images.device).manual_seed(0)

    def run(imgs):
        x, rope, dims = P.init_state(imgs)
        states = [x]
        for i in range(K):
            states.append(P.step(states[-1], i, K, rope, dims[0], dims[1]))
        pr = {k: v.float() for k, v in P.decode(states[-1], rope, dims).items() if k in ("depth", "rays")}
        return states, pr, rope, dims

    clean_states, clean_pred, rope, dims = run(images)
    clean_gt = gt_metrics(P, clean_pred) if P.gt_depth is not None else None
    print("clean:" + fmt_gt(clean_gt))
    rows = []
    for eps in a.eps:
        for _ in range(a.reps):
            noise = torch.randn(images.shape, generator=gen, device=images.device, dtype=torch.float32).to(images.dtype)
            imgs = (images + eps * noise).clamp(0, 1)
            states, pred, _, _ = run(imgs)
            # output change vs clean (decoded space) and trajectory divergence at the end
            d_out = out_dist(pred, clean_pred)
            traj_div = (patch_part(states[-1] - clean_states[-1], P.m).float().norm(dim=-1)
                        / patch_part(clean_states[-1], P.m).float().norm(dim=-1)).mean().item()
            gm = gt_metrics(P, pred) if P.gt_depth is not None else None
            rows.append(dict(eps=eps, depth_change=d_out["depth"], ray_change=d_out["ray"],
                             traj_div=traj_div, gt=gm))
            gchg = f"AbsRel {clean_gt['absrel']:.4f}->{gm['absrel']:.4f}" if gm else ""
            print(f"input eps={eps:<6} depth_change={d_out['depth']:.4f}  ray_change={d_out['ray']:.3f}  "
                  f"traj_div={traj_div:.3f}  {gchg}")
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(clean_gt=clean_gt, rows=rows), open(f"{a.out}/smoothinput.json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epss = sorted(set(r["eps"] for r in rows))
        def avg(key, e):
            vs = [r[key] for r in rows if r["eps"] == e]
            return float(np.mean(vs))
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(epss, [avg("depth_change", e) for e in epss], "-o", label="depth change")
        ax[0].plot(epss, [avg("ray_change", e) for e in epss], "-o", label="ray change")
        ax[0].set_title("output change vs input noise (linear & bounded => smooth)")
        ax[1].plot(epss, [avg("traj_div", e) for e in epss], "-o", color="#c44", label="final traj divergence")
        ax[1].set_title("trajectory divergence vs input noise")
        for x_ in ax:
            x_.set_xlabel("input pixel-noise eps"); x_.legend(fontsize=8); x_.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{a.out}/smoothinput.png", dpi=130)
        print(f"saved {a.out}/smoothinput.png")
    except Exception as e:
        print("plot skipped:", e)
    print("\nReading: output/trajectory change growing linearly and staying bounded with input noise => the flow"
          "\n         is smooth in the input (endpoint is a stable function of the images, not chaotically sensitive).")


# ----------------------------------------------------------------------------- exp 13: big perturbation sweep (all kinds x all steps x all levels, every quantity)
@torch.no_grad()
def cmd_sweep(P, a):
    """One big sweep: inject every (kind, at, eps), run to K, and measure EVERY quantity at the
    final state -- norm ratio, token DIRECTION recovery (cos to the clean final), decoded distance
    to the K-step standard output, error vs GT, and injected-vs-final token damage (recovery).
    All configs land in one multi-panel summary figure.
    """
    images = P.load(a.input)
    B, S, _, H, W = images.shape
    K = a.K
    gen = torch.Generator(device=images.device).manual_seed(0)
    x0, rope, dims = P.init_state(images)
    clean = [x0]
    for i in range(K):
        clean.append(P.step(clean[-1], i, K, rope, B, S))
    clean_final = clean[-1]
    clean_pred = {k: v.float() for k, v in P.decode(clean_final, rope, dims).items() if k in ("depth", "rays")}
    clean_gt = gt_metrics(P, clean_pred)
    cf_dir = _unit(patch_part(clean_final, P.m).float())
    cf_norm = patch_part(clean_final, P.m).float().norm(dim=-1).mean()
    print("clean final:" + fmt_gt(clean_gt))

    rows = []
    for kind in a.kinds:
        eps_list = a.eps if kind != "swap" else [a.eps[-1]]   # swap ignores eps magnitude
        for j in a.at:
            for eps in eps_list:
                xp = make_perturbation(clean[j], kind, eps, P, gen)
                tok_inj = (patch_part(xp - clean[j], P.m).float().norm(dim=-1)
                           / patch_part(clean[j], P.m).float().norm(dim=-1)).mean().item()
                x = xp
                for i in range(j, K):
                    x = P.step(x, i, K, rope, B, S)
                p = {k: v.float() for k, v in P.decode(x, rope, dims).items() if k in ("depth", "rays")}
                d = out_dist(p, clean_pred)
                gm = gt_metrics(P, p)
                row = dict(
                    kind=kind, at=j, eps=eps,
                    tok_rel_inj=tok_inj,
                    tok_rel_final=(patch_part(x - clean_final, P.m).float().norm(dim=-1)
                                   / patch_part(clean_final, P.m).float().norm(dim=-1)).mean().item(),
                    dir_cos_final=(_unit(patch_part(x, P.m).float()) * cf_dir).sum(-1).mean().item(),
                    norm_ratio=(patch_part(x, P.m).float().norm(dim=-1).mean() / cf_norm).item(),
                    depth_to_std=d["depth"], ray_to_std=d["ray"],
                    gt_absrel=(gm or {}).get("absrel", float("nan")),
                    gt_ate=(gm or {}).get("ate_rel", float("nan")),
                )
                rows.append(row)
                print(f"{kind:>8} @{j:>2} eps={eps:<5} | tok {tok_inj:.2f}->{row['tok_rel_final']:.2f}  "
                      f"dir_cos={row['dir_cos_final']:.3f}  |x|/std={row['norm_ratio']:.2f}  "
                      f"d->std={d['depth']:.4f}  GT AbsRel={row['gt_absrel']:.4f} ATE={row['gt_ate']:.4f}")

    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(clean_gt=clean_gt, rows=rows), open(f"{a.out}/sweep.json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [f"{r['at']}/{r['eps']}" for r in rows]
        kinds = a.kinds
        cmap = {k: plt.cm.tab10(i / 10) for i, k in enumerate(kinds)}
        colors = [cmap[r["kind"]] for r in rows]
        xpos = np.arange(len(rows))
        panels = [
            ("dir_cos_final", "direction recovery  cos(perturbed, clean)   1=recovered", None),
            ("tok_rel_final", "token damage after loop (rel norm)", None),
            ("gt_absrel", "depth AbsRel vs GT", (clean_gt or {}).get("absrel")),
            ("gt_ate", "camera ATE vs GT", (clean_gt or {}).get("ate_rel")),
            ("depth_to_std", "depth dist -> K-step standard", 0.0),
            ("norm_ratio", "|x| / standard |x|", 1.0),
        ]
        fig, axes = plt.subplots(3, 2, figsize=(max(12, len(rows) * 0.32), 12))
        for ax, (key, title, base) in zip(axes.ravel(), panels):
            ax.bar(xpos, [r[key] for r in rows], color=colors)
            if base is not None:
                ax.axhline(base, ls="--", color="grey", lw=1, alpha=0.7, label="clean baseline")
                ax.legend(fontsize=7)
            ax.set_title(title, fontsize=10)
            ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=5, rotation=90)
        # one shared legend for kind colors
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=cmap[k], label=k) for k in kinds],
                   loc="upper center", ncol=len(kinds), fontsize=9, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"perturbation sweep @ {S} frames, K={K}  (x = at/eps, color = kind)",
                     fontsize=13, weight="bold", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(f"{a.out}/sweep.png", dpi=130)
        print(f"saved {a.out}/sweep.png")
    except Exception as e:
        print("plot skipped:", e)
    print("\nReading: dir_cos near 1 = direction fully recovered (loop self-corrected). gauss/scale/cam/register"
          "\n         should recover; swap (wrong view) and large dir should not. Compare GT panels to the dashed"
          "\n         clean baseline to see which perturbations actually cost real accuracy.")


# ----------------------------------------------------------------------------- logging
class Tee:
    """Duplicate everything written to stdout into a log file as well."""

    def __init__(self, file_handle):
        self.file = file_handle
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# ----------------------------------------------------------------------------- main
# ----------------------------------------------------------------------------- exp 14: unified sweep
# ONE uniform measurement pass. Every variant (the K=16 baseline + each perturbation kind x level x
# injection step) is driven through the SAME loop, and the SAME full metric vector is recorded at
# EVERY step. Then one figure per metric DIMENSION overlays every variant as a line -- e.g. a single
# "|x| norm" figure with 30+ lines. Two anchors: baseline (the K=16 standard) and GT (the gt_* keys).
# Nothing here touches dvlt source or any other command.
UNIFIED_METRICS = [
    ("tok_norm",         "|x|  token norm"),
    ("cam_norm",         "|x|  camera-token norm"),
    ("upd_abs",          "|dx|  absolute per-step update"),
    ("upd_rel",          "|dx|/|x|  relative per-step update"),
    ("upd_cos_prev",     "cos(dx_t, dx_{t-1})  update coherence"),
    ("tok_cos_step",     "cos(x_t, x_{t-1})  direction stability"),
    ("top1_share",       "top-1 channel energy share"),
    ("top8_share",       "top-8 channel energy share"),
    ("eff_chan",         "effective #channels"),
    ("cluster_cos",      "mean inter-token cos (clustering)"),
    ("gate_frame_out",   "s_out  (frame attn)"),
    ("gate_global_out",  "s_out  (global attn)"),
    ("gt_absrel",        "depth AbsRel vs GT"),
    ("gt_d1",            "depth delta<1.25 vs GT"),
    ("gt_ate",           "camera ATE vs GT"),
    ("depth_to_std",     "depth dist to K=16 standard (final)"),
    ("ray_to_std",       "ray angle (deg) to K=16 standard (final)"),
    ("tok_rel_to_clean", "token rel dist to clean trajectory"),
]


def _full_step_metrics(P, x, x_new, delta, prev_delta):
    """The intrinsic per-step metric vector (same quantities as run_traj), shared by every variant."""
    pn = patch_part(x_new, P.m).float().norm(dim=-1)
    dn = patch_part(delta, P.m).norm(dim=-1)
    r = dict(
        tok_norm=pn.mean().item(),
        cam_norm=x_new[:, 0].float().norm(dim=-1).mean().item(),
        upd_abs=dn.mean().item(),
        upd_rel=(dn / pn.clamp_min(1e-6)).mean().item(),
        upd_cos_prev=(F.cosine_similarity(delta.flatten(1), prev_delta.flatten(1), dim=-1).mean().item()
                      if prev_delta is not None else float("nan")),
    )
    xo, xn = patch_part(x, P.m).float(), patch_part(x_new, P.m).float()
    r["tok_cos_step"] = F.cosine_similarity(xo, xn, dim=-1).mean().item()
    e = xn.pow(2).mean(dim=(0, 1)); tot = e.sum().clamp_min(1e-9)
    r["top1_share"] = (e.max() / tot).item()
    r["top8_share"] = (e.topk(8).values.sum() / tot).item()
    r["eff_chan"] = ((e.sum() ** 2) / e.pow(2).sum().clamp_min(1e-12)).item()
    u = _unit(xn).reshape(-1, xn.shape[-1])
    sub = u[torch.randperm(u.shape[0], device=u.device)[:512]]
    npair = sub.shape[0] * (sub.shape[0] - 1) / 2
    r["cluster_cos"] = ((sub @ sub.t()).triu(1).sum() / max(npair, 1)).item()
    r["gate_frame_out"] = P.gates["frame"][2]
    r["gate_global_out"] = P.gates["global"][2]
    return r


def _ablate_on(mode, i, K):
    if mode == "all":
        return True
    if mode == "first_half":
        return i < K // 2
    if mode == "second_half":
        return i >= K // 2
    return False


@torch.no_grad()
def _measure(P, x, x_new, delta, prev_delta, step, rope, dims, ref_final, clean_states, aligned):
    """The full 18-metric vector for one step of ANY variant (same quantities for every variant)."""
    x, x_new, delta = x.detach(), x_new.detach(), delta.detach()
    r = _full_step_metrics(P, x, x_new, delta, prev_delta)
    r["step"] = step
    pred = {k: v.float().cpu() for k, v in P.decode(x_new, rope, dims).items() if k in ("depth", "rays")}
    g = gt_metrics(P, pred)
    r["gt_absrel"] = g["absrel"] if g else float("nan")
    r["gt_d1"] = g["d1"] if g else float("nan")
    r["gt_ate"] = (g.get("ate_rel", float("nan")) if g else float("nan"))
    r["depth_to_std"] = log_depth_dist(pred["depth"], ref_final["depth"])
    r["ray_to_std"] = ray_angle_dist(pred["rays"], ref_final["rays"])
    if aligned and step + 1 < len(clean_states):
        cl = clean_states[step + 1]
        r["tok_rel_to_clean"] = (patch_part(x_new - cl, P.m).float().norm(dim=-1)
                                 / patch_part(cl, P.m).float().norm(dim=-1).clamp_min(1e-6)).mean().item()
    else:
        r["tok_rel_to_clean"] = float("nan")   # only meaningful for K=16-aligned variants
    return r


def _run_variant(P, images, spec, ref_final, clean_states, gen, conv):
    """Drive ONE variant through the loop and return its per-step metric rows. Handles every family:
    base / K / sched / perturb / ablate / renorm / photo. Each changes ONE thing vs the baseline."""
    fam = spec["family"]
    K = spec.get("K", 16)
    P.renorm = bool(spec.get("renorm", False))
    P.renorm_from = spec.get("renorm_from", 0)
    P._renorm_frozen = None
    x, rope, dims = P.init_state(images)
    B, S, H, W = dims
    rows, prev_delta = [], None

    # --- time-schedule family: hand-made (t_now,t_next) table, variable length ---
    if fam == "sched":
        pairs = build_schedule(spec["sched"], spec.get("sched_K", 16), spec.get("frozen_t", 0.5),
                               spec.get("sched_steps", 16))
        for i, (tn, tnx) in enumerate(pairs):
            with torch.no_grad():
                x_new = P.step_t(x, tn, tnx, rope, B, S)
            delta = (x_new - x).float()
            rows.append(_measure(P, x, x_new, delta, prev_delta, i, rope, dims, ref_final, clean_states, False))
            prev_delta, x = delta, x_new
        return rows

    # --- photometric gradient-guided family: std steps up to start, grad injection, then finish ---
    if fam == "photo":
        start, step_idx = spec["start"], 0
        for i in range(start):
            with torch.no_grad():
                x_new = P.step(x, i, K, rope, B, S)
            delta = (x_new - x).float()
            rows.append(_measure(P, x, x_new, delta, prev_delta, step_idx, rope, dims, ref_final, clean_states, True))
            prev_delta, x, step_idx = delta, x_new, step_idx + 1
        xcur = x.detach().clone()
        for _it in range(spec["iters"]):
            xcur.requires_grad_(True)
            with torch.enable_grad(), P.acc.autocast():
                pr = _pred_f(P, xcur, rope, dims)
                L, _ = photometric_residual(pr, images, dims, conv, stride=spec.get("stride", 2))
                gg, = torch.autograd.grad(L, xcur)
            with torch.no_grad():
                gn = gg.float().norm().clamp_min(1e-12)
                x_new = (xcur.float() - spec["eta"] * xcur.float().norm() * (gg.float() / gn)).to(x.dtype).detach()
            delta = (x_new - xcur).float()
            rows.append(_measure(P, xcur, x_new, delta, prev_delta, step_idx, rope, dims, ref_final, clean_states, False))
            prev_delta, xcur, step_idx = delta, x_new, step_idx + 1
        x = xcur
        for i in range(start, K):
            with torch.no_grad():
                x_new = P.step(x, i, K, rope, B, S)
            delta = (x_new - x).float()
            rows.append(_measure(P, x, x_new, delta, prev_delta, step_idx, rope, dims, ref_final, clean_states, False))
            prev_delta, x, step_idx = delta, x_new, step_idx + 1
        return rows

    # --- base / K / perturb / ablate / renorm: ordinary K-step loop with one knob changed ---
    pert = spec.get("perturb")
    ab = spec.get("ablate_mode")
    aligned = spec.get("aligned", fam == "base")
    for i in range(K):
        if pert is not None and i == pert[2]:
            x = make_perturbation(x, pert[0], pert[1], P, gen)
        with torch.no_grad():
            if ab:
                x_new = P.step_ablate(x, i, K, rope, B, S, ablate_global=_ablate_on(ab, i, K))
            else:
                x_new = P.step(x, i, K, rope, B, S)
        delta = (x_new - x).float()
        rows.append(_measure(P, x, x_new, delta, prev_delta, i, rope, dims, ref_final, clean_states, aligned))
        prev_delta, x = delta, x_new
    P.renorm = False   # restore (renorm variant must not leak into later variants)
    return rows


def _build_specs(a):
    """Enumerate EVERY variant as its own line: baseline + K-sweep + schedules + renorm + ablate +
    all perturbation kinds x levels x injection-steps + photometric guidance. Each gets a plot style."""
    from matplotlib import cm
    palette = {"gauss": "#1f77b4", "scale": "#ff7f0e", "dir": "#2ca02c",
               "swap": "#d62728", "cam": "#9467bd", "register": "#8c564b"}
    specs = [dict(name="baseline", family="base", K=16, color="k", ls="--", lw=2.6, alpha=1.0, legend="baseline K=16")]

    Kother = [k for k in a.Ks if k != 16]
    for n, K in enumerate(Kother):
        c = cm.winter(n / max(len(Kother) - 1, 1))
        specs.append(dict(name=f"K={K}", family="K", K=K, color=c, ls="-", lw=1.4, alpha=0.9, legend=f"K={K}"))

    schedcols = cm.tab10(np.linspace(0, 1, 10))
    for n, s in enumerate([s for s in a.schedules if s != "baseline"]):
        specs.append(dict(name=f"sched:{s}", family="sched", sched=s, sched_K=a.sched_K, frozen_t=a.frozen_t,
                          sched_steps=a.sched_steps, color=schedcols[n % 10], ls="-.", lw=1.6, alpha=0.95, legend=f"sched:{s}"))

    specs.append(dict(name="renorm", family="renorm", K=16, renorm=True, color="#e377c2", ls="-", lw=1.8, alpha=0.95, legend="renorm"))

    abcols = {"first_half": "#17becf", "second_half": "#bcbd22", "all": "#7f7f7f"}
    for ab in [x for x in a.ablate if x != "none"]:
        specs.append(dict(name=f"ablate:{ab}", family="ablate", K=16, ablate_mode=ab, aligned=True,
                          color=abcols.get(ab, "#555"), ls=":", lw=1.9, alpha=0.95, legend=f"ablate:{ab}"))

    epsvals = sorted(set(a.eps)) or [1.0]
    def alpha_of(eps):
        return 0.9 if len(epsvals) == 1 else 0.28 + 0.62 * (epsvals.index(eps) / (len(epsvals) - 1))
    for at in a.at:
        for kind in a.kinds:
            eps_list = [a.eps[0]] if kind == "swap" else a.eps   # swap ignores eps
            for eps in eps_list:
                specs.append(dict(name=f"{kind}@{at}_e{eps:g}", family="perturb", K=16, perturb=(kind, eps, at),
                                  aligned=True, color=palette.get(kind, "#333"), ls="-", lw=1.1, alpha=alpha_of(eps),
                                  legend=None))

    specs.append(dict(name="photo-guided", family="photo", K=16, start=a.start, eta=a.eta, iters=a.iters,
                      stride=a.stride, color="#ff9900", ls=(0, (1, 1)), lw=2.3, alpha=1.0, legend="photo-guided"))
    return specs, palette


def cmd_unified(P, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    images = P.load(a.input)
    gen = torch.Generator(device=images.device).manual_seed(0)

    # ---- anchor 1: K=16 baseline (clean states + final decoded output). anchor 2: GT (via gt_* metrics). ----
    x, rope, dims = P.init_state(images)
    B, S, H, W = dims
    clean_states = [x]
    for i in range(16):
        with torch.no_grad():
            clean_states.append(P.step(clean_states[-1], i, 16, rope, B, S))
    with torch.no_grad():
        ref_final = {k: v.float().cpu() for k, v in P.decode(clean_states[-1], rope, dims).items() if k in ("depth", "rays")}
        conv = detect_convention(_pred_f(P, clean_states[-1], rope, dims), dims)

    specs, palette = _build_specs(a)
    print(f"{len(specs)} variants, {len(UNIFIED_METRICS)} metrics each (conv={conv})\n")
    variants = {}
    for sp in specs:
        try:
            variants[sp["name"]] = _run_variant(P, images, sp, ref_final, clean_states, gen, conv)
        except Exception as e:
            print(f"!!! variant {sp['name']} FAILED: {e}")
            continue
        last = variants[sp["name"]][-1]
        print(f"{sp['name']:<22} final: |x|={last['tok_norm']:8.1f} dir_cos={last['tok_cos_step']:.3f} "
              f"eff_chan={last['eff_chan']:5.2f} cluster={last['cluster_cos']:.3f} "
              f"gt_absrel={last['gt_absrel']:.4f} depth->std={last['depth_to_std']:.4f}")

    os.makedirs(a.out, exist_ok=True)
    json.dump({k: v for k, v in variants.items()}, open(f"{a.out}/unified.json", "w"), indent=1)

    style = {sp["name"]: sp for sp in specs}
    # legend: one entry per non-perturb variant + one per perturbation KIND
    leg = [mlines.Line2D([], [], color=sp["color"], ls=sp["ls"], lw=sp["lw"], label=sp["legend"])
           for sp in specs if sp.get("legend")]
    leg += [mlines.Line2D([], [], color=palette[k], lw=1.4, label=f"perturb:{k}") for k in a.kinds]

    made = 0
    for key, label in UNIFIED_METRICS:
        allvals = [r.get(key, float("nan")) for v in variants.values() for r in v]
        if all((z != z) for z in allvals):   # all-NaN (e.g. GT metrics with no --gt) -> skip
            continue
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for nm, rows in variants.items():
            sp = style[nm]
            xs = [r["step"] for r in rows]
            ys = [r.get(key, float("nan")) for r in rows]
            ax.plot(xs, ys, color=sp["color"], ls=sp["ls"], lw=sp["lw"], alpha=sp["alpha"],
                    zorder=10 if sp["family"] == "base" else 5)
        ax.legend(handles=leg, fontsize=7, ncol=3, loc="best",
                  title="perturb: alpha increases with eps")
        ax.set_xlabel("loop step"); ax.set_ylabel(label); ax.grid(alpha=0.3)
        ax.set_title(f"{label}   ({len(variants)} variants)")
        fig.tight_layout(); fig.savefig(f"{a.out}/unified_{key}.png", dpi=130); plt.close(fig)
        made += 1
    print(f"\n{len(variants)} variants -> {made} per-dimension figures (unified_<metric>.png) in {a.out}/")


# ----------------------------------------------------------------------------- exp 15: Geshkovski diagnostics
# Measure DVLT's token dynamics with the "interacting particles on a sphere" lens of
# Geshkovski et al. (A mathematical perspective on Transformers). Four quantities, per step:
#   1. interaction energy  E_beta = mean_{i,j} exp(beta <z_i,z_j>)   (the Lyapunov functional of the
#      IDEAL symmetric flow; monotone rise = progressive clustering).
#   2. #clusters: k_eff = participation ratio of the token-Gram eigenvalues (threshold-free
#      "effective number of distinct directions") + connected components of the {<z_i,z_j> > tau} graph.
#   3. attention symmetry: asym(B_h)=||B-B^T||/||B+B^T|| for B_h = Wq_h^T Wk_h (STATIC, per head), plus a
#      data-driven logit-matrix corr(L_ij,L_ji). Near-symmetric => the "gradient flow / energy" story holds.
#   4. attractiveness: cos(update, weighted_center - self), tangential. ~1 => update really pulls each
#      token toward its attention-weighted neighbour mean (ideal attraction); ~0 => W_V does something else.
# Conventions (user-approved): uncentered unit directions z=x/|x|; a FIXED token subsample across steps;
# symmetry from the weights (B_h) primarily; attractiveness from the FULL step update; on GLOBAL attention.
# All derived from captured activations -- dvlt source untouched.

def _patch_global_idx(P, S, P_full):
    """Global-layout indices (in the (S*P_full) flattened sequence) of the image patch tokens."""
    pm, Np = P.m.patch_start_idx, P_full - P.m.patch_start_idx
    base = torch.arange(Np)
    return torch.cat([s * P_full + pm + base for s in range(S)])


def lyap_energy(zhat, betas):
    """E_beta = mean_{i,j} exp(beta <z_i,z_j>).  zhat: (n, C) unit rows. Higher = more clustered."""
    G = zhat @ zhat.t()                                  # (n,n) pairwise cosines
    return {float(b): torch.exp(b * G).mean().item() for b in betas}


def cluster_stats(zhat, taus):
    """k_eff (Gram-eigenvalue participation ratio) + number of NON-SINGLETON clusters (connected
    components of size >= 2) at each cosine threshold. Singletons are excluded so a spread-out cloud
    reads as "0 clusters" rather than n (which would just be counting isolated points)."""
    G = zhat @ zhat.t()
    ev = torch.linalg.eigvalsh(G.double()).clamp_min(0)  # PSD Gram eigenvalues
    k_eff = (ev.sum() ** 2 / ev.pow(2).sum().clamp_min(1e-12)).item()
    comps = {}
    Gc = G.cpu().numpy()
    n = Gc.shape[0]
    for tau in taus:
        parent = list(range(n))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for i in range(n):
            js = np.nonzero(Gc[i, i + 1:] > tau)[0] + (i + 1)
            for j in js:
                ri, rj = find(i), find(int(j))
                if ri != rj:
                    parent[ri] = rj
        size = {}
        for i in range(n):
            r = find(i); size[r] = size.get(r, 0) + 1
        comps[float(tau)] = sum(1 for s in size.values() if s >= 2)   # non-singleton clusters only
    return k_eff, comps


def attn_interaction_asym(attn):
    """Static per-head asymmetry of the interaction matrix B_h = Wq_h^T Wk_h (dim x dim).
    Decompose B = Sym + Antisym; asym = ||Antisym||_F / ||B||_F in [0,1] (0 = symmetric =
    gradient-flow-compatible, 1 = purely antisymmetric). This is the PRE-qk-norm proxy; the
    realized (post-LayerNorm) asymmetry is measured per step in cmd_geshkovski."""
    W = attn.qkv.weight.detach().float()                 # (3*dim, dim)
    dim = W.shape[1]; Hh = attn.num_heads; hd = dim // Hh
    Wq, Wk = W[:dim], W[dim:2 * dim]                      # rows = heads*head_dim
    out = []
    for h in range(Hh):
        wqh, wkh = Wq[h * hd:(h + 1) * hd], Wk[h * hd:(h + 1) * hd]   # (hd, dim)
        B = wqh.t() @ wkh                                # (dim, dim) input-space interaction
        anti = 0.5 * (B - B.t())
        out.append((anti.norm() / B.norm().clamp_min(1e-12)).item())   # ||Antisym||/||B|| in [0,1]
    return out


def _tangential(v, x):
    """component of v orthogonal to x (per row): v - (v.x/|x|^2) x."""
    xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return v - (v * xn).sum(-1, keepdim=True) * xn


@torch.no_grad()
def cmd_geshkovski(P, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images = P.load(a.input)
    B, S, _, H, W = images.shape
    assert B == 1, "geshkovski assumes B=1 (one scene)"
    K = a.K
    betas = a.betas
    taus = a.taus

    g_attn = P.m.recurrent_blocks[0].global_attn.attn
    f_attn = P.m.recurrent_blocks[0].frame_attn.attn
    # --- (3) STATIC attention symmetry (weights only; same every step) ---
    g_asym, f_asym = attn_interaction_asym(g_attn), attn_interaction_asym(f_attn)
    print(f"[symmetry] global B_h asym: mean={np.mean(g_asym):.3f} min={np.min(g_asym):.3f} per-head={[round(v,2) for v in g_asym]}")
    print(f"[symmetry] frame  B_h asym: mean={np.mean(f_asym):.3f} min={np.min(f_asym):.3f}")
    print("  (0=symmetric=gradient-flow-compatible; 1=purely antisymmetric)\n")

    cap = {}
    hk = g_attn.register_forward_pre_hook(
        lambda m, args, kw=None: cap.__setitem__("x", args[0].detach()), with_kwargs=True)
    # also capture the attention sublayer OUTPUT (what global attention actually adds this step),
    # so attractiveness compares the ISOLATED attention velocity -- not the full step (which also
    # contains frame attention + MLP). Attention.forward returns a plain tensor.
    hk2 = g_attn.register_forward_hook(
        lambda m, args, out: cap.__setitem__("o", (out[0] if isinstance(out, tuple) else out).detach()))

    x, rope, dims = P.init_state(images)
    P_full = x.shape[1]
    SP = S * P_full
    pidx = _patch_global_idx(P, S, P_full).to(x.device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    sub = pidx[torch.randperm(pidx.numel(), generator=gen)[:min(a.nsub, pidx.numel())]].to(x.device)  # FIXED across steps
    qsub = sub[:min(a.nq, sub.numel())]                  # fewer queries for the all-keys attractiveness pull
    scale = g_attn.head_dim ** -0.5

    rows = []
    try:
        for step in range(K):
            x_new = P.step(x, step, K, rope, dims[0], dims[1])
            xg = cap["x"][0]                              # (SP, C) global-attn input this step
            # all diagnostic math in fp32 (autocast off) so Gram/corrcoef/eigvalsh are stable
            with torch.autocast(device_type=P.acc.device.type, enabled=False):
                flat_new = x_new.reshape(SP, x_new.shape[-1]).float()

                # (1)+(2) energy & clusters on the END-OF-STEP state, uncentered unit dirs, FIXED subsample
                z = F.normalize(flat_new[sub], dim=-1)
                E = lyap_energy(z, betas)
                k_eff, comps = cluster_stats(z, taus)

                # (3) data-driven global logit symmetry on the fixed subsample
                qkv = g_attn.qkv(xg).reshape(SP, 3, g_attn.num_heads, g_attn.head_dim).permute(1, 2, 0, 3)
                q, k, _ = qkv.unbind(0); q, k = g_attn.q_norm(q).float(), g_attn.k_norm(k).float()   # (heads, SP, hd)
                qs, ks = q[:, sub], k[:, sub]                                        # (heads, n, hd)
                L = ((qs * scale) @ ks.transpose(-1, -2)).mean(0)                    # (n,n) head-mean logit
                off = ~torch.eye(L.shape[0], dtype=torch.bool, device=L.device)
                lu, ll = L[off], L.t()[off]
                sym_corr = torch.corrcoef(torch.stack([lu, ll]))[0, 1].item()
                # realized (post-qk-norm) logit asymmetry, bounded in [0,1]  (0 = symmetric)
                sym_asym = ((0.5 * (L - L.t())).norm() / L.norm().clamp_min(1e-9)).item()

                # (4) attractiveness: head-averaged global attention pull vs the ISOLATED global-attention
                #     OUTPUT (not the full step), both tangential at the attention input position xg.
                xg_all = xg.float()
                qq = q[:, qsub]                                                      # (heads, nq, hd)
                c = torch.zeros(qsub.numel(), xg.shape[-1], device=x.device)
                for h in range(g_attn.num_heads):
                    aw = ((qq[h] * scale) @ k[h].transpose(-1, -2)).softmax(-1)      # (nq, SP)
                    c = c + aw @ xg_all                                              # weighted centre (positions)
                c = c / g_attn.num_heads
                xi = xg_all[qsub]                                                    # self = attention input
                o = cap["o"][0].float()[qsub]                                        # global-attn sublayer output
                pull = _tangential(c - xi, xi)
                upd = _tangential(o, xi)                                             # isolated attention velocity
                attract = F.cosine_similarity(upd, pull, dim=-1).mean().item()

            r = dict(step=step, k_eff=k_eff, sym_corr=sym_corr, sym_asym=sym_asym, attract=attract,
                     **{f"E_b{int(b)}": E[float(b)] for b in betas},
                     **{f"ncomp_t{t}": comps[float(t)] for t in taus})
            rows.append(r)
            print(f"step {step:>2}  E_b{int(betas[0])}={E[float(betas[0])]:8.2f}  k_eff={k_eff:6.2f}  "
                  f"ncomp@{taus[0]}={comps[float(taus[0])]:>4}  sym_corr={sym_corr:+.3f}  attract={attract:+.3f}")
            x = x_new
    finally:
        hk.remove(); hk2.remove()

    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(rows=rows, global_asym=g_asym, frame_asym=f_asym), open(f"{a.out}/geshkovski.json", "w"), indent=1)

    st = [r["step"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for b in betas:
        ax[0, 0].plot(st, [r[f"E_b{int(b)}"] for r in rows], "-o", ms=3, label=f"beta={int(b)}")
    ax[0, 0].set_title("(1) interaction energy E_beta  (rise = clustering)"); ax[0, 0].set_yscale("log"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(st, [r["k_eff"] for r in rows], "-o", ms=3, color="#c44", label="k_eff (Gram)")
    for t in taus:
        ax[0, 1].plot(st, [r[f"ncomp_t{t}"] for r in rows], "-s", ms=3, label=f"#comp @cos>{t}")
    ax[0, 1].set_title("(2) #clusters"); ax[0, 1].set_yscale("log"); ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(st, [r["sym_corr"] for r in rows], "-o", ms=3)
    ax[1, 0].axhline(1, ls="--", color="grey", alpha=0.6)
    ax[1, 0].set_title(f"(3) global logit symmetry corr(L_ij,L_ji)\nstatic B_h asym mean={np.mean(g_asym):.3f}")
    ax[1, 1].plot(st, [r["attract"] for r in rows], "-o", ms=3)
    ax[1, 1].axhline(0, ls="--", color="grey", alpha=0.6)
    ax[1, 1].set_title("(4) attractiveness cos(update, weighted_centre - self)")
    for row in ax:
        for a_ in row:
            a_.set_xlabel("loop step"); a_.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{a.out}/geshkovski.png", dpi=130)
    print(f"\nsaved {a.out}/geshkovski.png  and  geshkovski.json")


# ----------------------------------------------------------------------------- exp 16: GRAND uniform table
# ONE clean pass with strict per-version logging. Two anchors (GT + K=16 baseline). Every version
# (baseline + each K / schedule / renorm / ablate / perturbation-level / photo = ~113) is run ONCE
# and logged to its OWN tsv with an IDENTICAL 52-column schema (one row per loop step). Then one
# figure per metric column overlays all versions as lines. Data is always saved so plots can be
# redrawn later. Attention-matrix / Lipschitz probes are intentionally NOT here (they are not a
# single per-step scalar); run `attn` / `smoothx` / `smoothinput` separately.
GRAND_STATE  = ["tok_norm", "norm_std", "cam_norm", "reg_norm"]
GRAND_CHANGE = ["upd_abs", "upd_rel", "upd_cos_prev", "tok_cos_step", "cam_upd_rel", "reg_upd_rel"]
GRAND_CHAN   = ["top1_share", "top8_share", "eff_chan"]
GRAND_CLUST  = ["cluster_cos", "E_b1", "E_b4", "E_b10", "k_eff", "ncomp_t0.8", "ncomp_t0.9", "ncomp_t0.95"]
GRAND_GATES  = ["g_frame_attn", "g_frame_mlp", "g_frame_out", "g_global_attn", "g_global_mlp", "g_global_out"]
GRAND_ATTN   = ["attn_entropy", "attn_self", "attn_push_rel", "attn_tang_frac", "sym_corr", "sym_asym", "attract"]
GRAND_GT     = ["gt_absrel", "gt_d1", "gt_d2", "gt_d3", "gt_rmse", "gt_ate"]
GRAND_VS     = ["depth_to_std", "ray_to_std", "depth_to_prev", "ray_to_prev"]
GRAND_REC    = ["tok_rel_to_clean"]
GRAND_METRICS = (GRAND_STATE + GRAND_CHANGE + GRAND_CHAN + GRAND_CLUST + GRAND_GATES
                 + GRAND_ATTN + GRAND_GT + GRAND_VS + GRAND_REC)
GRAND_IDS = ["version", "family", "kind", "eps", "at", "step", "t"]
GRAND_COLS = GRAND_IDS + GRAND_METRICS
NAN = float("nan")


def depth_metrics_full(pred_depth, gt):
    """AbsRel / delta<1.25^{1,2,3} / RMSE after per-sequence median-scale alignment."""
    p = pred_depth.float().reshape(gt.shape); mask = gt > 1e-6
    if mask.sum() < 10:
        return dict(gt_absrel=NAN, gt_d1=NAN, gt_d2=NAN, gt_d3=NAN, gt_rmse=NAN)
    sc = (gt[mask] / p[mask].clamp_min(1e-6)).median()
    q, g = (p * sc)[mask], gt[mask]
    rat = torch.maximum(q / g, g / q)
    return dict(gt_absrel=((q - g).abs() / g).mean().item(),
                gt_d1=(rat < 1.25).float().mean().item(),
                gt_d2=(rat < 1.25 ** 2).float().mean().item(),
                gt_d3=(rat < 1.25 ** 3).float().mean().item(),
                gt_rmse=((q - g) ** 2).mean().sqrt().item())


@torch.no_grad()
def _measure_grand(P, x, x_new, delta, prev_delta, rope, dims, ref_final, clean_states, aligned, step,
                   prev_depth, prev_rays, caps, g_attn, sub, qsub, flags):
    """Full 45-metric row for one step of any version. flags = (frame_gate_ok, global_gate_ok, attn_ok)."""
    m = P.m; pm = m.patch_start_idx
    frame_ok, global_ok, attn_ok = flags
    with torch.autocast(device_type=P.acc.device.type, enabled=False):
        x_new = x_new.detach(); delta = delta.detach()
        xp, xpo = x_new[:, pm:].float(), x[:, pm:].float()       # (BS,Np,C) patch tokens new/old
        dp = delta[:, pm:].float()
        pn = xp.norm(dim=-1); dn = dp.norm(dim=-1)
        r = {}
        # --- state ---
        r["tok_norm"] = pn.mean().item(); r["norm_std"] = pn.std().item()
        r["cam_norm"] = x_new[:, 0].float().norm(dim=-1).mean().item()
        r["reg_norm"] = x_new[:, 1:pm].float().norm(dim=-1).mean().item()
        # --- change ---
        r["upd_abs"] = dn.mean().item(); r["upd_rel"] = (dn / pn.clamp_min(1e-6)).mean().item()
        r["upd_cos_prev"] = (F.cosine_similarity(delta.flatten(1), prev_delta.flatten(1), dim=-1).mean().item()
                             if prev_delta is not None else NAN)
        r["tok_cos_step"] = F.cosine_similarity(xpo, xp, dim=-1).mean().item()
        cdo, cdn = delta[:, 0].float().norm(dim=-1), x_new[:, 0].float().norm(dim=-1)
        r["cam_upd_rel"] = (cdo / cdn.clamp_min(1e-6)).mean().item()
        rdo = delta[:, 1:pm].float().norm(dim=-1); rnn = x_new[:, 1:pm].float().norm(dim=-1)
        r["reg_upd_rel"] = (rdo / rnn.clamp_min(1e-6)).mean().item()
        # --- channel concentration ---
        e = xp.pow(2).mean(dim=(0, 1)); tot = e.sum().clamp_min(1e-9)
        r["top1_share"] = (e.max() / tot).item(); r["top8_share"] = (e.topk(8).values.sum() / tot).item()
        r["eff_chan"] = ((e.sum() ** 2) / e.pow(2).sum().clamp_min(1e-12)).item()
        # --- clustering on the FIXED subsample (end-of-step state) ---
        SP = x_new.shape[0] * x_new.shape[1]
        flat = x_new.reshape(SP, x_new.shape[-1]).float()
        zc = _unit(flat[sub])                                    # centered+normalized (cluster_cos)
        npair = zc.shape[0] * (zc.shape[0] - 1) / 2
        r["cluster_cos"] = ((zc @ zc.t()).triu(1).sum() / max(npair, 1)).item()
        z = F.normalize(flat[sub], dim=-1)                       # uncentered unit (energy/clusters)
        E = lyap_energy(z, [1.0, 4.0, 10.0]); r["E_b1"], r["E_b4"], r["E_b10"] = E[1.0], E[4.0], E[10.0]
        k_eff, comps = cluster_stats(z, [0.8, 0.9, 0.95])
        r["k_eff"] = k_eff
        r["ncomp_t0.8"], r["ncomp_t0.9"], r["ncomp_t0.95"] = comps[0.8], comps[0.9], comps[0.95]
        # --- gates (NaN where the sub-block did not run) ---
        gf, gg = P.gates.get("frame"), P.gates.get("global")
        r["g_frame_attn"], r["g_frame_mlp"], r["g_frame_out"] = (gf if frame_ok else (NAN, NAN, NAN))
        r["g_global_attn"], r["g_global_mlp"], r["g_global_out"] = (gg if global_ok else (NAN, NAN, NAN))
        # --- global-attention shape / symmetry / attractiveness ---
        for key in GRAND_ATTN:
            r[key] = NAN
        if attn_ok and caps.get("x") is not None and caps.get("o") is not None:
            xg = caps["x"][0]
            qkv = g_attn.qkv(xg).reshape(xg.shape[0], 3, g_attn.num_heads, g_attn.head_dim).permute(1, 2, 0, 3)
            q, k, _ = qkv.unbind(0); q, k = g_attn.q_norm(q).float(), g_attn.k_norm(k).float()
            scale = g_attn.head_dim ** -0.5
            qs, ks = q[:, sub], k[:, sub]
            L = ((qs * scale) @ ks.transpose(-1, -2)).mean(0)
            off = ~torch.eye(L.shape[0], dtype=torch.bool, device=L.device)
            r["sym_corr"] = torch.corrcoef(torch.stack([L[off], L.t()[off]]))[0, 1].item()
            r["sym_asym"] = ((0.5 * (L - L.t())).norm() / L.norm().clamp_min(1e-9)).item()
            xg_all = xg.float(); c = torch.zeros(qsub.numel(), xg.shape[-1], device=xg.device)
            ent = 0.0; selfw = 0.0
            for h in range(g_attn.num_heads):
                aw = ((q[h, qsub] * scale) @ k[h].transpose(-1, -2)).softmax(-1)   # (nq, SP)
                c = c + aw @ xg_all
                ent = ent + (-(aw.clamp_min(1e-12).log() * aw).sum(-1)).mean().item()
                selfw = selfw + aw[torch.arange(qsub.numel(), device=aw.device), qsub].mean().item()
            c = c / g_attn.num_heads
            o = caps["o"][0].float()[qsub]; xi = xg_all[qsub]
            r["attn_entropy"] = ent / g_attn.num_heads; r["attn_self"] = selfw / g_attn.num_heads
            r["attn_push_rel"] = (o.norm(dim=-1) / xi.norm(dim=-1).clamp_min(1e-9)).mean().item()
            ot = _tangential(o, xi)
            r["attn_tang_frac"] = (ot.norm(dim=-1) / o.norm(dim=-1).clamp_min(1e-9)).mean().item()
            r["attract"] = F.cosine_similarity(ot, _tangential(c - xi, xi), dim=-1).mean().item()
        # --- decode-based: GT, vs-standard, vs-previous ---
        pred = {kk: v.float().cpu() for kk, v in P.decode(x_new, rope, dims).items() if kk in ("depth", "rays")}
        if getattr(P, "gt_depth", None) is not None:
            S = pred["depth"].shape[1]
            r.update(depth_metrics_full(pred["depth"].to(P.gt_depth.device), P.gt_depth[:S]))
            if P.gt_c2w is not None:
                o3 = pred["rays"].to(P.gt_depth.device)[0, :, :, :, 3:6].mean(dim=(1, 2))
                r["gt_ate"] = sim3_ate(o3, P.gt_c2w[:S, :3, 3])
            else:
                r["gt_ate"] = NAN
        else:
            for kk in GRAND_GT:
                r[kk] = NAN
        r["depth_to_std"] = log_depth_dist(pred["depth"], ref_final["depth"])
        r["ray_to_std"] = ray_angle_dist(pred["rays"], ref_final["rays"])
        r["depth_to_prev"] = log_depth_dist(pred["depth"], prev_depth) if prev_depth is not None else NAN
        r["ray_to_prev"] = ray_angle_dist(pred["rays"], prev_rays) if prev_rays is not None else NAN
        if aligned and step + 1 < len(clean_states):
            cl = clean_states[step + 1].to(x_new.device)
            r["tok_rel_to_clean"] = (patch_part(x_new - cl, m).float().norm(dim=-1)
                                     / patch_part(cl, m).float().norm(dim=-1).clamp_min(1e-6)).mean().item()
        else:
            r["tok_rel_to_clean"] = NAN
    return r, pred["depth"], pred["rays"]


def _run_variant_grand(P, images, spec, ref_final, clean_states, gen, g_attn, sub, qsub, caps):
    """Drive ONE version through the loop, emitting a full GRAND row per step. Handles every family."""
    fam = spec["family"]; K = spec.get("K", 16)
    P.renorm = bool(spec.get("renorm", False)); P.renorm_from = spec.get("renorm_from", 0); P._renorm_frozen = None
    x, rope, dims = P.init_state(images); B, S, H, W = dims
    rows = []; prev_delta = None; pdep = pray = None
    pert = spec.get("perturb")
    id_kind, id_eps, id_at = (pert[0], pert[1], pert[2]) if pert else ("", "", "")

    def emit(xa, xb, step, t, aligned, flags):
        nonlocal prev_delta, pdep, pray
        delta = (xb - xa).float()
        r, pdep, pray = _measure_grand(P, xa, xb, delta, prev_delta, rope, dims, ref_final, clean_states,
                                       aligned, step, pdep, pray, caps, g_attn, sub, qsub, flags)
        r.update(version=spec["name"], family=fam, kind=id_kind, eps=id_eps, at=id_at, step=step, t=t)
        rows.append(r); prev_delta = delta

    if fam == "sched":
        pairs = build_schedule(spec["sched"], spec.get("sched_K", 16), spec.get("frozen_t", 0.5), spec.get("sched_steps", 16))
        for i, (tn, tnx) in enumerate(pairs):
            caps.clear()
            with torch.no_grad():
                xb = P.step_t(x, tn, tnx, rope, B, S)
            emit(x, xb, i, tn, False, (True, True, True)); x = xb
        P.renorm = False; return rows

    if fam == "photo":
        ts = torch.linspace(0, 1, K).tolist(); start = spec["start"]; si = 0
        for i in range(start):
            caps.clear()
            with torch.no_grad():
                xb = P.step(x, i, K, rope, B, S)
            emit(x, xb, si, ts[i], True, (True, True, True)); x = xb; si += 1
        xcur = x.detach().clone()
        for _ in range(spec["iters"]):
            caps.clear(); xcur.requires_grad_(True)
            with torch.enable_grad(), P.acc.autocast():
                pr = _pred_f(P, xcur, rope, dims)
                Lp, _ = photometric_residual(pr, images, dims, spec.get("conv", "c2w"), stride=spec.get("stride", 2))
                gg, = torch.autograd.grad(Lp, xcur)
            with torch.no_grad():
                gn = gg.float().norm().clamp_min(1e-12)
                xb = (xcur.float() - spec["eta"] * xcur.float().norm() * (gg.float() / gn)).to(x.dtype).detach()
            emit(xcur, xb, si, NAN, False, (False, False, False)); xcur = xb; si += 1
        x = xcur
        for i in range(start, K):
            caps.clear()
            with torch.no_grad():
                xb = P.step(x, i, K, rope, B, S)
            emit(x, xb, si, ts[i], False, (True, True, True)); x = xb; si += 1
        P.renorm = False; return rows

    # base / K / perturb / ablate / renorm
    ts = torch.linspace(0, 1, K).tolist(); ab = spec.get("ablate_mode"); aligned = spec.get("aligned", fam == "base")
    for i in range(K):
        if pert is not None and i == pert[2]:
            x = make_perturbation(x, pert[0], pert[1], P, gen)
        caps.clear()
        if ab:
            on = _ablate_on(ab, i, K)
            with torch.no_grad():
                xb = P.step_ablate(x, i, K, rope, B, S, ablate_global=on)
            flags = (True, not on, not on)
        else:
            with torch.no_grad():
                xb = P.step(x, i, K, rope, B, S)
            flags = (True, True, True)
        emit(x, xb, i, ts[i], aligned, flags); x = xb
    P.renorm = False; return rows


def _grand_fmt(v):
    if v is None or (isinstance(v, str) and v == ""):
        return "" if v is None else ""
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.6g}"
    return str(v)


def cmd_grand(P, a):
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    images = P.load(a.input); B, S, _, H, W = images.shape
    assert B == 1, "grand assumes B=1 (one scene)"
    g_attn = P.m.recurrent_blocks[0].global_attn.attn
    f_attn = P.m.recurrent_blocks[0].frame_attn.attn
    g_asym, f_asym = attn_interaction_asym(g_attn), attn_interaction_asym(f_attn)

    # chunk the per-frame decoders over frames so decoding 20 frames every step does not OOM
    # (mathematically identical -- the ray/depth heads are fully per-frame; see decode_chunk_size doc).
    P.m.decode_chunk_size = max(1, min(getattr(P.m, "decode_chunk_size", None) or 128, a.decode_chunk))

    # --- anchors: K=16 baseline (clean states + final decoded) ; GT via P.gt_* ---
    # MUST be under no_grad: otherwise the 16-step x20-frame graph is retained and OOMs the GPU.
    with torch.no_grad():
        x, rope, dims = P.init_state(images); P_full = x.shape[1]
        clean_states = [x]
        for i in range(16):
            clean_states.append(P.step(clean_states[-1], i, 16, rope, dims[0], dims[1]))
        ref_final = {k: v.float().cpu() for k, v in P.decode(clean_states[-1], rope, dims).items() if k in ("depth", "rays")}
        conv = detect_convention(_pred_f(P, clean_states[-1], rope, dims), dims)
        clean_states = [c.detach().cpu() for c in clean_states]   # keep anchors off-GPU

    pidx = _patch_global_idx(P, S, P_full).to(x.device)
    g2 = torch.Generator(device="cpu").manual_seed(0)
    sub = pidx[torch.randperm(pidx.numel(), generator=g2)[:min(a.nsub, pidx.numel())]].sort().values.to(x.device)
    qsub = sub[:min(a.nq, sub.numel())]

    caps = {}
    hk = g_attn.register_forward_pre_hook(lambda m, args, kw=None: caps.__setitem__("x", args[0].detach()), with_kwargs=True)
    hk2 = g_attn.register_forward_hook(lambda m, args, out: caps.__setitem__("o", (out[0] if isinstance(out, tuple) else out).detach()))

    specs, palette = _build_specs(a)
    for sp in specs:
        if sp["family"] == "photo":
            sp["conv"] = conv
    gen = torch.Generator(device=images.device).manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    vdir = os.path.join(a.out, "versions"); os.makedirs(vdir, exist_ok=True)
    print(f"{len(specs)} versions, {len(GRAND_METRICS)} metric cols each (conv={conv})")
    print(f"global B_h asym mean={np.mean(g_asym):.3f} | frame mean={np.mean(f_asym):.3f}\n")

    all_rows = {}
    try:
        for sp in specs:
            try:
                rows = _run_variant_grand(P, images, sp, ref_final, clean_states, gen, g_attn, sub, qsub, caps)
            except Exception as ex:
                print(f"!!! {sp['name']} FAILED: {ex}"); continue
            all_rows[sp["name"]] = rows
            fn = os.path.join(vdir, re.sub(r"[^A-Za-z0-9._-]", "_", sp["name"]) + ".tsv")
            with open(fn, "w") as fh:
                fh.write("\t".join(GRAND_COLS) + "\n")
                for r in rows:
                    fh.write("\t".join(_grand_fmt(r.get(c)) for c in GRAND_COLS) + "\n")
            la = rows[-1]
            print(f"{sp['name']:<22} rows={len(rows):>3}  gt_absrel={_grand_fmt(la.get('gt_absrel'))}  "
                  f"k_eff={_grand_fmt(la.get('k_eff'))}  attract={_grand_fmt(la.get('attract'))}")
    finally:
        hk.remove(); hk2.remove()

    with open(os.path.join(a.out, "all.tsv"), "w") as fh:
        fh.write("\t".join(GRAND_COLS) + "\n")
        for nm, rows in all_rows.items():
            for r in rows:
                fh.write("\t".join(_grand_fmt(r.get(c)) for c in GRAND_COLS) + "\n")
    json.dump(dict(global_asym=g_asym, frame_asym=f_asym, n_versions=len(all_rows), cols=GRAND_COLS),
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)

    # --- one figure per metric column: every version a line ---
    style = {sp["name"]: sp for sp in specs}
    leg = [mlines.Line2D([], [], color=sp["color"], ls=sp["ls"], lw=sp["lw"], label=sp["legend"])
           for sp in specs if sp.get("legend")]
    leg += [mlines.Line2D([], [], color=palette[k], lw=1.4, label="perturb:" + k) for k in a.kinds]
    fdir = os.path.join(a.out, "figs"); os.makedirs(fdir, exist_ok=True); made = 0
    for col in GRAND_METRICS:
        vals = [r.get(col) for rows in all_rows.values() for r in rows]
        if all((v is None) or (isinstance(v, float) and v != v) for v in vals):
            continue
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for nm, rows in all_rows.items():
            sp = style[nm]
            ax.plot([r["step"] for r in rows], [r.get(col, NAN) for r in rows],
                    color=sp["color"], ls=sp["ls"], lw=sp["lw"], alpha=sp["alpha"],
                    zorder=10 if sp["family"] == "base" else 5)
        ax.legend(handles=leg, fontsize=7, ncol=3, title="perturb: alpha ~ eps")
        ax.set_xlabel("loop step"); ax.set_ylabel(col); ax.grid(alpha=0.3)
        ax.set_title(f"{col}   ({len(all_rows)} versions)")
        fig.tight_layout(); fig.savefig(os.path.join(fdir, f"{col}.png"), dpi=120); plt.close(fig); made += 1
    print(f"\n{len(all_rows)} versions -> {vdir}/*.tsv + all.tsv + meta.json ; {made} figures in {fdir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["traj", "ksweep", "perturb", "photo", "kgt", "sched", "ablate", "attn",
                                    "flowdump", "channels", "smoothx", "smoothinput", "sweep", "unified",
                                    "geshkovski", "grand"])
    ap.add_argument("--input", default="/storage/group/dataset_mirrors/01_incoming/scannetpp/data/00777c41d4/dslr/resized_images")
    ap.add_argument("--ckpt", default="nvidia/dvlt")
    ap.add_argument("--out", default="probe_out",
                    help="output subdir name; always nested under the top-level 'out/' folder")
    ap.add_argument("--out-root", default="out", help="top-level folder that wraps every run's output")
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--ref-K", type=int, default=16)
    ap.add_argument("--Ks", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--at", type=int, nargs="+", default=[2, 8, 14])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.2])
    ap.add_argument("--kind", default="gauss", choices=["gauss", "scale", "dir", "swap", "cam", "register"])
    ap.add_argument("--start", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--gt", help="npz with depth (S,h,w) and optional c2w (S,4,4); enables GT columns in every command")
    ap.add_argument("--every", type=int, default=1, help="kgt/sched: decode every N steps (use 4 for K=64)")
    # --- sched (exp 6: time-schedule tricks) ---
    ap.add_argument("--schedules", nargs="+", default=["baseline", "half", "reverse", "shuffle", "frozen", "double"],
                    choices=["baseline", "half", "reverse", "shuffle", "frozen", "double", "skip", "repeat"],
                    help="sched: which time schedules to run and overlay")
    ap.add_argument("--sched-K", type=int, default=16, help="sched: base linspace resolution for the t-table")
    ap.add_argument("--sched-steps", type=int, default=16, help="sched: number of steps for the 'frozen' schedule")
    ap.add_argument("--frozen-t", type=float, default=0.5, help="sched/channels: the fixed t used by the 'frozen' schedule")
    ap.add_argument("--frozen", action="store_true", help="channels: iterate the SAME map at --frozen-t for --K steps (autonomous iteration -> attractor), instead of a linspace(0,1,K) grid")
    ap.add_argument("--mod-at", type=int, default=8, help="sched: step index to skip/repeat (for 'skip'/'repeat')")
    ap.add_argument("--mod-n", type=int, default=2, help="sched: how many times to apply the repeated step ('repeat')")
    # --- ablate (exp 7: global-attention ablation) ---
    ap.add_argument("--ablate", nargs="+", default=["none", "first_half", "second_half", "all"],
                    choices=["none", "first_half", "second_half", "all"],
                    help="ablate: which global-attention-off conditions to run and overlay")
    # --- attn (exp 8: attention maps vs GT correspondences) ---
    ap.add_argument("--attn-steps", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16],
                    help="attn: which steps to export + score the global attention at")
    ap.add_argument("--attn-stride", type=int, default=2, help="attn: query-patch subsampling stride (bigger=fewer queries)")
    # --- flowdump (exp 9: dump flow.json for dvlt_flow_viewer.html) ---
    ap.add_argument("--ntok", type=int, default=1500, help="flowdump: number of sampled tokens for the particle view")
    ap.add_argument("--no-depth", action="store_true", help="flowdump: skip decoding depth at every state")
    # --- smoothx / smoothinput (exp 11/12) ---
    ap.add_argument("--reps", type=int, default=4, help="smoothx/smoothinput: random repeats averaged per eps")
    # --- geshkovski (exp 15: interacting-particle diagnostics) ---
    ap.add_argument("--betas", type=float, nargs="+", default=[1.0, 4.0, 10.0], help="geshkovski: inverse temperatures for the interaction energy")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.8, 0.9, 0.95], help="geshkovski: cosine thresholds for connected-component cluster counting")
    ap.add_argument("--nsub", type=int, default=1024, help="geshkovski: fixed #tokens subsampled (across steps) for energy/clusters/symmetry")
    ap.add_argument("--nq", type=int, default=256, help="geshkovski: #query tokens for the all-keys attractiveness pull")
    ap.add_argument("--decode-chunk", type=int, default=4, help="grand: frames per decode chunk (lower = less GPU memory when decoding every step at many frames)")
    # --- sweep (exp 13: big perturbation sweep) ---
    ap.add_argument("--kinds", nargs="+", default=["gauss", "scale", "dir", "swap", "cam", "register"],
                    choices=["gauss", "scale", "dir", "swap", "cam", "register"],
                    help="sweep: which perturbation kinds to sweep over")
    ap.add_argument("--finish", action="store_true", help="photo[B]: run remaining standard steps after injection")
    ap.add_argument("--renorm", action="store_true", help="photo[B]: renormalize patch tokens to a fixed norm")
    ap.add_argument("--renorm-target", type=float, default=None, help="photo[B]: target norm for renormalization")
    ap.add_argument("--renorm-from", type=int, default=0, help="photo[B]: step index from which to start renormalization")

    a = ap.parse_args()

    # Always nest each run under the top-level out/ folder for tidiness, unless the
    # user already passed a path that lives under it (e.g. --out out/foo).
    if os.path.normpath(a.out).split(os.sep)[0] != a.out_root:
        a.out = os.path.join(a.out_root, a.out)

    P = Probe(a.ckpt, precision=a.precision, max_frames=a.max_frames,
          renorm=a.renorm, renorm_target=a.renorm_target, renorm_from=a.renorm_from)

    # Load ground truth once (shared by gt_metrics in every command). cmd_kgt reloads
    # its own copy, but this makes the GT columns available to traj/ksweep/perturb/photo too.
    if a.gt:
        P.gt_depth, P.gt_c2w, P.gt_K = load_gt(a.gt, P.img_size, device=P.acc.device)

    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, f"{a.cmd}.txt")

    # Tee stdout into out/<run>/<cmd>.txt so the printed tables can be re-read later.
    with open(log_path, "w") as logf:
        sys.stdout = Tee(logf)
        try:
            with P.acc.autocast():
                {"traj": cmd_traj, "ksweep": cmd_ksweep, "perturb": cmd_perturb,
                 "photo": cmd_photo, "kgt": cmd_kgt, "sched": cmd_sched,
                 "ablate": cmd_ablate, "attn": cmd_attn, "flowdump": cmd_flowdump,
                 "channels": cmd_channels, "smoothx": cmd_smoothx, "smoothinput": cmd_smoothinput,
                 "sweep": cmd_sweep, "unified": cmd_unified,
                 "geshkovski": cmd_geshkovski, "grand": cmd_grand}[a.cmd](P, a)
        finally:
            sys.stdout = sys.__stdout__   # restore even if the command raises

    print(f"[log] console output saved to {log_path}")


if __name__ == "__main__":
    main()
