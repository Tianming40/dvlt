#!/usr/bin/env python3
"""camswap.py -- can the camera token "process" patch tokens? Swap frame A's special tokens with frame B's.

Frozen DVLT, no training. Baseline = normal K-step run, decoded by DVLT's own heads (no GT needed).
Variants (for each pair A:B and each swap set):
  A  : decode only. After the normal K steps, decode frame A with frame B's special tokens (loop untouched).
       Decoding is per-frame independent, so only frame A's outputs can change.
  B1 : in the loop. After `swap_step` steps, overwrite frame A's special tokens in the STATE with frame B's,
       then run the remaining steps and decode (both frame- and global-attention see the swap).
  B2 : in the loop, FRAME attention only. From `swap_step` on, frame A's PATCHES are computed by frame attention
       that sees frame B's special tokens; frame A's special tokens themselves and everything in global attention
       stay as in the normal computation (frame attention is run twice for frame A: normal + swapped).
  D1 : inside the RAY DECODER (loop untouched). Ray decoder = proj_in -> block0 -> block1 -> norm -> per-patch linear
       head -> rays; the camera head reads token 0 after block1. D1 swaps frame A's special tokens with frame B's
       right BEFORE block1 (state after proj_in+block0), so A's patches read B's partially-processed token 0 in the
       last decoder block. No extra computation.
  D2 : inside the ray decoder. Run both blocks, swap in frame B's FULLY processed special tokens, then apply block1
       ONE EXTRA time to frame A, then the head. The extra application changes A even without a swap: compare with
       the control pair A:A, which measures exactly that.
  (D1/D2 only touch the ray decoder; the depth decoder is unchanged, so depth_change is 0 by construction and the
   camera head reads the swapped token 0 directly, so pose_toB is not informative for them.)
Swap sets: cam = token 0;  camreg = tokens 0..4 (cam + 4 registers).
Use pair A:A as a control (must give zero change). Frame 0 (reference cam variant) can be given as B.

Metrics on frame A (all from DVLT's own decoded rays/depth; convention-free):
  centre_toB   = |c_A' - c_B| / |c_A - c_B|        (0 = A's camera centre moved onto B's, 1 = unchanged)
  dir_toB_deg  = mean angle(d_A', d_B) per pixel    (vs dir_AB_deg = mean angle(d_A, d_B): 0 = A's rays became B's)
  rot_resid    = angle between R(d_A -> d_A') and R(d_A -> d_B) (Kabsch over pixels)
  depth_change = median |D_A'/D_A - 1|              (expected ~0 if only the camera moved)
  pts_vs_exp   = mean |X_A' - (o_B + d_B * D_A)| / scene_scale   (expected 0 if A's local geometry is re-placed under B's camera)
  pts_moved    = mean |X_A' - X_A| / scene_scale
  pose_toB     = |pe_A' - pe_B| / |pe_A - pe_B|     (camera-head pose_enc)
  others_depth / others_centre : change on frames != A (B1/B2: the swap can propagate through global attention)
  persistence (B1/B2): cos(A's cam token, baseline A) and cos(A's cam token, baseline B) at the last step
  patchA_change: relative change of frame A's patch tokens at the last step vs baseline

  PYTHONPATH=$(pwd) python shovel/camswap.py --input out/data/gt_00777c41d4_s10/probe_input --max-frames 27 \
      --K 16 --swap-step 8 --variants A B1 B2 --swap cam camreg --pairs 5:5 4:5 12:13 3:20 8:24 6:0 --out out/stage3_latent/camswap
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlt_probe import Probe  # noqa: E402

SETS = {"cam": slice(0, 1), "camreg": slice(0, 5)}


def kabsch(a, b):
    """rotation R with R a ~ b. a,b: (n,3)."""
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def to_np(pred):
    rays = pred["rays"][0].float().cpu().numpy()                 # (S,H,W,6)
    return dict(o=rays[..., 3:6], d=rays[..., :3], D=pred["depth"][0, ..., 0].float().cpu().numpy(),
                pe=pred["pose_enc"][0].float().cpu().numpy())


def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)


def metrics(sw, ref, a, b, sub=4):
    c = lambda P, f: P["o"][f].reshape(-1, 3).mean(0)
    cA, cB, cS = c(ref, a), c(ref, b), c(sw, a)
    dAB = np.linalg.norm(cA - cB)
    flat = lambda P, f: unit(P["d"][f][::sub, ::sub].reshape(-1, 3))
    dA, dB, dS = flat(ref, a), flat(ref, b), flat(sw, a)          # A's rays, B's rays, A's rays after the swap
    ang = lambda u, v: float(np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1))).mean())
    R_obs, R_exp = kabsch(dA, dS), kabsch(dA, dB)
    XA = ref["o"][a] + ref["d"][a] * ref["D"][a][..., None]
    XS = sw["o"][a] + sw["d"][a] * sw["D"][a][..., None]
    Xexp = ref["o"][b] + ref["d"][b] * ref["D"][a][..., None]
    scale = np.linalg.norm(XA - XA.reshape(-1, 3).mean(0), axis=-1).mean()
    peAB = np.linalg.norm(ref["pe"][a] - ref["pe"][b])
    others = [f for f in range(ref["D"].shape[0]) if f != a]
    traj = np.linalg.norm(ref["o"].reshape(ref["o"].shape[0], -1, 3).mean(1) -
                          ref["o"].reshape(ref["o"].shape[0], -1, 3).mean((0, 1)), axis=-1).mean()
    if a == b:                                   # control pair: A-B distances are 0 -> use trajectory / mean scales
        dAB, peAB = traj, float(np.linalg.norm(ref["pe"], axis=-1).mean())
    return dict(
        centre_toB=float(np.linalg.norm(cS - cB) / max(dAB, 1e-9)),
        centre_moved=float(np.linalg.norm(cS - cA) / max(dAB, 1e-9)),
        dir_toB_deg=ang(dS, dB), dir_toA_deg=ang(dS, dA), dir_AB_deg=ang(dA, dB),
        rot_obs_deg=rot_angle(R_obs), rot_exp_deg=rot_angle(R_exp), rot_resid_deg=rot_angle(R_obs.T @ R_exp),
        depth_change=float(np.median(np.abs(sw["D"][a] / np.clip(ref["D"][a], 1e-6, None) - 1))),
        pts_vs_exp=float(np.linalg.norm(XS - Xexp, axis=-1).mean() / scale),
        pts_moved=float(np.linalg.norm(XS - XA, axis=-1).mean() / scale),
        pose_toB=float(np.linalg.norm(sw["pe"][a] - ref["pe"][b]) / max(peAB, 1e-9)),
        others_depth=float(np.mean([np.median(np.abs(sw["D"][f] / np.clip(ref["D"][f], 1e-6, None) - 1)) for f in others])),
        others_centre=float(np.mean([np.linalg.norm(c(sw, f) - c(ref, f)) for f in others]) / max(traj, 1e-9)),
    )


def make_ray_forward(dec, mode, A, Bf, sl):
    """Replicates DecoderHead.forward (linear head) with a swap of frame A's special tokens inside the decoder."""
    ps = dec.patch_size

    def fwd(x, H, W, patch_start_idx=0, pos=None):
        Bx = x.shape[0]
        ph, pw = H // ps, W // ps
        x = dec.proj_in(x)
        if mode == "D1":
            x = dec.blocks[0](x, pos=pos)
            x = x.clone(); x[A, sl] = x[Bf, sl]
            for blk in dec.blocks[1:]:
                x = blk(x, pos=pos)
        else:                                                          # D2
            for blk in dec.blocks:
                x = blk(x, pos=pos)
            x = x.clone(); x[A, sl] = x[Bf, sl]
            xa = dec.blocks[-1](x[A:A + 1], pos=pos[A:A + 1] if pos is not None else None)
            x[A] = xa[0]
        features = x
        y = dec.norm(x)[:, patch_start_idx:]
        with torch.amp.autocast("cuda", enabled=False):
            y = dec.head(y.float())
            y = y.transpose(-1, -2).view(Bx, -1, ph, pw)
            y = F.pixel_shuffle(y, ps)
        return y, features
    return fwd


@torch.no_grad()
def run_loop(P, x, rope, dims, K, start, record=False):
    B, S = dims[0], dims[1]
    rec = []
    for i in range(start, K):
        x = P.step(x, i, K, rope, B, S)
        if record:
            rec.append(x[:, :5].float().clone())
    return x, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/data/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--ckpt", default="nvidia/dvlt")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--swap-step", type=int, default=8, help="number of normal steps before the swap takes effect")
    ap.add_argument("--variants", nargs="+", default=["A", "B1", "B2"])
    ap.add_argument("--swap", nargs="+", default=["cam", "camreg"])
    ap.add_argument("--pairs", nargs="+", default=["5:5", "4:5", "12:13", "3:20", "8:24", "6:0"])
    ap.add_argument("--out", default="out/stage3_latent/camswap")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pairs = [tuple(int(v) for v in p.split(":")) for p in a.pairs]

    P = Probe(a.ckpt, max_frames=a.max_frames)
    images = P.load(a.input)
    x0, rope, dims = P.init_state(images)
    B, S, H, W = dims
    K, s0 = a.K, a.swap_step
    assert all(0 < p[0] < S and 0 <= p[1] < S for p in pairs), "A must be 1..S-1 (frame 0 = reference), B in 0..S-1"

    # ---- baseline
    with torch.no_grad():
        x_s, _ = run_loop(P, x0.clone(), rope, dims, s0, 0)            # state after s0 steps
        x_T, base_rec = run_loop(P, x_s.clone(), rope, dims, K, s0, record=True)
        ref = to_np(P.decode(x_T, rope, dims))
    print(f"baseline done: S={S} K={K} swap after {s0} steps; pairs {pairs}")

    fa = P.m.recurrent_blocks[0].frame_attn
    fa_orig = fa.forward
    results = []
    for var in a.variants:
        for sname in a.swap:
            sl = SETS[sname]
            for (A, Bf) in pairs:
                with torch.no_grad():
                    rec = []
                    if var == "A":
                        xm = x_T.clone(); xm[A, sl] = x_T[Bf, sl]
                        x_fin = x_T
                        sw = to_np(P.decode(xm, rope, dims))
                    elif var in ("D1", "D2"):
                        dec = P.m.ray_decoder
                        assert dec.head_type == "linear", "D1/D2 replicate the linear ray head only"
                        dec_orig, chunk_orig = dec.forward, P.m.decode_chunk_size
                        dec.forward = make_ray_forward(dec, var, A, Bf, sl)
                        P.m.decode_chunk_size = None                     # all frames in one decoder call -> global frame indices
                        try:
                            sw = to_np(P.decode(x_T, rope, dims))
                        finally:
                            dec.forward, P.m.decode_chunk_size = dec_orig, chunk_orig
                        x_fin = x_T
                    elif var == "B1":
                        xm = x_s.clone(); xm[A, sl] = x_s[Bf, sl]
                        x_fin, rec = run_loop(P, xm, rope, dims, K, s0, record=True)
                        sw = to_np(P.decode(x_fin, rope, dims))
                    else:                                                  # B2: frame attention only
                        def fwd(x, k=None, pos=None, A=A, Bf=Bf, sl=sl):
                            out = fa_orig(x, k, pos=pos)
                            xa = x[A:A + 1].clone(); xa[:, sl] = x[Bf:Bf + 1, sl]
                            ka = k[A:A + 1] if (k is not None and torch.is_tensor(k) and k.dim() > 0 and k.shape[0] == x.shape[0]) else k
                            pa = pos[A:A + 1] if pos is not None else None
                            oa = fa_orig(xa, ka, pos=pa)
                            out = out.clone(); out[A, 5:] = oa[0, 5:]           # only A's patches see the swapped specials
                            return out
                        fa.forward = fwd
                        try:
                            x_fin, rec = run_loop(P, x_s.clone(), rope, dims, K, s0, record=True)
                        finally:
                            fa.forward = fa_orig
                        sw = to_np(P.decode(x_fin, rope, dims))
                m = metrics(sw, ref, A, Bf)
                m.update(variant=var, swap=sname, A=A, B=Bf)
                if var in ("B1", "B2"):
                    camA = x_fin[A, 0].float()
                    m["cam_cos_baseA"] = float(F.cosine_similarity(camA, x_T[A, 0].float(), dim=0))
                    m["cam_cos_baseB"] = float(F.cosine_similarity(camA, x_T[Bf, 0].float(), dim=0))
                    pa, pb = x_fin[A, 5:].float(), x_T[A, 5:].float()
                    m["patchA_change"] = float((pa - pb).norm() / pb.norm())
                    m["cam_cos_baseA_per_step"] = [float(F.cosine_similarity(r[A, 0], br[A, 0], dim=0)) for r, br in zip(rec, base_rec)]
                results.append(m)
                print(f"[{var:>2} {sname:>6} {A:>2}->{Bf:>2}] centre_toB={m['centre_toB']:.3f} dir_toB={m['dir_toB_deg']:.2f}° "
                      f"(A-B {m['dir_AB_deg']:.2f}°) rot_resid={m['rot_resid_deg']:.2f}° depthΔ={m['depth_change']:.4f} "
                      f"pts_vs_exp={m['pts_vs_exp']:.3f} pts_moved={m['pts_moved']:.3f} pose_toB={m['pose_toB']:.3f} "
                      f"others(depth {m['others_depth']:.4f}, centre {m['others_centre']:.4f})"
                      + (f" camcos A/B={m['cam_cos_baseA']:.3f}/{m['cam_cos_baseB']:.3f} patchΔ={m['patchA_change']:.4f}" if var in ("B1", "B2") else ""))

    tag = f"K{K}_s{s0}" + ("" if a.variants == ["A", "B1", "B2"] else "_" + "-".join(a.variants))
    json.dump(dict(K=K, swap_step=s0, S=S, pairs=a.pairs, results=results), open(f"{a.out}/camswap_{tag}.json", "w"), indent=1)
    keys = ["centre_toB", "centre_moved", "dir_toB_deg", "dir_AB_deg", "rot_resid_deg", "depth_change", "pts_vs_exp",
            "pts_moved", "pose_toB", "others_depth", "others_centre", "cam_cos_baseA", "cam_cos_baseB", "patchA_change"]
    with open(f"{a.out}/summary_{tag}.txt", "w") as fh:
        fh.write(" ".join(sys.argv) + "\n")
        fh.write("variant swap A B | " + " | ".join(keys) + "\n")
        for m in results:
            fh.write(f"{m['variant']} {m['swap']} {m['A']} {m['B']} | " +
                     " | ".join(f"{m[k]:.4f}" if k in m else "-" for k in keys) + "\n")
    print(f"saved {a.out}/camswap_{tag}.json and summary_{tag}.txt")


if __name__ == "__main__":
    main()
