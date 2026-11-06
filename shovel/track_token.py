#!/usr/bin/env python3
"""Single-token tracking: run the DVLT loop for K steps and, at EVERY step, record the full
dossier of a SMALL fixed set of tracked tokens (default 8 = cam[0] + 4 register[0] + 3 patches
on frame 0, from pick_patches.py). One K per run, one file per step.

What is stored (per step), for the T tracked tokens, Hh heads, S views, P per-frame tokens:
  frame_rows   (T,Hh,P)      attention over the token's OWN frame (frame 0), with RoPE
  global_rows  (T,Hh,S,P)    attention over ALL views, sliced per view (patch part -> 24x36)
  z_frame      (T,C)         state after frame_attn sub-block
  z_out        (T,C)         state after global_attn sub-block (= next step's input)
  anorm_frame/anorm_global  (T,Hh)   per-head attention-output norm (who pushes hardest)
  gates_frame/gates_global  (3,C)    time-conditioned gates s_attn/s_mlp/s_out
Static (meta.npz): z0 (T,C), gt_target (nP,S,2) + gt_valid (nP,S) for the patches (GT reprojection),
  tracked token table, K/S/P/C/Hh/grid/t-schedule, paths.

Attention is NOT materialized (model uses fused/flash SDPA); rows are recomputed from the module's
OWN qkv/q_norm/k_norm/rope (reused, not re-implemented) -> mathematically identical to SDPA. Only
the tracked query rows are formed, so cost/memory stay tiny (~4.5 MB/step for 8 tokens).

  PYTHONPATH=$(pwd) python shovel/track_token.py \
      --input out/gt_00777c41d4_s10/probe_input --gt out/gt_00777c41d4_s10/gt.npz \
      --picks out/gt_00777c41d4_s10/patch_pick_frame0.json --max-frames 27 --K 16 --tag s10
"""
import argparse, json, os
import numpy as np
import torch

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import load_model, load_images, init_state, load_gt


def attn_rows(module, x, pos, tok):
    """Recompute, for the tracked query indices `tok`, the attention rows + output norm, using the
    module's OWN layers (identical to SDPA). x: (Bx,N,C). Returns rows (T,Hh,N) fp16, outnorm (T,Hh).
    Tracked tokens live in frame 0 -> batch index 0 of x (both frame_attn's B*S and global's B=1)."""
    Bx, N, C = x.shape
    qkv = module.qkv(x).reshape(Bx, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)                                   # (Bx,Hh,N,hd)
    q, k = module.q_norm(q), module.k_norm(k)
    if module.rope is not None and pos is not None:            # frame_attn has RoPE; global does not
        q, k = module.rope(q, pos), module.rope(k, pos)
    scale = module.head_dim ** -0.5
    q0, k0, v0 = q[0].float(), k[0].float(), v[0].float()      # frame 0: (Hh,N,hd)
    qsel = q0[:, tok, :]                                       # (Hh,T,hd)
    aw = ((qsel * scale) @ k0.transpose(-1, -2)).softmax(-1)   # (Hh,T,N) == SDPA weights
    o = aw @ v0                                                # (Hh,T,hd)
    rows = aw.permute(1, 0, 2).contiguous()                    # (T,Hh,N)
    outnorm = o.norm(dim=-1).transpose(0, 1).contiguous()      # (T,Hh)
    return rows.half().cpu(), outnorm.float().cpu()


def gt_correspondence(gd, c2w, K, picks_rc, patch=14):
    """Reproject each frame-0 patch centre into every view via GT depth+pose. Pure geometry
    (ignores occlusion). Returns tgt (nP,S,2) patch (row,col), valid (nP,S)."""
    S, H, W = gd.shape
    gh, gw = H // patch, W // patch
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    inv = np.linalg.inv(c2w)                                   # (S,4,4) world->cam_j
    tgt = np.full((len(picks_rc), S, 2), -1, np.int32)
    val = np.zeros((len(picks_rc), S), bool)
    for pi, (r, c) in enumerate(picks_rc):
        u, vv = c * patch + patch // 2, r * patch + patch // 2
        d = float(gd[0, vv, u])
        if d <= 1e-6:
            continue
        Xw = c2w[0] @ np.array([(u - cx) / fx * d, (vv - cy) / fy * d, d, 1.0])
        for j in range(S):
            Pc = inv[j] @ Xw
            if Pc[2] <= 1e-6:
                continue
            uj, vj = fx * Pc[0] / Pc[2] + cx, fy * Pc[1] / Pc[2] + cy
            rj, cj = int(vj // patch), int(uj // patch)
            if 0 <= rj < gh and 0 <= cj < gw:
                tgt[pi, j] = (rj, cj)
                val[pi, j] = True
    return tgt, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--picks", required=True, help="patch_pick_frameX.json from pick_patches.py")
    ap.add_argument("--ckpt", default="nvidia/dvlt")
    ap.add_argument("--img-size", type=int, default=504)
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--tag", default="s10")
    ap.add_argument("--out-root", default="out")
    a = ap.parse_args()

    pk = json.load(open(a.picks))
    frame = pk["frame"]
    assert frame == 0, "this script tracks frame-0 tokens (local index == global index)"
    # tracked token order: cam, register x4, then the 3 patches (near_tex, flat, far)
    order = ["near_tex", "flat", "far"]
    patch_rc = [(pk["patches"][k]["row"], pk["patches"][k]["col"]) for k in order if k in pk["patches"]]
    tok_tbl = [dict(name="cam", index=0, type="cam")]
    tok_tbl += [dict(name=f"reg{i}", index=i, type="register") for i in range(1, pk["patch_start_idx"])]
    tok_tbl += [dict(name=k, index=pk["patches"][k]["token_index"], type="patch",
                     row=pk["patches"][k]["row"], col=pk["patches"][k]["col"]) for k in order if k in pk["patches"]]
    tok = torch.tensor([t["index"] for t in tok_tbl])

    out_dir = os.path.join(a.out_root, f"track_{a.tag}_K{a.K}")
    os.makedirs(out_dir, exist_ok=True)

    model, m, acc = load_model(a.ckpt, img_size=a.img_size)
    images = load_images(a.input, a.img_size, a.max_frames, acc.device)
    gd, c2w, Kmat = load_gt(a.gt, a.img_size, device="cpu")
    tok = tok.to(acc.device)

    block = m.recurrent_blocks[0]
    Hh = block.frame_attn.attn.num_heads
    C = m.register_token.shape[-1]
    cur = {}

    def pre_frame(mod, args, kwargs):
        cur["frame_rows"], cur["anorm_frame"] = attn_rows(mod, args[0], kwargs.get("pos"), tok)

    def pre_global(mod, args, kwargs):
        cur["global_rows"], cur["anorm_global"] = attn_rows(mod, args[0], kwargs.get("pos"), tok)

    def post_frame(mod, args, out):
        o = out[0] if isinstance(out, tuple) else out          # (B*S,P,C)
        cur["z_frame"] = o[0, tok, :].float().cpu()

    def post_global(mod, args, out):
        o = out[0] if isinstance(out, tuple) else out          # (1,S*P,C)
        cur["z_out"] = o[0, tok, :].float().cpu()

    hooks = [
        block.frame_attn.attn.register_forward_pre_hook(pre_frame, with_kwargs=True),
        block.global_attn.attn.register_forward_pre_hook(pre_global, with_kwargs=True),
        block.frame_attn.register_forward_hook(post_frame),
        block.global_attn.register_forward_hook(post_global),
    ]

    def gates(sub, t_rows):
        if getattr(sub, "depth_scale", None) is None:
            return None
        s = sub.depth_scale(t_rows)[0].float().cpu()           # (num_gates*C,)
        return s.reshape(-1, C).numpy()                        # (3,C) gated / (2,C) no_sout

    ts = torch.linspace(0.0, 1.0, a.K).tolist()
    with torch.no_grad(), acc.autocast():
        x, rope, dims = init_state(m, images)
        B, S, H, W = dims
        assert B == 1
        z0 = x[0, tok, :].float().cpu().numpy()
        t_pair_dev = lambda tn, tx: torch.tensor([[tn, tx]], device=x.device, dtype=torch.float32)
        for k in range(a.K):
            tn, tx = ts[k], (ts[k + 1] if k + 1 < a.K else 1.0)
            cur.clear()
            gf = gates(block.frame_attn, t_pair_dev(tn, tx).expand(B * S, -1))
            gg = gates(block.global_attn, t_pair_dev(tn, tx).expand(B, -1))
            x = m._interval_step(x, tn, tx, rope, B, S)        # hooks fire here
            gr = cur["global_rows"].reshape(len(tok), Hh, S, -1).numpy()   # (T,Hh,S,P)
            np.savez_compressed(
                os.path.join(out_dir, f"step_{k:02d}.npz"),
                step=k, t_now=tn, t_next=tx,
                frame_rows=cur["frame_rows"].numpy(),          # (T,Hh,P)
                global_rows=gr,                                # (T,Hh,S,P)
                z_frame=cur["z_frame"].numpy(), z_out=cur["z_out"].numpy(),
                anorm_frame=cur["anorm_frame"].numpy(), anorm_global=cur["anorm_global"].numpy(),
                gates_frame=gf, gates_global=gg,
            )
            print(f"step {k+1}/{a.K}  frame_rows{tuple(cur['frame_rows'].shape)} "
                  f"global_rows{gr.shape}  |z_out|={np.linalg.norm(cur['z_out'].numpy(),axis=-1).mean():.1f}")

    for h in hooks:
        h.remove()

    tgt, val = gt_correspondence(gd.numpy(), c2w.numpy(), Kmat.numpy(), patch_rc)
    P = (H // m.patch_size) * (W // m.patch_size) + m.patch_start_idx
    np.savez_compressed(
        os.path.join(out_dir, "meta.npz"),
        K=a.K, S=S, P=P, C=C, Hh=Hh, H=H, W=W,
        grid=np.array([H // m.patch_size, W // m.patch_size]), patch=m.patch_size,
        patch_start_idx=m.patch_start_idx, t_schedule=np.array(ts, np.float32),
        z0=z0, gt_target=tgt, gt_valid=val,
        tok_index=np.array([t["index"] for t in tok_tbl]),
        tok_name=np.array([t["name"] for t in tok_tbl]),
        tok_type=np.array([t["type"] for t in tok_tbl]),
        patch_names=np.array([k for k in order if k in pk["patches"]]),
        input=a.input, gt=a.gt, ckpt=a.ckpt,
    )
    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 1e6
    print(f"\nwrote {out_dir}/  ({a.K} step files + meta.npz, {total:.0f} MB)")
    print(f"  tracked {len(tok_tbl)} tokens: {[t['name'] for t in tok_tbl]}")


if __name__ == "__main__":
    main()
