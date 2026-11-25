#!/usr/bin/env python3
"""attn_undo_swap.py -- "undo one frame-attention update, swap the camera/register tokens, redo it, decode".

At three time points of a frozen DVLT run (K=16): x0 (before step 0), after step 0, after step 16 (final state).
For each state x and frame pair (A,B):
  a(x)  = s_attn * ls1( frame_attn( norm1(x) ) )        the frame-attention branch of DVLT's frame sub-block
          (gate s_attn from the time pair of step `gate_step`; attention only, no MLP / s_out)
  base      : decode(x)
  sub       : x' = x with patch tokens minus a(x)  (the 5 special tokens unchanged)            -> decode
  sub_add   : x' + a(x')                                                                       -> decode
  swap      : x with A<->B cam+registers exchanged                                             -> decode
  main      : x' -> exchange A<->B cam+registers -> x'' = x'_sw + a(x'_sw)  (normal add)      -> decode
Metrics for frame A vs the base decode at the same time point (shovel/camswap.metrics): does A's camera centre / rays /
points move toward B's? (centre_toB, dir_toB_deg vs dir_AB_deg, pts_moved, ...).

  PYTHONPATH=$(pwd) python shovel/attn_undo_swap.py --max-frames 27 --K 16 --pairs 5:5 4:5 12:13 3:20 8:24 --out out/stage3_latent/attnundo
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvlt_probe import Probe  # noqa: E402
from camswap import to_np, metrics  # noqa: E402

SPEC = slice(0, 5)


def make_branch(P, rope, dims, K):
    fa = P.m.recurrent_blocks[0].frame_attn
    B, S, H, W = dims
    ts = torch.linspace(0.0, 1.0, K).tolist()

    @torch.no_grad()
    def branch(x, step):
        t_now, t_next = ts[step], (ts[step + 1] if step + 1 < K else 1.0)
        tp = torch.tensor([[t_now, t_next]], device=x.device, dtype=torch.float32).expand(x.shape[0], -1)
        s_attn = fa.depth_scale(tp).unsqueeze(1).chunk(3, dim=-1)[0]
        pos = rope.reshape(x.shape[0], x.shape[1], 2)
        return (s_attn * fa.ls1(fa.attn(fa.norm1(x), pos=pos))).to(x.dtype)
    return branch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/data/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--pairs", nargs="+", default=["5:5", "4:5", "12:13", "3:20", "8:24"])
    ap.add_argument("--out", default="out/stage3_latent/attnundo")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pairs = [tuple(int(v) for v in p.split(":")) for p in a.pairs]

    P = Probe("nvidia/dvlt", max_frames=a.max_frames)
    images = P.load(a.input)
    x0, rope, dims = P.init_state(images)
    B, S, H, W = dims
    K = a.K
    branch = make_branch(P, rope, dims, K)

    states = {}
    with torch.no_grad():
        states["before_step0"] = (x0.clone(), 0)
        x = P.step(x0.clone(), 0, K, rope, B, S)
        states["after_step0"] = (x.clone(), 1)
        for i in range(1, K):
            x = P.step(x, i, K, rope, B, S)
        states[f"after_step{K}"] = (x.clone(), K - 1)

    results, log = [], []
    with torch.no_grad():
        for tname, (xs, gstep) in states.items():
            ref = to_np(P.decode(xs, rope, dims))
            a1 = branch(xs, gstep)
            x_sub = xs.clone(); x_sub[:, 5:] = xs[:, 5:] - a1[:, 5:]
            dec_sub = to_np(P.decode(x_sub, rope, dims))
            x_sa = x_sub + branch(x_sub, gstep)
            dec_sa = to_np(P.decode(x_sa, rope, dims))
            rel = float((a1[:, 5:].float().norm() / xs[:, 5:].float().norm()).item())
            hdr = f"\n=== {tname} (gate of step {gstep}) | |a(x)| / |x| on patches = {rel:.3f}"
            print(hdr); log.append(hdr)
            for (A, Bf) in pairs:
                x_sw = xs.clone(); x_sw[A, SPEC], x_sw[Bf, SPEC] = xs[Bf, SPEC], xs[A, SPEC]
                dec_sw = to_np(P.decode(x_sw, rope, dims))
                x1 = x_sub.clone(); x1[A, SPEC], x1[Bf, SPEC] = x_sub[Bf, SPEC], x_sub[A, SPEC]
                x_main = x1 + branch(x1, gstep)
                dec_main = to_np(P.decode(x_main, rope, dims))
                for cname, dec in [("sub", dec_sub), ("sub_add", dec_sa), ("swap", dec_sw), ("main", dec_main)]:
                    m = metrics(dec, ref, A, Bf)
                    m.update(time=tname, cond=cname, A=A, B=Bf)
                    results.append(m)
                    line = (f"  {A:>2}->{Bf:<2} {cname:>8}: centre_toB={m['centre_toB']:.3f} centre_moved={m['centre_moved']:.3f} "
                            f"dir_toB={m['dir_toB_deg']:.2f}° (A-B {m['dir_AB_deg']:.2f}°) depthΔ={m['depth_change']:.4f} "
                            f"pts_moved={m['pts_moved']:.3f} pts_vs_exp={m['pts_vs_exp']:.3f}")
                    print(line); log.append(line)
    json.dump(results, open(f"{a.out}/attnundo_K{K}.json", "w"), indent=1)
    with open(f"{a.out}/summary_K{K}.txt", "w") as fh:
        fh.write(" ".join(sys.argv) + "\n" + "\n".join(log) + "\n")
    print(f"saved {a.out}/attnundo_K{K}.json and summary_K{K}.txt")


if __name__ == "__main__":
    main()
