"""gate_fix.py -- switch DVLT's time-conditioned gates to CONSTANT gates (Solv3R-style), on the frozen model, no training.

DVLT gates every sub-block (frame_attn, global_attn) with s = depth_scale(t_now, t_next) in R^{3x768}:
    x += s_attn * attn(...);  x += s_mlp * mlp(...);  x = s_out * x
Solv3R replaces f(t) by a learned constant (3,768) per sub-block. Measured (2026-11-11): Solv3R's s_attn / s_mlp are
within ~5% (rel-L2) of DVLT's own f(t~0.47); only s_out differs (Solv3R tighter: frame mean 0.739, global 0.780 vs
DVLT f(0.5,0.5) ~0.86 / ~0.87).

Modes (all built from DVLT's OWN depth_scale network; nothing taken from Solv3R except, in B, two scalar targets):
  off : untouched (time schedule as usual)
  A   : every gate = f(t0, t0)                          (Stage-1 "frozen t" setting)
  B   : as A, but each sub-block's s_out vector is rescaled so its MEAN equals sout_frame / sout_global
        (default Solv3R's 0.739 / 0.780); per-channel shape of DVLT's s_out is kept.

Usage:
    fg = FixedGates(P.m, mode="B")      # P = dvlt_probe.Probe; P.m = DVLTModel
    fg.enable()  ...  fg.restore()      # or: with FixedGates(...): ...
With fixed gates the (t_now, t_next) the loop passes in is ignored, so "K" is just the number of steps.
"""
from __future__ import annotations

import torch

SOLV3R_SOUT = {"frame": 0.739, "global": 0.780}   # mean of Solv3R's learned s_out (frame / global)


class FixedGates:
    def __init__(self, m, mode="B", t0=0.5, sout_frame=SOLV3R_SOUT["frame"], sout_global=SOLV3R_SOUT["global"]):
        assert mode in ("off", "A", "B"), mode
        self.mode, self.t0 = mode, t0
        self.blk = m.recurrent_blocks[0]
        self.targets = {"frame": sout_frame, "global": sout_global}
        self.subs = {"frame": self.blk.frame_attn.depth_scale, "global": self.blk.global_attn.depth_scale}
        self.orig = {k: ds.forward for k, ds in self.subs.items()}
        self.const = {}          # filled lazily on the first call, under the caller's autocast (exact dtype match)

    def _build(self, k, t):
        ref = self.orig[k](torch.tensor([[self.t0, self.t0]], device=t.device, dtype=t.dtype))[0]   # DVLT f(t0,t0)
        c = ref
        if self.mode == "B":
            C = c.numel() // 3
            s_out = c[2 * C:].float()
            c = torch.cat([c[:2 * C], (s_out * (self.targets[k] / s_out.mean())).to(c.dtype)])
        self.const[k] = c
        return c

    def summary(self):
        if self.mode == "off":
            return "gates: off (DVLT time schedule)"
        if not self.const:
            return f"gates: FIXED mode {self.mode} (t0={self.t0}) -- constants built on first step"
        parts = []
        for k, c in self.const.items():
            C = c.numel() // 3
            m = [c[i * C:(i + 1) * C].float().mean().item() for i in range(3)]
            parts.append(f"{k}: s_attn={m[0]:.3f} s_mlp={m[1]:.3f} s_out={m[2]:.3f}")
        return f"gates: FIXED mode {self.mode} (t0={self.t0}) | " + " | ".join(parts)

    def enable(self):
        if self.mode == "off":
            return self
        for k, ds in self.subs.items():
            def fwd(t, k=k):                      # t: (B,2) -> (B, 3*C) constant, ignores t's value
                c = self.const.get(k)
                if c is None or c.dtype != self.orig[k](t[:1]).dtype:
                    c = self._build(k, t)
                return c.unsqueeze(0).expand(t.shape[0], -1)
            ds.forward = fwd
        return self

    def restore(self):
        for k, ds in self.subs.items():
            ds.forward = self.orig[k]

    def __enter__(self):
        return self.enable()

    def __exit__(self, *exc):
        self.restore()


SOLV3R_TARGET_RMS = 0.852   # Solv3R's learned target_rms[3] (state RMS after every call)


def rms_normalize(x, target=SOLV3R_TARGET_RMS):
    """Solv3R-style state normalization (SolverBlock.forward): ONE scalar RMS over all tokens and channels of the
    scene, then x * (target / rms). x: (B*S, P, C) DVLT state (cam + registers + patches)."""
    rms = x.float().pow(2).mean().clamp_min(1e-12).sqrt()
    return x * (target / rms).to(x.dtype)
