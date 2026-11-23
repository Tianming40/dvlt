"""Flow-matching refiner on per-patch geometry, conditioned on the frozen DVLT.

State per patch (7 numbers, normalised): [log depth, ray direction (3), ray origin (3)].
Bridge from DVLT's prediction to GT:  x_t = (1-t) x_dvlt + t x_gt (+ small bridge noise),  target v = x_gt - x_dvlt.
Per-patch input token = LN(W_tok * DVLT token) + W_rgb * (14x14 RGB patch) + W_x * [x_t, x_dvlt] + pos(row, col) + emb(t)
Blocks alternate FRAME attention (patches of one view) and GLOBAL attention (all patches of all views).
The output head is zero-initialised: the untrained model predicts v = 0, i.e. returns DVLT's answer.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000.0 * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    def __init__(self, d, heads, mlp=4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp)), nn.GELU(), nn.Linear(int(d * mlp), d))
        self.ada = nn.Linear(d, 2 * d)                                  # time-conditioned scale/shift
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, temb):
        sc, sh = self.ada(temb)[:, None].chunk(2, -1)
        h = self.n1(x) * (1 + sc) + sh
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class FlowRefiner(nn.Module):
    def __init__(self, grid=(27, 36), d=256, heads=8, depth=6, tok_dim=768, xdim=7, patch=14):
        super().__init__()
        self.gh, self.gw, self.d = grid[0], grid[1], d
        self.tok = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, d))
        self.rgb = nn.Conv2d(3, d, patch, patch)
        self.xin = nn.Linear(2 * xdim, d)
        self.pos = nn.Parameter(torch.zeros(1, grid[0] * grid[1], d)); nn.init.normal_(self.pos, std=0.02)
        self.temb = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, xdim))
        nn.init.zeros_(self.out[1].weight); nn.init.zeros_(self.out[1].bias)

    def forward(self, xt, xd, tokens, image, t, geo=None):
        """xt, xd: (B,S,Np,7) normalised; tokens: (B,S,Np,768) patch tokens; image: (B,S,3,H,W) in [0,1]; t: (B,)"""
        B, S, Np, _ = xt.shape
        rgb = self.rgb((image.flatten(0, 1) - 0.45) / 0.25).flatten(2).transpose(1, 2)        # (B*S,Np,d)
        h = self.tok(tokens.flatten(0, 1)) + rgb + self.xin(torch.cat([xt, xd], -1).flatten(0, 1)) + self.pos
        te = self.temb(time_embedding(t, self.d))                                          # (B,d)
        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:                                                                  # frame attention
                h = blk(h, te.repeat_interleave(S, 0))
            else:                                                                           # global attention
                h = blk(h.reshape(B, S * Np, self.d), te).reshape(B * S, Np, self.d)
        return self.out(h).reshape(B, S, Np, -1)


# ============================================================================= closed-loop version
@torch.no_grad()
def closed_loop_geometry(x, gh, gw, nbrs=(-2, -1, 1, 2), ps=14, win=2):
    """Reproject every patch with the CURRENT geometry x (S,Np,7, un-normalised) into neighbour frames.
    Returns pixel-space sampling positions etc. (no gradient: the geometry is an input, the comparison is learned).
      grid  (S,nb,Np,(2w+1)^2,2) pixel coords of a (2w+1)^2 window of patch-step offsets around each projection
      ctr   (S,nb,Np,2) pixel coords of the projection;  valid (S,nb,Np);  dlog (S,nb,Np) log(z_proj / depth_j(proj))
      disp  (S,nb,Np,2) projection minus own patch centre, / image size;  src (nb, S) neighbour frame index"""
    S, Np, _ = x.shape
    dev = x.device
    H, W = gh * ps, gw * ps
    jj, ii = torch.meshgrid(torch.arange(gw, device=dev), torch.arange(gh, device=dev), indexing="xy")
    uc = (jj.flatten() * ps + ps / 2 - 0.5).float(); vc = (ii.flatten() * ps + ps / 2 - 0.5).float()
    d = x[..., 1:4] / x[..., 1:4].norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c = x[..., 4:7].mean(1)                                                                # (S,3) camera centres
    # DLT camera fit per frame: A d ~ [u,v,1]  (Hartley-normalised pixel coords)
    su, sv = 2.0 / W, 2.0 / H
    un, vn = uc * su - 1, vc * sv - 1
    z0 = torch.zeros_like(d)
    M = torch.cat([torch.cat([d, z0, -un[None, :, None] * d], -1), torch.cat([z0, d, -vn[None, :, None] * d], -1)], 1)
    _, V = torch.linalg.eigh((M.transpose(1, 2) @ M).double())
    T = torch.tensor([[1 / su, 0, 1 / su], [0, 1 / sv, 1 / sv], [0, 0, 1.0]], device=dev, dtype=torch.float64)
    A = (T @ V[..., 0].reshape(S, 3, 3)).float()                                           # (S,3,3)
    sgn = torch.sign((A[:, 2] * d[:, Np // 2]).sum(-1))
    A = A * sgn[:, None, None]
    zax = A[:, 2] / A[:, 2].norm(dim=-1, keepdim=True)                                     # optical axis per frame
    D = x[..., 0].exp()
    X = x[..., 4:7] + (D / (d * zax[:, None]).sum(-1).clamp_min(1e-3))[..., None] * d     # (S,Np,3) points
    src = torch.stack([(torch.arange(S, device=dev) + o) for o in nbrs])                   # (nb,S)
    ok_f = (src >= 0) & (src < S)
    srcc = src.clamp(0, S - 1)
    Aj, cj, zj = A[srcc], c[srcc], zax[srcc]                                               # (nb,S,...)
    q = torch.einsum("nsab,nspb->nspa", Aj, X[None] - cj[:, :, None])                      # (nb,S,Np,3)
    zc = q[..., 2]
    u, v = q[..., 0] / zc.clamp_min(1e-6), q[..., 1] / zc.clamp_min(1e-6)
    valid = ok_f[..., None] & (zc > 1e-3) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    zproj = ((X[None] - cj[:, :, None]) * zj[:, :, None]).sum(-1)
    Dmap = D.reshape(S, 1, gh, gw)[srcc.flatten()].reshape(len(nbrs) * S, 1, gh, gw)
    g = torch.stack([(u - (ps / 2 - 0.5)) / (W - ps) * 2 - 1, (v - (ps / 2 - 0.5)) / (H - ps) * 2 - 1], -1)
    Dj = F.grid_sample(Dmap, g.reshape(len(nbrs) * S, Np, 1, 2), align_corners=True, padding_mode="border")
    dlog = zproj.clamp_min(1e-4).log() - Dj.reshape(len(nbrs), S, Np).clamp_min(1e-4).log()
    off = torch.arange(-win, win + 1, device=dev).float() * ps
    oy, ox = torch.meshgrid(off, off, indexing="ij")
    grid = torch.stack([u[..., None] + ox.flatten(), v[..., None] + oy.flatten()], -1)    # (nb,S,Np,w2,2)
    disp = torch.stack([(u - uc) / W, (v - vc) / H], -1)
    perm = lambda t: t.transpose(0, 1).contiguous()
    return dict(grid=perm(grid), ctr=perm(torch.stack([u, v], -1)), valid=perm(valid), dlog=perm(dlog.nan_to_num(0)),
                disp=perm(disp), src=src.clamp(0, S - 1))


class FlowRefinerCL(FlowRefiner):
    """FlowRefiner + CLOSED-LOOP input: at every integration step the current geometry decides WHERE each patch is
    compared in neighbour frames; WHAT is compared is learned:
      token correlation  cos(P t_i, P t_j(window around the projection)), P a learned 768->64 projection (25 values)
      photometric patch  RGB 7x7 (2-px step) around own centre minus the same around the projection (147 values)
      geometry           log depth inconsistency, displacement of the projection (3 values)
    Per-neighbour features -> MLP -> masked mean over valid neighbours -> added to the patch token."""

    def __init__(self, *args, corr_dim=64, win=2, **kw):
        super().__init__(*args, **kw)
        self.win = win
        self.tproj = nn.Linear(768, corr_dim)
        nwin = (2 * win + 1) ** 2
        self.cl = nn.Sequential(nn.Linear(nwin + 147 + 3 + 1, self.d), nn.GELU(), nn.Linear(self.d, self.d))
        self.cl_gate = nn.Parameter(torch.zeros(1))                       # starts closed (= open-loop model)
        oy, ox = torch.meshgrid(torch.arange(-3, 4).float() * 2, torch.arange(-3, 4).float() * 2, indexing="ij")
        self.register_buffer("pix_off", torch.stack([ox.flatten(), oy.flatten()], -1))  # (49,2)

    def cl_features(self, geo, tokens, image):
        """geo from closed_loop_geometry (S,...); tokens (S,Np,768); image (S,3,H,W). returns (S,Np,d)"""
        S, Np, _ = tokens.shape
        H, W = image.shape[-2:]
        nb = geo["src"].shape[0]
        ps = H // self.gh
        P = F.normalize(self.tproj(tokens - tokens.mean(1, keepdim=True)), dim=-1)         # (S,Np,64)
        Pmap = P.transpose(1, 2).reshape(S, -1, self.gh, self.gw)
        feats = []
        for k in range(nb):
            src = geo["src"][k]                                                             # (S,)
            gw_ = geo["grid"][:, k]                                                         # (S,Np,w2,2)
            gn = torch.stack([(gw_[..., 0] - (ps / 2 - 0.5)) / (W - ps) * 2 - 1, (gw_[..., 1] - (ps / 2 - 0.5)) / (H - ps) * 2 - 1], -1)
            samp = F.grid_sample(Pmap[src], gn, align_corners=True, padding_mode="zeros")   # (S,64,Np,w2)
            corr = (samp * P.transpose(1, 2)[..., None]).sum(1)                              # (S,Np,w2)
            ctr = geo["ctr"][:, k]                                                          # (S,Np,2)
            pj = ctr[:, :, None] + self.pix_off                                             # (S,Np,49,2)
            pn = torch.stack([pj[..., 0] / (W - 1) * 2 - 1, pj[..., 1] / (H - 1) * 2 - 1], -1)
            rj = F.grid_sample(image[src], pn, align_corners=True, padding_mode="border")    # (S,3,Np,49)
            jj, ii = torch.meshgrid(torch.arange(self.gw, device=image.device), torch.arange(self.gh, device=image.device), indexing="xy")
            own = torch.stack([jj.flatten() * ps + ps / 2 - 0.5, ii.flatten() * ps + ps / 2 - 0.5], -1).float()
            pi = own[None, :, None] + self.pix_off
            pin = torch.stack([pi[..., 0] / (W - 1) * 2 - 1, pi[..., 1] / (H - 1) * 2 - 1], -1).expand(S, -1, -1, -1)
            ri = F.grid_sample(image, pin, align_corners=True, padding_mode="border")
            photo = (ri - rj).permute(0, 2, 1, 3).reshape(S, Np, -1)                        # (S,Np,147)
            val = geo["valid"][:, k].float()
            f = torch.cat([corr, photo, geo["dlog"][:, k, :, None].clamp(-1, 1), geo["disp"][:, k], val[..., None]], -1)
            feats.append((self.cl(f), val))
        num = sum(f * v[..., None] for f, v in feats); den = sum(v for _, v in feats)[..., None]
        return num / den.clamp_min(1.0)

    def forward(self, xt, xd, tokens, image, t, geo=None):
        B, S, Np, _ = xt.shape
        rgb = self.rgb((image.flatten(0, 1) - 0.45) / 0.25).flatten(2).transpose(1, 2)
        h = self.tok(tokens.flatten(0, 1)) + rgb + self.xin(torch.cat([xt, xd], -1).flatten(0, 1)) + self.pos
        if geo is not None:
            h = h + self.cl_gate * self.cl_features(geo, tokens[0], image[0])
        te = self.temb(time_embedding(t, self.d))
        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:
                h = blk(h, te.repeat_interleave(S, 0))
            else:
                h = blk(h.reshape(B, S * Np, self.d), te).reshape(B * S, Np, self.d)
        return self.out(h).reshape(B, S, Np, -1)
