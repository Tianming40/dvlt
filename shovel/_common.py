"""Shared loaders for the shovel scripts -- repo functions directly, no Probe wrapper.

load_model / load_images / init_state are thin and call ONLY repo functions; init_state is the
verbatim assembly from DVLTModel.forward_inference (model.py:621-627) so there is no second copy
of the math hidden in a wrapper. load_gt is ours (we feed a gt.npz, not the repo dataset class).
"""
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from dvlt.common.constants import DataField
from dvlt.model.dvlt.model import DVLT, _slice_expand_flatten
from dvlt.util.preprocess import load_sequence, preprocess_images


def load_model(ckpt="nvidia/dvlt", img_size=504, precision="bf16"):
    """Repo-standard load. Returns (wrapper DVLT, DVLTModel, Accelerator)."""
    acc = Accelerator(mixed_precision=precision)
    model = DVLT(img_size=img_size)
    model.load_pretrained(ckpt, strict=True)
    model.setup_test(acc)
    return model, model.model, acc


def load_images(path, img_size, max_frames, device):
    """Repo image pipeline: load_sequence + preprocess_images. Returns (1,S,3,H,W) in [0,1]."""
    _, frames = load_sequence(path)
    if max_frames:
        frames = frames[:max_frames]
    batch = preprocess_images(frames, img_size=img_size, patch_size=14, device=device)
    return batch[DataField.IMAGES]


@torch.no_grad()
def init_state(m, images):
    """== DVLTModel.forward_inference assembly (model.py:621-627), using repo functions only."""
    B, S, _, H, W = images.shape
    z0 = m._encode_images(images)
    reg = m.register_token.expand(B, S, -1, -1).reshape(B * S, m.num_register_tokens, -1)
    cam = _slice_expand_flatten(m.camera_token, B, S)
    x = torch.cat([cam, reg, z0], dim=1)
    rope = m._get_rope_positions(B * S, H, W, images.device)
    return x, rope, (B, S, H, W)


def load_gt(path, img_size, patch=14, device="cpu"):
    """OURS: read a gt.npz (depth/c2w/intrinsics) and push depth+K through the same resize/crop as
    preprocess_images (longest side -> img_size, centre crop to patch multiple; NEAREST for depth).
    Returns (depth, c2w, K) at model input resolution. (No repo equivalent for a standalone npz.)"""
    g = np.load(path)
    d = torch.from_numpy(np.nan_to_num(g["depth"].astype(np.float32), nan=0.0))
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
        K = torch.from_numpy(g["intrinsics"].astype(np.float32)).clone()
        K[0, 0] *= sc; K[1, 1] *= sc
        K[0, 2] = K[0, 2] * sc - left
        K[1, 2] = K[1, 2] * sc - top
    return (d.to(device),
            c2w.to(device) if c2w is not None else None,
            K.to(device) if K is not None else None)
