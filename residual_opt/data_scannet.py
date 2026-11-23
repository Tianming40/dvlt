"""ScanNet clips for the flow-matching refiner.

A clip = S frames of one ScanNet scene, sampled with a fixed frame stride (parallax), with
  images  : list of PIL images (color resized to the depth resolution 640x480, as is standard for ScanNet)
  depth   : (S, 480, 640) float32 metres, 0 = invalid
  c2w     : (S, 4, 4) float32 camera-to-world (OpenCV convention, ScanNet's pose/*.txt)
  K       : (3, 3) float32 depth intrinsics (intrinsic/intrinsic_depth.txt)
`to_model_res` applies DVLT's preprocessing (longest side -> img_size, centre crop to a patch multiple) to depth and K,
so they are pixel-aligned with the images produced by dvlt.util.preprocess.preprocess_images.
"""
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCANNET = "/storage/group/dataset_mirrors/scannet/scans"
W0, H0 = 640, 480


@dataclass
class Clip:
    scene: str
    frames: list
    images: list
    depth: np.ndarray
    c2w: np.ndarray
    K: np.ndarray


def list_scenes(root=SCANNET):
    return sorted(d for d in os.listdir(root) if d.startswith("scene") and os.path.isdir(os.path.join(root, d, "color")))


def _pose(path):
    p = np.loadtxt(path, dtype=np.float32)
    return p if p.shape == (4, 4) and np.isfinite(p).all() else None


def sample_clip(scene, S=8, stride=30, rng=None, root=SCANNET, min_valid=0.5, tries=20):
    """random clip of S frames (frame ids start + i*stride) with finite poses and enough valid depth."""
    rng = rng or np.random.default_rng()
    d = os.path.join(root, scene)
    n = len(os.listdir(os.path.join(d, "color")))
    if n < S * stride + 1:
        return None
    K = np.loadtxt(os.path.join(d, "intrinsic", "intrinsic_depth.txt"), dtype=np.float32)[:3, :3]
    for _ in range(tries):
        start = int(rng.integers(0, n - S * stride))
        ids = [start + i * stride for i in range(S)]
        poses = [_pose(os.path.join(d, "pose", f"{i}.txt")) for i in ids]
        if any(p is None for p in poses):
            continue
        depth = np.stack([np.asarray(Image.open(os.path.join(d, "depth", f"{i}.png")), dtype=np.float32) / 1000.0 for i in ids])
        if (depth > 0).mean() < min_valid:
            continue
        images = [Image.open(os.path.join(d, "color", f"{i}.jpg")).convert("RGB").resize((W0, H0), Image.BILINEAR) for i in ids]
        return Clip(scene, ids, images, depth, np.stack(poses), K)
    return None


def to_model_res(depth, K, img_size=504, patch=14):
    """same resize / centre-crop as preprocess_images: depth (S,h,w) nearest, K rescaled + shifted."""
    S, h, w = depth.shape
    sc = img_size / max(h, w)
    nh, nw = int(round(h * sc)), int(round(w * sc))
    d = F.interpolate(torch.from_numpy(depth)[:, None], size=(nh, nw), mode="nearest")[:, 0]
    ch, cw = max(patch, nh // patch * patch), max(patch, nw // patch * patch)
    top, left = (nh - ch) // 2, (nw - cw) // 2
    d = d[:, top:top + ch, left:left + cw]
    K = K.copy()
    K[0, 0] *= nw / w; K[1, 1] *= nh / h
    K[0, 2] = K[0, 2] * nw / w - left; K[1, 2] = K[1, 2] * nh / h - top
    return d.numpy(), K


# ----------------------------------------------------------------------------- BundleFusion (out-of-distribution eval)
BUNDLEFUSION = "/storage/group/dataset_mirrors/dai2017bundlefusion"
BF_SCENES = ["apt0", "apt1", "apt2", "copyroom", "office0", "office1", "office2", "office3"]


def _bf_intrinsics(d):
    for line in open(os.path.join(d, "info.txt")):
        if line.startswith("m_calibrationDepthIntrinsic"):
            v = [float(x) for x in line.split("=")[1].split()]
            return np.array(v, dtype=np.float32).reshape(4, 4)[:3, :3]
    raise ValueError(d)


def sample_clip_bf(scene, S=8, stride=30, rng=None, root=BUNDLEFUSION, min_valid=0.5, tries=20):
    """BundleFusion: <root>/<scene>/<scene>/frame-XXXXXX.{color.jpg, depth.png, pose.txt}; 640x480, depth in mm."""
    rng = rng or np.random.default_rng()
    d = os.path.join(root, scene, scene)
    n = len([f for f in os.listdir(d) if f.endswith(".pose.txt")])
    if n < S * stride + 1:
        return None
    K = _bf_intrinsics(d)
    for _ in range(tries):
        start = int(rng.integers(0, n - S * stride))
        ids = [start + i * stride for i in range(S)]
        poses = [_pose(os.path.join(d, f"frame-{i:06d}.pose.txt")) for i in ids]
        if any(p is None for p in poses):
            continue
        depth = np.stack([np.asarray(Image.open(os.path.join(d, f"frame-{i:06d}.depth.png")), dtype=np.float32) / 1000.0 for i in ids])
        if (depth > 0).mean() < min_valid:
            continue
        images = [Image.open(os.path.join(d, f"frame-{i:06d}.color.jpg")).convert("RGB").resize((W0, H0), Image.BILINEAR) for i in ids]
        return Clip(scene, ids, images, depth, np.stack(poses), K)
    return None
