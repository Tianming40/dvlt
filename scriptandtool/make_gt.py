"""
make_gt.py -- build a ground-truth .npz for dvlt_probe's --gt, by reusing the repo's
ScanNet++ fisheye depth renderer (pytorch3d).

For the first N DSLR frames of a scene it:
  1. renders metric z-depth from the aligned laser mesh through the fisheye camera,
  2. undistorts both the RGB image and the depth to a pinhole model,
  3. writes undistorted pinhole images (probe --input) + gt.npz (probe --gt).

The model expects *pinhole* images, so the probe must be fed the undistorted images
written here, NOT the raw fisheye resized_images.

Usage:
  python make_gt.py --scene-dir out/scene_00777c41d4 --n 6 --out out/gt_00777c41d4
  # then:
  python dvlt_probe.py kgt --input out/gt_00777c41d4/undistorted_images \
         --gt out/gt_00777c41d4/gt.npz --max-frames 6 --Ks 4 8 12 16 20 24 32 64

Frame order matches load_sequence (sorted filenames), so the npz lines up with the
images the probe loads.  Needs: <scene-dir>/dslr/{colmap/images.txt, resized_images,
nerfstudio/transforms.json} and <scene-dir>/scans/mesh_aligned_0.05.ply.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes

# reuse the repo's proven fisheye helpers so the camera/distortion math is identical
from dvlt.scripts.preprocess.scannetpp.processor import (
    make_fisheye_camera,
    read_colmap_poses,
    rectified_intrinsics,
)


def load_mesh(path: Path, device) -> Meshes:
    m = o3d.io.read_triangle_mesh(str(path))
    verts = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(m.triangles), dtype=torch.int64, device=device)
    return Meshes(verts=[verts], faces=[faces])


def render_depth(rasterizer, mesh, pose_w2c, K, distortion, h, w, device) -> np.ndarray:
    """Fisheye z-depth in metres; 0 where no mesh is hit."""
    cam = make_fisheye_camera(pose_w2c, K, distortion, h, w).to(device)
    with torch.no_grad():
        zbuf = rasterizer(mesh, cameras=cam).zbuf[0, :, :, 0]
    depth = zbuf.cpu().numpy()
    depth[depth == -1] = 0.0
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, help="dir with dslr/ and scans/ (e.g. out/scene_00777c41d4)")
    ap.add_argument("--n", type=int, default=6, help="number of frames (matches probe --max-frames)")
    ap.add_argument("--out", required=True, help="output dir for undistorted_images/ and gt.npz")
    ap.add_argument("--sampling", default="uniform", choices=["first", "uniform"],
                    help="'first' = first n posed frames; 'uniform' = n evenly spaced across the sequence "
                         "(matches the repo eval's uniform view sampling; more diverse viewpoints)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scene = Path(args.scene_dir)
    dslr = scene / "dslr"
    out = Path(args.out)
    img_out = out / "probe_input"   # undistorted pinhole images -- feed this as probe --input
    img_out.mkdir(parents=True, exist_ok=True)

    # intrinsics + fisheye distortion from transforms.json
    tf = json.loads((dslr / "nerfstudio" / "transforms.json").read_text())
    K = np.array([[tf["fl_x"], 0, tf["cx"]], [0, tf["fl_y"], tf["cy"]], [0, 0, 1]], dtype=np.float64)
    distortion = np.array([tf[k] for k in ("k1", "k2", "k3", "k4")], dtype=np.float64)
    w, h = int(tf["w"]), int(tf["h"])

    # rectified pinhole intrinsics + the fisheye->pinhole remap
    new_K = rectified_intrinsics(K, distortion, w, h)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (w, h), cv2.CV_32FC1)

    # poses (name -> w2c OpenCV) from COLMAP
    poses = read_colmap_poses(dslr / "colmap" / "images.txt")

    # candidate frames: sorted (like load_sequence) and posed
    img_dir = dslr / "resized_images"
    posed = [n for n in sorted(p.name for p in img_dir.iterdir()
                               if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if n in poses]
    assert len(posed) >= args.n, f"only {len(posed)} posed frames available, need {args.n}"
    if args.sampling == "uniform":
        # same formula as the repo's EvalDataset uniform sampling; indices stay increasing so
        # the selection is already in sorted (load_sequence) order.
        idx = np.linspace(0, len(posed), args.n, endpoint=False, dtype=np.int32)
        names = [posed[i] for i in idx]
    else:
        names = posed[: args.n]
    print(f"sampling={args.sampling}  frames: {names}")

    mesh = load_mesh(scene / "scans" / "mesh_aligned_0.05.ply", device)
    rasterizer = MeshRasterizer(
        raster_settings=RasterizationSettings(image_size=(h, w), blur_radius=0.0, faces_per_pixel=1, cull_to_frustum=True)
    )

    # copy the picked raw (fisheye) frames too, so it is obvious which 20 were sampled
    src_out = out / "source_frames"
    src_out.mkdir(parents=True, exist_ok=True)
    (out / "frames.txt").write_text("\n".join(names) + "\n")

    depths, c2ws = [], []
    for name in names:
        shutil.copyfile(img_dir / name, src_out / name)  # the raw sampled frame (fisheye)
        # undistort the RGB image (what the probe will actually consume)
        img = cv2.imread(str(img_dir / name))
        rect_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        cv2.imwrite(str(img_out / name), rect_img)

        # render fisheye depth, then undistort it to pinhole (NEAREST so values stay exact)
        depth = render_depth(rasterizer, mesh, poses[name], K, distortion, h, w, device)
        rect_depth = cv2.remap(
            depth, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        depths.append(rect_depth.astype(np.float32))

        c2w = np.linalg.inv(poses[name]).astype(np.float32)  # w2c -> c2w (OpenCV)
        c2ws.append(c2w)
        valid = (rect_depth > 0).mean()
        print(f"  {name}: depth valid={valid:.1%} range=[{rect_depth[rect_depth>0].min():.2f},{rect_depth.max():.2f}]m")

    depth_arr = np.stack(depths)  # (S,h,w)
    c2w_arr = np.stack(c2ws)      # (S,4,4)
    # rectified pinhole intrinsics (shared across frames), at the full (h,w) render resolution;
    # load_gt rescales it to the model input resolution. Needed for GT correspondences (attn exp).
    intr = new_K.astype(np.float32)  # (3,3)
    np.savez_compressed(out / "gt.npz", depth=depth_arr, c2w=c2w_arr, intrinsics=intr, names=np.array(names))
    print(f"\nsaved {out}/gt.npz        depth{depth_arr.shape}  c2w{c2w_arr.shape}  intrinsics{intr.shape}")
    print(f"saved {out}/probe_input/   ({args.n} undistorted pinhole images -- feed as probe --input)")
    print(f"saved {out}/source_frames/ ({args.n} raw sampled frames, for reference)")
    print(f"saved {out}/frames.txt     (the {args.n} frame names)")
    print(f"\nNext:\n  python dvlt_probe.py kgt --input {out}/probe_input "
          f"--gt {out}/gt.npz --max-frames {args.n} --Ks 4 8 12 16 20 24 32 64 --every 4")


if __name__ == "__main__":
    main()
