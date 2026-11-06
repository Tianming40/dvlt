#!/usr/bin/env python3
"""Visualize the PER-STEP model reconstruction vs GT, via the REPO's official pipeline.

For each saved loop step (from step_decode's dump), it rebuilds the raw prediction dict, then runs
the repo's own:
  - DVLT._postprocess_predictions  (raw rays/depth -> CAMERAS, DEPTHS, WORLD_POINTS, ...)
  - index_batch_and_predictions    (pick batch idx 0 -> per-frame tensors)
  - align_predictions_to_gt        (Sim3 align predictions to GT)  [--align-mode]
and logs pred + GT point clouds on a Rerun "step" timeline, one toggleable entity per frame
(scrub the step slider to watch the reconstruction form; toggle frames in the entity tree).

--align-mode picks how the gauge-free prediction is placed against the fixed GT:
  per-step (default): fit a fresh Sim3 each step  -> best for per-step pred-vs-GT error
  fixed              : fit once on the final step, reuse it -> frames don't wobble (watch motion)
  none               : no alignment, prediction's own frame

Only thin glue is ours: reading the npz, assembling the GT batch, apply_fixed (= the repo's own
alignment application with a frozen transform), and the per-step Rerun logging (visualize_scene
has no step timeline).

  PYTHONPATH=$(pwd) python shovel/predict_viz.py --dir out/stepdecode_s10_K16 \
      --input out/gt_00777c41d4_s10/probe_input --gt out/gt_00777c41d4_s10/gt.npz \
      --max-frames 27 --conf 25 --align-mode per-step \
      --out out/stepdecode_s10_K16/predict_viz.rrd
"""
import argparse, glob, os
import numpy as np, torch
import rerun as rr

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import load_model, load_images, load_gt
from dvlt.common.constants import DataField, PredictionField
from dvlt.common.geometry import depth_to_world_coords_points
from dvlt.callbacks.util import align_predictions_to_gt, index_batch_and_predictions
from dvlt.common.pose import to4x4
from dvlt.common.geometry import transform_points
from dvlt.struct.util import extri_intri_to_cameras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="stepdecode_<tag>_K<K> folder")
    ap.add_argument("--input", required=True, help="probe_input images dir")
    ap.add_argument("--gt", required=True, help="gt.npz")
    ap.add_argument("--ckpt", default="nvidia/dvlt")
    ap.add_argument("--max-frames", type=int, default=27)
    ap.add_argument("--steps", default="all", help="'all' or comma list of step indices")
    ap.add_argument("--conf", type=float, default=25.0, help="confidence percentile threshold (repo default 25)")
    ap.add_argument("--align-mode", default="per-step", choices=["per-step", "fixed", "none"],
                    help="per-step: align each step to GT independently (best for pred-vs-GT error); "
                         "fixed: align once (final step) and reuse for all steps (best for watching motion); "
                         "none: no alignment (prediction's own frame)")
    ap.add_argument("--per-frame-points", type=int, default=12000, help="points kept per frame per step")
    ap.add_argument("--gt-color", default="0,180,0", help="uniform RGB for GT points+cameras (set 'rgb' to keep true color)")
    ap.add_argument("--pred-cam-color", default="230,40,40", help="RGB for the predicted camera markers")
    ap.add_argument("--cam-marker-radius", type=float, default=0.0, help="absolute camera-marker radius; 0 = auto (0.8%% of robust scene scale)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gt_col = None if a.gt_color.lower() == "rgb" else np.array([int(v) for v in a.gt_color.split(",")], np.uint8)
    pred_cam_col = np.array([int(v) for v in a.pred_cam_color.split(",")], np.uint8)
    out = a.out or os.path.join(a.dir, "predict_viz.rrd")

    # --- model (repo DVLT) + GT at model resolution (our load_gt) ---
    model, m, acc = load_model(a.ckpt, img_size=504)
    gd_all, gc_all, gK = load_gt(a.gt, 504, device="cpu")
    images = load_images(a.input, 504, a.max_frames, acc.device).cpu().float()  # (1,S,3,H,W)
    S = images.shape[1]; H, W = images.shape[-2:]
    gd = gd_all[:S].float(); gc = gc_all[:S].float(); gK = gK.float()

    # --- assemble the GT batch the repo pipeline expects (all on CPU) ---
    gt_world = torch.stack([depth_to_world_coords_points(gd[s], gc[s], gK)[0] for s in range(S)])  # (S,H,W,3)
    batch = {
        DataField.IMAGES: images,
        DataField.EXTRINSICS_C2W: gc[None],                # (1,S,4,4)
        DataField.INTRINSICS: gK[None, None].repeat(1, S, 1, 1),
        DataField.WORLD_POINTS: gt_world[None],            # (1,S,H,W,3)
        DataField.POINT_MASKS: (gd > 1e-6)[None],          # (1,S,H,W)
    }
    def _stepnum(p):   # numeric step index from "step_123.npz" (NOT basename[5:7], which breaks at >=100)
        return int(os.path.basename(p).split("_")[1].split(".")[0])
    all_steps = sorted(glob.glob(os.path.join(a.dir, "step_*.npz")), key=_stepnum)
    steps = all_steps
    if a.steps != "all":
        keep = set(int(x) for x in a.steps.split(","))
        steps = [s for s in all_steps if _stepnum(s) in keep]

    def raw_indexed(sp):
        """npz -> repo postprocess -> indexed (no align). Returns (ib, ip)."""
        z = np.load(sp)
        rays = torch.from_numpy(z["rays"].astype("f4"))[None]
        depth = torch.from_numpy(z["depth"].astype("f4"))[None]
        cf = torch.from_numpy(z["depth_conf"].astype("f4"))[None]
        wp = rays[..., 3:6] + rays[..., :3] * depth
        raw = {"rays": rays, "depth": depth, "depth_conf": cf, "world_points": wp, "world_points_conf": cf}
        pred = model._postprocess_predictions(batch, raw)
        return index_batch_and_predictions(batch, pred, batch_idx=0, seq_idxs=None, inplace=False)

    def apply_fixed(ip, T, s):
        """Apply a GIVEN Sim3 (T=[R|t] 3x4, scale s) to predictions -- mirrors align's own application."""
        ip = dict(ip)
        ip[PredictionField.WORLD_POINTS] = transform_points(s * ip[PredictionField.WORLD_POINTS], T)
        cams = ip[PredictionField.CAMERAS]
        extr = cams.camera_to_worlds.clone()
        extr[:, :3, :3] = T[:3, :3] @ extr[:, :3, :3]
        extr[:, :3, 3] = transform_points(s * extr[:, :3, 3], T)
        ip[PredictionField.CAMERAS] = extri_intri_to_cameras(extr, cams.get_intrinsics_matrices(),
                                                             batch[DataField.IMAGES].shape[-2:])
        return ip

    # fixed mode: compute the Sim3 once from the final (most-converged) step, reuse for all steps
    fixed_T = fixed_s = None
    if a.align_mode == "fixed":
        ibf, ipf = raw_indexed(all_steps[-1])
        ipf = align_predictions_to_gt(ibf, ipf, inplace=False)
        fixed_T, fixed_s = ipf["transform"], ipf["scale_factor"]
        print(f"fixed alignment from {os.path.basename(all_steps[-1])}: scale={fixed_s:.4f}")

    rng = np.random.default_rng(0)
    imgs_np = (images[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)  # (S,H,W,3)

    # camera-marker radius from a ROBUST scene scale (2-98 pct bbox, ignores stray far points)
    _gw_valid = gt_world.numpy()[(gd > 1e-6).numpy()]
    lo, hi = np.percentile(_gw_valid, [2, 98], axis=0)
    scene_diag = float(np.linalg.norm(hi - lo))
    cam_r = a.cam_marker_radius if a.cam_marker_radius > 0 else 0.008 * scene_diag

    def log_frame(ns, s, pts, col, c2w, K, static, cam_color):
        # TYPE-FIRST layout so a Rerun path filter can isolate a whole type with a trailing /** :
        #   <ns>/points/frame_XX , <ns>/cam/frame_XX , <ns>/campos/frame_XX
        # (Rerun only supports the trailing /** subtree wildcard, not mid-path /**/cam.)
        #   only point cloud:  + $origin/**   - /gt/cam/**  - /gt/campos/**  - /pred/cam/**  - /pred/campos/**
        #   only cameras:      + /gt/cam/**  + /gt/campos/**  + /pred/cam/**  + /pred/campos/**
        rr.log(f"{ns}/points/frame_{s:02d}", rr.Points3D(pts, colors=col, radii=0.004), static=static)
        rr.log(f"{ns}/cam/frame_{s:02d}", rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]), static=static)
        rr.log(f"{ns}/cam/frame_{s:02d}", rr.Pinhole(image_from_camera=K, width=W, height=H), static=static)
        # a big coloured dot at the camera centre, in WORLD space (NOT under the pinhole) -> clearly visible
        rr.log(f"{ns}/campos/frame_{s:02d}", rr.Points3D([c2w[:3, 3]], colors=[cam_color], radii=cam_r), static=static)

    def subs(p, c, n):
        if len(p) <= n:
            return p, c
        i = rng.choice(len(p), n, replace=False)
        return p[i], c[i]

    rr.init("predict_viz"); rr.save(out)
    rr.log("/", rr.ViewCoordinates.RFU, static=True)

    # --- GT per frame: static (points + camera), one toggleable entity per frame ---
    gw = gt_world.numpy(); gmask = (gd > 1e-6).numpy()
    gK_np = gK.numpy()
    for s in range(S):
        v = gmask[s]
        p, c = subs(gw[s][v], imgs_np[s][v], a.per_frame_points)
        if gt_col is not None:
            c = np.tile(gt_col, (len(p), 1))                    # uniform GT colour (distinct from pred)
        log_frame("gt", s, p, c, gc[s].numpy(), gK_np, static=True, cam_color=(gt_col if gt_col is not None else pred_cam_col))

    # --- predictions per step, per frame ---
    for sp in steps:
        k = _stepnum(sp)
        z = np.load(sp)
        rays = torch.from_numpy(z["rays"].astype("f4"))[None]      # (1,S,H,W,6)
        depth = torch.from_numpy(z["depth"].astype("f4"))[None]    # (1,S,H,W,1)
        conf = torch.from_numpy(z["depth_conf"].astype("f4"))[None]  # (1,S,H,W)
        wp = rays[..., 3:6] + rays[..., :3] * depth                # (1,S,H,W,3)
        raw = {"rays": rays, "depth": depth, "depth_conf": conf,
               "world_points": wp, "world_points_conf": conf}
        # ----- repo official pipeline -----
        pred = model._postprocess_predictions(batch, raw)           # repo -> CAMERAS/DEPTHS/WORLD_POINTS...
        ib, ip = index_batch_and_predictions(batch, pred, batch_idx=0, seq_idxs=None, inplace=False)
        if a.align_mode == "per-step":
            ip = align_predictions_to_gt(ib, ip, inplace=False)     # Sim3 fit to GT, this step
        elif a.align_mode == "fixed":
            ip = apply_fixed(ip, fixed_T, fixed_s)                  # reuse final-step Sim3 for every step
        # "none": leave ip in the prediction's own (gauge-free) frame
        world = ip[PredictionField.WORLD_POINTS].detach().cpu().numpy()   # (S,H,W,3) aligned
        cf = ip[PredictionField.DEPTHS_CONF].detach().cpu().numpy()       # (S,H,W)
        cams = ip[PredictionField.CAMERAS]
        c2w_all = to4x4(cams.camera_to_worlds).detach().cpu().numpy()     # (S,4,4) aligned
        K_all = cams.get_intrinsics_matrices().detach().cpu().numpy()     # (S,3,3)
        thr = np.percentile(cf.reshape(-1), a.conf)                       # repo-style global conf threshold
        rr.set_time_sequence("step", k)
        tot = 0
        for s in range(S):
            v = cf[s] >= thr
            p, c = subs(world[s][v], imgs_np[s][v], a.per_frame_points)
            log_frame("pred", s, p, c, c2w_all[s], K_all[s], static=False, cam_color=pred_cam_col)
            tot += len(p)
        print(f"step {k}: {tot:,} pred pts over {S} frames (conf>= p{a.conf:g}, align={a.align_mode})")

    print(f"\nwrote {out}  ({len(steps)} steps x {S} frames)")
    print(f"  open:  rerun {out}")
    print(f"  Rerun: scrub the 'step' slider; toggle frames in the entity tree (pred/frame_XX, gt/frame_XX)")


if __name__ == "__main__":
    main()
