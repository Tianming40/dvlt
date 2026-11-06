#!/usr/bin/env python3
"""Visualize the GROUND-TRUTH scene using ONLY the repo's own visualization code.

Builds GT world points with the repo's `depth_to_world_coords_points`, cameras with
`extri_intri_to_cameras`, then renders with the repo's:
  - `scene_to_plotly`  -> interactive .html  (whole scene + camera frustums)
  - `visualize_scene`  -> Rerun .rrd         (per-frame entities: toggle each frame in the tree)

  PYTHONPATH=$(pwd) python shovel/gt_viz.py --gt out/gt_00777c41d4_s10/gt.npz \
      --images out/gt_00777c41d4_s10/probe_input --frames all --mode both
"""
import argparse, os
import numpy as np, torch
from PIL import Image
from dvlt.common.geometry import depth_to_world_coords_points
from dvlt.struct.util import extri_intri_to_cameras
from dvlt.viz.scene_plotly import scene_to_plotly

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="out/gt_00777c41d4_s10/gt.npz")
    ap.add_argument("--images", default="out/gt_00777c41d4_s10/probe_input")
    ap.add_argument("--frames", default="all", help="'all' or comma list, e.g. 0,13,26")
    ap.add_argument("--out", default=None, help="output dir (default <gt_dir>/gt_viz)")
    ap.add_argument("--mode", default="both", choices=["plotly", "rerun", "both"])
    ap.add_argument("--per-frame-points", type=int, default=30000)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(a.gt), "gt_viz"); os.makedirs(out, exist_ok=True)

    d = np.load(a.gt); depth, c2w, K, names = d["depth"], d["c2w"], d["intrinsics"], [str(n) for n in d["names"]]
    S, H, W = depth.shape
    idxs = list(range(S)) if a.frames == "all" else [int(x) for x in a.frames.split(",")]
    rng = np.random.default_rng(0)

    pts_dict, rgb_dict = {}, {}
    for i in idxs:
        world,_,mask = depth_to_world_coords_points(torch.from_numpy(depth[i]).float(),
            torch.from_numpy(c2w[i]).float(), torch.from_numpy(K).float())
        world = world.numpy(); mask = mask.numpy().astype(bool)
        name = names[i] if names[i].endswith(".JPG") else names[i] + ".JPG"
        img = np.array(Image.open(os.path.join(a.images, name)).convert("RGB").resize((W, H)))
        p, c = world[mask], img[mask]
        if len(p) > a.per_frame_points:
            s = rng.choice(len(p), a.per_frame_points, replace=False); p, c = p[s], c[s]
        key = names[i].replace(".JPG", "")
        pts_dict[key], rgb_dict[key] = p, c.astype(np.uint8)

    # repo's Cameras object (image_size arg is (width, height))
    cams = extri_intri_to_cameras(torch.from_numpy(c2w[idxs]).float(),
                                  torch.from_numpy(K).float().unsqueeze(0).repeat(len(idxs), 1, 1), (W, H))

    if a.mode in ("plotly", "both"):
        allp = np.concatenate(list(pts_dict.values()), 0)
        allc = np.concatenate(list(rgb_dict.values()), 0)
        fig = scene_to_plotly("GT_scene", allp, allc, cams)
        html = os.path.join(out, "gt_scene_plotly.html"); fig.write_html(html)
        print(f"wrote {html}  (repo scene_to_plotly)")

    if a.mode in ("rerun", "both"):
        from dvlt.viz.scene_rerun import visualize_scene
        rrd = os.path.join(out, "gt_scene.rrd")
        # per-frame dict -> each frame is its own toggleable entity in the Rerun tree
        visualize_scene(log_path="gt", cameras=cams, points=pts_dict, rgb=rgb_dict, save_path=rrd)
        print(f"wrote {rrd}  (repo visualize_scene; toggle frames in the entity tree;  rerun {rrd})")

if __name__ == "__main__":
    main()
