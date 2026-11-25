# residual_opt — flow-matching refiner on top of a frozen DVLT

Goal: learn a refinement that moves DVLT's output geometry toward the truth, in **output (ray / depth) space** rather
than in the latent space (the latent has a huge decoder null space, so "the correct token" is not unique; the GT rays and
depths are). Inspired by RayDiffusion / DiffusionSfM (diffusion over rays), PosePlaner / FlowR (flow from a
feed-forward estimate to the truth). DVLT (encoder, loop, decoder) stays frozen.

## Pipeline
1. `data_scannet.py` — ScanNet clips (S=8 frames, frame stride 30), color resized to the depth resolution 640×480,
   depth in metres, camera-to-world poses, depth intrinsics; `to_model_res` reproduces DVLT's preprocessing
   (504×378, grid 27×36 patches of 14 px).
2. `cache_dvlt.py` — runs the frozen DVLT (original K=16 schedule, fp32) and stores per clip: final tokens, the
   preprocessed image, DVLT's per-patch geometry `x_dvlt` = [log depth, unit ray direction (3), ray origin (3)] and the
   GT `x_gt` in the same frame (frame-0 camera coordinates) and scale (median pred/GT depth), plus a valid-depth mask.
   Train/val split by ScanNet room (sceneXXXX), so scans of one room never cross the split.
   Sanity check: DVLT vs GT ray directions differ by 1.6–1.8°, camera centres agree to a few cm (frames aligned).
3. `model.py` — `FlowRefiner` (6.3 M params): one token per patch = DVLT token (LN + linear) + 14×14 RGB patch embedding
   + embedding of [x_t, x_dvlt] + learned 2D position; time t enters through adaLN-style scale/shift; 6 blocks
   alternating frame attention and global attention (d=256, 8 heads); zero-initialised output head (untrained model
   returns DVLT's answer).
4. `train.py` — bridge flow matching: `x_t = (1-t) x_dvlt + t x_gt + σ_b σ_v sqrt(t(1-t)) ε`, target `v = x_gt - x_dvlt`
   (per-channel normalised), log-depth loss only on valid patches; inference = Euler integration from `x_dvlt` in
   n = 1, 2, 4, 8 steps. Metrics per clip (patch level): depth AbsRel (median scale), ray-direction error (deg),
   camera-centre ATE (Sim3, relative to trajectory extent).

## Usage
```bash
. ~/miniconda3/etc/profile.d/conda.sh && conda activate dvlt && export PYTHONPATH=$(pwd)
python residual_opt/cache_dvlt.py --n-train 40 --n-val 10 --clips 2 --out /storage/user/zhtia/resopt_cache/small
python residual_opt/train.py --max-train 4 --eval-on-train --steps 1500 --accum 1 --run overfit4    # overfit check
python residual_opt/train.py --steps 6000 --run small                                               # small run
```
Outputs: `out/resopt/<run>/{best.pt, log.json}`; logs in `out/resopt/*.log`.

## Results so far (2026-11-17)
| run | data | DVLT (AbsRel / dir / ATE) | refiner, best |
|---|---|---|---|
| smoke | 1 clip, eval on itself | 0.0198 / 1.82° / 0.052 | 0.0052 / 0.17° / 0.006 (300 steps) |
| overfit4 | 4 clips, eval on themselves | 0.0200 / 1.64° / 0.091 | 0.0095 / 0.078° / 0.0031 (1500 steps) |
| small | 80 train clips (40 rooms), eval on 20 clips of 10 unseen rooms | 0.0154 / 1.45° / 0.055 | **no gain**: best = step 500 (0.0154 / 1.42° / 0.063); end (6000 steps, train loss 0.024): 0.0189 / 1.46° / 0.075 |

The overfit checks only show that the code and the mechanism can fit. The `small` run does NOT generalise: train loss
keeps falling while unseen-room depth and camera get worse (memorisation). Likely reasons: (i) the refiner is OPEN-LOOP —
its conditioning (DVLT token, RGB patch) is fixed during integration and never looks at the input at the current
estimate, so it can only learn dataset-level biases of DVLT; (ii) little data (80 clips); (iii) DVLT is already very
accurate on ScanNet, which is in its training set (DéjàView Table 6: ScanNet 6.2%, ScanNet++ 4.0%, DL3DV 5.8%).
Next: closed-loop conditioning phi(x, y_t) (reproject with the current geometry into neighbour frames and sample/compare
features, RAFT/DROID/SDM style), more data, and out-of-distribution evaluation (BundleFusion, TUM RGB-D).

## Results 2026-11-18: 300-clip runs and photometric BA (all three FAIL to beat DVLT)
Data: 300 train clips (150 rooms), val 20 clips (10 unseen rooms), OOD = BundleFusion 32 clips (`ood_bf`).
AbsRel / ray-dir deg / ATE, patch level.

| run | how | val best | val end | ood_bf end |
|---|---|---|---|---|
| DVLT | frozen K16 | 0.0154 / 1.45° / 0.055 | | 0.0201 / 1.43° / 0.098 |
| `ol_300` | open-loop FM refiner, 6000 steps | 0.0152 | 0.0169 / 1.37° / 0.067 | 0.0224 / 1.75° / 0.108 |
| `cl_300` | closed-loop FM refiner (`--closed-loop`) | 0.0152 | 0.0175 / 1.39° / 0.064 | 0.0228 / 1.80° / 0.111 |
| `photo_ba` cam+depth λ1 | per-scene direct photometric BA, no learning, no GT | | 0.0282 / 1.67° / 0.068 | 0.0307 / 1.45° / 0.099 |
| `photo_ba` cam+depth λ10 | | | 0.0179 / 1.74° / 0.072 | 0.0221 / 1.46° / 0.098 |
| `photo_ba` cam only | | | 0.0154 / 1.75° / 0.073 | 0.0201 / 1.51° / 0.100 |

- Learned refiners: memorise (train loss 0.55 → ~0.1), OOD worse from the first eval on; val dir-error gain is a
  ScanNet-specific bias that reverses on BundleFusion. Closed-loop = open-loop: the reprojection branch is used (cl_gate
  0 → 0.075 around step 2000–2500, when val starts degrading) but adds nothing that generalises. Patch-level (14 px)
  comparison is too coarse for DVLT's ~1.3 px reprojection error.
- Photometric BA: every config worse than DVLT; GT-picked best iteration at most ties DVLT. The energy does not identify
  the truth at this precision: E(GT) is only 3.6% (val) / 5.6% (ood) below E(DVLT) and lower on only 15/20 and 21/32
  clips, while the optimiser drives E 20–30% BELOW E(GT) with worse geometry. Caveats: per-patch (piecewise-constant)
  depth, no exposure compensation — so this falsifies THIS simplified photometric BA, not photometric BA in principle.
- Take-away: DVLT's remaining error on these scenes is below what our GT-free observations can resolve; beating it needs
  observations MORE precise than DVLT (e.g. sub-pixel matches from an external matcher, as in DeepBA arXiv 2611.18256).

## Stationary loop (option A), 2026-11-19 → 11-23 — `train_stationary.py`
Frozen DVLT, constant gates, restoring branch R beside the global MLP, optional linear pull to x0 / RMS. Best:
`statA4_af_bptt8` (fixed α=0.05 + R, BPTT 8) — val AbsRel .0163 (K16) … .0178 (K128), OOD .0220 … .0224; Solv3R on the
same clips .0168 … .0173 / .0215 … .0213; DVLT schedule K16 .0151 / .0205 (collapses at K64). Full table in
experiment.md §3.9, figure `out/resopt/stationary_ksweep.png`, Solv3R eval `eval_solv3r.py` → `out/resopt/solv3r_eval/`,
inspection of R `inspect_stationary.py`, all report figures `plot_report.py`.

## Known limitations of v0
- patch-level geometry (27×36), not per pixel; origin is predicted per patch although it is constant per frame;
- one shared time t for all frames (no per-frame progress yet → streaming comes later);
- only ScanNet, 8-frame clips; DVLT was very likely trained on ScanNet, so these are in-distribution scenes.
