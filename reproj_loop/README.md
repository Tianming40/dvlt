# reproj_loop — real reprojection residuals inside the DVLT loop

Stage 4 (from 2026-11-23). Stage 3 showed that the loop can be made stable with an anchor (constant gates + pull to x0,
see `residual_opt/`), but accuracy stays at the prior. This folder asks the next question.

## Question
A looped transformer iterates in a purely latent space — a black box whose steps correspond to no physical quantity.
**If we open the loop (every step, or every few steps), decode the current tokens into geometry, compare that geometry
with REAL observations through a reprojection residual, and feed the residual back — does the loop get better?**
- regime 1: prevent / repair the collapse past the trained budget (K = 64, 128): large errors, strong signal expected;
- regime 2: beat DVLT@16 inside the budget: remaining error ≈ 1.3 px, comparable to matcher noise — the hard case.

Base model: the ORIGINAL DVLT (no pull, no R from stage 3), so any gain can only come from the residual.

## What counts as a meaningful observation
1. **fixed** — independent of the current estimate, does not change during the loop;
2. **tied to geometry by a known rule** — the projection, not a learned comparison;
3. **discriminative before anything is trained** — its energy must be clearly lower for GT than for DVLT, on almost every
   clip (photometric BA failed exactly here: E(GT) only 4–6 % below E(DVLT)).

Candidate: RoMa v2 dense sub-pixel matches with per-pixel precision Σ⁻¹ (arXiv 2511.15706; ScanNet++ v2 EPE 4.0 px,
45.5 % within 1 px) — to be filtered by certainty, used with a robust kernel.

## Residual
For a match (u_a in frame a, u_b in frame b) and the decoded per-pixel depth d and rays (origin o, direction v):
X = o_a + d·v_a / (v_a·axis_a), then either
- pixel reprojection: project X with the camera of frame b (fitted from its rays) → r = û_b − u_b, E = Σ ρ(rᵀ Σ⁻¹ r);
- angular form (no intrinsics): r = ∠(v_b(u_b), X − o_b).
Both directions (a→b, b→a). Invariant to the similarity gauge.

## Routes (to be decided step by step)
- **Route 1 — residual → tokens (no training):** z ← z − η ∇_z E through the decoder, every (few) step(s); compare
  none / RoMa residual / GT residual (upper bound, = the stage-2 oracle) at K = 16 and K = 64.
- **Route 2 — residual → weights:** test-time adaptation of a few parameters per scene (gates, LoRA), or E as a
  self-supervised training loss.
- (later) learned update U(h, φ(r)) beside the global MLP (BA-T / DROID style), on top of the stage-3 stable loop.

## Step 0 (before any of the above): energy check, no training
On the analysis scene (and later the cached clips): E for GT geometry, DVLT@16, DVLT@64 (collapsed).
E(collapsed) ≫ E(DVLT@16) → regime 1 is feasible; E(GT) clearly < E(DVLT@16) on almost every pair/clip → regime 2 is feasible.
Also measure RoMa's actual error on our data: ‖u_b − u_b_gt‖ with GT-projected matches (u_b_gt from GT depth + poses).

---
# Design notes (discussion 2026-11-23, not final)

## Data
- **Analysis scene: the 27-frame s10 scene** used throughout stages 1–3 (ScanNet++ 00777c41d4, every 10th DSLR frame),
  now in `out/data/gt_00777c41d4_s10/` (`probe_input/` 27 images, `gt.npz` per-pixel GT) — results comparable with all
  earlier analyses, per-pixel GT available. 27 frames → 351 pairs (702 directed); filter pairs by RoMa overlap / frame gap.
- **Training data (only if training is needed later):** `/storage/user/zhtia/resopt_cache/{small,large}` — ScanNet 8-frame
  clips (stride 30), 378×504 DVLT images, PATCH-level x_dvlt / x_gt (+ tokens in `small`); val 20 clips, OOD BundleFusion
  32 clips. Missing for reprojection: per-pixel GT (reload from ScanNet via scene + frame ids, `residual_opt/data_scannet.py`)
  and RoMa matches (1196 clips × 56 directed pairs). ScanNet is inside DVLT's training set → BundleFusion is the honest test.
- Results go to `out/reproj_loop/<experiment>/`.

## RoMa v2 (observation provider) — computed ONCE per frame pair, offline
- Code: `third_party/RoMaV2` (user's fork of Parskatt/RoMaV2, commit 95c9968). Env: conda `romav2` (py3.12, torch 2.8 cu128,
  torchvision 0.23, editable install, `--no-deps` + einops/rich/tqdm). Weights cached in `/storage/user/zhtia/.cache/torch/hub/`.
- The fused local-corr CUDA kernel is NOT installed (PyPI wheel built for CUDA 13); the native torch fallback is used.
  Local A2000 (shared, 12 GB): `base` setting works (1.5 s/pair, 3.2 GB); `precise` OOMs → run on 48 GB cluster GPUs.
- Input: an image pair (0–1). Output (dense): `warp_AB` (H,W,2) in [-1,1] (pixel x = (x+1)/2·W, pixel-centre convention),
  `overlap_AB` (sigmoid), `precision_AB` (H,W,2,2) = Σ⁻¹ (new in v2), and the BA direction if bidirectional.
  Settings: precise 800→1280 bidirectional (default), base 640, fast 512, turbo 320.
  `sample()` returns only one row of the precision matrix in the non-bidirectional path → read Σ⁻¹ from the dense map ourselves.
  `geometry.warp_and_overlap_from_depth` gives GT warps from depth + K + pose.
- Matches never depend on the current estimate (no geometry-guided re-matching — that would make the observation
  self-consistent like BA-T). Recompute only for new frames (new pairs) or a changed preprocessing.
- **What to store per scene/clip:** directed pair indices (a,b); N≈5000 matches per direction: u_a, u_b in DVLT 378×504
  pixel coordinates (state the pixel-centre convention), overlap, Σ⁻¹ of u_b in DVLT pixel units; GT for diagnostics:
  per-pixel GT depth at DVLT resolution, K, poses rel. to frame 0, GT-projected u_b_gt + visibility; metadata (frame ids,
  RoMa setting, original vs DVLT input, resize/crop mapping). Not the dense warps (~85 MB per 8-frame clip).
  Coordinates must follow DVLT's preprocessing exactly (long side 504, centre crop to multiples of 14).

## Residual formulation — depth-parameterised, pairwise (DROID-SLAM style), no shared 3-D points
- DVLT decodes each frame separately → for a match two points X_a (depth of a along ray of a) and X_b (depth of b along
  ray of b), both already in the world (frame-0) frame. They are NOT merged; each frame's depth is checked against the
  observation separately:  r_a→b = π_b(X_a) − u_b  and  r_b→a = π_a(X_b) − u_a  (or the angular form with v_b(u_b)).
  r_a→b constrains depth_a, camera_a, camera_b. Many pairs → all frames pulled to mutual and observed consistency.
- Rejected: 3-D point-to-point X_a − X_b (shrinking the scene makes it vanish → trivial consistent-but-wrong solution;
  metric units, not matched to RoMa's pixel precision). Variant kept in mind: triangulate X* from both rays, compare depths.
- Worked example (2 frames, f=500, baseline 0.2 m, true X=(0.4,0,2), u_a=(420,240), u_b=(370,240)): DVLT depths 2.2 / 1.9 →
  r_a→b = +4.5 px, r_b→a = +2.6 px; Gauss–Newton on each depth (du/dd = 20.7 / −27.7 px/m) → 1.98 / 1.994 → both → 2.0;
  a 5 cm camera-baseline error alone gives r_a→b = −12.5 px. Classical BA on the same pair: ONE shared X (3 DOF) moved to
  the ray intersection in ~2 GN steps. Ours = per-frame (inverse-)depth parameterisation as in DSO / DROID-SLAM; the
  variables are TOKENS (decoder = parameterisation) — a change of one token moves a region of depth and the camera.
- Energy: E = Σ_pairs Σ_matches overlap · ρ(rᵀ Σ⁻¹ r), Huber ρ, occlusion test (depth of X in b vs DVLT depth at u_b).
- Apply from loop step ~4 on: A2 (stage 3) shows the loop CONTRACTS in steps 0–2 (g = 0.76–0.95), g ≈ 1 at step 3,
  and AMPLIFIES from step 4 (peak ×1.09 at step 6–7 for K16, ×1.13 at step 7–8 for K64).

## What can be modified (options; order to be decided)
No training (per scene, test time):
1. tokens (state) — cleanest test of whether the residual carries information; must be applied every step (one-shot
   edits are erased); direct comparison with the stage-2 oracle;
2. gates (per-step s_attn/s_mlp/s_out) — step size / stopping only, not direction;
3. register + camera token initial values (5×768, "prompt tuning", cf. Test3R) — few params, effect persists;
4. the input z0 (shooting through all K steps — expensive, start information decays);
5. attention guided by the matches (GT-visibility masks failed before, but those were coarse, not precise matches).
Training (offline, or test-time training with E as the loss and no GT):
- global attention (cross-view exchange, matching heads 0/9) — e.g. LoRA; most directly related to reprojection;
- global MLP — learned update rule U(h, φ(r)) at the stage-3 R slot;
- frame attention — single-view prior, little relation to reprojection; gate network — magnitude only;
- decoder heads — AVOID alone: they can make the output consistent while the loop state stays ungrounded.
Tentative order: tokens → few-parameter test-time adaptation (gates / register+camera tokens) → LoRA on global attention
or a trained U.

## Open questions
residual form (angular vs pixel); RoMa input (original resolution vs DVLT 378×504); pair selection on 27 frames;
where to store matches (`/storage/user/zhtia/reproj_cache/` vs `out/data/`); which modification to start with.

## Status
2026-11-23: folder + design notes; `romav2` env ready and smoke-tested; nothing implemented yet.
