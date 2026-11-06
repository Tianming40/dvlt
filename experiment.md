# Experiment: Perturbation Stability of the Recurrent Loop

**Question.** Is DVLT's weight-tied recurrence stable as a dynamical system? If we
perturb the intermediate state at step *k*, do the remaining steps pull the output
back toward the unperturbed trajectory (directional attractor), or does the
perturbation persist / amplify (brittle, same mechanism as the >K_max collapse)?

This directly tests the paper's "directional refinement" claim, which is only
characterized indirectly (cosine similarity, norm growth; Fig. 3) and never probed
with perturbations.

---

## Core protocol

- [ ] **Reference run.** Run inference normally, cache every intermediate state
      `z_1 … z_16` and the final outputs (depth, pose, point cloud) as the baseline.
- [ ] **Perturbed run.** Re-run; at step 8 inject a perturbation into `z_8`
      (additive random noise, or swap a subset of tokens), then continue the
      remaining 8 steps with the *same* weights.
- [ ] **Compare.** Measure how far the perturbed final output is from the reference.

---

## Design fixes (so the result is conclusive, not a toy)

- [ ] **Do not measure the difference on `z`.** The state norm grows with *k*, so a
      raw L2 on `z` is confounded. Measure on the **decoded outputs** (depth / pose /
      pointmap) and on the **direction** of the state (cosine to the unperturbed `z_k`).
- [ ] **Record the recovery curve**, not just the final gap: after injecting at
      step 8, log the output distance to the reference at every subsequent step —
      is it pulled back monotonically or does it drift further?
- [ ] **Sweep perturbation position:** inject at `z_2`, `z_8`, `z_14`. Hypothesis:
      more remaining steps → better recovery. This is the key figure.
- [ ] **Sweep perturbation magnitude:** find the linear (recoverable) regime vs. the
      nonlinear collapse threshold.
- [ ] **Two probe types:** (a) additive noise → tests raw sensitivity;
      (b) token swap (e.g. replace one view's tokens) → tests semantic robustness /
      self-correction.

---

## The high-value control (ties to the paper's central claim)

- [ ] **Shared vs. decoupled.** Run the same perturbation protocol on the decoupled
      ablation (`recurrence_mode=none`, 16 independent blocks; config
      `dvlt-large-ablation-decoupled`).
- [ ] **Answer:** does weight sharing make the model *more* or *less* robust to
      mid-trajectory perturbations? If shared recovers better, that is a new,
      stronger piece of evidence for "iteration is a stronger inductive bias"
      than anything currently in the paper.

---

## Implementation notes

- [ ] Injection point: `DVLTModel._solve_inference_linspace_k` in
      `src/dvlt/model/dvlt/model.py` — it loops `_interval_step`; add the
      perturbation to `x` after step *k*, weights unchanged.
- [ ] Decode intermediate states with the existing `_decode(...)` to get per-step
      outputs for the recovery curve.
- [ ] Start by validating the harness with random weights (shape/flow only), then
      run with the released checkpoint on real test images.

## Deliverables

- [ ] Recovery-curve plot: output distance vs. step, one line per injection position.
- [ ] Magnitude-threshold plot: final output distance vs. perturbation strength.
- [ ] Shared-vs-decoupled comparison on the above.

---

# Experiment 2: Opening the Box — What Happens at Each Iteration Step

**Question.** What does the model actually "see" at each intermediate step? Decode
every state `z_1 … z_16` into full outputs (point cloud, depth, pose, confidence)
and watch the geometry form step by step.

**Caveat (overlap with the paper).** Fig. 3a already decodes `z_k` per step and
shows *scalar metrics* (Pose AUC, pointmap L2) improving monotonically. So the bare
"metrics vs. step" plot is not new. The value here is in the **qualitative /
spatial** story the paper does not show — where and how the geometry resolves.

---

## Core protocol

- [ ] At inference, decode **every** intermediate state `z_k` (k = 1 … 16) with the
      existing `_decode(...)`, not just the final one.
- [ ] Save per-step: depth map, world point cloud, camera poses, confidence map.
- [ ] Render a step-by-step sequence (image grid or video) of the point cloud and
      depth evolving across iterations.

## Angles the paper does NOT cover (the actually-new part)

- [ ] **Spatial error maps per step.** Where does error concentrate early and get
      fixed late? (textureless regions, object boundaries, far depth, thin
      structures). Overlay per-pixel error on the image at each k.
- [ ] **Per-quantity convergence rates.** Do pose, depth, and rays converge at
      *different* speeds? E.g. does global pose lock in early while fine depth keeps
      refining? Plot each quantity's distance-to-final vs. step.
- [ ] **Confidence evolution.** How does the predicted confidence map change across
      steps, and does rising confidence actually track falling error?
- [ ] **Cross-view agreement over steps.** Measure multi-view consistency (do the
      per-view point clouds align better as k grows?) — this probes when the
      global-attention sub-block "stitches" the views together.
- [ ] **Early-exit quality.** At which step is the output "good enough" for a given
      tolerance? Characterize the accuracy/compute trade-off per scene.
- [ ] **Where the camera token vs. patch tokens settle.** Track the camera-token
      readout (pose) vs. dense patch readout (geometry) separately across steps.

## Implementation notes

- [ ] Same hook point as Experiment 1: `_solve_inference_linspace_k` in
      `src/dvlt/model/dvlt/model.py` — call `_decode(...)` on `x` after each
      `_interval_step`, collect a list of per-step predictions.
- [ ] Reuse the existing viz stack (`src/dvlt/viz/`: glb / pointcloud / rerun) to
      render the per-step point clouds; Rerun is well suited to a step slider.

## Deliverables

- [ ] Per-step point-cloud / depth sequence (grid or video) for a few scenes.
- [ ] Spatial error-map sequence overlaid on the input images.
- [ ] Per-quantity (pose / depth / rays / confidence) convergence-rate plot.
- [ ] Cross-view consistency vs. step plot.

---

# Grand probe: the 45 per-step metrics

Every version (baseline + each K / schedule / renorm / ablate / perturbation-level / photo) is logged to a TSV with one row per loop step and an identical 52-column schema. **Each metric below has its own figure `<metric>.png`** (one line per version), in `out/full_grand/figs/` (all versions), `figs_le64/` (drop K=128/256 & frozen-256), `figs_le16/` (steps ≤16 only) — same filename in every folder.

Notation: `x` = state before the step, `x'` = after, `Δ = x' − x`, shaped (S frames, N tokens, C=768). Patch tokens = `x[:,5:]`, camera token = `x[:,0]`, register tokens = `x[:,1:5]`. `‖·‖` is the L2 norm over channels; `ẑ = x/‖x‖` is the unit direction (uncentered). Clustering/symmetry use a FIXED 1024 patch-token subsample. `tang(v) = v − (v·x̂)x̂`. GT depth uses median-scale alignment (q = aligned prediction, g = GT). Global-attention quantities recompute `q,k` (after qk-norm) from the captured attention input, `A = softmax(scale·q·kᵀ)`; NaN on ablated steps.

Identifier/time columns (not plotted): `version, family, kind, eps, at, step, t` (t = time fed to the gates this step).

| metric | figure | formula | meaning |
|---|---|---|---|
| tok_norm | tok_norm.png | mean_token ‖x'_patch‖ | patch-token magnitude (= paper's feature norm ‖z_k‖; norm is a free gauge) |
| norm_std | norm_std.png | std_token ‖x'_patch‖ | spread of token norms (are a few blowing up?) |
| cam_norm | cam_norm.png | mean_frame ‖x'[:,0]‖ | camera-token magnitude |
| reg_norm | reg_norm.png | mean ‖x'[:,1:5]‖ | register-token magnitude |
| upd_abs | upd_abs.png | mean_token ‖Δ_patch‖ | absolute per-step update |
| upd_rel | upd_rel.png | mean_token ‖Δ_patch‖/‖x'_patch‖ | relative per-step update (= paper's relative update norm ‖Δz‖/‖z‖) |
| upd_cos_prev | upd_cos_prev.png | mean_frame cos(Δ_flat, Δ_prev_flat) | update coherence (steady vs jitter); step 0 = NaN |
| tok_cos_step | tok_cos_step.png | mean_token cos(x_patch, x'_patch) | direction stability (→1 = converging) |
| cam_upd_rel | cam_upd_rel.png | mean ‖Δ[:,0]‖/‖x'[:,0]‖ | camera-token relative update |
| reg_upd_rel | reg_upd_rel.png | mean ‖Δ[:,1:5]‖/‖x'[:,1:5]‖ | register-token relative update |
| top1_share | top1_share.png | max(e)/Σe, eₖ=mean_{frame,token}(x'_patch)²ₖ | energy share of the dominant channel |
| top8_share | top8_share.png | Σ(top-8 e)/Σe | energy share of the top 8 channels |
| eff_chan | eff_chan.png | (Σe)²/Σ(e²) | participation ratio = effective #active channels (low = collapsed) |
| cluster_cos | cluster_cos.png | mean_{i<j} ⟨ẑcᵢ,ẑcⱼ⟩, ẑc = normalize(v − mean_c v) | mean pairwise direction cosine (centered); token alignment |
| E_b1 | E_b1.png | mean_{i,j} exp(1·⟨ẑᵢ,ẑⱼ⟩) | Geshkovski interaction energy, β=1 (Lyapunov; rise = clustering) |
| E_b4 | E_b4.png | mean_{i,j} exp(4·⟨ẑᵢ,ẑⱼ⟩) | interaction energy, β=4 |
| E_b10 | E_b10.png | mean_{i,j} exp(10·⟨ẑᵢ,ẑⱼ⟩) | interaction energy, β=10 (sharper) |
| k_eff | k_eff.png | (Σλ)²/Σλ² of Gram G=ẑẑᵀ = n²/Σ_{i,j}⟨ẑᵢ,ẑⱼ⟩² | effective #distinct directions (=k for k orthogonal clusters) |
| ncomp_t0.8 | ncomp_t0.8.png | #connected components (size≥2) of {⟨ẑᵢ,ẑⱼ⟩>0.8} | #non-singleton clusters at cos>0.8 |
| ncomp_t0.9 | ncomp_t0.9.png | same at threshold 0.9 | #clusters at cos>0.9 |
| ncomp_t0.95 | ncomp_t0.95.png | same at threshold 0.95 | #clusters at cos>0.95 |
| g_frame_attn | g_frame_attn.png | frame-attn gate s_attn (mean) | time-conditioned gate on frame attention branch |
| g_frame_mlp | g_frame_mlp.png | frame-attn gate s_mlp (mean) | gate on frame MLP branch |
| g_frame_out | g_frame_out.png | frame-attn gate s_out (mean) | gate on whole frame-attn state |
| g_global_attn | g_global_attn.png | global-attn gate s_attn (mean) | gate on global attention branch |
| g_global_mlp | g_global_mlp.png | global-attn gate s_mlp (mean) | gate on global MLP branch |
| g_global_out | g_global_out.png | global-attn gate s_out (mean) | gate on whole global-attn state |
| attn_entropy | attn_entropy.png | mean_{head,query} −Σⱼ Aᵢⱼ log Aᵢⱼ | attention peakedness (low = sharp matching) |
| attn_self | attn_self.png | mean_{head,query} A[query,query] | self-attention weight |
| attn_push_rel | attn_push_rel.png | mean ‖oᵢ‖/‖xgᵢ‖ | size of the attention push relative to the state |
| attn_tang_frac | attn_tang_frac.png | mean ‖tang(oᵢ)‖/‖oᵢ‖ | fraction of the attention output that is tangential (on-sphere) |
| sym_corr | sym_corr.png | Pearson corr(Lᵢⱼ, Lⱼᵢ) off-diag, L = head-mean logit | correlation of logit matrix with its transpose |
| sym_asym | sym_asym.png | ‖½(L−Lᵀ)‖_F/‖L‖_F ∈[0,1] | realized logit asymmetry (0 = symmetric = gradient-flow-compatible) |
| attract | attract.png | mean cos( tang(oᵢ), tang(cᵢ−xgᵢ) ), cᵢ=head-mean Σⱼ Aᵢⱼ·xgⱼ | how much the attention output pulls toward the weighted neighbour mean |
| gt_absrel | gt_absrel.png | mean |q−g|/g | depth absolute relative error vs GT |
| gt_d1 | gt_d1.png | fraction[ max(q/g,g/q) < 1.25 ] | depth accuracy at 1.25 |
| gt_d2 | gt_d2.png | fraction[ … < 1.25² ] | depth accuracy at 1.25² |
| gt_d3 | gt_d3.png | fraction[ … < 1.25³ ] | depth accuracy at 1.25³ |
| gt_rmse | gt_rmse.png | sqrt(mean (q−g)²) | depth RMSE vs GT |
| gt_ate | gt_ate.png | Umeyama Sim(3) RMSE of pred camera centres vs GT ÷ trajectory extent | scale-free camera-trajectory error |
| depth_to_std | depth_to_std.png | median|log d − log d_ref16| (median-shifted) | depth distance to the K=16 standard output (anchor 1) |
| ray_to_std | ray_to_std.png | mean angle(pred rays, ref16 rays) (deg) | ray-direction distance to the 16-step standard |
| depth_to_prev | depth_to_prev.png | depth_to_std but reference = previous step | per-step output change (output convergence) |
| ray_to_prev | ray_to_prev.png | ray_to_std but reference = previous step | per-step ray change |
| tok_rel_to_clean | tok_rel_to_clean.png | mean ‖x'_patch − clean_patch[same step]‖ / ‖clean_patch‖ | distance to the clean trajectory (injection→recovery); K=16-aligned versions only, else NaN |

Static (in `meta.json`, not per-step, no figure): `global_asym` / `frame_asym` = per-head ‖½(B−Bᵀ)‖/‖B‖ with B_h = Wq_hᵀWk_h (pre-LN asymmetry of each head's interaction matrix).

---

# Perturbation definitions (how each kind is injected)

Perturbations are produced by `make_perturbation(x, kind, eps, ...)` in `dvlt_probe.py` and applied to the state **just before loop step `at`** (`x ← make_perturbation(x)`, then the loop continues normally from step `at`). Noise is always scaled by each token's own norm, so `eps` is a **relative** fraction (comparable across tokens). Below, `patch` = a token vector, `‖patch‖` = its channel-L2 norm, `n` = a random direction, `pm = patch_start_idx = 5` (tokens: 0 = camera, 1:5 = 4 registers, 5: = image patches). Random seed fixed (reproducible).

| kind | what it does | formula | which tokens |
|---|---|---|---|
| gauss | isotropic noise (perturbs both direction and norm) | n = normalize(randn)·‖patch‖ ; patch ← patch + eps·n | all patch tokens, all frames |
| scale | pure magnitude change (direction unchanged) | patch ← patch·(1+eps) | all patch tokens |
| dir | pure direction change (norm unchanged) | v = normalize(patch + eps·normalize(randn)·‖patch‖)·‖patch‖ ; patch ← v | all patch tokens |
| cam | gauss noise on the camera token only | same as gauss but on x[:,0] | camera token (idx 0) |
| register | gauss noise on the register tokens only | same as gauss but on x[:,1:5] | register tokens (idx 1:5) |
| swap | structured wrong-view error (eps-independent) | x[0,5:] ← x[1,5:] (frame 0's patches ← frame 1's) | frame 0 patch tokens |

Injection: `kind` ∈ {gauss, scale, dir, cam, register, swap}; `eps` ∈ {0.02, 0.05, 0.1, 0.2, 0.5, 1.0} (swap uses one eps only); `at` ∈ {1, 4, 12} (early / mid / late). The TSV columns `kind / eps / at` record which perturbation each row came from.

Contrasts worth remembering:
- **scale vs dir** = "does norm matter vs does direction matter": scale (norm only) is nearly harmless even at eps=1 (LayerNorm absorbs it); dir (direction only) damages like gauss → the clean "direction matters, norm is a gauge" demonstration.
- **cam / register** perturbations have ~zero effect on the output at any eps → those special tokens are re-derived by the loop and are not critical carriers.
- **early vs late (at)** for strong gauss/dir: early strong noise can derail the trajectory into a bad basin (ate → ~0.95); late strong noise is more survivable (geometry already formed).

Per-kind figures (6 panels each, with baseline + GT anchors) are in `out/full_grand/perturb/<kind>.png`, generated by `scriptandtool/plot_perturb.py`.

---

# STAGE 2 — mechanistic deep-dive  【PLANNED / NOT STARTED】

Stage 1 characterized the loop quantitatively (scalar metrics, per-step curves, 115 versions). Stage 2 goes DEEPER and more concrete: open up the model to the level of individual attention heads, specific tensors and dimensions, and SEE what actually happens — not just aggregate statistics. All exploratory / qualitative / mechanistic.

## Goals (as stated)
1. **Deep code dive** — down to specific attention heads, individual tensors, and the various dimensions (what each head/tensor/dim actually carries and does). [TODO / not done]
2. **Deep attention tracking + visualization** — not just scoring attention vs GT; actually trace and visualize what each head attends to, head-by-head and step-by-step through the loop. [TODO / not done]
3. **Per-step decoded output** — what each loop step ACTUALLY decodes to (depth / point cloud / rays / camera at every step k): watch the reconstruction form, see concretely what each step does. [TODO / not done]
4. **Norm constraint, deeper** — go beyond Stage-1's quantitative renorm result: understand mechanistically what happens when the norm is constrained (and explore variants of constraining it). [DONE 2026-10-30 — `renormsweep` (per-token norm clamp, direction-preserving) + `refeed` (reversibility): clamping ‖z‖ does NOT fix K64 collapse (k_eff stays low → collapse is directional); re-feeding re-diversifies direction once but can't restore the answer and then diverges. See "Collapse-intervention experiments" below.]

## Spirit
- Stage 1 = quantitative / aggregate (scalars across 115 versions).
- Stage 2 = mechanistic / specific — individual heads, specific tensors, concrete dimensions, actual decoded artifacts, qualitative "what is it really doing".

## Notes
- Constraints: Stage-2 code lives in `shovel/` (+ `scriptandtool/` for sbatch); reuse repo functions directly (via `shovel/_common.py`), no `Probe` wrapper; never touch dvlt source (runtime hooks/monkeypatch from `shovel/` are OK); keep old Stage-1 commands working.
- In progress: per-step decode dump (`shovel/step_decode.py`) + per-step pred-vs-GT viz (`shovel/predict_viz.py`, `--align-mode per-step|fixed|none`) = Stage-2 #3. Attention tracking (#1/#2) not yet implemented — see the two subsections below.

## Reference — "Understanding multi-view transformers" (Stary, Gaubil, Tewari, Sitzmann; arXiv 2510.24907, Oct 2025)

Highly relevant to Stage 2: it is essentially the mechanistic-interpretability program we plan, done on DUSt3R (same multi-view-transformer family as DVLT; DUSt3R has distinct decoder blocks ≈ DVLT's loop steps). Code: github.com/JulienGaubil/und3rstand.

### Method
- Train a probe on the residual/skip connection after each decoder layer to regress POINTMAPS (per-pixel 3D points), then visualize the latent 3D geometry developing layer-by-layer (4D).
- Probe = **5-layer non-linear MLP**, restricted to **individual patch features** (no cross-patch communication) so it reflects per-patch content and cannot solve the task itself. Trained with DUSt3R's confidence-aware scale-invariant loss. 3 probes per block (after self-attn, cross-attn, MLP).

### Findings
1. Iterative refinement: encoder already does first-view monocular depth+intrinsics; decoder refines — rotation resolved early, scale/translation over blocks; hard (opposing) views need many iterations.
2. Layer roles (error measured after Procrustes alignment): first view = stable anchor, second view transformed. **Cross-attention = cross-view correspondence** (moves "transported" patches toward matches) but INCREASES aligned error +11%; MLP +7%. **Self-attention does 94% of the error reduction** (re-establishes intra-view geometry).
3. Global pose NOT used: 4/12 heads attend to fixed register-like tokens (candidate pose store), but attention-knockout on them leaves pointmap/pose UNCHANGED → explicit global pose not crucial.
4. Relies on correspondences: 5/12 cross-attn heads do correspondence; accuracy 40%→60%+ over first 6 blocks; correspondences evolve SEMANTIC→GEOMETRIC; fall back to register tokens on failure; correspondences co-refined with geometry.

### Convergence with our Stage-1 DVLT findings
- Register/cam-token knockout harmless (theirs) == our cam/register perturbation = zero effect.
- "No global pose, relies on correspondence" == our attn result (global attention does cross-view matching).
- Correspondences semantic→geometric, sharpening == our attn (sharpest ~step 8).
- Encoder does monocular depth, pose built iteratively == our ate 0.95→0.01 (pose constructed from scratch).
- Contrast to analyze on DVLT: they find self-attn (≈ our frame attn) does the intra-view geometry while cross-attn (≈ our global attn) does correspondence but temporarily raises the Procrustes-aligned error — a different decomposition than our "global attention = pose machine" (ate collapses under ablate:all). Worth reconciling in Stage 2.

### Lessons for Stage 2
- Use a **non-linear per-patch MLP probe** (they show linear is insufficient), not a linear ridge, for the "what's in the state" probe (angle A).
- Probe target: **pointmaps** (geometric, scale-invariant, matches training) rather than depth alone.
- Their findings = a ready list of predictions to test on DVLT: frame-vs-global attention division of labor, register-token fallback, correspondence semantic→geometric, early rotation.

## Stage-2 resources already in the codebase

### Codebase lineage (relevant: enables the "what does looping buy" comparison for free)
- DVLT is **not built from scratch**: it is based on NVIDIA's **VGGT** — the recurrent block `LoopedAABlock` IS VGGT's **Alternating Attention** (frame ↔ global), made weight-shared/looped + time-gated. Evidence: `loss/{util,common,camera}.py` carry `LICENSES/VGGT-LICENSE.txt`; `model/` ships VGGT, Pi3, MapAnything side-by-side; `model_registry.py` lists them.
- Lineage: DUSt3R → VGGT (alternating attention, NVIDIA) → **DVLT = VGGT's AA block looped + time-conditioned**. Parts: DINOv2 encoder, MoGe UV grid, DUSt3R-style pointmap/scale-invariant loss, Nerfstudio colormaps.
- **Implication for Stage 2**: the feed-forward baseline VGGT (non-looped AA) is in the SAME repo → "what does the loop buy?" (angle F) can reuse the identical probe with near-zero extra cost.

### Reusable visualization components (do NOT rebuild these for Stage-2 #3)
The per-step decoded-output visualization (depth / point cloud / rays / camera) can be built almost entirely from existing code — our probe already decodes {depth, rays} at every step; feed that into:
- `viz/pointcloud.py` : `build_pointcloud` / `build_pointcloud_with_frame_ids` — depth+rays → point cloud, outlier filtering, per-frame coloring.
- `viz/scene_rerun.py` : `visualize_scene` — export a Rerun (.rrd) scene (points + cameras). **Rerun has a time axis → use loop-step as time → scrub to watch the reconstruction form over the 16 steps.**
- `viz/scene_plotly.py` : `scene_to_plotly` — interactive 3D plotly (web).
- `viz/camera_overlay.py` : camera frustum / trajectory meshes — watch per-step pose convergence.
- `viz/glb.py` : `pointcloud_to_glb` — export .glb.
- `viz/depth.py` : `overlay_depth_map` — depth overlaid on the image (per-step depth sequence).
- High-level wrappers: `scripts/visualize.py` `visualize_predictions_rerun` (model → pointcloud → rerun, end-to-end); `callbacks/visualization.py` `SceneVisualizationCallback` (`_prepare_pointcloud_data`, plotly + rerun).

Minimal plan for Stage-2 #3 (per-step 3D): for each loop step, decode {depth, rays} (probe already does this) → `build_pointcloud(...)` → `visualize_scene(...)` with step as the Rerun time axis. No point-cloud/rendering code to write.

### NOT available (must build)
- Per-head attention visualization (Stage-2 #2): `viz/` has no attention tooling. Reuse the Stage-1 `cmd_attn` machinery (global-attn capture via forward-pre-hook + `gt_correspondence`) as the starting point.

### Attention is NOT materialized (flash) — the key barrier for Stage-2 #1/#2  【verified 2026-10-27】
The model uses **fused / flash attention**, so the N×N attention matrix is never a tensor you can read. To inspect attention you must recompute `q·kᵀ` yourself. Verified facts:
- **Token layout per frame** (`patch_start_idx=5`, model.py:287): `[ cam(1) | register(4) | patch(gh·gw) ]` → `P = 5 + gh·gw`, where `gh=H//14, gw=W//14` for the ACTUAL (preprocessed) image H×W — **not** fixed, it depends on the image aspect ratio. Preprocessing scales the longest side to `img_size=504` and crops to a multiple of 14. Examples: a square 504×504 image → 36×36=1296 patches → `P=1301`; the **stride-10 dataset (frame 336×504 → grid 24×36=864)** → **`P=869`**, and at 27 frames the global sequence is **`N = 27·869 = 23463`**. The cam token has a special first-frame variant (`camera_token[:,0]` for frame 0, `[:,1]` for the rest — `_slice_expand_flatten`, model.py:80), so frame 0 is the designed reference/anchor.
- **Dims**: `embed_dim=768`, `num_heads=12`, `head_dim=64` (attention.py:75-76).
- **One loop step = one weight-shared `LoopedAABlock`** (model.py:413, `recurrent_blocks[0]`), containing exactly **2 attention sub-blocks**: `frame_attn` (intra-frame, **with RoPE**, `P` keys) and `global_attn` (cross-frame, **no RoPE**, `S·P` keys).
- Nothing in the Stage-1 probe hardcodes a token count (`patch_part = x[:, patch_start_idx:]`, indices via actual dims) → non-square images were handled correctly throughout Stage 1; `P=1301` appeared only in discussion, never in experiment code.
- **Why the matrix is gone**: `Attention.forward` defaults to `fused_attn=True` → `F.scaled_dot_product_attention` (attention.py:113-128), which computes `softmax(q·kᵀ)·v` inside a fused kernel and returns only the value output `(B,N,C)` after `proj` (attention.py:136-139). The explicit `attn=(q@kᵀ).softmax()` path exists ONLY in the non-default `else` branch (`fused_attn=False`). A `forward_hook` on the `.attn` module therefore yields `(B,N,768)` features, **not** an `(B,H,N,N)` matrix. (This is the exact bug in the pasted `SinglePatchRecorder.frame_hook`/`global_hook` sketch, which did `output[:,:,idx,:]` expecting the attention row.)
- **Correct way to read one attention row (no source edit, no math duplication)**: `register_forward_hook(..., with_kwargs=True)` on the `Attention` module; inside the hook use the module's OWN layers to recompute q,k from `args[0]` (the already-normed input) and `kwargs["pos"]`:
  `qkv = module.qkv(x).reshape(B,N,3,H,Dh).permute(2,0,3,1,4); q,k,_ = qkv.unbind(0); q,k = module.q_norm(q), module.k_norm(k); if module.rope and pos is not None: q,k = module.rope(q,pos), module.rope(k,pos)` → then for tracked token `i`: `row = softmax(scale · q[:,:,i,:] @ k.transpose(-2,-1))` `(B,H,N)`. Mathematically identical to SDPA; reuses the module's own weights/RoPE so there is no second copy of the math. Frame-attn needs RoPE on q,k; global-attn does not. (Verified: recomputed rows sum to 1; a patch's global attention peaks on its own cell at step 0.) Same approach already used in Stage-1 `cmd_attn` (global only); Stage-2 adds the frame-attn RoPE case.

### Single-token tracking (Stage-2 #1/#2)  【DONE 2026-10-28 — ran K8/16/32, viz + quantitative analysis below; K64/128 submitted】
Full `N×N` per-step per-head attention is TB-scale; instead track a SMALL fixed set of token indices and store only THEIR rows — MB-scale (~4.5 MB/step for 8 tokens). Pipeline: `pick_patches.py` (choose + visualize patches) → `track_token.py` (run loop, record) → `viz_attention.py` (per-head heatmap overlays) → `attn_video.py` (step/frame videos) → analysis.
- **Dataset / loop**: stride-10 set (27 frames, scene 00777c41d4), `_interval_step` linspace loop, run once per **K ∈ {8,16,32}** (`submit_track.sbatch`).
- **Tracked set = 8 frame-0 tokens**: `cam[0]` (idx 0), `register[0]×4` (idx 1–4), and **3 patches** auto-picked on frame 0 by GT depth + image gradient (`pick_patches.py`): `near_tex` high-texture+near (row15,col10 → idx 555), `flat` low-texture (row18,col31 → idx 684), `far` background (row7,col17 → idx 274). Frame-0 local index == global index.
- **Stored per step** (`out/track_<tag>_K<K>/step_kk.npz`): `frame_rows (T,Hh,P)`, `global_rows (T,Hh,S,P)` (per-view sliced; patch part reshapes to 24×36), `z_frame`/`z_out (T,C)` (state after each sub-block; `z_in` is the previous `z_out`, not re-stored), `anorm_frame`/`anorm_global (T,Hh)` (per-head attention-output norm), `gates_frame`/`gates_global (3,C)`.
- **Static** (`meta.npz`): `z0 (T,C)`, `gt_target (3,S,2)` + `gt_valid (3,S)` (each frame-0 patch centre reprojected into every view via GT depth+pose — pure geometry, ignores occlusion), tracked-token table, K/S/P/C/Hh/grid/t-schedule/paths.
- **Questions it targets**: (1) frame-vs-global division of labor (via `z_frame` vs `z_out` movement + the two row sets); (2) cross-view matching timeline — does a patch's global attention converge onto its `gt_target` across steps? (3) token-type differences (cam/register/near-tex/flat/far: who converges, who gets ignored).
- **Viz** (`shovel/viz_attention.py`): per (tracked token, step, frame, head) a 24×36 attention heatmap overlaid with alpha on the frame image — `hot` colormap (black=weak → white=strong), dimmed-background glow (matches the DVLT paper Fig 6 style), white cross = GT correspondence, white box = query cell. `--norm head` (faithful across frames: a frame truly attended less stays dim). `--attn global` (all frames, cross-view) or `--attn frame` (the query's OWN frame only, intra-view, from `frame_rows`). `shovel/attn_video.py` stitches per-step overlays into videos (single-head or a 12-head+mean grid), with `--attn frame` for frame-attn.

### Attention FINDINGS (K8/16/32, 3 patches × 12 heads)  【2026-10-28】
Method note: our attention recompute (reuse the module's own qkv/q_norm/k_norm/RoPE, then `softmax(qkᵀ/√d)`) is EXACTLY the DVLT paper's probe (Supp. C: head-averaged, post q-/k-LayerNorm, sliced per target view). We extend it to **per-head** and to **frame_attn** (the paper only shows global head-averaged).

- **Correspondence lives in a FEW specialized heads, not the head-mean.** For `near_tex`→view2, per-head attention-centroid distance to the GT correspondence: ~4/12 heads are sharp matchers (h0,h4,h8,**h9≈0.4 patch ≈ exact**), the rest attend elsewhere (10–17 patches off). The **head-mean is misleading** (averages matchers with non-matchers → looks like it "drifts"); the matcher heads stay sharp and largely persist across steps. Matches Stary et al. (~5/12 heads do correspondence). So the SOFT correspondence is actually accurate — in the right heads.
- **global_attn is NOT masked** (verified: no `block_mask`/`attn_mask` anywhere; `global_attn(x, pos=None)` → full SDPA). So the diagonal (own-frame) block IS included: a frame-0 query attends to its own frame WITHIN global too, and that self-frame slice holds **~38–40% of the global attention mass**. Hence the own frame has TWO attentions: `frame_rows` (frame_attn, with RoPE) and `global_rows[:,0]` (global_attn diagonal, no RoPE).
- **Division of labor (quantified, final step):** `frame_attn` is a **near-uniform intra-frame smoother** (normalized entropy ≈ **0.90**, self-attention ≈ 0.006, peak ≈ 0.02 — it barely selects, spreads almost uniformly over its own frame). `global_attn` is the **selector** (entropy ≈ **0.66**, self-attention starts high ≈ 0.065, peak ≈ 0.047). So "intra-view geometry" (Stary's self-attn role) is done by frame_attn via broad averaging, NOT sharp selection; the selective matching + self-focus live in global_attn.
- **Temporal dynamic (global):** starts strongly self-focused (step0 self-attn 0.065, peak 0.067) → redistributes to cross-view matches over the loop (self-attn → 0.02, entropy flat ≈ 0.66). "Look at yourself first, then find your matches." frame_attn instead diffuses monotonically toward uniform (entropy 0.85 → 0.92).
- **Trend with K (toward collapse):** both entropies climb with more steps (frame 0.899→0.912→0.915; global 0.650→0.674→0.682) and peaks drop — attention spreads more with larger K. global self-attention collapses at K32 (0.034→**0.021**). Same direction as the state-norm blow-up: approaching the "everything-attends-everything" / uniform attractor. (Figures: `out/attn_metrics_K8-16-32.png`, `out/attn_frame_vs_global_K.png`.)

### Collapse past Kmax — official DVLT-paper explanation (Supp. D, Fig 7)  【matches our K-sweep】
The paper confirms the divergence we measured: Pose AUC peaks near the trained budget and Pointmap Rel.L2 stays stable "up to approximately K_inf=30 but eventually collapses". Mechanism (their words): "**some feature channels grow unbounded** as we scale beyond K_max … cosine similarity to z_Kmax saturates near 1 and the relative feature update decays, [but] the state norm ‖z_k‖₂ keeps growing monotonically through K_max … iterating beyond K_max extrapolates this persistent drift: a handful of channels grow without bounds, producing the collapse." Our Stage-1 K-sweep agrees exactly: gt_absrel 0.031(K16)→0.038(K32)→**0.353(K64)**→0.585(K128), tok_norm 25→31→52→770→**2.0e6(K256)**, k_eff 2.6→2.5→**1.6(K64)**. So two complementary views of the same drift: **channel-level norm blow-up** (paper) + **directional clustering / k_eff drop** (our Geshkovski metrics) + **attention diffusing to uniform** (above).

### Collapse-intervention experiments — can we fix K64 collapse by touching norm or re-running the clock?  【2026-10-30, `dvlt_probe.py renormsweep`/`refeed`, out/experiment1/】
Two targeted Stage-1 interventions on the K64 collapsed run. **Both fail — and the failures pin down the mechanism and motivate the residual.**

- **renormsweep — clamp per-token norm, keep the directional update (K64, renorm_from ∈ {0,16,24,32,48}).** Freeze each token's norm at `renorm_from` and, every step after, rescale the new state back to that frozen norm (direction update kept, magnitude overridden). Result (final AbsRel / k_eff): **no-renorm 0.358/1.51; from0 0.441/1.30; from16 0.435/1.33; from24 0.372/1.45; from32 0.368/1.53; from48 0.372/1.53.** Holding ‖z‖ flat does **NOT** recover the answer — k_eff stays ~1.3–1.5 regardless, and clamping *earlier* makes it slightly worse. → **The collapse is DIRECTIONAL (patches re-clustering, k_eff↓), not a norm artifact. Pressing on the magnitude is the wrong lever** (confirms the Geshkovski-vs-channel split in the section above: K64 fails on direction, norm is only a symptom there).
- **refeed — is the collapse reversible? (K64 base, then re-feed the K16 schedule ×3; boundaries at steps 64/80/96/112.)** Pretend the collapsed state is at the START of a fresh K16 run and re-apply the early, large-s_attn gates. Result (|z| / k_eff / AbsRel): **step64 52/1.51/0.358 → step80 (1st re-feed) 62/2.67/0.354 → step96 (2nd) 120/2.38/0.555 → step112 (3rd) 287/1.95/0.681.** The **first** re-feed DOES re-diversify direction (k_eff 1.51→**2.67** — the clustering is not a permanent attractor, the schedule can pull directions apart again) — **but the decoded answer does not improve** (AbsRel stuck ~0.35), and **repeated** re-feeding **diverges** (‖z‖ 52→287, AbsRel→0.68). → **Collapse is not a simple reversible excursion: re-running the clock can nudge the geometry of the state but cannot restore the reconstruction, and over-iterating destabilizes.**
- **Joint takeaway (strengthens the north star).** Neither *mechanical* intervention — norm clamp or schedule re-feed — repairs the collapse. You cannot fix it from inside the dynamics; it needs a **real, data-grounded corrective signal** (the external residual). This is a clean set of NEGATIVE results that both (a) predicted correctly (we expected direction, not norm, to be the disease) and (b) directly justify the residual program over "just control the norm / just loop more".

### Token-level mechanistic FINDINGS (single-token trace + PCA + clustering + output/state)  【2026-10-29】
Tools (all in `shovel/`, from the K8/16/32 `track_token` dumps): `pca_trace.py` (trajectory PCA), `near_tex_token_trace.txt` / `patch_depth_trace.txt` (raw per-step 768-dim state + gates + per-step decoded depth vs GT), `token_clustering.png`, `output_vs_state.png`. 8 tracked frame-0 tokens: cam, register×4, near_tex/flat/far.

- **Collapse is multi-faceted but one event.** Energy concentrates into a FEW CHANNELS past budget: `eff_chan` 2.7→**11** within budget (energy SPREADS), then back to **1.6** past budget (K128 top8_share→0.999, tok_norm→2e6). But **"few channels" ≠ "few heads"**: per-head output-norm participation is stable across K8/16/32 with a few heads dominant (h0/h2/h8). **[CORRECTED 2026-11-03 — that "decoupled from heads" claim was an artifact of only having K8–32; see the head bullet below: with K64/128 the heads DO change.]** The collapse = {channel concentration + norm blow-up + directional clustering (k_eff↓) + attention→uniform}, all facets of the same drift. **K64 fails on direction** (k_eff 1.6, norm only 52); K128+ is the norm/channel explosion.
- **Collapse = GLOBAL-attention HEAD DE-SPECIALIZATION (new, K64/128; corrects the "heads decoupled" claim above).** `shovel/norm_gate_head_K.py` over the 5 K track dumps (K8/16/32/64/128), per-head attention-output norm `anorm`. Within budget (K8–32) GLOBAL attention is **head-specialized**: a single head (**head 0**) carries ~0.26 of the output share and the effective #heads (participation ratio of the 12 `anorm`) sits at ~**7.7**, STABLE across K8/16/32 (why K8–32-only looked "decoupled"). Past budget it **de-specializes**: at K64/128 head 0's share drops to ~0.11–0.14, the output spreads across all heads, and eff#heads rises to ~**11** (all 12 heads ≈ equally active). FRAME attention stays multi-head/diffuse (eff#heads ~10–11.6) for ALL K — the "smoother" role is K-invariant; the head collapse is GLOBAL-only. So the per-head structure is NOT decoupled from the collapse — it is intact within budget and dissolves past it. Unifying view: the collapse is one "loss of learned structure" showing up on three axes — **tokens cluster (k_eff 2.6→1.5), heads de-specialize (eff#heads 7.7→11.2), attention diffuses spatially (global entropy 0.66→0.85, self-frame mass 0.43→0.04≈1/27)** — token directions all align, heads all do the same thing, attention spreads everywhere. (norm.png just re-confirms the known |x| blow-up 21→~670 at true K128 endpoint; gates.png re-confirms frame s_out opening — the only gate addition is that s_out is near-SCALAR, channel participation ~0.95, so it opens all 768 channels together, carrying no head/direction structure itself.)
- **Norm growth = net RADIAL sign-flip, driven by s_out opening.** The per-step update is ~98% tangential (`attn_tang_frac`≈0.98), yet the norm grows. Decomposing Δ=z_k−z_{k−1} into radial (changes norm) + tangential (changes direction): early = big radial SHRINK (s_out≈0.73 crushes norm 27.7→16), mid = near-pure tangential (direction only), late = radial turns POSITIVE and grows → Δ‖z‖≈radial (Pythagoras from the tangential part is tiny, ~0.02). The driver is **frame s_out opening 0.73→0.93 over steps** (brake released); global s_out stays ~0.865, global s_attn tightens. So "progressively released gates" = frame s_out specifically; others stable.
- **Direction: big early, converges late.** per-step rotation 48° (step0) → 2.5° (step15); cumulative 63.5° from z0; `upd_cos_prev` 0.2→0.99 (updates become consistent). So "the flow changes direction" is true but concentrated early; late is convergence, not zero change.
- **Single-token trajectory is ~2D.** PCA of a token's 17-state path: eff dim **2.02** (PC1 64%+PC2 29%=93%), a J-curve (step0 plunge → turn → smooth late arc). Low trajectory dimension ↔ consistent late updates. (PCA coord = distance from the state centroid, NOT norm — a near-zero special token start plots far out because the centroid sits at norm 17.4.)
- **Two opposite token-type behaviors.** **Special tokens (cam/register) grow from ≈0 to huge (|z| 1.5→40)** — the loop's accumulator/scratchpad; **patches start informative (|z|≈27.7) and barely move (→24)**. The **4 register tokens converge to cos=1.00 (fully REDUNDANT — learned into one vector)**; cam joins them → 5 special tokens form ONE cluster. **Patches DIFFERENTIATE** (mutual cos 0.88→0.72). → the k_eff collapse = **patches reversing from differentiate back to cluster**; special tokens cluster normally (harmless).
- **Error concentrates at depth EDGES (full-frame, `out/track_s10_K16/perpixel_convergence.png`).** Per-pixel analysis of frame 0, K16 (decoded depth per step vs GT, global median-scale aligned): the final error map traces object outlines — **edge pixels (top-10% GT depth-gradient) end at absrel 8.3% vs 2.2% for flat pixels**, corr(edge, final error)=**+0.39**. This is the solid, visually clear finding (and matches the known difficulty of depth discontinuities: sub-pixel boundaries, the model predicts a smoothed depth across a sharp GT jump). It is a DEPTH-boundary effect, NOT a texture effect: classical "texture→easy depth" is a CORRESPONDENCE intuition, but DVLT is a learned predictor where smooth planes are easy and edges hard.
- **Output-vs-state convergence — corrected (an earlier 3-patch impression was an overgeneralization).** Full-frame, convergence TIMING is only weakly tied to edges (corr(edge, convergence-step)=+0.14): flat pixels converge at ~step 7.7, edge pixels at ~step 9.8 — both converge around step 8–10, and the convergence-step map looks more like "background/floor early, objects/machinery late" than "flat early, edge late". So: the earlier "flat locks at step 1" was a single lucky patch (not general — flat pixels average ~step 8), and "edges never converge" is wrong — edges DO stabilize (~step 10) but to a WORSE value (8.3%). The variable-per-region convergence still mildly supports adaptive/residual-driven budgeting, but the clean, defensible conclusion is just **edges = higher final error**. (Also corrected: an even earlier "near_tex wrong by 15%" was a BUG — invalid GT pixels averaged into the patch mean; real ~8%, an edge patch.) Caveat: still ONE scene — multi-scene needed for full confidence.

### Useful for "deep code dive" (Stage-2 #1)
- `tests/test_*.py` mirror `src/dvlt/` and give concrete input/output examples for geometry/utility functions (e.g. `rays_to_pose`, projection, rotation) — often the fastest way to understand what a function computes.

---

# NORTH STAR — residual-driven iteration (replace K)  【long-term goal】

## ★★ TOP-LEVEL MOTIVATION — the collapse is an INCOMPLETE relational mechanism; the residual COMPLETES it  【framing 2026-11-05】
The deepest why behind everything. **Attention is a relation operator that only AGGREGATES (attract/smooth): its output is a convex combination `O_i=Σ_j A_{ij}V_j` (A≥0, Σ=1) → contraction toward the mean → tokens attract → cluster.** The architecture has NO complementary operator that DIFFERENTIATES along relations (pushes tokens that should differ apart). MLP is per-token (ignores relations), residual is identity — neither is a *relational repulsion*. So the relational mechanism is ONE-SIDED (attraction only), and a one-sided attractive dynamic **must** collapse to a single cluster (k_eff→1; this is the Geshkovski asymptotic attractor). It is exactly **contrastive learning with no negatives / GNN–transformer oversmoothing**: without a repulsion term, representations collapse. Stable structure needs BOTH attraction and repulsion (gravity-only collapses; activator needs inhibitor; bonding needs Pauli exclusion). Training only PATCHES this incompleteness by calibrating the model to STOP early (the K-budget + bounded gate schedule) inside the metastable window — "works within budget" ≠ "mechanism is complete"; past budget it marches to the inevitable collapse, accelerated by the positive feedback *clustering ⟹ uniform/degenerate attention ⟹ more clustering*.
- **⇒ The residual IS the missing function.** Attention's differentiation is INTERNAL and self-referential (token-token similarity) so it can only reach consensus. A geometric residual is an EXTERNAL, data-grounded statement of "these points actually ARE distinct" — it supplies the relational **differentiation / repulsion** the architecture lacks, grounded in real data. `internal attraction (attention) + external data-grounded differentiation (residual) = a COMPLETE relational mechanism`. This reframes the north star: residual injection is **not a patch, it completes the mechanism**. It also explains the whole lineage — plain weight-tying diverges (attraction only, no differentiation source); ICL looped transformers converge (the prompt's labeled examples ARE an in-input differentiation source); DVLT at inference has no such source (GT is train-only) → must INJECT a residual to supply it.
- **Design corollary:** the most root-level cure is to inject the differentiation at the ATTENTION level (a repulsion term / δQ,δK on the logits — "add the missing negatives"), not just the state. State injection rescues the OUTPUT but the next attention step re-collapses the latent (confirmed in the rescue experiment: |x|/k_eff collapse regardless of state injection), precisely because the missing repulsion was never supplied. See the gate-vs-attention split and [[stage2-injection-plan]].

The overarching aim behind Stage 2: turn DVLT from a **fixed-K looped transformer** into a **residual-driven learned optimizer** — inject external GEOMETRIC residuals (photometric, reprojection, other data terms) into the iteration so the loop is driven by "how wrong the reconstruction still is" rather than by a fixed step budget K. This fuses the learned flow (fast, strong prior/initialization) with explicit, test-time-correctable geometric objectives (bundle-adjustment-like), making the optimization principled and data-grounded.

## Why this is coherent with the Stage-1 findings
- The loop is (plausibly) a gradient flow (effective symmetry ~93%) descending a geometric/differentiation energy → replacing the implicit energy's *budget* (K) with an explicit external residual makes the objective explicit, interpretable, and correctable.
- K is a calibrated horizon, t is a soft control knob; the flow converges in direction but has no fixed point (diverges past K_max) → "replacing K" means replacing the fixed horizon with a residual-driven drive/convergence criterion.
- Injecting into the STATE lets global attention propagate the correction across views next step → multi-view consistency for free.
- The `photo` experiment is a prototype of exactly this (gradient of photometric residual → state).

## ★ MECHANISM VALIDATED — residual-gradient → state WORKS (oracle diagnostic)  【milestone 2026-11-04, `dvlt_probe.py photo --residual`, out/experiment1/photodiag/】
Test-time latent optimization: run the frozen loop to step 8, then inject `x ← x − η·‖x‖·∇_x L / ‖∇_x L‖` (gradient of a residual L backprop'd through the decoder to the STATE, no weight training), then finish the loop. Compared **photo** (self-supervised photometric) vs **gtdepth** (ORACLE = scale-invariant GT-depth residual, an upper bound), swept η∈{0.002,0.005,0.01,0.02}. Baseline (no inject) AbsRel=0.0336.
- **Oracle IMPROVES**: gtdepth at small η drives L down monotonically AND improves AbsRel: **0.0336 → 0.0273** (η=0.01; also 0.0278/0.0284 at η=0.002/0.005) — ~**19% relative gain**. So "**residual gradient → state**" genuinely corrects geometry when the objective is good and the step is small.
- **Photo FAILS**: self-supervised photometric is at best neutral (0.0333 at η=0.002) and worsens with η (→0.059 at η=0.02); its L barely descends even at small η (degenerate landscape).
- **The earlier "photo FAILED" was TWO things, now separated**: (a) **η too big** (0.02 overshoots → L oscillates 0.065→0.217→…, sawtooth); (b) **photometric objective is degenerate** (many 3D configs give the same reprojected photometry → its gradient doesn't align with true geometry).
- **⇒ The bottleneck is OBJECTIVE QUALITY, not the injection mechanism.** The mechanism (gradient→state, small η) is proven. What's needed is a non-degenerate residual, or a learned module that produces an oracle-like correction from the available (non-GT) signal.
- **Design convergence → frozen-DVLT + small two-head adapter (stage-3, no full retrain).** A small trainable network reads the residual and outputs TWO things: (1) a STATE guidance Δz (direction — targets the oracle gradient, which is proven to work) injected into z; (2) a GATE intervention δs (magnitude/pacing/stop — residual size → step size / adaptive-K). DVLT frozen; train only the adapter with GT supervision (like stage-2). state-head = direction, gate-head =量/stop — the two levers the gate-vs-attention analysis identified. Zero-init the adapter so it starts as no-op. Key engineering point: the state head must output a COMPACT correction (per-patch, not a full 23463×768 tensor). See [[stage2-injection-plan]].

## ★ COLLAPSE-RESCUE — injection rescues the OUTPUT on a collapsing K, but NOT the latent  【milestone 2026-11-05, `dvlt_probe.py rescue`, out/experiment1/rescue_*/; previews are 12-frame on the login A2000, full 27-frame queued】
Test-time interleaved injection on a COLLAPSING K (K64): after every loop step from `--i-start`, take `--i-per` oracle-residual gradient steps on the state (eta=0.002, from the photodiag sweep), then continue. Also a `burst` (one-shot) and a `baseline` (no inject). Per step logs |x|/k_eff/AbsRel/ATE, decoding both post-step AND post-inject (sawtooth visible).
- **Continuous injection RESCUES the depth OUTPUT through the collapse.** K64 baseline depth AbsRel collapses to ~0.35; interleave keeps it ~0.08 (≈**4× better**, near the healthy-K16 level). So residual injection keeps the DECODED output good even past the budget.
- **burst (one-shot) is temporary — it RE-collapses.** A 15-iter burst at step 30 drops AbsRel 0.39→0.19, but the loop re-climbs to ~0.32 by step 64. Confirms NO FIXED POINT: a one-shot correction is undone by continued iteration → you need CONTINUOUS injection.
- **But the LATENT STATE collapses regardless** — |x| (→53) and k_eff (→1.54) are IDENTICAL across baseline/interleave/burst/all-λ (lines perfectly overlap). **Injection is a SYMPTOMATIC output-rescue** (continuous data-pull keeps the decoded channels good) — it does NOT cure the state-level collapse. Curing the latent needs a different lever (norm control / a real attractor), i.e. the adapter's GATE head, not the STATE head.
- **Depth-only injection does NOT fix the camera** (ATE 0.83→0.88, slightly WORSE). Added an oracle camera residual `gtcam_residual` (predicted ray-origin camera centres vs GT poses, scale+translation gauge detached). Combined `L = L_depth + λ·L_cam`.
- **★ WHICH TOKENS the residual gradient reaches — MEASURED 2026-11-06 (corrects an earlier wrong claim).** Computed `g = ∂L/∂x` and read ‖g‖ by token type (cam / 4 register / 864 patch): **gtdepth → mainly the PATCH tokens** (total ‖g_patch‖ 1.1e-2 vs cam 2.7e-3, reg 7.4e-3; per-token the special tokens are actually comparable/larger, but the 864 patches dominate the mass). **gtcam → overwhelmingly the CAM + REGISTER tokens** (‖g_cam‖ **3.0**, ‖g_reg‖ **4.2**, ‖g_patch‖ only 0.12). Reason: depth is the DPT dense head on patches (`norm(x)[:, patch_start_idx:]` drops cam/reg), whereas the ray ORIGIN = camera centre is read from the camera-token pose, not the dense patches. **(This REPLACES the earlier note that said gtcam "flows to the PATCH tokens via the ray decoder / ray head drops the cam token" — that was wrong; gtcam lands on cam+register.)** This is a clean token-level confirmation of the division of labor: **patches = dense geometry (depth residual lands here); cam+register = camera/pose accumulator (camera residual lands here)** — and it is the TOKEN-LEVEL root of the depth↔camera opposition: the two residuals push DIFFERENT token groups, but both decode into the shared world point `X = R·o + D·R·d`, so correcting one perturbs the other's output → must jointly balance with λ. (register gets the largest camera gradient, consistent with registers = shared global/camera scratchpad.)
- **depth↔camera are OPPOSED (depth-pose ambiguity).** λ sweep (K64 interleave endpoint): λ0 → depth 0.084/ATE 0.882; **λ0.03 → 0.091/0.319**; λ0.1 → 0.110/0.299; λ0.3 → 0.143/0.397; λ1 → 0.271/0.276. Pure depth hurts camera and vice versa — because both decode from the SAME z and world-points `X=R^o+D·R^d` lets depth trade against camera translation (classic scale-translation ambiguity). So you MUST jointly correct (a mini bundle-adjustment), weighted by λ. **Sweet spot λ≈0.03–0.1 rescues BOTH** (depth ~4× + camera ~2.6× vs baseline); λ too big (1.0) wrecks depth. Tradeoff plot: `out/experiment1/rescue_lambda_tradeoff.png`.
- **Injection start step barely matters** (i_start 1 vs 4 ≈ identical) — early steps are already healthy (nothing to fix); what matters is injecting DURING the collapse zone.
- **Implications for the adapter.** (1) The residual MUST be joint (depth + camera/pointmap), with a learned/tuned balance — a single-term residual gets cancelled by the opposing term. (2) The STATE head rescues the output; curing the latent collapse needs the GATE head (norm/stop) — confirms the two-head split. (3) Continuous (every-step) injection, not one-shot. (Caveat: oracle residual + 12-frame preview; 27-frame full run queued as `1989026`.)

## ★ REPULSION TEST — correspondence-gated attention does NOT cure the collapse (clean negative result)  【2026-11-05, `dvlt_probe.py repel`, out/experiment1/repel/】
Direct test of the "attention is attract-only, add the missing repulsion" hypothesis in its simplest form: bias the GLOBAL-attention logits with `γ·S`, where `S = cos(z_0^i, z_0^j)` is the affinity of the (un-collapsed) frozen DINOv2 encoder features — the TRUE relatedness structure, before the loop homogenizes it. Monkeypatch `global_attn.attn.forward` to `SDPA(q,k,v, attn_mask=γ·S)`; run baseline + a γ-sweep on K64 (8-frame login-A2000 preview). `S` off-diag **mean=0.867, std=0.112** (patches are ~87% similar in z_0 — a huge DC component).
- **It does not help — if anything it ACCELERATES the collapse.** γ=20/50 vs baseline (step 8): k_eff **2.13/2.11 vs 2.55** (more clustered), AbsRel **0.12/0.14 vs 0.076** (worse); endpoints track baseline or slightly worse. γ=5 was a no-op (effect swamped by the O(8) base logits). **Hypothesis FALSIFIED in this form.**
- **Why (the real lesson): reweighting a softmax CANNOT add repulsion — it is structurally attract-only.** Softmax weights are a probability simplex (nonneg, sum to 1), so `O_i = Σ_j A_ij V_j ∈ conv(V)` for ANY weights A. The output is always inside the convex hull of the values ⇒ `‖O_i‖ ≤ max_j‖V_j‖` and the map contracts toward the (weighted) mean. Changing WHICH tokens are averaged (what the `γ·S` bias does) cannot escape the hull — it only picks a different interior point. Biasing toward z_0 affinity even sharpens attention onto the most-similar tokens → faster clustering, which is what we measured.
- **⇒ True repulsion needs to BREAK the simplex (negative net coefficients), i.e. a term OUTSIDE the convex combination.** That is exactly the residual / differentiation function: `X ← X − β·(something)` subtracts, pushing a token AWAY from others — a direction the simplex structurally forbids. This CONFIRMS the top-level framing from the other side: you cannot fix an attract-only mechanism by re-routing the attraction; you must ADD an external subtractive (repulsive / data-grounded) term. It also explains why state/residual injection (which subtracts a gradient) is the right lever and attention-weight reshaping is not.
- **SUBTRACTIVE form ALSO fails (2026-11-05, `repel --repel-mode sub`).** Implemented the genuine "add negatives": manual attention `O_i = (A@V)_i − β·(R@V)_i`, `R = softmax(−γ·S)` (attend-AWAY from z_0-dissimilar tokens), net coeffs `A−β·R` can go negative → leaves conv(V). Sweep β=0.1/0.3, γ=30 on K64. Result: still no rescue — k_eff endpoint 1.53/1.48 vs baseline 1.56 (β=0.3 slightly WORSE), AbsRel 0.39/0.40 vs 0.37, |x| climbs higher (54 vs 52, as expected once the conv-hull norm bound is broken). So negative coefficients are necessary but NOT sufficient in this form.
- **Why this parameterization is weak (diagnosis).** `R = softmax(−γ·S)` concentrates each row on the FEW globally-most-dissimilar tokens (likely shared outliers — cam/register/degenerate patches), so `R@V` is nearly the SAME vector for every query i → subtracting it is a near-COMMON SHIFT of all tokens, which does not change cosine-based k_eff (only grows |x|). True de-clustering needs a PER-TOKEN-DISTINCT repulsion that targets the bulk being over-averaged, not one shared outlier.

### ★★ Repulsion CAN cure the latent collapse — but k_eff and output quality are DECOUPLED  【2026-11-05, `repel --repel-mode {center,corr}`】
Two per-token-distinct repulsions: **center** (high-pass / anti-oversmoothing, `O_i = (A@V)_i − β·mean_j V_j` — subtract the per-head token mean everything collapses toward); **corr** (z_0-gated, `O_i = (A@V)_i − β·(W@V)_i`, `W_ij ∝ A_ij·relu(S̄_i − S_ij)` row-normalized — undo only the attention mass flowing to tokens z_0 says are UNRELATED to i).
- **Both HOLD k_eff UP, monotonically in β — the FIRST forms that cure the latent directional collapse** (bias/sub could not). center: k_eff@64 1.56(base)→1.64→1.77→**2.05** (β=0.3/0.6/1). corr: 1.56→1.60→1.84→**4.80** (β=1 over-repels, disperses past even the healthy K16 level). So negative-coefficient repulsion DOES prevent the Geshkovski clustering — the "attention is attract-only, add repulsion" claim is mechanistically TRUE for the LATENT.
- **But NONE improves the depth OUTPUT — AbsRel stays flat or gets WORSE.** center@64 AbsRel 0.37→0.39/0.42/0.50; corr 0.37→0.41/0.46/0.49. Holding k_eff up (even restoring it to a healthy ~2–3, or overshooting to 4.8) does NOT recover geometry.
- **⇒ KEY FINDING: k_eff (latent clustering) and AbsRel (output quality) are DECOUPLED / orthogonal levers.** This is the MIRROR of the rescue experiment: there, oracle STATE injection rescued the OUTPUT while k_eff collapsed regardless; here, repulsion rescues k_eff (the LATENT) while the output stays bad. Together they prove the two are independent — the post-budget output degradation is NOT simply "caused by" the directional clustering, and de-clustering is neither necessary nor sufficient for good geometry. **Practical lesson: do NOT chase k_eff as a proxy for quality** — the output fix is the DATA-GROUNDED residual (state head), while repulsion is only a structural conditioner of the latent (geometry-blind: center is blind; even z_0-gated corr, which is data-structure-grounded but not TASK/GT-grounded, doesn't help the output). Refines the top-level framing: "incomplete relational mechanism" is real for the latent, but completing it (adding repulsion) does not by itself restore the task output — that still needs the external GT/objective-grounded correction.

### Which "who-corresponds" signal is best: A₁ (first-step attention) vs cos(z₀) vs GT 3D  【2026-11-05, scratchpad `corr_check.py`, 8-frame s10】
Motivation (user): the repulsion needs a reliable "which patches TRULY correspond" signal (only push apart genuinely-distinct points). Is the FIRST-STEP global attention A₁ a better correspondence signal than cos(z₀)? GROUND TRUTH = real 3D correspondence: back-project each patch-centre with GT depth+pose+K to a world point; patches i,j (different frames) "correspond" iff ‖W_i−W_j‖ < ε. Metric = per-query ROC-AUC (how well each signal ranks the true cross-frame correspondents) + precision@1; swept ε = 0.02/0.05/0.10 × scene-radius.
- **Both signals carry strong correspondence (far above the 0.5 random AUC).** cos(z₀): AUC **0.84–0.85**, prec@1 0.20–0.34 (random ~0). DINOv2 appearance cosine is already a strong matcher, as expected.
- **A₁ head-MEAN is WORSE than cos(z₀)** (AUC 0.81 vs 0.85): averaging 12 heads dilutes the signal — the heads are wildly heterogeneous (per-head AUC from **0.45** (slightly ANTI-correspondence) to **0.91**), many doing non-correspondence work (context/position).
- **But the BEST single head beats cos(z₀): head 0 / head 9 reach AUC 0.89–0.91 > 0.85.** So the trained Q/K DID learn a correspondence matcher SHARPER than raw appearance cosine — it just lives in specific specialized heads (consistent with the earlier "global attn is head-specialized, head-0 dominant" finding). ⇒ the user's intuition is half-right: A₁ *contains* a better correspondence signal, but you must SELECT the correspondence head(s) (0/9), not use raw/averaged A₁.
- **Caveats / implications.** (1) Neither inference-available signal is perfect (AUC ≤0.91); the gold "who" signal is GT reprojection (AUC=1) but it needs geometry solved. (2) This only improves lever (a) "WHO to repel"; it does NOT touch lever (b) "where to push to" (correct target/magnitude), which still needs the GT/objective-grounded residual — so a better correspondence gate (A₁-head instead of z₀) sharpens `corr`'s selectivity but is not expected to fix the output on its own (see the decoupling finding above). **Confirmed below.**

### corrhead — a better who-signal makes repulsion OUTPUT-NEUTRAL (not harmful), but still does NOT cure collapse or improve output  【2026-11-05, `repel --repel-mode corrhead --repel-head 0 9`】
Replaced `corr`'s gate `ref = cos(z₀)` (AUC 0.85) with `ref = first-step attention CORRESPONDENCE-HEAD mean` (heads 0+9, AUC 0.91), captured once (space-cheap: keep only the selected heads). Same per-token gated repulsion `O_i=(A@V)_i − β·(W@V)_i`, `W∝A·relu(rowmean(ref)−ref)`. K64, 8-frame.
- **It does NOT cure the collapse and does NOT improve the output.** corrhead β=0.6 per-step AbsRel: 0.085→0.21→0.40→0.39→0.35→0.38→0.39→**0.368**, vs baseline 0.076→0.20→0.41→0.41→0.39→0.42→0.40→**0.375**. Same collapse shape (0.08→0.37), and the endpoint 0.368 vs 0.375 is NOISE-level identical — NOT a real improvement. (β=1's 0.294 endpoint is a damped OSCILLATION — mid-run it spikes to 0.66 — not a stable gain; earlier read of it as a "21% win" was over-interpretation of noise, retracted.)
- **Its ONLY merit vs corr(z₀): it stops HURTING.** At matched k_eff (β=0.6, both k_eff≈1.83): corrhead AbsRel 0.37 vs z₀'s 0.46. The cleaner who-signal avoids pushing apart TRULY-corresponding patches, so repulsion goes from HARMFUL (z₀) to NEUTRAL (≈baseline). It is "from harmful to harmless", NOT "from harmless to helpful".
- **⇒ CONFIRMS (does NOT walk back) the decoupling.** Attention repulsion — even with the best inference-available who-signal — holds k_eff higher but leaves the OUTPUT at the collapsed ~0.37. The output degradation is not fixable by any attention-weight edit; the fix needs the DATA/GT-grounded residual (lever b, the correct TARGET), which corrhead does not supply. Correspondence-gated repulsion's realistic role is at most an output-SAFE latent conditioner to run ALONGSIDE the residual, never a substitute for it.

## Why input injection alone is NOT enough — it must be a RESIDUAL/objective (key 2026-10-29 finding)
- **Looped transformers CAN extrapolate without collapse — if the input is re-injected every step.** Yang et al. (ICL looped transformer, `looptransformer.pdf`) show plain weight-tying `Y_{t+1}=M(Y_t)` (input only at `Y_0`) diverges past the trained budget (the input's influence vanishes → "essentially random", citing Bai/DEQ + Bansal/overthinking), while **input injection `Y_{t+1}=M(Y_t+P)`** reaches a stable fixed point and extrapolates. **DVLT is exactly the diverging weight-tying case** (DINOv2 encodes once into `z_0`, then `_interval_step` iterates with no re-injection).
- **But DVLT already TRIED this and it only PLATEAUS.** DVLT paper (Supp. D limitations): "preliminary attempts with looped-transformer stabilization [Yang et al.] plateau past the trained budget rather than continuing to improve, at significant additional training cost."
- **Why the difference: ICL's input CONTAINS a solvable objective; DVLT's doesn't.** ICL's prompt holds labeled in-context examples `(x_i,y_i)` — the answers are IN THE INPUT, so the loop minimizes a real objective (fit those examples) that is computable at inference and keeps improving toward the optimum. 3D reconstruction's input (images) has NO answer in it — the GT point cloud is a TRAINING loss only, unavailable at inference. So re-injecting images just stabilizes (plateau); there is nothing to optimize toward.
- **⇒ the missing ingredient is an OBJECTIVE/RESIDUAL, not the input.** The photometric/reprojection residual is exactly the "answer the model can check against at inference from the input images alone" (does my reconstruction reproject consistently to the actual images?) — DVLT's self-supervised analogue of ICL's in-context labels. This is the precise reason the north star is a RESIDUAL-driven loop, not merely input injection.

## Known hard challenges (surfaced by Stage 1)
1. **Divergence**: the model is only stable within its ~16-step calibrated horizon; run longer and it diverges (norm explodes, effective symmetry breaks). So "iterate until the residual converges" needs either norm control (renorm) or treating the residual as a corrective force that keeps the state on the normal manifold.
2. **Non-autonomy / time**: the flow needs the time coordinate t (via the gates) to converge/brake. After replacing K, how is t set? Options: keep the schedule and add the residual as an extra force; or learn a "residual → t" controller.
3. **Tangential + scheduled guidance**: the naive `photo` gradient must be (a) projected to the tangent of the sphere (radial component is absorbed by LayerNorm, wasted) and (b) applied over a window / schedule-aware, not one-shot (the contractive, self-correcting flow washes out one-shot nudges).

## One line
Replace the fixed K-step budget with an external-residual-driven iteration (photometric / reprojection / latent), turning the looped transformer into a learned + residual-grounded optimizer. Stage 1 shows this is compatible with the model's nature (gradient flow, state injection propagates, tangential+scheduled guidance needed) and identifies the main obstacle (the model diverges past its calibrated horizon, so stable long iteration needs norm control or a corrective-force formulation).
