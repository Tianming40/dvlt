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
