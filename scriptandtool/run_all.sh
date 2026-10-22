#!/usr/bin/env bash
# ONE clean experiment, strict per-version logging (no duplicated probe/unified runs).
# Logic: two anchors (GT + K=16 baseline). Every version = baseline OR baseline + ONE change
# (K / schedule / renorm / ablate / perturbation-level / photo) ~= 113 versions. Each version runs
# ONCE and is logged to its own TSV with an identical 52-column schema (one row per loop step);
# then one figure per metric column overlays all versions. All at 20 frames.
#   out/full_grand/versions/<version>.tsv   per-version table (52 cols)
#   out/full_grand/all.tsv                  everything stacked
#   out/full_grand/figs/<metric>.png        45 metric figures, one line per version
#   out/full_grand/meta.json                static attn-asym + column list
# attn / smoothx / smoothinput are NOT per-step single scalars, so they run SEPARATELY below.
# Run from repo root:  bash scriptandtool/run_all.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { echo "+ $*"; "$@" || echo "!!! step FAILED (continuing): $*"; }

IN=out/gt_00777c41d4/probe_input
GT=out/gt_00777c41d4/gt.npz
F=20          # ALWAYS 20 frames (standard) -- every command below
PY="python dvlt_probe.py"

echo "##### 1. GRAND -- all ~113 versions, 45 metrics each, per-version TSV + per-metric figures"
# --Ks now goes up to 256 (long-horizon over-processing), and --sched-steps 256 makes the FROZEN
# version iterate the same map 256x (the attractor / divergence study) -- both show up as lines.
run $PY grand --input $IN --gt $GT --max-frames $F \
              --Ks 4 8 12 16 20 24 32 64 128 256 \
              --schedules baseline half reverse shuffle frozen double skip repeat \
              --sched-steps 256 \
              --ablate none first_half second_half all \
              --kinds gauss scale dir swap cam register \
              --eps 0.02 0.05 0.1 0.2 0.5 1.0 \
              --at 1 4 12 \
              --start 8 --eta 0.02 --iters 5 --stride 2 \
              --nsub 1024 --nq 256 \
              --decode-chunk 4 \
              --out full_grand

echo "##### 2. attn -- global attention vs GT correspondences (separate: matrix measurement)"
# 20 frames: the full attention matrix is memory-heavy -> needs the a40 (48 GB). Raise --attn-stride if it OOMs.
run $PY attn      --input $IN --gt $GT --max-frames $F --K 16 --attn-steps 1 2 4 6 8 10 12 14 16 --out full_attn

echo "##### 3. smoothx -- local Lipschitz of the step map (separate: pairwise measurement)"
run $PY smoothx   --input $IN --max-frames $F --at 2 8 14 --eps 0.005 0.01 0.02 0.05 0.1 0.2 0.5 --out full_smoothx

echo "##### 4. smoothinput -- input-perturbation sensitivity (separate: pairwise measurement)"
run $PY smoothinput --input $IN --gt $GT --max-frames $F --eps 0.005 0.01 0.02 0.05 0.1 --out full_smoothinput

echo "##### 5. flowdump -- export flow.json for the dvlt_flow_viewer.html particle viewer"
run $PY flowdump  --input $IN --gt $GT --max-frames $F --K 16 --out full_flow

echo ""
echo "===== ALL DONE. main table + figures in out/full_grand/ ; separate probes in out/full_{attn,smoothx,smoothinput}/ ; viewer data in out/full_flow/flow.json ====="
