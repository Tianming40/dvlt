#!/usr/bin/env bash
# Run the classic probe experiments at 20 frames (undistorted + GT) in one shot,
# then assemble a single comparison dashboard. Activate the dvlt env first.
# Run from the repo root:  bash scriptandtool/run_probe20.sh
set -e

IN=out/gt_00777c41d4/probe_input
GT=out/gt_00777c41d4/gt.npz
F=20

echo "===== traj (per-step trajectory, several K) ====="
python dvlt_probe.py traj    --input $IN --gt $GT --max-frames $F --Ks 8 12 16 24 --out probe20_traj

echo "===== ksweep (final quality vs K) ====="
python dvlt_probe.py ksweep  --input $IN --gt $GT --max-frames $F --Ks 8 12 16 20 24 32 --out probe20_ksweep

echo "===== perturb (gauss + swap) ====="
python dvlt_probe.py perturb --input $IN --gt $GT --max-frames $F --K 16 --at 2 8 14 --kind gauss --out probe20_perturb_gauss
python dvlt_probe.py perturb --input $IN --gt $GT --max-frames $F --K 16 --at 2 8 14 --kind swap  --out probe20_perturb_swap

echo "===== photo (photometric residual + grad injection) ====="
python dvlt_probe.py photo   --input $IN --gt $GT --max-frames $F --K 16 --start 8 --eta 0.02 --iters 5 --out probe20_photo

echo "===== dashboard (one figure) ====="
python scriptandtool/make_dashboard.py --out-root out --fig out/dashboard20.png

echo "done -> out/dashboard20.png"
