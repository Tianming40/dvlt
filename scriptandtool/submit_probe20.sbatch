#!/bin/bash
#SBATCH --job-name=dvlt_probe20
#SBATCH --output=/home/stud/zhtia/dvlt/out/slurmout/probe20_%j.out
#SBATCH --error=/home/stud/zhtia/dvlt/out/slurmout/probe20_%j.err
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -e
# NOTE: must request a modern GPU (Ampere+). Old GPUs (e.g. TITAN X Pascal, 12 GB)
# have no flash attention, so SDPA falls back to the math backend and materialises
# the full (S*N)^2 global-attention matrix -> ~13.5 GB OOM even on the forward pass.
# a40 = Ampere, 48 GB (also fine: rtx_a6000 / l40s / nvidia_rtx_6000_ada_generation).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. /home/stud/zhtia/miniconda3/etc/profile.d/conda.sh
conda activate dvlt
echo "=== dvlt_probe20  job=$SLURM_JOB_ID  node=$SLURMD_NODENAME  gpu=$CUDA_VISIBLE_DEVICES  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /home/stud/zhtia/dvlt
bash scriptandtool/run_probe20.sh
echo "=== Done $(date) ==="
