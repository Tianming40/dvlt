#!/bin/bash
#SBATCH --job-name=dvlt_all
#SBATCH --output=/home/stud/zhtia/dvlt/out/slurmout/all_%j.out
#SBATCH --error=/home/stud/zhtia/dvlt/out/slurmout/all_%j.err
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -e
# a40 (Ampere, 48 GB, flash-attention) -- needed so 20-frame global attention and the photo
# gradient path fit. Do NOT use gpu:1 (may land on an old 12 GB card with no flash attention).
. /home/stud/zhtia/miniconda3/etc/profile.d/conda.sh
conda activate dvlt
echo "=== dvlt_all  job=$SLURM_JOB_ID  node=$SLURMD_NODENAME  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /home/stud/zhtia/dvlt
bash scriptandtool/run_all.sh
echo "=== Done $(date) ==="
