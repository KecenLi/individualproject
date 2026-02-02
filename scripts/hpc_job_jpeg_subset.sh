#!/usr/bin/env bash
#SBATCH --job-name=jpeg_subset
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/jpeg_subset_%j.out
#SBATCH --error=logs/jpeg_subset_%j.err

set -euo pipefail

module load languages/python/3.12.3 || true
source /user/work/$USER/venvs/individualproject/bin/activate

cd /user/work/$USER/individualproject
mkdir -p logs

if [ -z "${JPEG_NORM:-}" ] || [ -z "${JPEG_EPS_LIST:-}" ]; then
  echo "[ERROR] JPEG_NORM and JPEG_EPS_LIST must be set via --export."
  exit 1
fi

python3 scripts/run_jpeg_subset.py \
  --jpeg-norm "$JPEG_NORM" \
  --jpeg-eps "$JPEG_EPS_LIST" \
  --jpeg-iters 200 \
  --model-name "Standard" \
  --model-alias "ResNet18_Std" \
  --layers "block1.layer.2,block2.layer.2,block3.layer.2" \
  --limit-samples 10000 \
  --profiling-samples 1000 \
  --output-dir total_benchmark_results
