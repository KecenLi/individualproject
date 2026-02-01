#!/usr/bin/env bash
#SBATCH --account=coms037985
#SBATCH --job-name=oodrb_nac_small
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/oodrb_small_%j.out
#SBATCH --error=logs/oodrb_small_%j.err

set -euo pipefail

module load languages/python/3.12.3 || true
source /user/work/$USER/venvs/individualproject/bin/activate

cd /user/work/$USER/individualproject
mkdir -p logs

# Quick smoke run (1 corruption × 1 severity, small sample)
python3 scripts/run_oodrb_nac.py \
  --data-dir ./data \
  --dataset cifar10 \
  --model-name Standard \
  --threat-model Linf \
  --batch-size 64 \
  --profile-samples 200 \
  --id-examples 200 \
  --corruptions gaussian_noise \
  --severities 1 \
  --output-dir oodrb_results
