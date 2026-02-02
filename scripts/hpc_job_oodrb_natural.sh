#!/usr/bin/env bash
#SBATCH --job-name=oodrb_nac_nat
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/oodrb_nat_%j.out
#SBATCH --error=logs/oodrb_nat_%j.err

set -euo pipefail

module load languages/python/3.12.3 || true
source /user/work/$USER/venvs/individualproject/bin/activate

cd /user/work/$USER/individualproject
mkdir -p logs

# Natural shifts only (skip corruption loop by passing a non-existent corruption name).
python3 scripts/run_oodrb_nac.py \
  --data-dir ./data \
  --dataset cifar10 \
  --model-name Standard \
  --threat-model Linf \
  --batch-size 64 \
  --profile-samples 1000 \
  --id-examples 10000 \
  --corruptions none \
  --run-natural-shifts \
  --natural-shifts all \
  --output-dir oodrb_results
