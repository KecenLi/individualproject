#!/usr/bin/env bash
#SBATCH --job-name=oodrb_nac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/oodrb_%j.out
#SBATCH --error=logs/oodrb_%j.err

set -euo pipefail

module load languages/python/3.12.3 || true
source /user/work/$USER/venvs/individualproject/bin/activate

cd /user/work/$USER/individualproject
mkdir -p logs

python3 scripts/run_oodrb_nac.py \
  --data-dir ./data \
  --dataset cifar10 \
  --model-name Standard \
  --threat-model Linf \
  --batch-size 64 \
  --profile-samples 1000 \
  --id-examples 10000 \
  --corruptions all \
  --severities 1,2,3,4,5 \
  --output-dir oodrb_results
