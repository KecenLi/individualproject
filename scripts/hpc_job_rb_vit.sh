#!/usr/bin/env bash
#SBATCH --job-name=rb_vit
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rb_vit_%j.out
#SBATCH --error=logs/rb_vit_%j.err

set -euo pipefail

module load languages/python/3.12.3 || true
source /user/work/$USER/venvs/individualproject/bin/activate

cd /user/work/$USER/individualproject
mkdir -p logs

# ImageNet model (RobustBench): Mo2022When_ViT-B
# Confirmed layers: model.blocks.10, model.blocks.11
python3 scripts/run_oodrb_nac.py \
  --data-dir ./data \
  --dataset imagenet \
  --model-name Mo2022When_ViT-B \
  --threat-model Linf \
  --batch-size 64 \
  --profile-samples 2000 \
  --id-examples 5000 \
  --layer-names model.blocks.10,model.blocks.11 \
  --corruptions all \
  --severities 1,2,3,4,5 \
  --run-natural-shifts \
  --natural-shifts all \
  --output-dir oodrb_results
