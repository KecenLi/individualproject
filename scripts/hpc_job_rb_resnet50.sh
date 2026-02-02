#!/usr/bin/env bash
#SBATCH --job-name=rb_r50
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rb_r50_%j.out
#SBATCH --error=logs/rb_r50_%j.err

set -euo pipefail

module load languages/python/3.12.3 || true
source /user/work/$USER/venvs/individualproject/bin/activate

cd /user/work/$USER/individualproject
mkdir -p logs

# ImageNet model (RobustBench): Standard_R50
# Confirmed layers: model.layer3, model.layer4
python3 scripts/run_oodrb_nac.py \
  --data-dir ./data \
  --dataset imagenet \
  --model-name Standard_R50 \
  --threat-model Linf \
  --batch-size 64 \
  --profile-samples 2000 \
  --id-examples 5000 \
  --layer-names model.layer3,model.layer4 \
  --corruptions all \
  --severities 1,2,3,4,5 \
  --run-natural-shifts \
  --natural-shifts all \
  --output-dir oodrb_results
