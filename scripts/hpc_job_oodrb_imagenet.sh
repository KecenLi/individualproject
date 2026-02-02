#!/usr/bin/env bash
#SBATCH --job-name=oodrb_imagenet
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/oodrb_imagenet_%j.out
#SBATCH --error=logs/oodrb_imagenet_%j.err

set -euo pipefail

module load languages/python/3.12.3 || true
source /user/work/$USER/venvs/individualproject/bin/activate

cd /user/work/$USER/individualproject
mkdir -p logs

# Requires ImageNet data under ./data/imagenet and OODRobustBench shift data under ./data
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
