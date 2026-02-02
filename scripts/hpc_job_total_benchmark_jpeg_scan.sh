#!/usr/bin/env bash
#SBATCH --job-name=rb_jpeg_scan
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rb_jpeg_%j.out
#SBATCH --error=logs/rb_jpeg_%j.err

set -euo pipefail

module load languages/python/3.12.3 || true
source /user/work/$USER/venvs/individualproject/bin/activate

cd /user/work/$USER/individualproject
mkdir -p logs

# This script runs the full run_total_benchmark.py (includes JPEG eps scan)
python3 run_total_benchmark.py
