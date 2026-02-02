#!/usr/bin/env bash
#SBATCH --job-name=phase3_comp
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase3_comp_%j.out
#SBATCH --error=logs/phase3_comp_%j.err

set -euo pipefail

module load languages/python/3.12.3 || true
source /user/work/$USER/venvs/individualproject/bin/activate

cd /user/work/$USER/individualproject
mkdir -p logs

# Full composition + order experiments defined in scripts/run_phase3_correct.py
AA_VERSION=standard \
ADVEX_ITERS=20 \
NAC_BATCH_SIZE=64 \
NAC_PROFILE_SAMPLES=1000 \
NAC_TEST_SAMPLES=2000 \
python3 scripts/run_phase3_correct.py
