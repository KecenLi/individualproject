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

# Ensure natural-shift datasets exist (CIFAR-10.1/10.2 are required).
# Do NOT download during the job; pre-download on login/local node.
DATA_DIR=./data
missing=()
[ ! -f "$DATA_DIR/cifar-10.1/cifar10.1_v6_data.npy" ] && missing+=("cifar-10.1")
[ ! -f "$DATA_DIR/cifar-10.2/cifar102_test.npz" ] && missing+=("cifar-10.2")
if [ ${#missing[@]} -gt 0 ]; then
  echo "[ERROR] Missing datasets: ${missing[*]}"
  echo "Please pre-download them on a login/local node:"
  echo "  DATA_DIR=./data bash scripts/download_oodrb_cifar_natural.sh"
  exit 1
fi

# Build natural-shift list based on available datasets.
shifts=()
[ -f "$DATA_DIR/cifar-10.1/cifar10.1_v6_data.npy" ] && shifts+=("cifar10.1")
[ -f "$DATA_DIR/cifar-10.2/cifar102_test.npz" ] && shifts+=("cifar10.2")
[ -d "$DATA_DIR/CINIC-10" ] && shifts+=("cinic")
[ -d "$DATA_DIR/cifar-10-r" ] && shifts+=("cifar10-r")
if [ ${#shifts[@]} -eq 0 ]; then
  echo "[ERROR] No natural-shift datasets found under $DATA_DIR."
  exit 1
fi
NATURAL_SHIFTS=$(IFS=,; echo "${shifts[*]}")
echo "[INFO] Natural shifts to run: $NATURAL_SHIFTS"

# Natural shifts only (skip corruption loop by passing a non-existent corruption name).
python3 scripts/run_oodrb_nac.py \
  --data-dir "$DATA_DIR" \
  --dataset cifar10 \
  --model-name Standard \
  --threat-model Linf \
  --batch-size 64 \
  --profile-samples 1000 \
  --id-examples 10000 \
  --corruptions none \
  --run-natural-shifts \
  --natural-shifts "$NATURAL_SHIFTS" \
  --output-dir oodrb_results
