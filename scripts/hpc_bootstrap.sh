#!/usr/bin/env bash
set -euo pipefail

# BluePebble bootstrap for the NAC project.
# Usage:
#   PROJECT_DIR=/user/work/$USER/individualproject \
#   VENV_DIR=/user/work/$USER/venvs/individualproject \
#   PY_MODULE=languages/python/3.12.3 \
#   bash scripts/hpc_bootstrap.sh

PROJECT_DIR="${PROJECT_DIR:-/user/work/$USER/individualproject}"
VENV_DIR="${VENV_DIR:-/user/work/$USER/venvs/individualproject}"
PY_MODULE="${PY_MODULE:-languages/python/3.12.3}"
PIP_TORCH_INDEX="${PIP_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

echo "[1/6] Load Python module (if available)"
module load "$PY_MODULE" || true

echo "[2/6] Create venv: $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[3/6] Upgrade pip/setuptools/wheel"
python -m pip install -U pip setuptools wheel

echo "[4/6] Install PyTorch (CUDA 12.1 wheels by default)"
python -m pip install torch torchvision --index-url "$PIP_TORCH_INDEX"

echo "[5/6] Install core dependencies"
python -m pip install \
  numpy pandas scipy scikit-learn matplotlib seaborn \
  pyyaml tqdm pillow click robustbench gdown

echo "[6/6] Ensure repo exists and submodules are ready"
if [ -d "$PROJECT_DIR/.git" ]; then
  git -C "$PROJECT_DIR" submodule update --init --recursive || true
  echo "Repo OK: $PROJECT_DIR"
else
  echo "Repo not found at $PROJECT_DIR (clone or rsync it first)."
fi

echo "Bootstrap complete."
