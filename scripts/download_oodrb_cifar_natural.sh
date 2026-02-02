#!/usr/bin/env bash
set -euo pipefail

# Download CIFAR-10 natural shift datasets for OODRobustBench.
# This script targets:
#   - CIFAR-10.1 (v4/v6)
#   - CIFAR-10.2 (train/test npz)
#   - CIFAR-10-R (requires manual URL if not bundled)
#   - CINIC-10 (large; optional URL)
#
# Usage:
#   DATA_DIR=./data bash scripts/download_oodrb_cifar_natural.sh
#
# Optional:
#   CIFAR10R_URL=<direct tar/tgz url>
#   CINIC_URL=<direct tar/tgz url>

DATA_DIR="${DATA_DIR:-./data}"
TMP_DIR="${TMP_DIR:-/tmp/oodrb_downloads}"

mkdir -p "$DATA_DIR" "$TMP_DIR"

echo "[1/4] CIFAR-10.1 (v4/v6)"
if [ -f "$DATA_DIR/cifar-10.1/cifar10.1_v6_data.npy" ]; then
  echo "  -> already present: $DATA_DIR/cifar-10.1"
else
  rm -rf "$TMP_DIR/CIFAR-10.1"
  git clone --depth 1 https://github.com/modestyachts/CIFAR-10.1 "$TMP_DIR/CIFAR-10.1"
  mkdir -p "$DATA_DIR/cifar-10.1"
  cp "$TMP_DIR/CIFAR-10.1/datasets/"*.npy "$DATA_DIR/cifar-10.1/"
  echo "  -> installed: $DATA_DIR/cifar-10.1"
fi

echo "[2/4] CIFAR-10.2 (train/test npz)"
if [ -f "$DATA_DIR/cifar-10.2/cifar102_test.npz" ]; then
  echo "  -> already present: $DATA_DIR/cifar-10.2"
else
  rm -rf "$TMP_DIR/cifar-10.2"
  git clone --depth 1 https://github.com/modestyachts/cifar-10.2 "$TMP_DIR/cifar-10.2"
  mkdir -p "$DATA_DIR/cifar-10.2"
  cp "$TMP_DIR/cifar-10.2/"cifar102_*.npz "$DATA_DIR/cifar-10.2/"
  echo "  -> installed: $DATA_DIR/cifar-10.2"
fi

echo "[3/4] CIFAR-10-R (manual download may be required)"
if [ -d "$DATA_DIR/cifar-10-r" ]; then
  echo "  -> already present: $DATA_DIR/cifar-10-r"
elif [ -n "${CIFAR10R_URL:-}" ]; then
  echo "  -> downloading from CIFAR10R_URL"
  mkdir -p "$DATA_DIR"
  curl -L "$CIFAR10R_URL" -o "$TMP_DIR/cifar10-r.tar"
  tar -xf "$TMP_DIR/cifar10-r.tar" -C "$DATA_DIR"
  echo "  -> extracted under $DATA_DIR"
else
  echo "  -> NOT downloaded. Set CIFAR10R_URL or download manually:"
  echo "     https://github.com/TreeLLi/cifar10-r"
fi

echo "[4/4] CINIC-10 (large; optional)"
if [ -d "$DATA_DIR/CINIC-10" ]; then
  echo "  -> already present: $DATA_DIR/CINIC-10"
elif [ -n "${CINIC_URL:-}" ]; then
  echo "  -> downloading from CINIC_URL"
  mkdir -p "$DATA_DIR"
  curl -L "$CINIC_URL" -o "$TMP_DIR/CINIC-10.tar.gz"
  tar -xzf "$TMP_DIR/CINIC-10.tar.gz" -C "$DATA_DIR"
  echo "  -> extracted under $DATA_DIR"
else
  echo "  -> NOT downloaded. Set CINIC_URL or download manually:"
  echo "     https://datashare.ed.ac.uk/handle/10283/3192"
fi

echo "Done. Verify structure under $DATA_DIR:"
echo "  cifar-10.1/, cifar-10.2/, cifar-10-r/, CINIC-10/"
