#!/usr/bin/env bash
set -euo pipefail

# Phase3 split runner (Scheme A: 4 jobs)
# Usage:
#   bash scripts/phase3_split_a.sh 1
#   bash scripts/phase3_split_a.sh 2
#   bash scripts/phase3_split_a.sh 3
#   bash scripts/phase3_split_a.sh 4

PART=${1:-}
if [[ -z "$PART" ]]; then
  echo "Usage: $0 {1|2|3|4}"
  exit 1
fi

# Common configuration (adjust as needed)
export NAC_PROFILE_SAMPLES=${NAC_PROFILE_SAMPLES:-50000}
export NAC_TEST_SAMPLES=${NAC_TEST_SAMPLES:-10000}
export NAC_BATCH_SIZE=${NAC_BATCH_SIZE:-64}
export AA_VERSION=${AA_VERSION:-standard}
export ADVEX_ITERS=${ADVEX_ITERS:-20}

case "$PART" in
  1)
    export EXP_SET="clean,gaussian_0.05,gaussian_0.10,rotate_15,rotate_30,deepg_rotate_range,deepg_translate_2px,deepg_scale_range,deepg_shear_range"
    export PHASE3_OUTPUT_DIR=${PHASE3_OUTPUT_DIR:-phase3_output_part1}
    ;;
  2)
    export EXP_SET="clean,autoattack_linf,autoattack_l2"
    export PHASE3_OUTPUT_DIR=${PHASE3_OUTPUT_DIR:-phase3_output_part2}
    ;;
  3)
    export EXP_SET="clean,advex_elastic,advex_fog,advex_snow,advex_gabor,brightness_1.3,contrast_1.5"
    export PHASE3_OUTPUT_DIR=${PHASE3_OUTPUT_DIR:-phase3_output_part3}
    ;;
  4)
    export EXP_SET="clean,advex_jpeg_linf,advex_jpeg_l2,advex_jpeg_l1,order_geom_A_rotate_translate,order_geom_B_translate_rotate,order_mix_A_rotate_noise,order_mix_B_noise_rotate,order_A_rotate_noise,order_B_noise_rotate"
    export PHASE3_OUTPUT_DIR=${PHASE3_OUTPUT_DIR:-phase3_output_part4}
    ;;
  *)
    echo "Usage: $0 {1|2|3|4}"
    exit 1
    ;;
esac

echo "Running Phase3 split part: $PART"
echo "EXP_SET=$EXP_SET"

python3 scripts/run_phase3_correct.py
