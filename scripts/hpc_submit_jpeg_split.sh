#!/usr/bin/env bash
set -euo pipefail

# Submit 8 jobs, splitting JPEG eps lists across norms.
# Each job uses a single RTX 3090.

mkdir -p logs

sbatch --account=coms037985 \
  --job-name=jpeg_linf_01 \
  --output=logs/jpeg_linf_01_%j.out \
  --error=logs/jpeg_linf_01_%j.err \
  --export=JPEG_NORM=linf,JPEG_EPS_LIST=0.03125,0.0625 \
  scripts/hpc_job_jpeg_subset.sh

sbatch --account=coms037985 \
  --job-name=jpeg_linf_02 \
  --output=logs/jpeg_linf_02_%j.out \
  --error=logs/jpeg_linf_02_%j.err \
  --export=JPEG_NORM=linf,JPEG_EPS_LIST=0.125,0.25 \
  scripts/hpc_job_jpeg_subset.sh

sbatch --account=coms037985 \
  --job-name=jpeg_linf_03 \
  --output=logs/jpeg_linf_03_%j.out \
  --error=logs/jpeg_linf_03_%j.err \
  --export=JPEG_NORM=linf,JPEG_EPS_LIST=0.5,1.0 \
  scripts/hpc_job_jpeg_subset.sh

sbatch --account=coms037985 \
  --job-name=jpeg_l2_01 \
  --output=logs/jpeg_l2_01_%j.out \
  --error=logs/jpeg_l2_01_%j.err \
  --export=JPEG_NORM=l2,JPEG_EPS_LIST=0.25,0.5 \
  scripts/hpc_job_jpeg_subset.sh

sbatch --account=coms037985 \
  --job-name=jpeg_l2_02 \
  --output=logs/jpeg_l2_02_%j.out \
  --error=logs/jpeg_l2_02_%j.err \
  --export=JPEG_NORM=l2,JPEG_EPS_LIST=1.0,2.0 \
  scripts/hpc_job_jpeg_subset.sh

sbatch --account=coms037985 \
  --job-name=jpeg_l2_03 \
  --output=logs/jpeg_l2_03_%j.out \
  --error=logs/jpeg_l2_03_%j.err \
  --export=JPEG_NORM=l2,JPEG_EPS_LIST=4.0,8.0 \
  scripts/hpc_job_jpeg_subset.sh

sbatch --account=coms037985 \
  --job-name=jpeg_l1_01 \
  --output=logs/jpeg_l1_01_%j.out \
  --error=logs/jpeg_l1_01_%j.err \
  --export=JPEG_NORM=l1,JPEG_EPS_LIST=2.0,8.0,64.0 \
  scripts/hpc_job_jpeg_subset.sh

sbatch --account=coms037985 \
  --job-name=jpeg_l1_02 \
  --output=logs/jpeg_l1_02_%j.out \
  --error=logs/jpeg_l1_02_%j.err \
  --export=JPEG_NORM=l1,JPEG_EPS_LIST=256.0,512.0,1024.0 \
  scripts/hpc_job_jpeg_subset.sh
