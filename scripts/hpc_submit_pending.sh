#!/usr/bin/env bash
set -euo pipefail

# Pending CIFAR-10 tasks (ready to run)
sbatch --account=coms037985 scripts/hpc_job_oodrb.sh
sbatch --account=coms037985 scripts/hpc_job_oodrb_natural.sh
sbatch --account=coms037985 scripts/hpc_job_phase3_composition.sh
sbatch --account=coms037985 scripts/hpc_job_total_benchmark_jpeg_scan.sh

# ImageNet tasks (postponed until ImageNet data is available)
# sbatch --account=coms037985 scripts/hpc_job_oodrb_imagenet.sh
# sbatch --account=coms037985 scripts/hpc_job_rb_resnet50.sh
# sbatch --account=coms037985 scripts/hpc_job_rb_vit.sh
