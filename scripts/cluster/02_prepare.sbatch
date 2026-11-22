#!/bin/bash
# Data preparation for every slide (CPU only). Submit from the repo root:
#   sbatch scripts/cluster/02_prepare.sbatch
#   sbatch scripts/cluster/02_prepare.sbatch --datasets JOHNSON     # restrict to cohorts / slide patterns
#
# 1. bench inspect    slide table: WSI path, size, pixel size, resampling scale. Stops on an unknown pixel
#                     size or a mismatch with xenium_metadata.csv: fix with mpp_overrides in configs/bench.yaml
#                     and resubmit. Review: $BENCH_ROOT/manifest/slides_inspect.csv
# 2. bench darkspots  black / no-data / unreadable block map per slide, slides in parallel on this node's
#                     CPUs. Review: $BENCH_ROOT/darkspots/*.png
# 3. bench manifest   the global cell manifest; prints the per-cohort drop counts (unknown label, border, dark spot)
#
# Resumable: resubmitting skips the dark-spot maps that are already done.
#SBATCH --job-name=bench-prepare
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=./bench_prepare_%j.out.txt
#SBATCH --error=./bench_prepare_%j.err.txt

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cluster/bench_env.sh || exit 1
set -eo pipefail
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) args: $*"

W="${SLURM_CPUS_PER_TASK:-16}"
echo "== 1/3 inspect";   bench inspect   --workers "$W" "$@"
echo "== 2/3 darkspots"; bench darkspots --local    "$@"
echo "== 3/3 manifest";  bench manifest  --workers "$W" "$@"
echo "end=$(date -Is)"
echo "review: $BENCH_ROOT/manifest/slides_inspect.csv, $BENCH_ROOT/darkspots/*.png, the drop counts above"
