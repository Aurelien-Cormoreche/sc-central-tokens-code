#!/bin/bash
# Real end-to-end smoke on one GPU (~30-90 min). Submit from the repo root, after 01_weights.sh:
#   sbatch scripts/cluster/03_smoke.sbatch
#   sbatch scripts/cluster/03_smoke.sbatch --models uni2 lemon        # subset of models
#   sbatch --gres=gpu:1 scripts/cluster/03_smoke.sbatch               # other GPU type name (also set slurm.gres)
#
# Independent of 02_prepare.sbatch (own outputs in $BENCH_ROOT/smoke), so both can run at the same time.
# On 7 small slides (<= 300 cells each) and every model: inspect -> darkspots -> manifest -> extract ->
# classify -> genexp -> stage 2a/2b -> scaling -> report -> status -> resume check, plus
#   parity       each model's `cls` (bf16, our loader) vs its model-card forward (fp32); PASS = mean cos >= 0.999
#   centring     $BENCH_ROOT/smoke/centring/{model}__{slide}.png: red cross (centroid) inside the blue box
#   calibration  GPU throughput per model -> $BENCH_ROOT/calibration.json (every later --dry-run uses it)
# The last table lists every step as PASS/FAIL; the job exits non-zero if any step failed.
#SBATCH --job-name=bench-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=./bench_smoke_%j.out.txt
#SBATCH --error=./bench_smoke_%j.err.txt

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cluster/bench_env.sh || exit 1
set -eo pipefail
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) args: $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

bench smoke "$@"
echo "end=$(date -Is)"
