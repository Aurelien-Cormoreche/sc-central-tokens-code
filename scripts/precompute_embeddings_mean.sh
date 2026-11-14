#!/bin/bash
# Precompute each slide's mean embedding (embeddings_mean.h5 next to embeddings_dataset.h5) so
# the classifier reads one dataset instead of four -- see src/python/utils/embeddings_mean.py.
# CPU only; bound by ceph reads (~250 MB/s per job), so split big folders over an array job.
#
# Usage (submit from the repo root, next to .env; arguments go to embeddings_mean.py):
#   sbatch scripts/precompute_embeddings_mean.sh --config configs/train_cross_cohort.yaml \
#       --model UNI2 --correction Base
#   sbatch --array=0-3 scripts/precompute_embeddings_mean.sh --shard-from-array \
#       --config configs/train_cross_cohort.yaml --model UNI2 --correction Base
#   python -m src.python.utils.embeddings_mean ... --dry-run     # on a login node: what's missing/stale
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --job-name=embeddings_mean
#SBATCH --output=./embeddings_mean_%A_%a.out.txt
#SBATCH --error=./embeddings_mean_%A_%a.err.txt

CONDA_ENV="${CONDA_ENV:-cellViT}"

source ~/.bashrc
conda activate "$CONDA_ENV"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

ENV_FILE="$REPO_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "ERROR: $ENV_FILE not found" >&2
    exit 1
fi

ARGS=()
for a in "$@"; do
    if [ "$a" = "--shard-from-array" ]; then
        ARGS+=(--shard "${SLURM_ARRAY_TASK_ID:?--shard-from-array needs sbatch --array}/${SLURM_ARRAY_TASK_COUNT}")
    else
        ARGS+=("$a")
    fi
done

python3 -u -m src.python.utils.embeddings_mean "${ARGS[@]}"
