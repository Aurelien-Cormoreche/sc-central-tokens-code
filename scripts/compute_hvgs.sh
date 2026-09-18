#!/bin/bash
# Precompute highly-variable genes per split for deepspot_training (see
# src/python/deepspot_training/compute_hvgs.py's module docstring for what this does
# and why it needs a big-memory node). Run once per config before any deepspot_training
# experiment, unless data.hvg_dir is set to null in that config.
#
# Usage:
#   sbatch scripts/compute_hvgs.sh                                   # configs/deepspot_train_gsm.yaml
#   sbatch scripts/compute_hvgs.sh --config-name=deepspot_train       # any other configs/*.yaml
#
# Requires the env vars in .env.example to be set (DATASETS_ROOT, XENIUM_PROCESSED_OUTPUT_ROOT).
#SBATCH --job-name=compute_hvgs
#SBATCH --partition=compute
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs_compute_hvgs/%j.out
#SBATCH --error=slurm_logs_compute_hvgs/%j.err
source ~/.bashrc
ENV_FILE="${SLURM_SUBMIT_DIR:-$PWD}/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "Loaded $ENV_FILE"
else
    echo "ERROR: $ENV_FILE not found" >&2
    exit 1
fi

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "$REPO_ROOT"

mkdir -p slurm_logs_compute_hvgs

conda activate "${CONDA_ENV_NAME:-cellViT}"

python -m src.python.deepspot_training.compute_hvgs "$@"
