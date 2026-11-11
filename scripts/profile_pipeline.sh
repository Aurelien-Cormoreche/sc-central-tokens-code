#!/bin/bash
# Profile extraction + classifier training on one GPU (see scripts/profile_pipeline.py).
# Submit from the repo root (next to .env), one job per GPU type, choosing the type with --gres:
#   sbatch --gres=gpu:rtx4090:1 --cpus-per-task=32 scripts/profile_pipeline.sh          # everything
#   sbatch --gres=gpu:rtx5090:1 scripts/profile_pipeline.sh --sections env extract
#   sbatch --gres=gpu:h200:1    scripts/profile_pipeline.sh --sections env extract
# The report lands in profile_reports/<gpu>_<jobid>.txt (+ .json, inventory csv).
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=profile_pipeline
#SBATCH --output=./profile_pipeline_%j.out.txt
#SBATCH --error=./profile_pipeline_%j.err.txt

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

# job/cluster-level HF settings (not in .env)
export HF_HOME=/cluster/customapps/biomed/boeva/acormoreche/HF-cache/
export HF_HUB_OFFLINE=1

python3 -u scripts/profile_pipeline.py "$@"
