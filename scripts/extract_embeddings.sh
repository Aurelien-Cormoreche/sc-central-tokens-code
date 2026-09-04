#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=60:00:00
#SBATCH --output=./pfm_extract_embeddings.out.txt
#SBATCH --job-name=pfm_extract_embeddings
#SBATCH --error=./pfm_extract_embeddings.err.txt

CONDA_ENV="${1:-cellViT}"

source ~/.bashrc
conda activate "$CONDA_ENV"

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

echo "========================================="
echo "Job Started (env: $CONDA_ENV)"
echo "========================================="
nvidia-smi

# job/cluster-level HF settings (not in .env)
export HF_HOME=/cluster/customapps/biomed/boeva/acormoreche/HF-cache/
export HF_HUB_OFFLINE=1

python3 -m src.python.extract_embeddings.extract_embeddings

echo "========================================="
echo "Job Completed"
echo "========================================="