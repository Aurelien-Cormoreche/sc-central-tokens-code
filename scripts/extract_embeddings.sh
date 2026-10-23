#!/bin/bash
# Extract embeddings on one GPU for a list of JSON run configs, run one after another
# (see src/python/extract_embeddings/run_configs.py and the "JSON run configs" section
# of src/python/extract_embeddings/extract_embeddings.py for the config format).
#
# Usage (submit from the repo root, next to .env):
#   sbatch scripts/extract_embeddings.sh configs/extract_embeddings/uni2_224.json
#   sbatch scripts/extract_embeddings.sh configs/extract_embeddings/a.json configs/extract_embeddings/b.json
#   sbatch scripts/extract_embeddings.sh configs/extract_embeddings/          # every *.json in the folder
#   CONDA_ENV=myenv sbatch --time=24:00:00 scripts/extract_embeddings.sh ...  # other env / resources
#
# Check what would run first (no GPU needed):
#   python -m src.python.extract_embeddings.run_configs configs/extract_embeddings/uni2_224.json --dry-run
#
# Requires the env vars in .env.example (DATASETS_ROOT, XENIUM_PROCESSED_OUTPUT_ROOT,
# WSI_RAW_ROOT, WSI_CONVERTED_ROOT, + the boundary/alignment roots if save_cell/save_nucleus).
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=extract_embeddings
#SBATCH --output=./extract_embeddings_%j.out.txt
#SBATCH --error=./extract_embeddings_%j.err.txt

if [ "$#" -eq 0 ]; then
    echo "ERROR: pass at least one JSON run config (or a directory of them)" >&2
    exit 1
fi

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
    echo "Loaded $ENV_FILE"
else
    echo "ERROR: $ENV_FILE not found" >&2
    exit 1
fi

echo "========================================="
echo "Job Started (env: $CONDA_ENV) -- configs: $*"
echo "========================================="
nvidia-smi

# job/cluster-level HF settings (not in .env)
export HF_HOME=/cluster/customapps/biomed/boeva/acormoreche/HF-cache/
export HF_HUB_OFFLINE=1

python3 -u -m src.python.extract_embeddings.run_configs "$@"
STATUS=$?

echo "========================================="
echo "Job Completed (exit code $STATUS)"
echo "========================================="
exit $STATUS
