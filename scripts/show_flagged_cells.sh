#!/bin/bash
# Find the black-patch cells of the given slides in their embeddings
# (scripts/check_black_patches.py) and save, per slide, an overview with them marked and a
# grid of their crops (scripts/show_flagged_cells.py). CPU only.
#
# Usage (submit from the repo root, next to .env):
#   sbatch scripts/show_flagged_cells.sh SLIDE_1 SLIDE_2 ...
#   OUT=my_dir MODEL=UNI2 CONFIG=configs/extract_embeddings/uni2_224.json \
#       sbatch scripts/show_flagged_cells.sh SLIDE_1 ...
# Results: $OUT/{slide}_overview.png, $OUT/{slide}_crops.png, $OUT/black_cells.csv and the
# summary in show_flagged_cells_<jobid>.out.txt.
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=show_flagged_cells
#SBATCH --output=./show_flagged_cells_%j.out.txt
#SBATCH --error=./show_flagged_cells_%j.err.txt

if [ "$#" -eq 0 ]; then
    echo "ERROR: pass at least one slide name" >&2
    exit 1
fi

CONDA_ENV="${CONDA_ENV:-cellViT}"
MODEL="${MODEL:-UNI2}"                                        # embeddings folder: $DATASETS_ROOT/${MODEL}_h5
CONFIG="${CONFIG:-configs/extract_embeddings/uni2_224.json}"  # extraction config that lists the slides
OUT="${OUT:-flagged_cells}"

source ~/.bashrc
conda activate "$CONDA_ENV"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
set -a; source .env; set +a

mkdir -p "$OUT"
printf '%s\n' "$@" > "$OUT/slides.txt"
echo "== check_black_patches: $MODEL, $# slide(s)"
python3 -u scripts/check_black_patches.py "$MODEL" "$OUT/slides.txt" --csv "$OUT/black_cells.csv"
echo "== show_flagged_cells"
python3 -u scripts/show_flagged_cells.py "$CONFIG" --slides "$@" --csv "$OUT/black_cells.csv" \
    --out "$OUT" --workers "${SLURM_CPUS_PER_TASK:-4}"
