#!/bin/bash
# Paper, UNI2 read-out x field-of-view sweep on SPLIT, stage 1: submit one extraction job
# per field of view S (configs/extract_embeddings/paper_uni2_fov/uni2_fov{S}_split.json).
# Each crops S x S native pixels around every cell of the 29 SPLIT slides, resizes it to
# 224 x 224 and saves the central tokens, CLS and nucleus token to
# $DATASETS_ROOT/UNI2_fov{S}_h5/raw/{sample}/embeddings_dataset.h5. Stage 2 (classifier):
# scripts/launch_paper_uni2_fov.sh.
#
# Usage (repo root, login node, conda env active):
#   bash scripts/extract_paper_uni2_fov.sh --dry-run        # inputs + progress per slide; no GPU or torch
#   bash scripts/extract_paper_uni2_fov.sh                  # submit the 6 jobs
#   FOVS="32 56" bash scripts/extract_paper_uni2_fov.sh     # only these sizes
#   bash scripts/extract_paper_uni2_fov.sh --time=24:00:00  # other args go to sbatch
#
# Finished slides are skipped (skip_existing), so resubmitting a size after a crash or
# timeout only extracts what's left. Each job runs ~8 h on an RTX 4090 at fp32 (~100
# cells/s, 2.9M cells), ~24 h on a V100, inside extract_embeddings.sh's 72 h. Disk: 6
# float32 datasets (4 central tokens, CLS, nucleus) x 2.9M cells x 1536 = ~108 GB per size.
FOVS="${FOVS:-32 56 112 224 448 896}"
CONFIG_DIR=configs/extract_embeddings/paper_uni2_fov

for S in $FOVS; do
    if [ ! -f "$CONFIG_DIR/uni2_fov${S}_split.json" ]; then
        echo "ERROR: no config for field of view $S ($CONFIG_DIR/uni2_fov${S}_split.json)" >&2
        exit 1
    fi
done

if [ "${1:-}" = "--dry-run" ]; then
    if [ -f .env ]; then set -a; source .env; set +a; fi
    for S in $FOVS; do
        echo "######## UNI2_fov$S ########"
        python3 -m src.python.extract_embeddings.extract_embeddings \
            --config "$CONFIG_DIR/uni2_fov${S}_split.json" --dry-run
    done
    exit 0
fi

for S in $FOVS; do
    sbatch --job-name="uni2_fov$S" \
        --output="./extract_uni2_fov${S}_%j.out.txt" --error="./extract_uni2_fov${S}_%j.err.txt" \
        "$@" scripts/extract_embeddings.sh "$CONFIG_DIR/uni2_fov${S}_split.json"
done
