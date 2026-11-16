#!/bin/bash
# Paper, UNI2 read-out x field-of-view sweep on SPLIT, stage 2: launch the classifier
# (configs/paper/uni2_readout_fov_within_cohort_split.yaml) for every field of view S and
# read-out -- 6 x 3 = 18 launches, each one joint job over the 20 within-cohort folds.
# Needs stage 1's embeddings (scripts/extract_paper_uni2_fov.sh) for the sizes you launch.
# Results: outputs_classifier_paper/uni2_readout_fov_within_cohort_SPLIT/UNI2_fov{S}/raw/...
#
# Usage (repo root, login node, conda env active):
#   bash scripts/launch_paper_uni2_fov.sh dry_run=true      # print the jobs, submit nothing
#   bash scripts/launch_paper_uni2_fov.sh                   # submit all 18
#   FOVS="32 224" READOUTS="cls nucleus" bash scripts/launch_paper_uni2_fov.sh   # a subset
# Other arguments are passed to every launcher call as Hydra overrides. Folds that already
# have results are skipped by the launcher, so relaunching only submits what's missing.
FOVS="${FOVS:-32 56 112 224 448 896}"
READOUTS="${READOUTS:-central cls nucleus}"

if [ -f .env ]; then set -a; source .env; set +a; fi

for S in $FOVS; do
    for R in $READOUTS; do
        case "$R" in
            central) E="[bottom_left,bottom_right,top_left,top_right]" ;;
            cls)     E="[cls]" ;;
            nucleus) E="[nucleus]" ;;
            *) echo "ERROR: unknown read-out '$R' (central, cls or nucleus)" >&2; exit 1 ;;
        esac
        echo "######## UNI2_fov$S / $R ########"
        python3 -m src.python.classifier_training.launcher \
            --config-name=paper/uni2_readout_fov_within_cohort_split \
            "data.model_name=UNI2_fov$S" data.correction_name=raw "data.embeddings_datasets=$E" "$@"
    done
done
