#!/bin/bash
# Download the six checkpoints into an HF cache, on a machine WITH internet. Needs only the `hf` CLI
# (huggingface_hub >= 0.34); the bench code and env are not needed there. Typical: on the customapps client,
# straight into the cache the cluster reads (read-only is fine, bench never writes to it):
#
#   bash scripts/cluster/download_weights.sh /path/to/HF_HOME        # = HF_HOME in the cluster's .env
#
# Files go to /path/to/HF_HOME/hub in the Hugging Face cache layout (blobs/, refs/, snapshots/ with symlinks).
# Do not use `hf download --local-dir`: that layout is not a cache. Then, on the cluster:
#   bash scripts/cluster/01_weights.sh       (checks every file offline, records the snapshot shas)
#
# Gated: log in first (`hf auth login`) and accept the terms on each model page:
#   https://huggingface.co/MahmoodLab/UNI2-h       https://huggingface.co/MahmoodLab/conchv1_5
#   https://huggingface.co/paige-ai/Virchow2       https://huggingface.co/bioptimus/H-optimus-1 (manual approval)
# LEMON and GenBio-PathFM are open. Files already in the cache are skipped, so rerunning is cheap.
# The file lists match bench/models/encoders.py::weight_spec (tests/bench/test_weights.py checks it).
set -eo pipefail
if [ -z "${1:-}" ]; then
    echo "usage: bash $0 /path/to/hf_home   (the cluster's HF_HOME, or a folder you then copy there)" >&2
    exit 1
fi
CACHE="$1/hub"
mkdir -p "$CACHE"
hf auth whoami || { echo "not logged in: run  hf auth login" >&2; exit 1; }

hf download MahmoodLab/UNI2-h                                      --cache-dir "$CACHE"
hf download paige-ai/Virchow2                                      --cache-dir "$CACHE"
hf download bioptimus/H-optimus-1                                  --cache-dir "$CACHE"
hf download MahmoodLab/conchv1_5 pytorch_model_vision.bin          --cache-dir "$CACHE"
hf download aliceblondel/LEMON mean_std.json lemon_vits8.pth.tar   --cache-dir "$CACHE"
hf download genbio-ai/genbio-pathfm                                --cache-dir "$CACHE"

du -sh "$CACHE"/models--* 2>/dev/null || true
echo "done: $CACHE"
