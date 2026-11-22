#!/bin/bash
# Download the six checkpoints on any machine WITH internet (your laptop, a transfer node). Needs only the `hf`
# CLI (pip install "huggingface_hub>=0.34"); the bench code and env are not needed there.
#
#   bash scripts/cluster/download_weights.sh /path/to/hf_home
#
# Files go to /path/to/hf_home/hub in the Hugging Face cache layout that bench reads offline (blobs/, refs/,
# snapshots/ with symlinks). Do not use `hf download --local-dir`: that layout is not a cache.
# Then, if this machine is not the cluster, copy the cache over, keeping the symlinks (rsync -a does):
#   rsync -a /path/to/hf_home/hub/ <cluster>:<HF_HOME in the cluster .env>/hub/
# and on the cluster run  bash scripts/cluster/01_weights.sh  (checks every file, records the snapshot shas).
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
