#!/bin/bash
# On a login node (needs internet), from the repo root:  bash scripts/cluster/01_fetch_weights.sh
#
# Downloads the six checkpoints into $HF_HOME and records the snapshot shas in
# $BENCH_ROOT/weights_revisions.json; every later job loads exactly those snapshots, offline.
# Gated repos: log in once (`huggingface-cli login`, or HF_TOKEN in .env) and accept the terms on
#   https://huggingface.co/MahmoodLab/UNI2-h          https://huggingface.co/MahmoodLab/conchv1_5
#   https://huggingface.co/paige-ai/Virchow2          https://huggingface.co/bioptimus/H-optimus-1 (manual approval)
#   LEMON (aliceblondel/LEMON) and GenBio-PathFM (genbio-ai/genbio-pathfm) are open.
# Re-running is cheap: files already in the cache are not downloaded again.
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set +e; source scripts/cluster/bench_env.sh || exit 1; set -e
unset HF_HUB_OFFLINE

python - <<'EOF'
from huggingface_hub import whoami
try:
    print("Hugging Face user:", whoami()["name"])
except Exception as e:
    raise SystemExit(f"not logged in to Hugging Face ({e}): run `huggingface-cli login` or set HF_TOKEN in .env")
EOF

bench fetch-weights "$@"
