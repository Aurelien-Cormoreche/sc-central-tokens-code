#!/bin/bash
# On the cluster (no internet needed), from the repo root:  bash scripts/cluster/01_weights.sh
#
# Checks that every model's files are in $HF_HOME/hub (put there by download_weights.sh, i.e. `hf download`)
# and records the snapshot shas in $BENCH_ROOT/weights_revisions.json; every later job loads exactly those
# snapshots, offline. Anything missing is listed with the `hf download` command that fetches it; the
# script then exits 1. Rerun after copying the missing files.
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set +e; source scripts/cluster/bench_env.sh || exit 1; set -e

bench fetch-weights --offline "$@"
