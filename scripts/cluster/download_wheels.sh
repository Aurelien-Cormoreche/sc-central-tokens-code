#!/bin/bash
# Only for a cluster WITHOUT a PyPI mirror (with one, 00_install.sh needs just the TRIDENT wheel). On a machine
# WITH internet (any OS; needs python3 with pip, and git): Linux x86_64 wheels of bench's extra packages
# (requirements-bench.txt) and of TRIDENT, so 00_install.sh can install them on the cluster without internet.
#
#   bash scripts/cluster/download_wheels.sh /path/to/wheels
#   PYVER=3.11 bash scripts/cluster/download_wheels.sh /path/to/wheels   # if cellViT is not Python 3.10
#
# Then copy the folder to the cluster and run:  WHEELS=/path/to/wheels bash scripts/cluster/00_install.sh
set -eo pipefail
if [ -z "${1:-}" ]; then
    echo "usage: bash $0 /path/to/wheels" >&2
    exit 1
fi
OUT="$1"
PYVER="${PYVER:-3.10}"
PIP="${PIP:-python3 -m pip}"
REQ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements-bench.txt"
TRIDENT="$(sed -n 's/^# trident: //p' "$REQ")"
mkdir -p "$OUT"

CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
echo "numpy==1.26.4" > "$CONSTRAINTS"      # the cellViT env's numpy (requirements.txt)

echo "== wheels for Linux x86_64, CPython $PYVER -> $OUT"
$PIP download -d "$OUT" -r "$REQ" -c "$CONSTRAINTS" --only-binary=:all: \
    --implementation cp --python-version "$PYVER" \
    --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64
echo "== TRIDENT ($TRIDENT)"
$PIP wheel --no-deps -w "$OUT" "$TRIDENT"
echo "== $(ls "$OUT" | wc -l | tr -d ' ') files, $(du -sh "$OUT" | cut -f1): copy $OUT to the cluster"
