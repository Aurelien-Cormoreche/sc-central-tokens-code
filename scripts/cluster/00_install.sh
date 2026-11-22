#!/bin/bash
# Once, on the cluster login node, from the repo root (safe to rerun):
#   TRIDENT=/path/to/trident-0.3.2-py3-none-any.whl bash scripts/cluster/00_install.sh
#
# 1. conda env $CONDA_ENV (default bench): cloned from $SRC_ENV (default cellViT) if it does not exist yet,
#    reused otherwise. cellViT itself is never modified.
# 2. bench's packages (scripts/cluster/requirements-bench.txt) with pip, through the cluster's PyPI mirror;
#    torch / torchvision / timm / numpy stay at the env's versions. No PyPI access at all: WHEELS=/path/to/wheels
#    (from download_wheels.sh).
# 3. TRIDENT with --no-deps (bench loads only its CONCH v1.5 model file). GitHub is blocked on the cluster, so
#    build the wheel where GitHub works (e.g. the customapps client), at the commit in requirements-bench.txt:
#      pip wheel --no-deps -w DIR "git+https://github.com/mahmoodlab/TRIDENT.git@<commit>"
#    and pass TRIDENT=DIR/trident-0.3.2-py3-none-any.whl (read-only location is fine).
# 4. `bench` on the env's path (no build step). The checks run on a compute node: sbatch scripts/cluster/00_check.sbatch
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
SRC_ENV="${SRC_ENV:-cellViT}"
CONDA_ENV="${CONDA_ENV:-bench}"
REQ=scripts/cluster/requirements-bench.txt
TRIDENT_URL="$(sed -n 's/^# trident: //p' "$REQ")"

set +e; source ~/.bashrc; set -e
if conda env list | awk '{print $1}' | grep -x "$CONDA_ENV" >/dev/null; then
    echo "== reusing conda env $CONDA_ENV"
else
    echo "== cloning conda env $SRC_ENV -> $CONDA_ENV"
    conda create -y -n "$CONDA_ENV" --clone "$SRC_ENV"
fi
conda activate "$CONDA_ENV"
PYVER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "== $CONDA_PREFIX (Python $PYVER)"

CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
python -m pip list --format=freeze 2>/dev/null | grep -iE '^(torch|torchvision|timm|numpy)==' > "$CONSTRAINTS" || true
echo "== kept at the env's versions: $(tr '\n' ' ' < "$CONSTRAINTS")"
SRC=()
if [ -n "${WHEELS:-}" ]; then
    SRC=(--no-index --find-links "$WHEELS")
fi

echo "== bench's packages${WHEELS:+ from $WHEELS}"
if ! python -m pip install "${SRC[@]}" -c "$CONSTRAINTS" -r "$REQ"; then
    echo "pip failed (above). Without a PyPI mirror: PYVER=$PYVER bash scripts/cluster/download_wheels.sh DIR" \
         "where there is internet, then WHEELS=DIR" >&2
    exit 1
fi
echo "   (pip may list conflicts of cellViT-only packages, e.g. cellvit/pathopatch want pydantic<2: bench does not use them)"

TRI="${TRIDENT:-}"
if [ -z "$TRI" ] && [ -n "${WHEELS:-}" ]; then
    TRI="$(ls "$WHEELS"/trident-*.whl 2>/dev/null | head -1 || true)"
fi
if [ -n "$TRI" ]; then
    echo "== TRIDENT from $TRI (--no-deps)"
    python -m pip install "${SRC[@]}" --no-deps --force-reinstall "$TRI"
elif python -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("trident") is None)'; then
    echo "== TRIDENT already installed (pass TRIDENT=<wheel> to replace it)"
else
    echo "== TRIDENT from $TRIDENT_URL (--no-deps)"
    if ! python -m pip install --no-deps "$TRIDENT_URL"; then
        echo >&2
        echo "TRIDENT: GitHub is not reachable from here. Where it is (e.g. the customapps client), run" >&2
        echo "  pip wheel --no-deps -w DIR \"$TRIDENT_URL\"" >&2
        echo "then rerun: TRIDENT=DIR/trident-0.3.2-py3-none-any.whl bash scripts/cluster/00_install.sh" >&2
        exit 1
    fi
fi

# bench itself: the repo on the env's path + a `bench` launcher (no build step)
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$REPO_ROOT" > "$SITE/bench_repo.pth"
printf '#!/bin/sh\nexec "%s" -m bench "$@"\n' "$CONDA_PREFIX/bin/python" > "$CONDA_PREFIX/bin/bench"
chmod +x "$CONDA_PREFIX/bin/bench"

echo
echo "install done. Checks (they import torch, whose CUDA libraries do not load on login nodes):"
echo "  sbatch scripts/cluster/00_check.sbatch"
