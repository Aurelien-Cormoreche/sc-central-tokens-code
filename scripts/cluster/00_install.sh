#!/bin/bash
# Once, on the cluster (a login node is fine), from the repo root:
#   WHEELS=/path/to/wheels bash scripts/cluster/00_install.sh   # no internet: wheels from download_wheels.sh
#   bash scripts/cluster/00_install.sh                          # if pip can reach PyPI and GitHub
#
# bench runs in the cellViT conda env plus a small overlay venv ($BENCH_VENV, default .venv-bench in the repo)
# that sees every cellViT package and adds bench's own (requirements-bench.txt). cellViT itself is not
# modified, and torch / torchvision / timm / numpy stay at its versions. TRIDENT goes in with --no-deps (bench
# loads only its CONCH v1.5 file). `bench` is put on the venv's path (no build step), then the CPU checks run:
# unit tests and the synthetic end-to-end smoke (~1 min).
#   CONDA_ENV=other BENCH_VENV=/path/to/venv ...     # other base env / overlay location (bench_env.sh reads both)
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
CONDA_ENV="${CONDA_ENV:-cellViT}"
VENV="${BENCH_VENV:-$REPO_ROOT/.venv-bench}"
REQ=scripts/cluster/requirements-bench.txt
TRIDENT="$(sed -n 's/^# trident: //p' "$REQ")"

set +e; source ~/.bashrc; set -e
conda activate "$CONDA_ENV"
PYVER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "== base: conda env $CONDA_ENV (Python $PYVER)"
if [ ! -x "$VENV/bin/python" ]; then
    echo "== creating overlay venv $VENV"
    python -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"

CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
python -m pip list --format=freeze 2>/dev/null | grep -iE '^(torch|torchvision|timm|numpy)==' > "$CONSTRAINTS" || true
echo "== kept at the cellViT versions: $(tr '\n' ' ' < "$CONSTRAINTS")"

if [ -n "${WHEELS:-}" ]; then
    SRC=(--no-index --find-links "$WHEELS")
    TRI=("$WHEELS"/trident-*.whl)
else
    SRC=()
    TRI=("$TRIDENT")
fi
echo "== installing bench's packages${WHEELS:+ from $WHEELS}"
if ! python -m pip install "${SRC[@]}" -c "$CONSTRAINTS" -r "$REQ"; then
    echo "pip failed. Offline, the wheels must be for Python $PYVER:" \
         "PYVER=$PYVER bash scripts/cluster/download_wheels.sh DIR" >&2
    exit 1
fi
python -m pip install "${SRC[@]}" --no-deps "${TRI[@]}"
echo "   (pip may say 'trident requires bioio, pyvips, ...': expected, bench uses only its CONCH v1.5 file)"

# bench itself: the repo on the venv's path + a `bench` launcher (no build step, works offline)
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$REPO_ROOT" > "$SITE/bench_repo.pth"
printf '#!/bin/sh\nexec "%s" -m bench "$@"\n' "$VENV/bin/python" > "$VENV/bin/bench"
chmod +x "$VENV/bin/bench"

echo "== versions"
python - <<'EOF'
import importlib
for m in ["torch", "timm", "numpy", "transformers", "huggingface_hub", "openslide", "tifffile", "pyarrow",
          "pydantic", "einops", "lightning", "deepspot", "skmisc", "bench"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:16s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {m:16s} MISSING ({type(e).__name__}: {e})")
from bench.models.encoders import trident_conchv15
trident_conchv15()
print("  trident          ok (CONCH v1.5 model file)")
EOF

echo "== unit tests + synthetic end-to-end smoke (CPU)"
if python -m pytest tests/bench; then
    echo
    echo "install OK. Next: BENCH_ROOT and HF_HOME in .env, weights (download_weights.sh), then 01_weights.sh"
else
    echo
    echo "install done, but tests FAILED (output above): send it to me before running anything else" >&2
    exit 1
fi
