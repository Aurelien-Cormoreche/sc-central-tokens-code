#!/bin/bash
# Once, on a login node, from the repo root:  bash scripts/cluster/00_install.sh
#
# Creates the conda env `bench` as a clone of `cellViT` (torch, timm, openslide, deepspot, lightning
# already work there), so installing bench's extra packages cannot break the existing pipelines.
# Then installs bench's additions, TRIDENT (CONCH v1.5 loader) and the `bench` command, and runs the
# CPU checks: unit tests and the synthetic end-to-end smoke (~1 min).
#
#   SRC_ENV=myenv CONDA_ENV=bench2 bash scripts/cluster/00_install.sh   # other source / target env
#   REUSE=1 bash scripts/cluster/00_install.sh                          # env exists: only (re)install packages
set -eo pipefail
SRC_ENV="${SRC_ENV:-cellViT}"
CONDA_ENV="${CONDA_ENV:-bench}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

set +e; source ~/.bashrc; set -e
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    if [ -z "${REUSE:-}" ]; then
        echo "conda env '$CONDA_ENV' already exists: REUSE=1 to install into it, or CONDA_ENV=<other name>" >&2
        exit 1
    fi
else
    echo "== cloning conda env $SRC_ENV -> $CONDA_ENV"
    conda create -y -n "$CONDA_ENV" --clone "$SRC_ENV"
fi
conda activate "$CONDA_ENV"

# keep the core stack at the versions the repo was verified with, whatever TRIDENT asks for
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
printf '%s\n' torch==2.9.1 torchvision==0.24.1 timm==1.0.22 numpy==1.26.4 > "$CONSTRAINTS"

echo "== installing bench's packages"
pip install -c "$CONSTRAINTS" \
    pydantic==2.11.7 pyarrow==17.0.0 tifffile==2024.8.30 huggingface_hub==0.34.4 \
    transformers==4.57.1 einops==0.8.1 pytest==9.2.0
python -c "import skmisc" 2>/dev/null || pip install -c "$CONSTRAINTS" scikit-misc   # seurat_v3 HVGs
python -c "import trident" 2>/dev/null || pip install -c "$CONSTRAINTS" git+https://github.com/mahmoodlab/TRIDENT.git
pip install --no-deps -e .
pip check || echo "WARNING: pip check reports conflicts (above); the checks below show whether they matter"

echo "== versions"
python - <<'EOF'
import importlib
for m in ["torch", "timm", "transformers", "huggingface_hub", "openslide", "tifffile", "pyarrow", "pydantic",
          "lightning", "deepspot", "skmisc", "trident"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:16s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {m:16s} MISSING ({type(e).__name__}: {e})")
EOF

echo "== unit tests + synthetic end-to-end smoke (CPU)"
python -m pytest tests/bench
echo
echo "done. Next: fill .env (BENCH_ROOT, HF_HOME, WSI/Xenium roots), then bash scripts/cluster/01_fetch_weights.sh"
