#!/bin/bash
# Checks of the install, on a CPU compute node (on login nodes torch's CUDA libraries fail to load: "failed to map
# segment from shared object", a memory cap). Submit from the repo root after 00_install.sh:
#   sbatch scripts/cluster/00_check.sbatch
# Lists package versions, loads TRIDENT's CONCH v1.5 file, runs the unit tests and the synthetic end-to-end smoke
# (generated slides + tiny ViT through the whole CLI, incl. gene expression when deepspot is installed). No GPU,
# no .env needed. Result: last lines of bench_check_<jobid>.out.txt.
#SBATCH --job-name=bench-check
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=./bench_check_%j.out.txt
#SBATCH --error=./bench_check_%j.err.txt

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ~/.bashrc
conda activate "${CONDA_ENV:-bench}" || exit 1
set -eo pipefail
echo "host=$(hostname) job=$SLURM_JOB_ID env=$CONDA_PREFIX"

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
python -m pytest tests/bench -rs
echo "CHECK OK"
