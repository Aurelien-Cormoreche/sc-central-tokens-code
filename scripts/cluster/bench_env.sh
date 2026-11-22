# Environment for every bench command, interactive or inside a SLURM job. Source it, don't run it:
#   source scripts/cluster/bench_env.sh
# Every bench array task sources it too (configs/bench.yaml: slurm.setup).
#
# 1. activates the conda env $CONDA_ENV (default: cellViT) and bench's overlay venv $BENCH_VENV on top of it
#    (default: .venv-bench in the repo, created by 00_install.sh)
# 2. loads the repo's .env (same file the rest of the repo uses, see .env.example) and checks the
#    variables bench needs: BENCH_ROOT WSI_RAW_ROOT WSI_CONVERTED_ROOT XENIUM_PROCESSED_OUTPUT_ROOT HF_HOME
# 3. inside SLURM jobs sets HF_HUB_OFFLINE=1: weights come from HF_HOME (download_weights.sh, checked by 01_weights.sh)

_bench_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_bench_conda="${CONDA_ENV:-cellViT}"
_bench_venv="${BENCH_VENV:-$_bench_repo/.venv-bench}"

_bench_env_main() {
    if [ "${VIRTUAL_ENV:-}" != "$_bench_venv" ]; then
        if [ "${CONDA_DEFAULT_ENV:-}" != "$_bench_conda" ]; then
            source ~/.bashrc
            if ! conda activate "$_bench_conda"; then
                echo "ERROR: conda env '$_bench_conda' not found (set CONDA_ENV)" >&2
                return 1
            fi
        fi
        if [ ! -f "$_bench_venv/bin/activate" ]; then
            echo "ERROR: $_bench_venv not found (run scripts/cluster/00_install.sh, or set BENCH_VENV)" >&2
            return 1
        fi
        source "$_bench_venv/bin/activate"
    fi
    if [ ! -f "$_bench_repo/.env" ]; then
        echo "ERROR: $_bench_repo/.env not found (copy .env.example and fill it in)" >&2
        return 1
    fi
    set -a
    source "$_bench_repo/.env"
    set +a
    local v
    for v in BENCH_ROOT WSI_RAW_ROOT WSI_CONVERTED_ROOT XENIUM_PROCESSED_OUTPUT_ROOT HF_HOME; do
        if [ -z "${!v:-}" ]; then
            echo "ERROR: $v is not set (add it to $_bench_repo/.env)" >&2
            return 1
        fi
    done
    for v in WSI_RAW_ROOT WSI_CONVERTED_ROOT XENIUM_PROCESSED_OUTPUT_ROOT; do
        if [ ! -d "${!v}" ]; then
            echo "ERROR: $v='${!v}' is not a directory (paths with & or spaces must be quoted in .env)" >&2
            return 1
        fi
    done
    mkdir -p "$BENCH_ROOT" "$HF_HOME" || return 1
    export BENCH_CONFIG="${BENCH_CONFIG:-$_bench_repo/configs/bench.yaml}"
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    fi
    echo "bench env: conda=$_bench_conda venv=$_bench_venv BENCH_ROOT=$BENCH_ROOT HF_HOME=$HF_HOME config=$BENCH_CONFIG" \
         "offline=${HF_HUB_OFFLINE:-0}"
}

_bench_env_main
