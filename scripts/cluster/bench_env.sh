# Environment for every bench command, interactive or inside a SLURM job. Source it, don't run it:
#   source scripts/cluster/bench_env.sh
# Every bench array task sources it too (configs/bench.yaml: slurm.setup).
#
# 1. activates the conda env $CONDA_ENV (default: bench, set up by 00_install.sh)
# 2. loads the repo's .env (same file the rest of the repo uses, see .env.example) and checks the
#    variables bench needs: BENCH_ROOT WSI_RAW_ROOT WSI_CONVERTED_ROOT XENIUM_PROCESSED_OUTPUT_ROOT HF_HOME
# 3. HF_HOME is only read, never written, so it can be a read-only cache (e.g. customapps, filled by `hf download`
#    on the customapps client; 01_weights.sh checks it): HF_HUB_OFFLINE=1 (set HF_HUB_OFFLINE=0 in .env if the
#    cluster has internet), and transformers' module cache (GenBio's remote code, by default $HF_HOME/modules)
#    goes to a writable folder: per job in $TMPDIR, else $BENCH_ROOT/.hf_modules

_bench_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_bench_conda="${CONDA_ENV:-bench}"

_bench_env_main() {
    if [ "${CONDA_DEFAULT_ENV:-}" != "$_bench_conda" ]; then
        source ~/.bashrc
        if ! conda activate "$_bench_conda"; then
            echo "ERROR: conda env '$_bench_conda' not found (run scripts/cluster/00_install.sh, or set CONDA_ENV)" >&2
            return 1
        fi
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
    if [ -n "${ALIGNMENT_MATRIX_ROOT:-}" ] && [ ! -d "$ALIGNMENT_MATRIX_ROOT" ]; then
        echo "ERROR: ALIGNMENT_MATRIX_ROOT='$ALIGNMENT_MATRIX_ROOT' is not a directory" >&2
        return 1
    fi
    if [ ! -d "$HF_HOME/hub" ]; then
        echo "ERROR: HF_HOME='$HF_HOME' has no hub/ folder (the HF cache that \`hf download\` fills)" >&2
        return 1
    fi
    mkdir -p "$BENCH_ROOT" || return 1
    export BENCH_CONFIG="${BENCH_CONFIG:-$_bench_repo/configs/bench.yaml}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        export HF_MODULES_CACHE="${TMPDIR:-/tmp}/bench_hf_modules_$SLURM_JOB_ID"   # per job: no concurrent writers
    else
        export HF_MODULES_CACHE="$BENCH_ROOT/.hf_modules"
    fi
    echo "bench env: conda=$_bench_conda BENCH_ROOT=$BENCH_ROOT HF_HOME=$HF_HOME config=$BENCH_CONFIG" \
         "offline=${HF_HUB_OFFLINE:-0}"
}

_bench_env_main
