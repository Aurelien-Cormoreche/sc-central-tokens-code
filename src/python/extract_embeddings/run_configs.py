"""Run a list of extraction JSON configs one after another.

Each config is a separate `python -m src.python.extract_embeddings.extract_embeddings
--config <json>` subprocess, so GPU memory is fully released between configs and a
crash in one config doesn't take the remaining ones down with it.

Usage (from the repo root):
    python -m src.python.extract_embeddings.run_configs configs/extract_embeddings/uni2_224.json
    python -m src.python.extract_embeddings.run_configs configs/extract_embeddings/        # every *.json, sorted
    python -m src.python.extract_embeddings.run_configs a.json b.json --dry-run
    sbatch scripts/extract_embeddings.sh a.json b.json                                     # same, on a GPU node
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

_EXIT_MEANING = {
    0: "ok",
    2: "ok, but some slides were skipped for missing input files",
    3: "some slides FAILED during extraction (others completed)",
}


def _expand(paths: list[str]) -> list[Path]:
    configs = []
    for p in map(Path, paths):
        if p.is_dir():
            configs.extend(sorted(p.glob('*.json')))
        elif p.is_file():
            configs.append(p)
        else:
            raise FileNotFoundError(f"config not found: {p}")
    return configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run extraction JSON configs sequentially.")
    parser.add_argument('configs', nargs='+', help="JSON run configs and/or directories of them")
    parser.add_argument('--dry-run', action='store_true', help="forwarded to every config (no model is loaded)")
    parser.add_argument('--stop-on-failure', action='store_true',
                        help="stop at the first config that exits with an error instead of running the rest")
    args = parser.parse_args(argv)

    configs = _expand(args.configs)
    print(f"{len(configs)} config(s) to run:")
    for c in configs:
        print(f"  {c}")

    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n######## [{i}/{len(configs)}] {config} ########", flush=True)
        cmd = [sys.executable, '-m', 'src.python.extract_embeddings.extract_embeddings', '--config', str(config)]
        if args.dry_run:
            cmd.append('--dry-run')
        t0 = time.time()
        returncode = subprocess.run(cmd).returncode
        results.append((config, returncode, time.time() - t0))
        if args.stop_on_failure and returncode != 0 and returncode != 2:
            print(f"--stop-on-failure: stopping after {config} (exit code {returncode})")
            break

    print("\n######## summary ########")
    for config, returncode, elapsed in results:
        meaning = _EXIT_MEANING.get(returncode, f"crashed / config error (exit code {returncode})")
        print(f"  {elapsed / 3600:6.2f} h  {meaning:<60} {config}")
    for config in configs[len(results):]:
        print(f"  {'':>8}  {'not run':<60} {config}")
    return 0 if all(rc == 0 for _, rc, _ in results) and len(results) == len(configs) else 1


if __name__ == '__main__':
    sys.exit(main())
