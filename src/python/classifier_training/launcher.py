"""
Layer 1 — Launcher: generates and submits one SLURM job per WSI split.

Usage:
    # Submit all jobs
    python -m src.python.classifier_training.launcher \
        data.model_name=virchow_v2 data.correction_name=scanorama

    # Dry run (print SLURM scripts, don't submit)
    python -m src.python.classifier_training.launcher \
        data.model_name=virchow_v2 data.correction_name=scanorama dry_run=true

    # Print SLURM log for a specific job (for debugging)
    python -m src.python.classifier_training.launcher \
        data.model_name=virchow_v2 data.correction_name=scanorama \
        debug_split=0
"""
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

EPOCH_LENGTH = 0.5

def _estimate_total_time(cfg: DictConfig, one_epoch_time: float):
    num_epochs = cfg.training.num_epochs
    params_combinations_number = 1
    for _, confs in cfg.hparams.items():
        if confs['tune']:
            params_combinations_number *= len(confs['values'])

    return one_epoch_time * num_epochs * params_combinations_number


def _estimate_total_time_grid(cfg: DictConfig, one_epoch_time: float) -> float:
    """Grid experiment runs 4 checkerboard offsets, each with the full grid search."""
    return 4 * _estimate_total_time(cfg, one_epoch_time)


def _exclude_directive(cfg: DictConfig) -> str:
    """SBATCH line steering around node(s) named in slurm.exclude (comma-separated),
    or a harmless comment if unset -- lets a config avoid a known-bad node (e.g. a
    GPU stuck in the ERR! state reported by nvidia-smi) without hand-editing every
    generated script.
    """
    exclude = str(OmegaConf.select(cfg, "slurm.exclude", default="") or "").strip()
    return f"#SBATCH --exclude={exclude}" if exclude else "# slurm.exclude not set"


def _minutes_to_hhmmss(minutes: float) -> str:
    total_seconds = int(minutes * 60)
    hours, remainder = divmod(total_seconds, 3600)
    mins, secs = divmod(remainder, 60)
    return f"{hours:02}:{mins:02}:{secs:02}"


# Mirrors experiment.py's identically-named helpers -- must stay in sync with how
# experiment.py itself builds output_dir, since that's the path we check for a
# finished results_summary.yaml before deciding whether a split still needs a job.
def _matching_folder_name(consider_matching: bool) -> str:
    return "matching" if consider_matching else "all_cells"


def _embeddings_folder_name(embeddings_datasets: list[str]) -> str:
    return "_".join(sorted(embeddings_datasets)) if embeddings_datasets else "default"


def _path_relevant_data_overrides(cfg: DictConfig) -> list[str]:
    """CLI overrides for every field _split_results_summary_path's output path depends
    on (mapping, data.consider_matching, data.embeddings_datasets). Must be forwarded
    into every generated SLURM script -- otherwise a launcher-level override changes
    where *this process* looks for results_summary.yaml without changing where the
    submitted job actually writes them (the subprocess only ever inherits
    data.model_name/data.correction_name plus whatever config-name's file has baked
    in), silently desyncing the skip-check from reality.
    """
    consider_matching = bool(OmegaConf.select(cfg, "data.consider_matching", default=True))
    embeddings_datasets = list(cfg.data.embeddings_datasets) if cfg.data.embeddings_datasets else []
    embeddings_literal = "[" + ",".join(embeddings_datasets) + "]"
    return [
        f"mapping={cfg.mapping}",
        f"data.consider_matching={consider_matching}",
        f"data.embeddings_datasets={embeddings_literal}",
    ]


def _split_results_summary_path(split_idx: int, cfg: DictConfig, repo_root: Path) -> Path:
    outputs_folder_name = str(OmegaConf.select(cfg, "output_dir", default="outputs"))
    split_label = "same_wsi_split" if split_idx == -1 else f"split_{split_idx}"
    consider_matching = bool(OmegaConf.select(cfg, "data.consider_matching", default=True))
    embeddings_datasets = list(cfg.data.embeddings_datasets) if cfg.data.embeddings_datasets else []
    return (
        repo_root / outputs_folder_name
        / cfg.data.model_name / cfg.data.correction_name
        / cfg.mapping
        / _matching_folder_name(consider_matching)
        / _embeddings_folder_name(embeddings_datasets)
        / split_label
        / "results_summary.yaml"
    )


def _generate_slurm_script(
    split_idx: int,
    cfg: DictConfig,
    repo_root: Path,
    log_dir: Path,
    config_name: str,
) -> str:
    split_label = "same_wsi_split" if split_idx == -1 else f"split_{split_idx}"
    log_out = log_dir / f"{split_label}.out"
    log_err = log_dir / f"{split_label}.err"

    overrides = " ".join([
        f"--config-name={config_name}",
        f"split_idx={split_idx}",
        f"data.model_name={cfg.data.model_name}",
        f"data.correction_name={cfg.data.correction_name}",
        *_path_relevant_data_overrides(cfg),
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
    ])
    
    if cfg.slurm.partition == 'gpu':

        return textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name=clf_{split_label}
            #SBATCH --partition={cfg.slurm.partition}
            #SBATCH --gres={cfg.slurm.gres}
            #SBATCH --mem={cfg.slurm.mem}
            #SBATCH --time={cfg.slurm.time}
            #SBATCH --cpus-per-task={cfg.slurm.cpus_per_task}
            {_exclude_directive(cfg)}
            #SBATCH --output={log_out}
            #SBATCH --error={log_err}
            
            nvidia-smi
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
            source ~/.bashrc
            conda activate {cfg.slurm.conda_env}
            cd {repo_root}

            python -m src.python.classifier_training.experiment {overrides}
        """)
    
    else:

        return textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name=clf_{split_label}
            #SBATCH --partition={cfg.slurm.partition}
            #SBATCH --mem={cfg.slurm.mem}
            #SBATCH --time={cfg.slurm.time}
            #SBATCH --cpus-per-task={cfg.slurm.cpus_per_task}
            {_exclude_directive(cfg)}
            #SBATCH --output={log_out}
            #SBATCH --error={log_err}
            
            source ~/.bashrc
            conda activate {cfg.slurm.conda_env}
            cd {repo_root}

            python -m src.python.classifier_training.experiment {overrides}
        """)


def _generate_slurm_script_sequential(
    split_indices: list[int],
    cfg: DictConfig,
    repo_root: Path,
    log_dir: Path,
    config_name: str,
) -> str:
    """Single SLURM script that runs every split sequentially on one GPU."""
    log_out = log_dir / "sequential.out"
    log_err = log_dir / "sequential.err"

    run_lines = []
    for split_idx in split_indices:
        split_label = "same_wsi_split" if split_idx == -1 else f"split_{split_idx}"
        overrides = " ".join([
            f"--config-name={config_name}",
            f"split_idx={split_idx}",
            f"data.model_name={cfg.data.model_name}",
            f"data.correction_name={cfg.data.correction_name}",
            *_path_relevant_data_overrides(cfg),
            "hydra.run.dir=.",
            "hydra.output_subdir=null",
        ])
        run_lines.append(f'echo "[sequential] running {split_label}"')
        run_lines.append(f"python -m src.python.classifier_training.experiment {overrides}")
    runs_block = "\n".join(run_lines)
    # textwrap.dedent needs every line to share the same leading whitespace as
    # the rest of the template, so re-indent the interpolated block to match
    # before substitution (its own internal newlines would otherwise have no
    # indentation and break dedent's common-prefix detection).
    runs_block_indented = textwrap.indent(runs_block, " " * 12).lstrip(" ")

    if cfg.slurm.partition == 'gpu':
        return textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name=clf_sequential
            #SBATCH --partition={cfg.slurm.partition}
            #SBATCH --gres={cfg.slurm.gres}
            #SBATCH --mem={cfg.slurm.mem}
            #SBATCH --time={cfg.slurm.time}
            #SBATCH --cpus-per-task={cfg.slurm.cpus_per_task}
            {_exclude_directive(cfg)}
            #SBATCH --output={log_out}
            #SBATCH --error={log_err}

            nvidia-smi
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
            source ~/.bashrc
            conda activate {cfg.slurm.conda_env}
            cd {repo_root}

            {runs_block_indented}
        """)
    else:
        return textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name=clf_sequential
            #SBATCH --partition={cfg.slurm.partition}
            #SBATCH --mem={cfg.slurm.mem}
            #SBATCH --time={cfg.slurm.time}
            #SBATCH --cpus-per-task={cfg.slurm.cpus_per_task}
            {_exclude_directive(cfg)}
            #SBATCH --output={log_out}
            #SBATCH --error={log_err}

            source ~/.bashrc
            conda activate {cfg.slurm.conda_env}
            cd {repo_root}

            {runs_block_indented}
        """)


def _generate_slurm_script_grid(
    cfg: DictConfig,
    repo_root: Path,
    log_dir: Path,
    config_name: str,
) -> str:
    """SLURM script for grid_experiment.py."""
    log_out = log_dir / "grid_patch_split.out"
    log_err = log_dir / "grid_patch_split.err"

    overrides = " ".join([
        f"--config-name={config_name}",
        f"data.model_name={cfg.data.model_name}",
        f"data.correction_name={cfg.data.correction_name}",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
    ])

    if cfg.slurm.partition == "gpu":
        return textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name=clf_grid
            #SBATCH --partition={cfg.slurm.partition}
            #SBATCH --gres={cfg.slurm.gres}
            #SBATCH --mem={cfg.slurm.mem}
            #SBATCH --time={cfg.slurm.time}
            #SBATCH --cpus-per-task={cfg.slurm.cpus_per_task}
            {_exclude_directive(cfg)}
            #SBATCH --output={log_out}
            #SBATCH --error={log_err}

            nvidia-smi
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
            source ~/.bashrc
            conda activate {cfg.slurm.conda_env}
            cd {repo_root}

            python -m src.python.classifier_training.grid_experiment {overrides}
        """)
    else:
        return textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name=clf_grid
            #SBATCH --partition={cfg.slurm.partition}
            #SBATCH --mem={cfg.slurm.mem}
            #SBATCH --time={cfg.slurm.time}
            #SBATCH --cpus-per-task={cfg.slurm.cpus_per_task}
            {_exclude_directive(cfg)}
            #SBATCH --output={log_out}
            #SBATCH --error={log_err}

            source ~/.bashrc
            conda activate {cfg.slurm.conda_env}
            cd {repo_root}

            python -m src.python.classifier_training.grid_experiment {overrides}
        """)


@hydra.main(config_path="../../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    repo_root = Path(hydra.utils.get_original_cwd())
    # The generated SLURM scripts invoke experiment.py/grid_experiment.py as
    # fresh processes -- they must be told which config this launcher itself
    # was invoked with (its @hydra.main default is 'train'), otherwise they
    # silently fall back to configs/train.yaml regardless of --config-name
    # passed here.
    config_name: str = HydraConfig.get().job.config_name or "train"
    log_dir = (
        repo_root
        / cfg.slurm.output_dir
        / cfg.data.model_name
        / cfg.data.correction_name
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    dry_run: bool = bool(OmegaConf.select(cfg, "dry_run", default=False))
    debug_split: Optional[int] = OmegaConf.select(cfg, "debug_split", default=None)
    grid_mode: bool = bool(OmegaConf.select(cfg, "patch_split.enabled", default=False))
    sequential: bool = bool(OmegaConf.select(cfg, "slurm.sequential", default=False))

    if grid_mode:
        total_time_minutes = _estimate_total_time_grid(cfg, EPOCH_LENGTH)
    else:
        total_time_minutes = _estimate_total_time(cfg, EPOCH_LENGTH)

    if sequential and not grid_mode:
        n_splits = len(cfg.splits) + (1 if cfg.same_wsi_split else 0)
        total_time_minutes *= n_splits

    cfg.slurm.time = _minutes_to_hhmmss(total_time_minutes)

    print(f'Requesting {cfg.slurm.time}')

    # ── debug mode: print captured output of an existing SLURM job ───────────
    if debug_split is not None:
        split_label = "same_wsi_split" if debug_split == -1 else f"split_{debug_split}"
        for suffix in (".out", ".err"):
            log_path = log_dir / f"{split_label}{suffix}"
            label = "STDOUT" if suffix == ".out" else "STDERR"
            print(f"\n{'='*60}\n{label}: {log_path}\n{'='*60}")
            if log_path.exists():
                print(log_path.read_text())
            else:
                print(f"  (file not found: {log_path})")
        return

    # ── grid experiment mode ──────────────────────────────────────────────────
    if grid_mode:
        mode_prefix = "DRY RUN — " if dry_run else ""
        print(f"[launcher] {mode_prefix}Grid patch split: 1 job for "
              f"{cfg.data.model_name}/{cfg.data.correction_name}")
        script = _generate_slurm_script_grid(cfg, repo_root, log_dir, config_name)
        script_path = log_dir / "job_grid_patch_split.sh"
        script_path.write_text(script)
        if dry_run:
            print(f"\n{'─'*60}\n[launcher] Script → {script_path}\n{script}")
        else:
            result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[launcher] Submitted grid_patch_split: {result.stdout.strip()}")
            else:
                print(f"[launcher] ERROR: {result.stderr.strip()}", file=sys.stderr)
        return

    # ── standard experiment mode ──────────────────────────────────────────────
    split_indices = list(range(len(cfg.splits)))
    if cfg.same_wsi_split:
        split_indices.append(-1)

    mode = "DRY RUN — " if dry_run else ""

    # Skip any split whose results_summary.yaml already exists -- lets you re-run the
    # launcher after a partial sweep (failed/never-submitted splits) without resubmitting
    # or overwriting splits that already finished.
    pending_indices = []
    for split_idx in split_indices:
        split_label = "same_wsi_split" if split_idx == -1 else f"split_{split_idx}"
        if _split_results_summary_path(split_idx, cfg, repo_root).exists():
            print(f"[launcher] Skipping {split_label}: results_summary.yaml already exists")
        else:
            pending_indices.append(split_idx)

    if sequential:
        if not pending_indices:
            print(f"[launcher] All {len(split_indices)} split(s) already have results — nothing to submit")
            return
        print(f"[launcher] {mode}Preparing 1 sequential job ({len(pending_indices)} of "
              f"{len(split_indices)} splits) for {cfg.data.model_name}/{cfg.data.correction_name}")
        script = _generate_slurm_script_sequential(pending_indices, cfg, repo_root, log_dir, config_name)
        script_path = log_dir / "job_sequential.sh"
        script_path.write_text(script)

        if dry_run:
            print(f"\n{'─'*60}\n[launcher] Script → {script_path}\n{script}")
        else:
            result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[launcher] Submitted sequential job: {result.stdout.strip()}")
            else:
                print(f"[launcher] ERROR: {result.stderr.strip()}", file=sys.stderr)
        return

    print(f"[launcher] {mode}Preparing {len(pending_indices)} of {len(split_indices)} job(s) for "
          f"{cfg.data.model_name}/{cfg.data.correction_name}")

    for split_idx in pending_indices:
        split_label = "same_wsi_split" if split_idx == -1 else f"split_{split_idx}"
        script = _generate_slurm_script(split_idx, cfg, repo_root, log_dir, config_name)
        script_path = log_dir / f"job_{split_label}.sh"
        script_path.write_text(script)

        if dry_run:
            print(f"\n{'─'*60}\n[launcher] Script for {split_label} → {script_path}\n{script}")
        else:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"[launcher] Submitted {split_label}: {result.stdout.strip()}")
            else:
                print(
                    f"[launcher] ERROR submitting {split_label}: {result.stderr.strip()}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
