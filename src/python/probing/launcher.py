"""
Layer 1 -- Launcher: generates and submits one SLURM job per WSI split for the
probing pipeline (src/python/probing/experiment.py). Mirrors
classifier_training/launcher.py's structure; the main difference is that there's
no per-trial training-time estimate to compute (a lambda-sweep ridge fit is a
handful of closed-form linear solves, not epoch-based NN training) -- SLURM
`--time` is taken directly from cfg.slurm.time instead.

Usage:
    # Submit all jobs
    python -m src.python.probing.launcher data.model_name=UNI2 data.correction_name=Base

    # Dry run (print SLURM scripts, don't submit)
    python -m src.python.probing.launcher data.model_name=UNI2 data.correction_name=Base dry_run=true

    # Print SLURM log for a specific job (for debugging)
    python -m src.python.probing.launcher data.model_name=UNI2 data.correction_name=Base debug_split=0
"""
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


# Mirrors experiment.py's identically-named helpers -- must stay in sync with how
# experiment.py itself builds output_dir, since that's the path we check for a
# finished results summary before deciding whether a split still needs a job.
def _matching_folder_name(consider_matching: bool) -> str:
    return "matching" if consider_matching else "all_cells"


def _embeddings_folder_name(embeddings_datasets: list[str]) -> str:
    return "_".join(sorted(embeddings_datasets)) if embeddings_datasets else "default"


def _exclude_directive(cfg: DictConfig) -> str:
    exclude = str(OmegaConf.select(cfg, "slurm.exclude", default="") or "").strip()
    return f"#SBATCH --exclude={exclude}" if exclude else "# slurm.exclude not set"


def _path_relevant_data_overrides(cfg: DictConfig) -> list[str]:
    """CLI overrides for every field _split_summary_path's output path depends on
    -- forwarded into every generated SLURM script, same reasoning as
    classifier_training/launcher.py's identically-named helper."""
    consider_matching = bool(OmegaConf.select(cfg, "data.consider_matching", default=True))
    embeddings_datasets = list(cfg.data.embeddings_datasets) if cfg.data.embeddings_datasets else []
    embeddings_literal = "[" + ",".join(embeddings_datasets) + "]"
    return [
        f"mapping={cfg.mapping}",
        f"data.consider_matching={consider_matching}",
        f"data.embeddings_datasets={embeddings_literal}",
    ]


def _split_summary_path(split_idx: int, cfg: DictConfig, repo_root: Path) -> Path:
    outputs_folder_name = str(OmegaConf.select(cfg, "output_dir", default="outputs_probing"))
    split_label = "same_wsi_split" if split_idx == -1 else f"split_{split_idx}"
    consider_matching = bool(OmegaConf.select(cfg, "data.consider_matching", default=True))
    embeddings_datasets = list(cfg.data.embeddings_datasets) if cfg.data.embeddings_datasets else []
    return (
        repo_root / outputs_folder_name / cfg.data.model_name / cfg.data.correction_name
        / cfg.mapping / _matching_folder_name(consider_matching)
        / _embeddings_folder_name(embeddings_datasets) / split_label / "probes_summary.yaml"
    )


def _generate_slurm_script(split_idx: int, cfg: DictConfig, repo_root: Path, log_dir: Path, config_name: str) -> str:
    split_label = "same_wsi_split" if split_idx == -1 else f"split_{split_idx}"
    log_out, log_err = log_dir / f"{split_label}.out", log_dir / f"{split_label}.err"

    overrides = " ".join([
        f"--config-name={config_name}", f"split_idx={split_idx}",
        f"data.model_name={cfg.data.model_name}", f"data.correction_name={cfg.data.correction_name}",
        *_path_relevant_data_overrides(cfg), "hydra.run.dir=.", "hydra.output_subdir=null",
    ])

    if cfg.slurm.partition == "gpu":
        return textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name=probe_{split_label}
            #SBATCH --partition={cfg.slurm.partition}
            #SBATCH --gres={cfg.slurm.gres}
            #SBATCH --mem={cfg.slurm.mem}
            #SBATCH --time={cfg.slurm.time}
            #SBATCH --cpus-per-task={cfg.slurm.cpus_per_task}
            {_exclude_directive(cfg)}
            #SBATCH --output={log_out}
            #SBATCH --error={log_err}

            nvidia-smi
            source ~/.bashrc
            conda activate {cfg.slurm.conda_env}
            cd {repo_root}

            python -m src.python.probing.experiment {overrides}
        """)
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=probe_{split_label}
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

        python -m src.python.probing.experiment {overrides}
    """)


def _generate_slurm_script_sequential(
    split_indices: list[int], cfg: DictConfig, repo_root: Path, log_dir: Path, config_name: str,
) -> str:
    """Single SLURM script that runs every split sequentially on one node/GPU."""
    log_out, log_err = log_dir / "sequential.out", log_dir / "sequential.err"

    run_lines = []
    for split_idx in split_indices:
        split_label = "same_wsi_split" if split_idx == -1 else f"split_{split_idx}"
        overrides = " ".join([
            f"--config-name={config_name}", f"split_idx={split_idx}",
            f"data.model_name={cfg.data.model_name}", f"data.correction_name={cfg.data.correction_name}",
            *_path_relevant_data_overrides(cfg), "hydra.run.dir=.", "hydra.output_subdir=null",
        ])
        run_lines.append(f'echo "[sequential] running {split_label}"')
        run_lines.append(f"python -m src.python.probing.experiment {overrides}")
    runs_block = "\n".join(run_lines)
    runs_block_indented = textwrap.indent(runs_block, " " * 12).lstrip(" ")

    if cfg.slurm.partition == "gpu":
        return textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name=probe_sequential
            #SBATCH --partition={cfg.slurm.partition}
            #SBATCH --gres={cfg.slurm.gres}
            #SBATCH --mem={cfg.slurm.mem}
            #SBATCH --time={cfg.slurm.time}
            #SBATCH --cpus-per-task={cfg.slurm.cpus_per_task}
            {_exclude_directive(cfg)}
            #SBATCH --output={log_out}
            #SBATCH --error={log_err}

            nvidia-smi
            source ~/.bashrc
            conda activate {cfg.slurm.conda_env}
            cd {repo_root}

            {runs_block_indented}
        """)
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=probe_sequential
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


@hydra.main(config_path="../../../configs", config_name="probing_colon", version_base="1.3")
def main(cfg: DictConfig) -> None:
    repo_root = Path(hydra.utils.get_original_cwd())
    config_name: str = HydraConfig.get().job.config_name or "probing_colon"
    log_dir = repo_root / cfg.slurm.output_dir / cfg.data.model_name / cfg.data.correction_name
    log_dir.mkdir(parents=True, exist_ok=True)

    dry_run: bool = bool(OmegaConf.select(cfg, "dry_run", default=False))
    debug_split: Optional[int] = OmegaConf.select(cfg, "debug_split", default=None)
    sequential: bool = bool(OmegaConf.select(cfg, "slurm.sequential", default=False))

    if debug_split is not None:
        split_label = "same_wsi_split" if debug_split == -1 else f"split_{debug_split}"
        for suffix in (".out", ".err"):
            log_path = log_dir / f"{split_label}{suffix}"
            label = "STDOUT" if suffix == ".out" else "STDERR"
            print(f"\n{'='*60}\n{label}: {log_path}\n{'='*60}")
            print(log_path.read_text() if log_path.exists() else f"  (file not found: {log_path})")
        return

    split_indices = list(range(len(cfg.splits)))
    if cfg.same_wsi_split:
        split_indices.append(-1)

    mode = "DRY RUN — " if dry_run else ""

    # Skip any split whose probes_summary.yaml already exists -- lets you re-run
    # the launcher after a partial sweep without resubmitting finished splits.
    pending_indices = []
    for split_idx in split_indices:
        split_label = "same_wsi_split" if split_idx == -1 else f"split_{split_idx}"
        if _split_summary_path(split_idx, cfg, repo_root).exists():
            print(f"[launcher] Skipping {split_label}: probes_summary.yaml already exists")
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
            result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[launcher] Submitted {split_label}: {result.stdout.strip()}")
            else:
                print(f"[launcher] ERROR submitting {split_label}: {result.stderr.strip()}", file=sys.stderr)


if __name__ == "__main__":
    main()
