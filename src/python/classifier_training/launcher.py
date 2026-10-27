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
from typing import Callable, Optional

import h5py
import numpy as np
import yaml
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# ── SLURM wall-time estimate ──────────────────────────────────────────────────
# Each job's --time is estimated from the cells its split actually holds (read from
# the embeddings h5 headers, no data loaded), so a cross-cohort fold with millions of
# train cells doesn't get the same request as a 7-slide LWO split:
#   minutes = overhead + safety * (load + n_trials * (epochs * (train + val) + test) * s_per_cell_epoch) / 60
#   load    = load_seconds_per_cell_dataset * n_embedding_datasets * cells read
# epochs is training.num_epochs (an upper bound under early stopping). Values below are
# defaults for any key missing from the config's slurm.time_estimate.
_TIME_ESTIMATE_DEFAULTS = {
    # Derived from the old fixed 2 min/epoch, set for ~0.8M-cell train+val sets.
    # experiment.py records the measured value in results_summary.yaml's `timing`.
    "seconds_per_cell_epoch": 1.5e-4,
    "load_seconds_per_cell_dataset": 3.0e-5,
    "overhead_minutes": 15,
    "safety_factor": 1.5,
    "max_time": "120:00:00",
}
# Used only when the h5 files can't be read (DATASETS_ROOT unset / not mounted).
FALLBACK_EPOCH_MINUTES = 2


def _n_trials(cfg: DictConfig) -> int:
    n = 1
    for _, confs in cfg.hparams.items():
        if confs['tune']:
            n *= len(confs['values'])
    return n


def _fallback_minutes(cfg: DictConfig) -> float:
    """The old size-blind estimate: a fixed duration per epoch per trial."""
    return FALLBACK_EPOCH_MINUTES * cfg.training.num_epochs * _n_trials(cfg)


def _time_estimate_params(cfg: DictConfig) -> dict:
    configured = OmegaConf.select(cfg, "slurm.time_estimate", default=None)
    return {**_TIME_ESTIMATE_DEFAULTS, **(OmegaConf.to_container(configured, resolve=True) if configured else {})}


def _hhmmss_to_minutes(time_str: str) -> float:
    """SLURM time ("[D-]H:M:S", "M:S" or "M") -> minutes."""
    days, _, rest = str(time_str).rpartition("-")
    parts = [int(p) for p in rest.split(":")]
    if len(parts) == 1:
        return int(days or 0) * 1440 + parts[0]
    parts = [0] * (3 - len(parts)) + parts
    return int(days or 0) * 1440 + parts[0] * 60 + parts[1] + parts[2] / 60


def _cell_counter(cfg: DictConfig) -> Callable[[str], int]:
    """WSI name -> number of cells in its embeddings_dataset.h5 (header only, cached).
    Raises if the data root can't be resolved or a file can't be opened."""
    base = Path(str(cfg.data.base_dir)) / f"{cfg.data.model_name}_h5" / cfg.data.correction_name
    cache: dict[str, int] = {}

    def count(wsi: str) -> int:
        if wsi not in cache:
            with h5py.File(base / wsi / "embeddings_dataset.h5", "r") as f:
                cache[wsi] = int(f["cell_labels"].shape[0])
        return cache[wsi]
    return count


def _side_props(side_cfg) -> dict[str, float]:
    return {k: float(v) for k, v in side_cfg.items() if k != "cell_type_proportions"} if side_cfg is not None else {}


def _split_wsis(cfg: DictConfig, split_idx: int) -> set[str]:
    """Every WSI a split's job loads (same_wsi_split: every WSI of every split)."""
    splits = cfg.splits if split_idx == -1 else [cfg.splits[split_idx]]
    return {w for sp in splits for side in ("train", "val", "test") for w in _side_props(OmegaConf.select(sp, side))}


def _missing_embeddings(cfg: DictConfig, wsis: set[str]) -> Optional[list[str]]:
    """WSIs without an embeddings_dataset.h5 under the data root; None if the root
    can't be resolved (e.g. DATASETS_ROOT unset), i.e. nothing could be checked."""
    try:
        base = Path(str(cfg.data.base_dir)) / f"{cfg.data.model_name}_h5" / cfg.data.correction_name
    except Exception:
        return None
    return sorted(w for w in wsis if not (base / w / "embeddings_dataset.h5").exists())


def _used_cells(cfg: DictConfig, count: Callable[[str], int], wsi: str) -> int:
    """Cells of a WSI that reach training: all of them, or data.max_cells_per_wsi."""
    cap = OmegaConf.select(cfg, "data.max_cells_per_wsi", default=None)
    return count(wsi) if cap is None else min(count(wsi), int(cap))


def _split_cell_counts(split_idx: int, cfg: DictConfig, count: Callable[[str], int]) -> dict[str, float]:
    """Cells read from disk ('read', whole files -- the per-WSI cap subsamples while
    reading, so IO stays full) and used per side after the cap and per-WSI proportions.
    Upper bounds: 'Unknown' cells and consider_matching aren't subtracted."""
    if split_idx == -1:
        props: dict[str, float] = {}
        for sp in cfg.splits:
            for side in ("train", "val", "test"):
                for wsi, p in _side_props(OmegaConf.select(sp, side)).items():
                    props[wsi] = max(props.get(wsi, 0.0), p)
        used = sum(_used_cells(cfg, count, w) * p for w, p in props.items())
        return {"read": sum(count(w) for w in props), "train": 0.7 * used, "val": 0.15 * used, "test": 0.15 * used}
    sp = cfg.splits[split_idx]
    sides = {side: _side_props(OmegaConf.select(sp, side)) for side in ("train", "val", "test")}
    counts = {side: sum(_used_cells(cfg, count, w) * p for w, p in props.items()) for side, props in sides.items()}
    counts["read"] = sum(count(w) for props in sides.values() for w in props)
    return counts


def _grid_cell_counts(cfg: DictConfig, count: Callable[[str], int]) -> dict[str, float]:
    """grid_experiment.py: every patch_split WSI is read once; all used cells counted
    as train (an upper bound -- each offset trains on part and tests on the rest)."""
    props = {k: float(v) for k, v in cfg.patch_split.wsis.items()}
    return {
        "read": sum(count(w) for w in props),
        # grid_experiment.py's loader (load_data_with_coords) ignores data.max_cells_per_wsi.
        "train": sum(count(w) * p for w, p in props.items()),
        "val": 0.0,
        "test": 0.0,
    }


def _estimate_minutes(cfg: DictConfig, counts: dict[str, float], n_grid_repeats: int = 1) -> float:
    p = _time_estimate_params(cfg)
    n_datasets = max(1, len(cfg.data.embeddings_datasets or []))
    load_s = p["load_seconds_per_cell_dataset"] * n_datasets * counts["read"]
    cell_epochs = cfg.training.num_epochs * (counts["train"] + counts["val"]) + counts["test"]
    train_s = n_grid_repeats * _n_trials(cfg) * cell_epochs * p["seconds_per_cell_epoch"]
    return p["overhead_minutes"] + p["safety_factor"] * (load_s + train_s) / 60


def _clip_minutes(cfg: DictConfig, minutes: float, label: str) -> float:
    max_minutes = _hhmmss_to_minutes(_time_estimate_params(cfg)["max_time"])
    if minutes > max_minutes:
        print(
            f"[launcher] WARNING {label}: estimated {minutes / 60:.1f}h exceeds "
            f"slurm.time_estimate.max_time ({max_minutes / 60:.0f}h) -- requesting the max; the job "
            f"may be killed before finishing. Reduce cells (WSI proportions), trials or num_epochs."
        )
    return min(minutes, max_minutes)


def _recorded_seconds_per_cell_epoch(results_root: Path) -> list[float]:
    """seconds_per_cell_epoch that finished runs under results_root measured
    (results_summary.yaml's `timing`, see experiment.py) -- for calibrating the estimate."""
    values = []
    for path in results_root.glob("**/results_summary.yaml"):
        try:
            with open(path) as f:
                timing = (yaml.safe_load(f) or {}).get("timing") or {}
        except yaml.YAMLError:  # e.g. an older file with python-specific tags
            continue
        if timing.get("seconds_per_cell_epoch"):
            values.append(float(timing["seconds_per_cell_epoch"]))
    return values


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


def _path_relevant_overrides(cfg: DictConfig) -> list[str]:
    """CLI overrides for every field _split_results_summary_path's output path depends
    on (experiment, mapping, data.consider_matching, data.embeddings_datasets). Must be
    forwarded into every generated SLURM script -- otherwise a launcher-level override
    changes where *this process* looks for results_summary.yaml without changing where
    the submitted job actually writes them (the subprocess only ever inherits
    data.model_name/data.correction_name plus whatever config-name's file has baked
    in), silently desyncing the skip-check from reality.
    """
    consider_matching = bool(OmegaConf.select(cfg, "data.consider_matching", default=True))
    embeddings_datasets = list(cfg.data.embeddings_datasets) if cfg.data.embeddings_datasets else []
    embeddings_literal = "[" + ",".join(embeddings_datasets) + "]"
    overrides = [
        f"mapping={cfg.mapping}",
        f"data.consider_matching={consider_matching}",
        f"data.embeddings_datasets={embeddings_literal}",
    ]
    # Only configs that declare `experiment` accept an override of it (Hydra rejects
    # overriding a key the config doesn't have).
    if "experiment" in cfg:
        overrides.append(f"experiment={cfg.experiment if cfg.experiment else 'null'}")
    return overrides


def _split_results_summary_path(split_idx: int, cfg: DictConfig, repo_root: Path) -> Path:
    outputs_folder_name = str(OmegaConf.select(cfg, "output_dir", default="outputs"))
    split_label = "same_wsi_split" if split_idx == -1 else f"split_{split_idx}"
    consider_matching = bool(OmegaConf.select(cfg, "data.consider_matching", default=True))
    embeddings_datasets = list(cfg.data.embeddings_datasets) if cfg.data.embeddings_datasets else []
    experiment = OmegaConf.select(cfg, "experiment", default=None)
    return (
        repo_root / outputs_folder_name
        / (experiment or "")
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
        *_path_relevant_overrides(cfg),
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
            *_path_relevant_overrides(cfg),
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
        / (OmegaConf.select(cfg, "experiment", default=None) or "")
        / cfg.data.model_name
        / cfg.data.correction_name
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    dry_run: bool = bool(OmegaConf.select(cfg, "dry_run", default=False))
    debug_split: Optional[int] = OmegaConf.select(cfg, "debug_split", default=None)
    grid_mode: bool = bool(OmegaConf.select(cfg, "patch_split.enabled", default=False))
    sequential: bool = bool(OmegaConf.select(cfg, "slurm.sequential", default=False))

    # ── wall time: slurm.time as given (auto_time: false), else estimated per job
    # from cell counts (see _estimate_minutes); minutes_for() returns None when the
    # h5 headers can't be read, in which case the old size-blind estimate is used.
    auto_time: bool = bool(OmegaConf.select(cfg, "slurm.auto_time", default=True))
    count: Optional[Callable[[str], int]] = None
    if auto_time:
        try:
            count = _cell_counter(cfg)
        except Exception as e:  # DATASETS_ROOT unset -> interpolation error
            print(f"[launcher] WARNING: can't resolve the embeddings root ({e}); "
                  f"falling back to {FALLBACK_EPOCH_MINUTES} min/epoch/trial for --time")

    def _job_time(
        label: str,
        counts_of: Callable[[Callable[[str], int]], list[dict[str, float]]],
        n_fallback_units: int,
        n_grid_repeats: int = 1,
    ) -> str:
        """--time for one job; counts_of(count) gives the cell counts of each unit
        (split) the job runs, summed here."""
        if not auto_time:
            return str(cfg.slurm.time)
        minutes = None
        if count is not None:
            try:
                total = {"read": 0.0, "train": 0.0, "val": 0.0, "test": 0.0}
                for unit in counts_of(count):
                    for k, v in unit.items():
                        total[k] += v
                minutes = _estimate_minutes(cfg, total, n_grid_repeats)
                print(f"[launcher] {label}: train={total['train']:,.0f} val={total['val']:,.0f} "
                      f"test={total['test']:,.0f} cells, {_n_trials(cfg)} trial(s) x "
                      f"<={cfg.training.num_epochs} epochs -> {minutes / 60:.1f}h")
            except OSError as e:
                print(f"[launcher] WARNING {label}: can't read cell counts ({e}); using the fallback estimate")
        if minutes is None:
            minutes = n_grid_repeats * n_fallback_units * _fallback_minutes(cfg)
        return _minutes_to_hhmmss(_clip_minutes(cfg, minutes, label))

    if not auto_time:
        print(f"[launcher] Requesting {cfg.slurm.time} per job (slurm.auto_time: false)")
    elif not grid_mode and len(cfg.splits):
        # .../{model}/{correction}/{mapping}/{matching}/{embeddings}/split_0/results_summary.yaml
        results_root = _split_results_summary_path(0, cfg, repo_root).parents[4]
        recorded = _recorded_seconds_per_cell_epoch(results_root) if results_root.exists() else []
        if recorded:
            print(f"[launcher] {len(recorded)} finished run(s) here measured a median of "
                  f"{np.median(recorded):.2e} s/cell-epoch (slurm.time_estimate uses "
                  f"{_time_estimate_params(cfg)['seconds_per_cell_epoch']:.2e})")

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
        # 4 checkerboard offsets, each running the full grid search.
        cfg.slurm.time = _job_time(
            "grid_patch_split", lambda c: [_grid_cell_counts(cfg, c)], 1, n_grid_repeats=4
        )
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

    # A job whose WSI has no embeddings file would only crash after queueing -- list
    # every missing one up front and submit nothing (dry runs just report them).
    missing = _missing_embeddings(cfg, {w for idx in pending_indices for w in _split_wsis(cfg, idx)})
    if missing is None:
        print("[launcher] WARNING: can't resolve the embeddings root -- not checking that every WSI has embeddings")
    elif missing:
        print(f"[launcher] ERROR: {len(missing)} WSI(s) have no embeddings_dataset.h5 under "
              f"{cfg.data.model_name}_h5/{cfg.data.correction_name}: {missing}", file=sys.stderr)
        if not dry_run:
            sys.exit(1)

    if sequential:
        if not pending_indices:
            print(f"[launcher] All {len(split_indices)} split(s) already have results — nothing to submit")
            return
        print(f"[launcher] {mode}Preparing 1 sequential job ({len(pending_indices)} of "
              f"{len(split_indices)} splits) for {cfg.data.model_name}/{cfg.data.correction_name}")
        cfg.slurm.time = _job_time(
            "sequential", lambda c: [_split_cell_counts(i, cfg, c) for i in pending_indices], len(pending_indices)
        )
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
        cfg.slurm.time = _job_time(split_label, lambda c: [_split_cell_counts(split_idx, cfg, c)], 1)
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
