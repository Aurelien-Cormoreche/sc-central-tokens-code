"""
Grid-based spatial split experiment.

Avoids embedding leakage by selecting non-adjacent cells via a checkerboard
patch grid: every other row AND every other column → 1/4 of all patches are
selected; these patches are pairwise non-adjacent, so the foundation model's
receptive field cannot span both train and test.

All 4 checkerboard offsets (row_off, col_off) ∈ {0,1}² are evaluated in a
single run, giving a 4-fold spatial cross-validation.

The WSIs to use are listed under cfg.patch_split.wsis.  Using a single WSI
is equivalent to a within-WSI split; using multiple WSIs is equivalent to an
across-WSI split — the splitting logic is identical either way.

Usage:
    python -m src.python.classifier_training.grid_experiment \\
        data.model_name=virchow_v2 data.correction_name=scanorama
"""

import itertools
import os
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

import hydra

from src.python.classifier_training.trainer import TrainConfig, train_and_evaluate
from src.python.code_configs.mappings import mapping_factory
from src.python.utils.loading_functions import load_data_with_coords


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_label_encoder(mapping: dict) -> tuple[list[str], dict[str, int]]:
    """"Unknown" cells are excluded entirely -- see _load_wsi_data_with_coords."""
    class_names = sorted(set(mapping.values()) - {"Unknown"})
    label_enc = {name: i for i, name in enumerate(class_names)}
    return class_names, label_enc


def _build_grid(hparams_cfg: DictConfig) -> list[dict[str, Any]]:
    param_lists: dict[str, list] = {}
    for name, spec in hparams_cfg.items():
        if spec["tune"]:
            param_lists[name] = list(OmegaConf.to_container(spec["values"], resolve=True))
        else:
            values = OmegaConf.to_container(spec["values"], resolve=True)
            param_lists[name] = [values[0] if isinstance(values, list) else values]
    keys = list(param_lists.keys())
    combos = list(itertools.product(*[param_lists[k] for k in keys]))
    return [dict(zip(keys, combo)) for combo in combos]


def _class_counts(labels: np.ndarray, num_classes: int, class_names: list[str]) -> dict[str, int]:
    counts = np.bincount(labels, minlength=num_classes)
    return {class_names[i]: int(counts[i]) for i in range(num_classes)}


def _matching_folder_name(consider_matching: bool) -> str:
    return "matching" if consider_matching else "all_cells"


def _embeddings_folder_name(embeddings_datasets: list[str]) -> str:
    return "_".join(sorted(embeddings_datasets)) if embeddings_datasets else "default"


def _get_patch_mask(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    patch_size: float,
    row_off: int,
    col_off: int,
) -> np.ndarray:
    """Boolean mask: True for cells whose patch has parity (row_off, col_off)."""
    row_parity = np.floor(y_coords / patch_size).astype(int) % 2
    col_parity = np.floor(x_coords / patch_size).astype(int) % 2
    return (row_parity == row_off) & (col_parity == col_off)


def _load_wsi_data_with_coords(
    wsi_proportions: dict[str, float],
    base_path: str,
    embeddings_datasets: list[str],
    mapping: dict,
    label_enc: dict[str, int],
    rng: np.random.Generator,
    consider_matching: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings + centroid coordinates for a set of WSIs.

    Returns all arrays as numpy so downstream masking/splitting is uniform:
      embeddings : (N, D) float32
      labels_int : (N,) int64
      wsi_orig   : (N,) int64 — WSI index per cell
      x_coords   : (N,) float
      y_coords   : (N,) float
    """
    wsi_names = list(wsi_proportions.keys())
    embeddings, cell_labels, _, wsi_origin, _, wsi_idx_map, x_coords, y_coords = load_data_with_coords(
        base_path, embeddings_datasets, wsi_names, mapping, consider_matching=consider_matching
    )

    # Drop "Unknown" cells before any proportion sampling, so e.g. proportion=1.0
    # keeps 100% of the WSI's typed cells rather than 100% of everything.
    known_mask = cell_labels != "Unknown"
    embeddings = embeddings[known_mask]
    cell_labels = cell_labels[known_mask]
    wsi_origin = wsi_origin[known_mask]
    x_coords = x_coords[known_mask]
    y_coords = y_coords[known_mask]

    keep_indices: list[np.ndarray] = []
    for wsi_idx, wsi_name in enumerate(wsi_idx_map):
        proportion = float(wsi_proportions[wsi_name])
        wsi_mask = np.where(wsi_origin == wsi_idx)[0]
        n_keep = max(1, int(len(wsi_mask) * proportion))
        keep_indices.append(rng.choice(wsi_mask, size=n_keep, replace=False))

    keep = np.concatenate(keep_indices)
    labels_int = np.array(
        [label_enc[lbl] for lbl in cell_labels[keep]],
        dtype=np.int64,
    )
    return embeddings[keep], labels_int, wsi_origin[keep], x_coords[keep], y_coords[keep]


# ── per-offset experiment ──────────────────────────────────────────────────────

def _run_one_offset(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    train_wsi: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    test_wsi: np.ndarray,
    row_off: int,
    col_off: int,
    cfg: DictConfig,
    class_names: list[str],
    num_classes: int,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the full hyperparameter grid search for one checkerboard offset."""
    seed = int(OmegaConf.select(cfg, "seed", default=42))
    offset_label = f"r{row_off}_c{col_off}"
    n_train, n_test = int(len(train_labels)), int(len(test_labels))
    print(f"[grid_experiment] offset ({row_off},{col_off}): train={n_train}, test={n_test} cells")

    if n_train == 0 or n_test == 0:
        print(f"[grid_experiment] WARNING: skipping offset ({row_off},{col_off}) — empty split")
        return {"offset": offset_label, "skipped": True}

    offset_dir = output_dir / f"offset_{offset_label}"
    offset_dir.mkdir(parents=True, exist_ok=True)

    with open(offset_dir / "split_info.yaml", "w") as f:
        yaml.dump(
            {
                "offset": {"row": row_off, "col": col_off},
                "train_size": n_train,
                "test_size": n_test,
                "class_names": class_names,
                "train_class_counts": _class_counts(train_labels, num_classes, class_names),
                "test_class_counts": _class_counts(test_labels, num_classes, class_names),
            },
            f,
        )

    # Convert to torch only here, once, right before training
    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    train_wsi_t = torch.tensor(train_wsi, dtype=torch.long)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long)
    test_wsi_t = torch.tensor(test_wsi, dtype=torch.long)

    input_dim = train_emb.shape[1]
    grid = _build_grid(cfg.hparams)
    trial_rows: list[dict] = []
    best_trial: Optional[dict] = None

    with mlflow.start_run(run_name=f"offset_{offset_label}", nested=True):
        mlflow.log_params({
            "offset_row": row_off,
            "offset_col": col_off,
            "train_size": n_train,
            "test_size": n_test,
            "num_trials": len(grid),
        })

        for trial_idx, hparams in enumerate(grid):
            pca_val: Optional[int] = (
                int(hparams["pca_components"]) if hparams["pca_components"] is not None else None
            )
            train_cfg = TrainConfig(
                input_dim=input_dim,
                num_classes=num_classes,
                class_names=class_names,
                learning_rate=float(hparams["learning_rate"]),
                dropout=float(hparams["dropout"]),
                hidden_dim=int(hparams["hidden_dim"]),
                pca_components=pca_val,
                normalization=str(hparams["normalization"]),
                loss_weights=str(hparams["loss_weights"]),
                num_epochs=int(cfg.training.num_epochs),
                batch_size=int(cfg.training.batch_size),
                seed=seed,
            )

            with mlflow.start_run(run_name=f"trial_{trial_idx}", nested=True):
                mlflow.log_params({k: (str(v) if v is None else v) for k, v in hparams.items()})
                metrics = train_and_evaluate(
                    train_emb, train_labels_t, train_wsi_t,
                    test_emb, test_labels_t, test_wsi_t,
                    train_cfg, device,
                )
                flat_metrics = {
                    k: v for k, v in metrics.items()
                    if k not in ("per_class_f1", "confusion_matrix",
                                 "train_loss_curve", "test_loss_curve")
                }
                mlflow.log_metrics(flat_metrics)
                for epoch, (trl, tel) in enumerate(
                    zip(metrics["train_loss_curve"], metrics["test_loss_curve"])
                ):
                    mlflow.log_metrics({"train_loss": trl, "test_loss": tel}, step=epoch)
                for cn, f1v in metrics["per_class_f1"].items():
                    mlflow.log_metric(f"f1_{cn}", f1v)
                mlflow.log_table(
                    pd.DataFrame(metrics["confusion_matrix"], index=class_names, columns=class_names),
                    artifact_file="confusion_matrix.json",
                )

            row: dict[str, Any] = {"trial_idx": trial_idx, **hparams, **flat_metrics}
            trial_rows.append(row)
            if best_trial is None or flat_metrics["macro_f1"] > best_trial["macro_f1"]:
                best_trial = {
                    **row,
                    "per_class_f1": metrics["per_class_f1"],
                    "confusion_matrix": metrics["confusion_matrix"],
                    "train_loss_curve": metrics["train_loss_curve"],
                    "test_loss_curve": metrics["test_loss_curve"],
                }
            print(
                f"[grid_experiment]   trial {trial_idx + 1}/{len(grid)} "
                f"macro_f1={flat_metrics['macro_f1']:.4f}"
            )

    assert best_trial is not None
    pd.DataFrame(trial_rows).to_csv(offset_dir / "grid_search_results.csv", index=False)
    np.save(offset_dir / "best_train_loss_curve.npy", np.array(best_trial["train_loss_curve"]))
    np.save(offset_dir / "best_test_loss_curve.npy", np.array(best_trial["test_loss_curve"]))

    summary = {
        "offset": {"row": row_off, "col": col_off},
        "best_config": {
            k: best_trial[k]
            for k in ("learning_rate", "dropout", "hidden_dim",
                       "pca_components", "normalization", "loss_weights")
        },
        "best_metrics": {
            k: best_trial[k]
            for k in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "loss")
        },
        "per_class_f1": best_trial["per_class_f1"],
        "all_trial_results": trial_rows,
    }
    with open(offset_dir / "results_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(
        f"[grid_experiment] offset ({row_off},{col_off}) done. "
        f"Best macro_f1={best_trial['macro_f1']:.4f}"
    )
    return {
        "offset": offset_label,
        "train_size": n_train,
        "test_size": n_test,
        "best_macro_f1": float(best_trial["macro_f1"]),
        "best_config": summary["best_config"],
        "skipped": False,
    }


# ── main experiment ────────────────────────────────────────────────────────────

def run_grid_experiment(cfg: DictConfig, device: str, output_dir: Path) -> None:
    """
    Load all WSIs listed in cfg.patch_split.wsis, then for each of the 4
    checkerboard offsets:
      1. Select the 1/4 of patches with the given (row_off, col_off) parity.
      2. Split those patches 80/20 at the PATCH level (all cells of a patch
         stay together) into train and test.
      3. Run the full hyperparameter grid search and log results to MLflow.
    """
    mapping = mapping_factory(cfg.mapping)
    class_names, label_enc = _build_label_encoder(mapping)
    num_classes = len(class_names)

    base_path = os.path.join(
        cfg.data.base_dir, f"{cfg.data.model_name}_h5", cfg.data.correction_name
    )
    embeddings_datasets: list[str] = list(cfg.data.embeddings_datasets) if cfg.data.embeddings_datasets else []
    consider_matching: bool = bool(OmegaConf.select(cfg, "data.consider_matching", default=True))
    seed = int(OmegaConf.select(cfg, "seed", default=42))
    rng = np.random.default_rng(seed)
    patch_size = float(cfg.patch_split.patch_size)

    wsi_props = {k: float(v) for k, v in cfg.patch_split.wsis.items()}
    emb, labels, wsi_orig, x_coords, y_coords = _load_wsi_data_with_coords(
        wsi_props, base_path, embeddings_datasets, mapping, label_enc, rng,
        consider_matching=consider_matching,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name = f"{cfg.data.model_name}__{cfg.data.correction_name}__grid_patch_split"

    offset_summaries: list[dict] = []

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_name": cfg.data.model_name,
            "correction_name": cfg.data.correction_name,
            "patch_size": patch_size,
            "mapping": cfg.mapping,
            "wsis": list(wsi_props.keys()),
            "total_cells": int(len(labels)),
            "consider_matching": consider_matching,
            "seed": seed,
        })

        for row_off, col_off in itertools.product(range(2), range(2)):
            sel_mask = _get_patch_mask(x_coords, y_coords, patch_size, row_off, col_off)
            sel_emb = emb[sel_mask]
            sel_labels = labels[sel_mask]
            sel_wsi = wsi_orig[sel_mask]
            sel_x = x_coords[sel_mask]
            sel_y = y_coords[sel_mask]

            # Split at patch level so all cells of a patch stay on the same side
            row_patch = np.floor(sel_y / patch_size).astype(int)
            col_patch = np.floor(sel_x / patch_size).astype(int)
            patch_ids = np.stack([sel_wsi, row_patch, col_patch], axis=1)
            unique_patches, cell_patch_idx = np.unique(patch_ids, axis=0, return_inverse=True)

            n_patches = len(unique_patches)
            perm = rng.permutation(n_patches)
            cut = int(n_patches * 0.8)

            is_train_patch = np.zeros(n_patches, dtype=bool)
            is_train_patch[perm[:cut]] = True
            cell_is_train = is_train_patch[cell_patch_idx]
            cell_is_test = ~cell_is_train

            summary = _run_one_offset(
                sel_emb[cell_is_train], sel_labels[cell_is_train], sel_wsi[cell_is_train],
                sel_emb[cell_is_test], sel_labels[cell_is_test], sel_wsi[cell_is_test],
                row_off, col_off, cfg, class_names, num_classes,
                device, output_dir,
            )
            offset_summaries.append(summary)

    valid = [s for s in offset_summaries if not s.get("skipped")]
    if valid:
        mean_f1 = float(np.mean([s["best_macro_f1"] for s in valid]))
        print(f"[grid_experiment] done. Mean macro_f1 across offsets: {mean_f1:.4f}")

    with open(output_dir / "grid_summary.yaml", "w") as f:
        yaml.dump(
            {"wsis": list(wsi_props.keys()), "consider_matching": consider_matching, "offsets": offset_summaries},
            f,
            default_flow_style=False,
        )


# ── Hydra entry point ──────────────────────────────────────────────────────────

@hydra.main(config_path="../../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[grid_experiment] device={device}")

    repo_root = Path(hydra.utils.get_original_cwd())
    consider_matching: bool = bool(OmegaConf.select(cfg, "data.consider_matching", default=True))
    embeddings_datasets: list[str] = list(cfg.data.embeddings_datasets) if cfg.data.embeddings_datasets else []
    output_dir = (
        repo_root
        / "outputs"
        / cfg.data.model_name
        / cfg.data.correction_name
        / cfg.mapping
        / _matching_folder_name(consider_matching)
        / _embeddings_folder_name(embeddings_datasets)
        / "grid_patch_split"
    )
    run_grid_experiment(cfg, device, output_dir)


if __name__ == "__main__":
    main()
