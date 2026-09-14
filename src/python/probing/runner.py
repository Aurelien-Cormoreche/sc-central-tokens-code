"""Pure lambda-sweep training/evaluation logic for one probe, given already-loaded
features and targets -- MLflow- and Hydra-free, mirrors classifier_training/
trainer.py's separation between pure ML logic (this file) and I/O/orchestration
(experiment.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.python.probing import solvers
from src.python.probing.probes.base import Probe, ProbeContext


@dataclass
class SplitFit:
    """A trained model's inputs/outputs for one split (val or test), already
    restricted to that probe's own valid mask."""
    cell_ids: np.ndarray
    wsi_names: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]


@dataclass
class ProbeRunResult:
    probe_name: str
    task_type: str
    target_columns: list[str]
    skipped: bool
    skip_reason: Optional[str]
    trial_rows: list[dict]           # one row per lambda: {"lambda": ..., "val_<metric>": ..., "test_<metric>": ...}
    best_lambda: Optional[float]
    n_train: int
    n_val: int
    n_test: int
    val_fit: Optional[SplitFit] = None
    test_fit: Optional[SplitFit] = None


def _select_valid(
    ctx: ProbeContext, features: np.ndarray, targets_values: np.ndarray, targets_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = targets_valid
    return features[mask], targets_values[mask], ctx.cell_ids[mask], ctx.wsi_names[mask]


def _fit(probe: Probe, X_train: torch.Tensor, y_train: torch.Tensor, lam: float):
    """One model, fit once per lambda -- shared between the val and test
    predictions below (they must come from the *same* train-only fit; see
    run_probe's docstring). Split out from prediction so a lambda's fit cost is
    paid once regardless of how many eval sets it's scored against: cheap
    redundancy for ridge_fit's closed-form solve, but solvers.logistic_fit's
    L-BFGS optimization is iterative and re-running it a second time on
    identical train data for no reason used to roughly double
    cell_type_classification's wall time (see probes/cell_level.py:CellTypeProbe)."""
    if probe.task_type == "regression":
        return solvers.ridge_fit(X_train, y_train, lam)
    if probe.task_type == "classification":
        # Extension point for a future classification probe (see solvers.logistic_fit's
        # docstring) -- expects target_columns() == ["class_idx"] and a `num_classes`
        # attribute on the probe.
        num_classes = getattr(probe, "num_classes")
        return solvers.logistic_fit(X_train, y_train.squeeze(-1).long(), lam, num_classes)
    raise ValueError(f"Unknown probe.task_type={probe.task_type!r} for probe {probe.name!r}")


def _predict(probe: Probe, model, X_eval: torch.Tensor) -> torch.Tensor:
    if probe.task_type == "regression":
        return model.predict(X_eval)
    if probe.task_type == "classification":
        return model.predict_proba(X_eval)
    raise ValueError(f"Unknown probe.task_type={probe.task_type!r} for probe {probe.name!r}")


def run_probe(
    probe: Probe,
    train_ctx: ProbeContext,
    val_ctx: ProbeContext,
    test_ctx: ProbeContext,
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    lambdas: list[float],
    device: str,
    min_train_cells: int = 10,
) -> ProbeRunResult:
    """Compute this probe's targets on train/val/test, fit a ridge (or logistic --
    see _fit_predict) model per lambda on train, evaluate every lambda on val,
    and pick the lambda with the best `selection_metric` (see Probe.eval_metrics)
    -- test is evaluated with that same train-only-fitted model (never refit on
    train+val), the same train/val/test discipline classifier_training/trainer.py
    uses. Returns every lambda's trial metrics plus the best lambda's full
    per-cell predictions (for the per-cell CSV export, see experiment.py).
    """
    train_targets = probe.compute_targets(train_ctx)
    val_targets = probe.compute_targets(val_ctx)
    test_targets = probe.compute_targets(test_ctx)
    cols = probe.target_columns()

    train_X, train_y, _tr_ids, _tr_wsi = _select_valid(train_ctx, train_features, train_targets.values, train_targets.valid)
    val_X, val_y, val_ids, val_wsi = _select_valid(val_ctx, val_features, val_targets.values, val_targets.valid)
    test_X, test_y, test_ids, test_wsi = _select_valid(test_ctx, test_features, test_targets.values, test_targets.valid)

    if len(train_X) < min_train_cells or len(val_X) == 0:
        reason = (f"only {len(train_X)} valid train cells (need >= {min_train_cells})" if len(train_X) < min_train_cells
                   else "0 valid val cells")
        print(f"[probing] SKIPPING probe '{probe.name}': {reason}")
        return ProbeRunResult(probe.name, probe.task_type, cols, True, reason, [], None,
                               len(train_X), len(val_X), len(test_X))

    train_X_t = torch.tensor(train_X, dtype=torch.float32, device=device)
    train_y_t = torch.tensor(train_y, dtype=torch.float32, device=device)
    val_X_t = torch.tensor(val_X, dtype=torch.float32, device=device)
    val_y_t = torch.tensor(val_y, dtype=torch.float32, device=device)
    has_test = len(test_X) > 0
    test_X_t = torch.tensor(test_X, dtype=torch.float32, device=device)
    test_y_t = torch.tensor(test_y, dtype=torch.float32, device=device)

    standardizer = solvers.Standardizer.fit(train_X_t)
    train_Xs = standardizer.transform(train_X_t)
    val_Xs = standardizer.transform(val_X_t)
    test_Xs = standardizer.transform(test_X_t)

    trial_rows: list[dict] = []
    best_trial: Optional[dict] = None

    for lam in lambdas:
        model = _fit(probe, train_Xs, train_y_t, lam)
        val_pred = _predict(probe, model, val_Xs)
        val_metrics = probe.eval_metrics(val_y_t, val_pred)

        if has_test:
            test_pred = _predict(probe, model, test_Xs)
            test_metrics = probe.eval_metrics(test_y_t, test_pred)
        else:
            test_pred = test_y_t.new_zeros((0, len(cols)))
            test_metrics = {}

        row = {"lambda": lam, **{f"val_{k}": v for k, v in val_metrics.items()},
               **{f"test_{k}": v for k, v in test_metrics.items()}}
        trial_rows.append(row)

        if best_trial is None or val_metrics["selection_metric"] > best_trial["val_metrics"]["selection_metric"]:
            best_trial = {
                "lambda": lam, "val_metrics": val_metrics, "test_metrics": test_metrics,
                "val_pred": val_pred.detach().cpu().numpy(),
                "test_pred": test_pred.detach().cpu().numpy(),
            }

    assert best_trial is not None
    val_fit = SplitFit(val_ids, val_wsi, val_y, best_trial["val_pred"], best_trial["val_metrics"])
    test_fit = (
        SplitFit(test_ids, test_wsi, test_y, best_trial["test_pred"], best_trial["test_metrics"])
        if has_test else None
    )

    return ProbeRunResult(
        probe_name=probe.name, task_type=probe.task_type, target_columns=cols,
        skipped=False, skip_reason=None, trial_rows=trial_rows, best_lambda=best_trial["lambda"],
        n_train=len(train_X), n_val=len(val_X), n_test=len(test_X),
        val_fit=val_fit, test_fit=test_fit,
    )
