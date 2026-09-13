"""Cell-level probes: area, eccentricity, and orientation (the latter regressed
only for cells at or above an eccentricity threshold, where "long axis" is a
meaningful, low-noise quantity -- see geometry.polygon_shape_descriptors and
src/python/ot/centering_correction.py's identical reasoning for
orientation_confidence), plus cell_type_classification. The morphology probes'
targets come straight from each cell's own Xenium boundary polygon (see
data/boundary_cache.py); cell_type_classification's target is the cell's own
mapped type (ctx.cell_types). All are independent of any context window --
contrast with probes/context.py's cell_type_composition, which targets the
*neighbourhood's* type fractions rather than the cell's own identity.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.python.probing import geometry, solvers
from src.python.probing.probes.base import Probe, ProbeContext, ProbeTargets, group_ranges, register_probe_factory


def cell_shape_descriptors(ctx: ProbeContext) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(area, eccentricity, orientation_deg, valid) for every cell in ctx, looked
    up from its WSI's cell-boundary polygon store. `valid` is False where no
    polygon was found for that cell_id, or the polygon was too degenerate for
    geometry.polygon_shape_descriptors to return a descriptor."""
    n = len(ctx)
    area = np.full(n, np.nan)
    ecc = np.full(n, np.nan)
    orient = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    order = np.argsort(ctx.wsi_names, kind='stable')
    sorted_wsi = ctx.wsi_names[order]
    for start, end in group_ranges(sorted_wsi):
        wsi_name = sorted_wsi[start]
        store = ctx.boundary_cache.get(wsi_name)
        for local in order[start:end]:
            poly = store.polygon_for(ctx.cell_ids[local])
            if poly is None:
                continue
            desc = geometry.polygon_shape_descriptors(poly)
            if desc is None:
                continue
            area[local], ecc[local], orient[local] = desc
            valid[local] = True
    return area, ecc, orient, valid


class AreaProbe(Probe):
    name = "area"
    task_type = "regression"

    def __init__(self, log_transform: bool = True):
        self.log_transform = log_transform

    def target_columns(self) -> list[str]:
        return ["log1p_area"] if self.log_transform else ["area"]

    def compute_targets(self, ctx: ProbeContext) -> ProbeTargets:
        area, _ecc, _orient, valid = cell_shape_descriptors(ctx)
        values = np.log1p(area) if self.log_transform else area
        return ProbeTargets(np.nan_to_num(values).reshape(-1, 1), valid)


class EccentricityProbe(Probe):
    name = "eccentricity"
    task_type = "regression"

    def target_columns(self) -> list[str]:
        return ["eccentricity"]

    def compute_targets(self, ctx: ProbeContext) -> ProbeTargets:
        _area, ecc, _orient, valid = cell_shape_descriptors(ctx)
        return ProbeTargets(np.nan_to_num(ecc).reshape(-1, 1), valid)


class OrientationProbe(Probe):
    """Regresses [cos(2*theta), sin(2*theta)] (theta = major-axis angle, mod 180)
    via ridge, restricted to cells with eccentricity >= eccentricity_threshold."""
    name = "orientation"
    task_type = "regression"

    def __init__(self, eccentricity_threshold: float = 0.5):
        self.eccentricity_threshold = eccentricity_threshold

    def target_columns(self) -> list[str]:
        return ["orientation_cos2", "orientation_sin2"]

    def compute_targets(self, ctx: ProbeContext) -> ProbeTargets:
        _area, ecc, orient, valid = cell_shape_descriptors(ctx)
        valid = valid & (ecc >= self.eccentricity_threshold)
        vec = geometry.angle_to_doubled_unit_vector(np.nan_to_num(orient))
        return ProbeTargets(vec, valid)

    def eval_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
        angle_true = geometry.doubled_unit_vector_to_angle(y_true.cpu().numpy())
        angle_pred = geometry.doubled_unit_vector_to_angle(y_pred.cpu().numpy())
        err = geometry.circular_angle_error_deg(angle_true, angle_pred)
        mean_abs_err = float(np.mean(err)) if len(err) else float('nan')
        r2 = solvers.r2_score(y_true, y_pred)
        return {
            "r2_orientation_cos2": float(r2[0]),
            "r2_orientation_sin2": float(r2[1]),
            "mean_angular_error_deg": mean_abs_err,
            # negated so "higher is better" holds for lambda selection, same
            # direction convention as every other probe's R^2-based metric.
            "selection_metric": -mean_abs_err if np.isfinite(mean_abs_err) else float('-inf'),
        }

    def predictions_to_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
        angle_true = geometry.doubled_unit_vector_to_angle(y_true)
        angle_pred = geometry.doubled_unit_vector_to_angle(y_pred)
        return {
            "true_orientation_deg": angle_true,
            "pred_orientation_deg": angle_pred,
            "angular_error_deg": geometry.circular_angle_error_deg(angle_true, angle_pred),
        }


class CellTypeProbe(Probe):
    """Multinomial-logistic-regression classification of the probed cell's own
    mapped type (ctx.cell_types) from its embedding -- "how much of a cell's own
    identity is linearly decodable", as opposed to cell_type_composition
    (probes/context.py), which regresses the *neighbourhood's* type fractions.

    "Unknown" cells are excluded entirely (not a real class to classify into),
    same convention as classifier_training/experiment.py's _build_label_encoder /
    _load_wsi_data -- so results are directly comparable to that pipeline's MLP
    classification metrics. The class list is read from the run's population
    cache (mapping-derived, identical for every WSI -- see
    data/cell_population.py's WSIPopulation.class_names / CompositionProbe's
    identical lazy-discovery pattern), minus "Unknown", the first time
    compute_targets() is called; see Probe's docstring re: target_columns()
    only being meaningful after that.
    """
    name = "cell_type_classification"
    task_type = "classification"

    def __init__(self):
        self._class_names: list[str] = []

    def target_columns(self) -> list[str]:
        return ["class_idx"]

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    def compute_targets(self, ctx: ProbeContext) -> ProbeTargets:
        if not self._class_names and len(ctx):
            all_classes = ctx.population_cache.get(ctx.wsi_names[0]).class_names
            self._class_names = [c for c in all_classes if c != "Unknown"]
        name_to_idx = {name: i for i, name in enumerate(self._class_names)}
        idx = np.array([name_to_idx.get(ct, -1) for ct in ctx.cell_types], dtype=np.int64)
        valid = idx >= 0
        return ProbeTargets(idx.astype(np.float64).reshape(-1, 1), valid)

    def eval_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
        # y_true: (N, 1) class indices (float, see compute_targets). y_pred: (N, C)
        # class probabilities (runner._fit_predict's classification path returns
        # predict_proba, not hard labels) -- argmax before any label-based metric.
        true_idx = y_true.squeeze(-1).long().cpu().numpy()
        pred_idx = y_pred.argmax(dim=1).cpu().numpy()
        labels = list(range(self.num_classes))
        macro_f1 = float(f1_score(true_idx, pred_idx, average="macro", zero_division=0, labels=labels))
        return {
            "accuracy": float(accuracy_score(true_idx, pred_idx)),
            "macro_f1": macro_f1,
            "weighted_f1": float(f1_score(true_idx, pred_idx, average="weighted", zero_division=0, labels=labels)),
            "balanced_accuracy": float(balanced_accuracy_score(true_idx, pred_idx)),
            # macro F1 (not accuracy) so lambda selection isn't dominated by the
            # majority class on an imbalanced cell-type population.
            "selection_metric": macro_f1,
        }

    def predictions_to_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
        true_idx = y_true[:, 0].astype(np.int64)
        pred_idx = y_pred.argmax(axis=1).astype(np.int64)
        names = np.array(self._class_names)
        return {
            "true_cell_type": names[true_idx],
            "pred_cell_type": names[pred_idx],
            "pred_confidence": y_pred[np.arange(len(pred_idx)), pred_idx],
        }


@register_probe_factory("cell_type_classification")
def _build_cell_type_classification(block: dict) -> list[Probe]:
    return [CellTypeProbe()]


@register_probe_factory("area")
def _build_area(block: dict) -> list[Probe]:
    return [AreaProbe(log_transform=bool(block.get("log_transform", True)))]


@register_probe_factory("eccentricity")
def _build_eccentricity(block: dict) -> list[Probe]:
    return [EccentricityProbe()]


@register_probe_factory("orientation")
def _build_orientation(block: dict) -> list[Probe]:
    return [OrientationProbe(eccentricity_threshold=float(block.get("eccentricity_threshold", 0.5)))]
