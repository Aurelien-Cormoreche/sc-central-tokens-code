"""Cell-level morphology probes: area, eccentricity, and orientation (the latter
regressed only for cells at or above an eccentricity threshold, where "long
axis" is a meaningful, low-noise quantity -- see geometry.polygon_shape_descriptors
and src/python/ot/centering_correction.py's identical reasoning for
orientation_confidence). Targets come straight from each cell's own Xenium
boundary polygon (see data/boundary_cache.py), independent of any context window.
"""
from __future__ import annotations

import numpy as np
import torch

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


@register_probe_factory("area")
def _build_area(block: dict) -> list[Probe]:
    return [AreaProbe(log_transform=bool(block.get("log_transform", True)))]


@register_probe_factory("eccentricity")
def _build_eccentricity(block: dict) -> list[Probe]:
    return [EccentricityProbe()]


@register_probe_factory("orientation")
def _build_orientation(block: dict) -> list[Probe]:
    return [OrientationProbe(eccentricity_threshold=float(block.get("eccentricity_threshold", 0.5)))]
