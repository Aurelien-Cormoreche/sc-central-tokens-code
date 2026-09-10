"""Object-oriented Probe framework: every probe (area, eccentricity, orientation,
cell-type composition, cell count, cell density) is a small self-contained Probe
subclass registered via @register_probe_factory -- add a new probe type by
subclassing Probe and registering a factory for it; the sweep/train/eval/save
driver in experiment.py is entirely probe-agnostic (it only calls the methods
declared here).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from src.python.probing import solvers
from src.python.probing.data.boundary_cache import BoundaryCache
from src.python.probing.data.cell_population import PopulationCache


@dataclass
class ProbeContext:
    """Everything a Probe.compute_targets() call might need, gathered in one
    place so every probe's signature stays identical regardless of what data it
    actually reads. One ProbeContext is built per split (train/val/test)."""
    cell_ids: np.ndarray          # (N,) str
    wsi_names: np.ndarray         # (N,) str
    cell_types: np.ndarray        # (N,) str -- this cell's own mapped type
    x: np.ndarray                 # (N,) float -- cell centre, px (nan where unresolved)
    y: np.ndarray                 # (N,) float
    position_valid: np.ndarray    # (N,) bool -- False where x/y could not be resolved
    boundary_cache: BoundaryCache
    population_cache: PopulationCache

    def __len__(self) -> int:
        return len(self.cell_ids)


@dataclass
class ProbeTargets:
    values: np.ndarray    # (N, K) float
    valid: np.ndarray     # (N,) bool -- False where this probe has no usable target for that cell


def group_ranges(sorted_keys: np.ndarray):
    """Yield (start, end) index ranges of each contiguous run of equal values in
    a sorted array -- used to group cells by WSI (see e.g. probes/cell_level.py,
    probes/context.py) without pulling in a pandas groupby for a plain numpy loop."""
    n = len(sorted_keys)
    if n == 0:
        return
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])
    for s, e in zip(starts.tolist(), ends.tolist()):
        yield s, e


class Probe(ABC):
    """One probeable quantity. Subclasses are stateless besides their own config
    (e.g. a context window size) -- all per-cell data flows through
    ProbeContext / ProbeTargets, never stored on self across splits, so one
    Probe instance is safely reused across train/val/test calls.

    NOTE (CompositionProbe only): target_columns() reflects the class list
    discovered by the most recent compute_targets() call -- call compute_targets
    at least once before reading target_columns() for that probe.
    """

    #: unique key, also the output sub-directory / mlflow run name
    name: str
    #: "regression" (ridge, solvers.ridge_fit) or "classification" (logistic,
    #: solvers.logistic_fit) -- none of the probes shipped in cell_level.py /
    #: context.py use "classification", but the framework supports it (see
    #: solvers.logistic_fit's docstring).
    task_type: str = "regression"

    @abstractmethod
    def target_columns(self) -> list[str]:
        """Names of this probe's target dimensions (K of them, matching
        compute_targets()'s second axis) -- used as column names when saving
        per-cell results and as per-output R^2 metric keys."""

    @abstractmethod
    def compute_targets(self, ctx: ProbeContext) -> ProbeTargets:
        ...

    def eval_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
        """Default: per-column R^2 + mean R^2 (regression). `selection_metric` is
        what the lambda sweep maximizes (see experiment.py) -- override this
        together with predictions_to_report for a probe whose natural metric
        isn't plain per-column R^2 (see OrientationProbe, which reports circular
        angular error instead)."""
        r2 = solvers.r2_score(y_true, y_pred)
        cols = self.target_columns()
        out = {f"r2_{c}": float(v) for c, v in zip(cols, r2.tolist())}
        out["r2_mean"] = solvers.mean_finite(r2.tolist())
        out["selection_metric"] = out["r2_mean"]
        return out

    def predictions_to_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
        """{column_name: (N,) array} for the per-cell true_<col>/pred_<col> CSV
        export (see experiment.py). Default: one true_/pred_ pair per target
        column; override (e.g. OrientationProbe) to additionally report derived
        quantities such as the recovered angle / angular error."""
        cols = self.target_columns()
        out: dict[str, np.ndarray] = {}
        for i, c in enumerate(cols):
            out[f"true_{c}"] = y_true[:, i]
            out[f"pred_{c}"] = y_pred[:, i]
        return out


#: probe-type key (matches a cfg.probes.<key> block) -> factory building the list
#: of Probe instances that block expands to (one, or one per context_px for a
#: context probe -- see probes/context.py).
PROBE_FACTORY_REGISTRY: dict[str, Callable[[dict], list[Probe]]] = {}


def register_probe_factory(key: str):
    """Decorator registering `build_fn(cfg_block: dict) -> list[Probe]` under
    `key`, matched against `cfg.probes.<key>` by build_active_probes()
    (see experiment.py). Add a new probe type by writing a Probe subclass plus a
    `@register_probe_factory("my_probe")` build function -- no other file needs
    to change."""
    def _decorator(build_fn: Callable[[dict], list[Probe]]):
        PROBE_FACTORY_REGISTRY[key] = build_fn
        return build_fn
    return _decorator
