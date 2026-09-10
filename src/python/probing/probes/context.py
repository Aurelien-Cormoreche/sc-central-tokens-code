"""Context (neighbourhood) probes: cell-type composition, raw cell count, and
areal cell density within a context_px x context_px square window centred on
each probed cell -- i.e. properties of the *patch*, matching the field of view
an embedding extracted at that context size actually covers (see
extract_embeddings.py's build_resized_cell_configs size_side=100/448/1344 and
build_configs' native 224 crop). The window's own centre cell is included (the
embedding is of the whole patch, not just its neighbours).

One Probe INSTANCE is created per configured context_px (see build_* factories
below): `cfg.probes.cell_type_composition.enabled=true` with
`context_px=[224,448,1344]` expands to three independent probes, each trained
against the very same embeddings -- so you can compare how much of e.g. a
448px-window target's information a given embedding encodes, regardless of
whether that embedding itself came from a 224 or 1344px crop.
"""
from __future__ import annotations

import numpy as np

from src.python.probing.probes.base import Probe, ProbeContext, ProbeTargets, group_ranges, register_probe_factory


def window_stats(ctx: ProbeContext, context_px: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(counts (N,), per_class_counts (N, K), class_names) for every valid-position
    cell in ctx, counted over its own WSI population within a context_px x
    context_px window centred on it. Rows with ctx.position_valid == False get
    count 0 / all-zero per_class_counts (callers must further gate on
    position_valid via each probe's `valid` mask -- see e.g. CellCountProbe)."""
    n = len(ctx)
    half = context_px / 2.0

    class_names: list[str] = []
    counts = np.zeros(n, dtype=np.int64)
    per_class = np.zeros((n, 0), dtype=np.int64)

    order = np.argsort(ctx.wsi_names, kind='stable')
    sorted_wsi = ctx.wsi_names[order]
    for start, end in group_ranges(sorted_wsi):
        wsi_name = sorted_wsi[start]
        pop = ctx.population_cache.get(wsi_name)
        if not class_names:
            class_names = pop.class_names
            per_class = np.zeros((n, len(class_names)), dtype=np.int64)

        local_idx = order[start:end]
        local_idx = local_idx[ctx.position_valid[local_idx]]
        if len(local_idx) == 0:
            continue
        neighbor_lists = pop.query_windows(ctx.x[local_idx], ctx.y[local_idx], half)
        for row, neighbors in zip(local_idx.tolist(), neighbor_lists):
            if len(neighbors) == 0:
                continue
            classes = pop.class_idx[np.asarray(neighbors, dtype=np.int64)]
            counts[row] = len(neighbors)
            per_class[row] = np.bincount(classes, minlength=len(class_names))
    return counts, per_class, class_names


class CompositionProbe(Probe):
    """NOTE: target_columns() only reflects the correct class list after
    compute_targets() has been called at least once (see Probe's docstring) --
    class_names come from the run's cell-type `mapping`, so this is the same
    every call, but is only known once a population has actually been loaded."""
    name_prefix = "cell_type_composition"
    task_type = "regression"

    def __init__(self, context_px: int):
        self.context_px = context_px
        self.name = f"{self.name_prefix}@{context_px}px"
        self._class_names: list[str] = []

    def target_columns(self) -> list[str]:
        return [f"frac_{c}" for c in self._class_names] if self._class_names else ["fraction"]

    def compute_targets(self, ctx: ProbeContext) -> ProbeTargets:
        counts, per_class, class_names = window_stats(ctx, self.context_px)
        self._class_names = class_names
        fractions = np.divide(
            per_class, counts[:, None],
            out=np.zeros(per_class.shape, dtype=np.float64),
            where=counts[:, None] > 0,
        )
        valid = ctx.position_valid & (counts > 0)
        return ProbeTargets(fractions, valid)


class CellCountProbe(Probe):
    name_prefix = "cell_count"
    task_type = "regression"

    def __init__(self, context_px: int, log_transform: bool = True):
        self.context_px = context_px
        self.log_transform = log_transform
        self.name = f"{self.name_prefix}@{context_px}px"

    def target_columns(self) -> list[str]:
        return ["log1p_cell_count"] if self.log_transform else ["cell_count"]

    def compute_targets(self, ctx: ProbeContext) -> ProbeTargets:
        counts, _per_class, _class_names = window_stats(ctx, self.context_px)
        values = np.log1p(counts) if self.log_transform else counts.astype(np.float64)
        return ProbeTargets(values.reshape(-1, 1), ctx.position_valid.copy())


class CellDensityProbe(Probe):
    """Cell count normalized by window area, reported as cells per 100x100px unit
    so windows of different context_px land on a comparable scale -- captures
    local tissue "crowdedness" independent of the window-size-driven raw-count
    scale CellCountProbe reports (see module docstring)."""
    name_prefix = "cell_density"
    task_type = "regression"

    def __init__(self, context_px: int, log_transform: bool = True):
        self.context_px = context_px
        self.log_transform = log_transform
        self.name = f"{self.name_prefix}@{context_px}px"

    def target_columns(self) -> list[str]:
        return ["log1p_cell_density_per_100px2"] if self.log_transform else ["cell_density_per_100px2"]

    def compute_targets(self, ctx: ProbeContext) -> ProbeTargets:
        counts, _per_class, _class_names = window_stats(ctx, self.context_px)
        density = counts.astype(np.float64) * (100.0 / self.context_px) ** 2
        values = np.log1p(density) if self.log_transform else density
        return ProbeTargets(values.reshape(-1, 1), ctx.position_valid.copy())


def _context_px_list(block: dict) -> list[int]:
    return [int(v) for v in block.get("context_px", [224, 448, 1344])]


@register_probe_factory("cell_type_composition")
def _build_composition(block: dict) -> list[Probe]:
    return [CompositionProbe(px) for px in _context_px_list(block)]


@register_probe_factory("cell_count")
def _build_count(block: dict) -> list[Probe]:
    log_t = bool(block.get("log_transform", True))
    return [CellCountProbe(px, log_transform=log_t) for px in _context_px_list(block)]


@register_probe_factory("cell_density")
def _build_density(block: dict) -> list[Probe]:
    log_t = bool(block.get("log_transform", True))
    return [CellDensityProbe(px, log_transform=log_t) for px in _context_px_list(block)]
