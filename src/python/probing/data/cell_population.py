"""Per-WSI ground-truth cell population (position + type), used by the probing
package's context probes (composition / count / density of the neighbourhood
around a probed cell -- see probes/context.py) and to look up a probed cell's
own centroid position.

Sourced from `patch_coordinates.h5` (see extract_embeddings/data/patch_dataset.py)
-- Marc's ground-truth pipeline output listing *every* detected cell in a WSI
(cell_id, x_start, y_start, cell_type), independent of which cells ended up with
an embedding extracted (a WSI/cell-type-proportion-subsampled, possibly
CellViT-matching-filtered subset -- see classifier_training/experiment.py's
_load_wsi_data). Context probes need the *full* population so a probed cell's
neighbourhood isn't artificially sparse just because most of its neighbours
happen to not be in the embedding-bearing subset.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import h5py
import numpy as np
from scipy.spatial import cKDTree

# Cell centre = x_start + CELL_OFFSET_PX, y_start + CELL_OFFSET_PX -- the standard
# "224x224 patch centred on the cell" convention used throughout extract_embeddings
# (see PatchDataset, MultiCellPatchDataset's default cell_offset_x/y=112, and
# src/python/ot/export_matched_patch_coordinates.py's PATCH_HALF=112).
CELL_OFFSET_PX = 112


def _decode(arr: np.ndarray) -> np.ndarray:
    """h5py reads HDF5 string datasets back as bytes by default (both fixed- and
    variable-length) -- decode once to plain Python str for clean joins/dict keys."""
    if len(arr) and isinstance(arr[0], (bytes, bytearray)):
        return np.array([x.decode('utf-8') for x in arr])
    return arr


@dataclass
class WSIPopulation:
    """Every detected cell in one WSI: id, centre position (H&E pixel space, same
    frame as PatchDataset / boundary polygons), and mapped cell-type class index.
    `tree` supports fast square-window queries via Chebyshev-distance search (see
    query_window/query_windows)."""
    wsi_name: str
    cell_id: np.ndarray         # (N,) str
    x: np.ndarray                # (N,) float -- cell centre, px
    y: np.ndarray                # (N,) float
    class_idx: np.ndarray        # (N,) int -- index into class_names
    class_names: list[str]
    tree: cKDTree = field(repr=False)
    id_to_row: dict = field(repr=False)

    def __len__(self) -> int:
        return len(self.cell_id)

    def position_for(self, cell_id: str) -> tuple[float, float] | None:
        """Centroid (x, y) of `cell_id` in this WSI, or None if it isn't present
        in this population (e.g. a cell_id convention mismatch between the
        embeddings source and this WSI's patch_coordinates.h5)."""
        row = self.id_to_row.get(cell_id)
        if row is None:
            return None
        return float(self.x[row]), float(self.y[row])

    def query_window(self, cx: float, cy: float, half_size: float) -> np.ndarray:
        """Row indices of every population cell whose centre lies in the square
        window [cx-half_size, cx+half_size] x [cy-half_size, cy+half_size] --
        i.e. within Chebyshev (L-infinity) distance half_size of (cx, cy), which
        is exactly a square-window query. Includes the probed cell itself when
        its own centre lies in that range (see probes/context.py: the embedding
        is of the *whole* patch, itself included, not just its neighbours)."""
        return np.asarray(self.tree.query_ball_point([cx, cy], r=half_size, p=np.inf), dtype=np.int64)

    def query_windows(self, cx: np.ndarray, cy: np.ndarray, half_size: float) -> list[np.ndarray]:
        """Vectorized query_window over many (cx, cy) centres at once."""
        points = np.column_stack([cx, cy])
        return self.tree.query_ball_point(points, r=half_size, p=np.inf, workers=-1)


def load_wsi_population(
    wsi_name: str,
    cells_info_root: os.PathLike | str,
    mapping: dict,
) -> WSIPopulation:
    """Load `{cells_info_root}/{wsi_name}/patch_coordinates.h5` and map every
    cell's raw cell_type through `mapping` (see code_configs/mappings.py) --
    unlike the classifier pipeline, "Unknown" is *kept* as its own class here
    rather than dropped: composition/count/density describe everything visibly
    present in a patch, not just confidently-typed cells.

    class_names is derived from `mapping`'s own value set (plus "Unknown"), not
    from what's actually observed in this WSI -- so it is identical across every
    WSI in a run and the resulting composition vectors are directly comparable.
    """
    path = os.path.join(str(cells_info_root), wsi_name, 'patch_coordinates.h5')
    with h5py.File(path, 'r') as f:
        cell_id = _decode(f['cell_id'][:])
        x_start = f['x_start'][:].astype(np.float64)
        y_start = f['y_start'][:].astype(np.float64)
        cell_type_raw = f['cell_type'][:]

    # mapping's keys are bytes (see code_configs/mappings.py). patch_coordinates.h5
    # usually stores cell_type the same way (read back as bytes by h5py, see
    # _decode's docstring), but a POSITIONS_CONVERTED_ROOT file written by
    # export_matched_patch_coordinates.py explicitly asstr()s bytes to a Python str
    # dataset instead -- in that case, decode the mapping's own keys once so lookups
    # still line up regardless of which convention this particular file used.
    lookup_mapping = mapping
    if len(cell_type_raw) and isinstance(cell_type_raw[0], str):
        lookup_mapping = {(k.decode('utf-8') if isinstance(k, (bytes, bytearray)) else k): v
                           for k, v in mapping.items()}

    class_names = sorted(set(mapping.values()) | {'Unknown'})
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    class_idx = np.array(
        [name_to_idx[lookup_mapping.get(ct, 'Unknown')] for ct in cell_type_raw], dtype=np.int64
    )

    x = x_start + CELL_OFFSET_PX
    y = y_start + CELL_OFFSET_PX
    tree = cKDTree(np.column_stack([x, y]))
    id_to_row = {cid: i for i, cid in enumerate(cell_id)}
    return WSIPopulation(wsi_name, cell_id, x, y, class_idx, class_names, tree, id_to_row)


class PopulationCache:
    """Lazily loads and caches one WSIPopulation per WSI name for the lifetime of
    a single experiment.py run (context probes and cell-position lookups both
    query every WSI touched by train/val/test, so this avoids re-reading
    patch_coordinates.h5 / rebuilding its KD-tree once per probe)."""

    def __init__(self, cells_info_root: os.PathLike | str, mapping: dict):
        self._cells_info_root = cells_info_root
        self._mapping = mapping
        self._cache: dict[str, WSIPopulation] = {}

    def get(self, wsi_name: str) -> WSIPopulation:
        if wsi_name not in self._cache:
            self._cache[wsi_name] = load_wsi_population(wsi_name, self._cells_info_root, self._mapping)
        return self._cache[wsi_name]


def resolve_positions(
    cell_ids: np.ndarray, wsi_names: np.ndarray, population_cache: PopulationCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centroid (x, y) of every (cell_id, wsi_name) pair, looked up via each WSI's
    population (see WSIPopulation.position_for). Used both to build context probes'
    query centres and, more generally, so cell-level probes' ProbeContext always
    carries a position even though they don't need one themselves.

    Returns (x, y, valid) -- valid is False (x/y left as nan) for a cell_id not
    found in its WSI's patch_coordinates.h5 population (e.g. a cell_id convention
    mismatch between the embeddings source and the population source for that
    WSI); callers should fold `valid` into any probe that actually uses x/y.
    """
    n = len(cell_ids)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    order = np.argsort(wsi_names, kind='stable')
    sorted_wsi = wsi_names[order]
    boundaries = np.flatnonzero(sorted_wsi[1:] != sorted_wsi[:-1]) + 1 if n else np.array([], dtype=int)
    starts = np.concatenate([[0], boundaries]) if n else np.array([], dtype=int)
    ends = np.concatenate([boundaries, [n]]) if n else np.array([], dtype=int)

    n_missing = 0
    for s, e in zip(starts.tolist(), ends.tolist()):
        wsi_name = sorted_wsi[s]
        pop = population_cache.get(wsi_name)
        for local in order[s:e]:
            pos = pop.position_for(cell_ids[local])
            if pos is None:
                n_missing += 1
                continue
            x[local], y[local] = pos
            valid[local] = True

    if n_missing:
        print(f"[cell_population] WARNING: {n_missing}/{n} cells had no matching position "
              f"in their WSI's patch_coordinates.h5 population -- dropped from any probe "
              f"that needs a cell position (context probes; cell-level probes are unaffected).")
    return x, y, valid
