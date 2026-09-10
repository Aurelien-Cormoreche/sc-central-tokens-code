"""Per-WSI Xenium boundary-polygon lookup for the probing package's cell-level
morphology probes (area / eccentricity / orientation -- see probes/cell_level.py
and geometry.py). Thin, probing-specific wrapper around
extract_embeddings/data/xenium_boundaries.BoundaryPolygons -- the same H&E-pixel-
space polygons extract_embeddings.py's save_cell/save_nucleus token-pooling
already uses, so a probed cell's shape target lives in the exact same pixel
frame as the patch its embedding was extracted from.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.python.extract_embeddings.data.xenium_boundaries import (
    BoundaryPolygons,
    load_alignment_matrix_inv,
)


@dataclass
class BoundaryRoots:
    """Same two roots build_configs(with_boundaries=True) wires in -- see
    extract_embeddings/extract_embeddings.py and src/python/code_configs/paths.py
    (defaults: XENIUM_OUTPUT_ROOT / ALIGNMENT_MATRIX_ROOT)."""
    cell_boundaries_root: os.PathLike | str
    alignment_matrix_root: os.PathLike | str


def _boundary_path(wsi_name: str, roots: BoundaryRoots, kind: str) -> str:
    filename = 'cell_boundaries.csv.gz' if kind == 'cell' else 'nucleus_boundaries.parquet'
    return os.path.join(str(roots.cell_boundaries_root), f'{wsi_name}_out', filename)


class BoundaryCache:
    """Lazily loads and caches one BoundaryPolygons store per WSI for the
    lifetime of a single experiment.py run."""

    def __init__(self, roots: BoundaryRoots, kind: str):
        """kind: 'cell' (whole-cell segmentation, cell_boundaries.csv.gz -- default,
        matches the literal "cell" wording of the area/eccentricity/orientation
        probes) or 'nucleus' (nucleus_boundaries.parquet)."""
        assert kind in ('cell', 'nucleus'), f"kind must be 'cell' or 'nucleus', got {kind!r}"
        self._roots = roots
        self._kind = kind
        self._cache: dict[str, BoundaryPolygons] = {}

    def get(self, wsi_name: str) -> BoundaryPolygons:
        if wsi_name not in self._cache:
            alignment_path = os.path.join(
                str(self._roots.alignment_matrix_root), f'{wsi_name}_he_imagealignment.csv'
            )
            M_inv = load_alignment_matrix_inv(alignment_path)
            boundary_path = _boundary_path(wsi_name, self._roots, self._kind)
            self._cache[wsi_name] = BoundaryPolygons(boundary_path, M_inv)
        return self._cache[wsi_name]
