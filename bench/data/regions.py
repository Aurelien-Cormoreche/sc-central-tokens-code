"""Group cell crops into tile-aligned read regions (one WSI read per region).

A crop of N px for a cell at extraction position (ex, ey) has origin
o = floor(e + 0.5) - N/2, so the centroid lies within half a pixel of the crop centre, i.e. on
the shared corner of the central 2x2 tokens for an even token grid, at the centre of the central
token for an odd one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


def crop_origins(ex: np.ndarray, ey: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if n % 2:
        raise ValueError(f"crop size must be even, got {n}")
    ox = np.floor(np.asarray(ex, dtype=np.float64) + 0.5).astype(np.int64) - n // 2
    oy = np.floor(np.asarray(ey, dtype=np.float64) + 0.5).astype(np.int64) - n // 2
    return ox, oy


@dataclass
class Region:
    x0: int
    y0: int
    w: int
    h: int
    cells: np.ndarray      # (k,) indices into the slide's extraction cell list
    dx: np.ndarray         # (k,) crop origin relative to the region
    dy: np.ndarray


def plan_regions(ox: np.ndarray, oy: np.ndarray, n: int, region_px: int) -> List[Region]:
    """Cells bucketed by the region_px tile holding their crop origin; each region is the tight
    bounding box of its cells' crops (<= region_px + n per side). Regions are ordered row-major
    and cells within a region by (dy, dx), so reads sweep the slide once."""
    if len(ox) == 0:
        return []
    tx = np.floor_divide(ox, region_px)
    ty = np.floor_divide(oy, region_px)
    order = np.lexsort((ox, oy, tx, ty))
    keys = np.stack([ty[order], tx[order]], axis=1)
    bounds = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1
    regions = []
    for chunk in np.split(order, bounds):
        x0, y0 = int(ox[chunk].min()), int(oy[chunk].min())
        x1, y1 = int(ox[chunk].max()) + n, int(oy[chunk].max()) + n
        regions.append(Region(x0, y0, x1 - x0, y1 - y0, chunk.astype(np.int64),
                              (ox[chunk] - x0).astype(np.int64), (oy[chunk] - y0).astype(np.int64)))
    return regions


def read_crops(read, dx: np.ndarray, dy: np.ndarray, n: int, out: np.ndarray, error=Exception) -> int:
    """Fill `out` (a region's array) at every crop (origins dx, dy in it, n x n) when the region cannot be read
    as a whole: read the bounding box of the crops; when that fails (`error`), split the crops in two along the
    box's longer side and read each half the same way. A corrupt tile inside the region but away from every crop
    (the manifest keeps crops clear of unreadable blocks) is skipped this way; a crop that cannot be read itself
    raises. `read(x, y, w, h)` reads in region coordinates. Returns the number of reads that failed."""
    failed = 0
    stack = [np.arange(len(dx))]
    while stack:
        idx = stack.pop()
        x0, y0 = int(dx[idx].min()), int(dy[idx].min())
        x1, y1 = int(dx[idx].max()) + n, int(dy[idx].max()) + n
        try:
            out[y0:y1, x0:x1] = read(x0, y0, x1 - x0, y1 - y0)
        except error:
            failed += 1
            if len(idx) == 1 or (x1 - x0 == n and y1 - y0 == n):     # one crop (or crops at one origin)
                raise
            key = dx if x1 - x0 >= y1 - y0 else dy
            o = idx[np.argsort(key[idx], kind="stable")]
            stack += [o[: len(o) // 2], o[len(o) // 2:]]
    return failed


def emission_order(regions: List[Region]) -> np.ndarray:
    """Cell indices in the order the extractor emits them (= row order of the output h5)."""
    if not regions:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([r.cells for r in regions])
