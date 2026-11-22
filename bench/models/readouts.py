"""Token indices for the `central` readout, for any token grid.

Single cell: the crop is centred on the cell (bench/data/regions.py), so the centroid sits on the
shared corner of the central 2x2 tokens of an even grid (rows/cols g/2-1, g/2), or inside the
central token of an odd grid (row/col (g-1)/2). `central` = mean of those tokens.

Multicell: a cell at pixel (px, py) of a T-px tile uses the 2x2 tokens around the token corner
nearest to it (corner index round(p/patch), clipped so all 4 tokens exist).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def central_tokens(grid: int) -> List[Tuple[int, int]]:
    if grid < 1:
        raise ValueError(f"grid must be >= 1, got {grid}")
    if grid % 2:
        c = (grid - 1) // 2
        return [(c, c)]
    a, b = grid // 2 - 1, grid // 2
    return [(a, a), (a, b), (b, a), (b, b)]


def central_flat(grid: int) -> torch.Tensor:
    return torch.tensor([r * grid + c for r, c in central_tokens(grid)], dtype=torch.long)


def central_pixel_box(grid: int, patch: int) -> Tuple[int, int]:
    """[lo, hi) pixel span (per axis) covered by the central tokens."""
    toks = central_tokens(grid)
    lo = min(r for r, _ in toks) * patch
    hi = (max(r for r, _ in toks) + 1) * patch
    return lo, hi


def nearest_2x2_flat(px: np.ndarray, py: np.ndarray, patch: int, grid: int) -> np.ndarray:
    """(k, 4) flat token indices of the 2x2 block around the token corner nearest each point."""
    if grid < 2:
        raise ValueError("multicell readout needs a grid of at least 2x2 tokens")
    cx = np.clip(np.floor(np.asarray(px, dtype=np.float64) / patch + 0.5).astype(np.int64), 1, grid - 1)
    cy = np.clip(np.floor(np.asarray(py, dtype=np.float64) / patch + 0.5).astype(np.int64), 1, grid - 1)
    rows = np.stack([cy - 1, cy - 1, cy, cy], axis=1)
    cols = np.stack([cx - 1, cx, cx - 1, cx], axis=1)
    return rows * grid + cols
