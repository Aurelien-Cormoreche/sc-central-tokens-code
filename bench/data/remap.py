"""Slides whose lower part is a shifted copy of the part above (the raw file repeats itself; measured with
scripts/find_copy_offset.py): the cells there are read from the part the copy repeats.

configs/bench.yaml `remap: {slide: {from_row, dy, dx}}`. A cell whose centroid lies at a row >= from_row is read at
(x - dx, y - dy), moved again while it is still at a row >= from_row; its crop then comes from the original part,
in the right colours. Only where a cell is *read* changes: the manifest keeps its own position (cx, cy), and its
extraction position (ex, ey) is the read position. A crop must lie in the original part (rows < from_row): cells
whose crop would reach into the copy, and moved cells whose crop would leave the image, are dropped ("remap").
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def remap_for(cfg, slide: str):
    return (getattr(cfg, "remap", None) or {}).get(slide)


def remap_key(rm) -> Optional[list]:
    """What a dark-spot map records, so a changed remap redoes it."""
    return None if rm is None else [int(rm.from_row), int(rm.dy), int(rm.dx)]


def read_positions(rm, cx, cy):
    """(qx, qy, steps): native px where each cell is read; steps = how many times it was moved (0 = in place)."""
    qx, qy = np.array(cx, dtype=np.float64), np.array(cy, dtype=np.float64)
    steps = np.zeros(len(qx), dtype=np.int64)
    if rm is None:
        return qx, qy, steps
    while True:                                     # dy > 0 (config), so this ends
        m = qy >= rm.from_row
        if not m.any():
            return qx, qy, steps
        qx[m] -= rm.dx
        qy[m] -= rm.dy
        steps[m] += 1


def remap_drop(rm, rx, ry, moved, scale: float, margin: float, width_ext: int) -> np.ndarray:
    """Cells whose crop (`margin` extraction px around the rounded read position rx, ry) reaches row from_row, or,
    for moved cells, leaves the image."""
    if rm is None:
        return np.zeros(len(rx), dtype=bool)
    out = ry + margin > rm.from_row * scale
    out |= moved & ((rx - margin < 0) | (ry - margin < 0) | (rx + margin > width_ext))
    return out
