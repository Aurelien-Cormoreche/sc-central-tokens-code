"""Xenium cell/nucleus boundary polygons, mapped into H&E pixel space.

Cell boundaries come from Xenium's `cell_boundaries.csv.gz` (columns: cell_id,
vertex_x, vertex_y, in Xenium physical microns); nucleus boundaries from
`nucleus_boundaries.parquet` (same column layout, see
src/python/ot/centering_correction.py's `load_xenium_nuclei`). Both are mapped to
H&E pixel space via the inverse of the 3x3 Xenium<->H&E affine alignment matrix,
the same transform ResizedCellDataset._box_from_vertices uses for its per-cell crop
bounding box -- this module keeps the full polygon instead of just its bounding box,
so PatchDataset can test which grid tokens actually overlap it (see
InferenceProvider.pool_boundary_tokens).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from os import PathLike
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

# Xenium instrument pixel size (µm/px) -- matches ResizedCellDataset's _XENIUM_MPP.
XENIUM_MPP = 0.2125


def load_alignment_matrix_inv(alignment_matrix_path: PathLike) -> np.ndarray:
    M = np.genfromtxt(str(alignment_matrix_path), delimiter=',')
    assert M.shape == (3, 3), f"Expected a 3x3 alignment matrix, got {M.shape}"
    return np.linalg.inv(M)


def _normalize_cell_id(cell_id_raw) -> str:
    cell_id_str = cell_id_raw.decode('utf-8') if isinstance(cell_id_raw, bytes) else str(cell_id_raw)
    return cell_id_str.split('_')[-1]


class BoundaryPolygons:
    """Loads a Xenium boundary file (cell_boundaries.csv.gz or
    nucleus_boundaries.parquet) once and looks up per-cell polygons, in H&E pixel
    space, by (possibly sample-prefixed) cell_id.
    """

    def __init__(self, boundaries_path: PathLike, M_inv: np.ndarray, xenium_mpp: float = XENIUM_MPP):
        self.M_inv = M_inv
        self.xenium_mpp = xenium_mpp
        path_str = str(boundaries_path)
        df = pd.read_parquet(path_str) if path_str.endswith('.parquet') else pd.read_csv(path_str)
        xcol = next(c for c in df.columns if 'vertex_x' in c or c == 'x')
        ycol = next(c for c in df.columns if 'vertex_y' in c or c == 'y')
        df = df.rename(columns={xcol: 'vertex_x', ycol: 'vertex_y'})
        df['cell_id'] = df['cell_id'].astype(str)
        self._by_id = df.groupby('cell_id', sort=False)

    def polygon_for(self, cell_id_raw) -> Polygon | MultiPolygon | None:
        cell_id_str = _normalize_cell_id(cell_id_raw)
        try:
            vertices = self._by_id.get_group(cell_id_str)
        except KeyError:
            return None
        if len(vertices) < 3:
            return None

        # Same homogeneous-coordinate inverse-alignment transform as
        # ResizedCellDataset._box_from_vertices, kept as a full polygon here.
        xen_hom = np.column_stack([
            vertices['vertex_x'].to_numpy(dtype=float),
            vertices['vertex_y'].to_numpy(dtype=float),
            np.full(len(vertices), self.xenium_mpp),
        ])
        pred_hom = (self.M_inv @ xen_hom.T).T
        pixel_x = pred_hom[:, 0] / self.xenium_mpp
        pixel_y = pred_hom[:, 1] / self.xenium_mpp

        try:
            poly = Polygon(np.column_stack([pixel_x, pixel_y]))
            if not poly.is_valid:
                poly = make_valid(poly)
        except Exception:
            return None
        poly = _polygonal_only(poly)
        return None if poly is None or poly.is_empty else poly


def _polygonal_only(geom) -> Polygon | MultiPolygon | None:
    """Reduce `geom` to just its polygonal area, dropping any degenerate
    points/lines `make_valid` can introduce for a self-intersecting or
    near-degenerate boundary ring (e.g. a bowtie collapses to a GeometryCollection
    of two triangles plus the crossing point). Returns None if nothing polygonal
    survives -- e.g. the ring degenerated entirely to a line.
    """
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        return unary_union(polys)
    return None


def token_overlap_mask(
    polygon: Polygon | MultiPolygon, x0: float, y0: float, token_size: float, grid_size: int,
) -> np.ndarray:
    """Boolean (grid_size, grid_size) mask, True at (row, col) where that token's
    pixel footprint -- [x0+col*token_size, x0+(col+1)*token_size) x
    [y0+row*token_size, y0+(row+1)*token_size) -- intersects `polygon`. `polygon`
    and (x0, y0) must be in the same absolute pixel frame (e.g. WSI pixel
    coordinates, as returned by PatchDataset.origin()).
    """
    minx, miny, maxx, maxy = polygon.bounds
    c_lo = max(0, int((minx - x0) // token_size))
    c_hi = min(grid_size - 1, int((maxx - x0) // token_size))
    r_lo = max(0, int((miny - y0) // token_size))
    r_hi = min(grid_size - 1, int((maxy - y0) // token_size))

    mask = np.zeros((grid_size, grid_size), dtype=bool)
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            cell = box(x0 + c * token_size, y0 + r * token_size,
                       x0 + (c + 1) * token_size, y0 + (r + 1) * token_size)
            if cell.intersects(polygon):
                mask[r, c] = True
    return mask


def nearest_token(polygon: Polygon | MultiPolygon, x0: float, y0: float, token_size: float, grid_size: int) -> tuple[int, int]:
    """Fallback (row, col) -- the grid token whose footprint center is closest to
    the polygon's centroid, clamped to the grid. Used when no token's footprint
    overlaps `polygon` (e.g. it pokes outside the fixed patch crop).
    """
    cx, cy = polygon.centroid.x, polygon.centroid.y
    col = int(round((cx - x0) / token_size - 0.5))
    row = int(round((cy - y0) / token_size - 0.5))
    return max(0, min(grid_size - 1, row)), max(0, min(grid_size - 1, col))
