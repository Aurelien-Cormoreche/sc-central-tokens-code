"""Pure math helpers for the probing package: polygon shape descriptors (area,
eccentricity, orientation) and circular (mod-180) angle utilities for the
OrientationProbe (see probes/cell_level.py).

Shape descriptors are derived from the *continuous* polygon boundary (Xenium
cell/nucleus boundary polygons, see data/boundary_cache.py) via the standard
analytic polygon area/second-moment formulas (shoelace formula and its
second-moment extension), not from a rasterized mask. This is the same
underlying quantity skimage.measure.regionprops' eccentricity/orientation are
built from (the inertia/covariance tensor of a uniform-density lamina), just
computed directly from the polygon vertices instead of pixel counts, so the
result isn't sensitive to a rasterization grid choice.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon


def _ring_moments(coords: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Raw (about-origin) signed area and first/second moments of one polygon ring.

    coords: (M, 2) array of vertices; need not be explicitly closed (first vertex
    repeated as last) -- the np.roll below wraps the last vertex back to the first.
    Returns (A, Mx, My, Ixx, Iyy, Ixy): signed area, first moments (Mx = A * cx,
    My = A * cy, so summing raw moments across rings before dividing by the summed
    area gives the correct combined centroid), and raw second moments about the
    origin, where Ixx = integral(y^2 dA), Iyy = integral(x^2 dA), Ixy = integral(x*y dA).
    """
    x = coords[:, 0]
    y = coords[:, 1]
    x1, y1 = x, y
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x1 * y2 - x2 * y1

    A = 0.5 * np.sum(cross)
    Mx = np.sum((x1 + x2) * cross) / 6.0
    My = np.sum((y1 + y2) * cross) / 6.0
    Ixx = np.sum((y1 ** 2 + y1 * y2 + y2 ** 2) * cross) / 12.0
    Iyy = np.sum((x1 ** 2 + x1 * x2 + x2 ** 2) * cross) / 12.0
    Ixy = np.sum((x1 * y2 + 2 * x1 * y1 + 2 * x2 * y2 + x2 * y1) * cross) / 24.0
    return A, Mx, My, Ixx, Iyy, Ixy


def _polygon_parts(geom: Polygon | MultiPolygon) -> list[Polygon]:
    return list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]


def polygon_shape_descriptors(geom: Polygon | MultiPolygon) -> tuple[float, float, float] | None:
    """(area, eccentricity, orientation_deg) of `geom`, treated as a uniform-density
    lamina -- interior rings (holes), if any, are included: shapely orients holes
    opposite the exterior ring, so their signed moments subtract automatically
    when summed alongside it.

    eccentricity is in [0, 1) (0 = circle, -> 1 = a needle); orientation_deg is
    the major-axis angle in degrees, mod 180 (a nucleus/cell has no front/back --
    same convention as src/python/ot/centering_correction.py's
    `_orientation_and_elongation`).

    Returns None for a degenerate shape: zero area, or a perfectly symmetric one
    (e.g. a circle or square) where the principal axis is undefined -- callers
    should drop the cell rather than invent an arbitrary orientation.
    """
    A = Mx = My = Ixx = Iyy = Ixy = 0.0
    for part in _polygon_parts(geom):
        rings = [np.asarray(part.exterior.coords)] + [np.asarray(r.coords) for r in part.interiors]
        for ring in rings:
            if len(ring) < 3:
                continue
            a, mx, my, ixx, iyy, ixy = _ring_moments(ring)
            A += a
            Mx += mx
            My += my
            Ixx += ixx
            Iyy += iyy
            Ixy += ixy

    area = abs(A)
    if area < 1e-6 or A == 0:
        return None

    cx, cy = Mx / A, My / A
    # Central second moments (parallel-axis theorem), expressed as a covariance-like
    # tensor Cov = [[Iyy_c, Ixy_c], [Ixy_c, Ixx_c]] over (x, y): Iyy_c = integral((x-cx)^2 dA)
    # is the "variance along x" term (Cov_xx), Ixx_c = integral((y-cy)^2 dA) is the
    # "variance along y" term (Cov_yy), and Ixy_c is the cross term (Cov_xy) directly
    # (no extra sign flip, unlike the physics "moment of inertia" tensor convention) --
    # framing it this way keeps the principal-axis formula unambiguous, exactly the PCA
    # of the shape's own area distribution.
    ixx_c = Ixx - A * cy ** 2
    iyy_c = Iyy - A * cx ** 2
    ixy_c = Ixy - A * cx * cy
    cov_xx, cov_yy, cov_xy = iyy_c, ixx_c, ixy_c

    common = np.hypot(cov_xx - cov_yy, 2 * cov_xy)
    lam_max = 0.5 * (cov_xx + cov_yy) + 0.5 * common
    lam_min = 0.5 * (cov_xx + cov_yy) - 0.5 * common
    if lam_max <= 1e-9 or common < 1e-9:
        # common ~ 0 -> cov_xx == cov_yy and cov_xy == 0: rotationally symmetric
        # (circle/square-like), no well-defined major axis.
        return None

    eccentricity = float(np.sqrt(max(0.0, 1.0 - lam_min / lam_max)))
    orientation_deg = float(np.degrees(0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)) % 180.0)
    return float(area), eccentricity, orientation_deg


# ── circular (mod-180) angle helpers, for OrientationProbe ─────────────────────

def angle_to_doubled_unit_vector(angle_deg: np.ndarray) -> np.ndarray:
    """angle_deg (mod 180, no front/back) -> (N, 2) [cos(2*theta), sin(2*theta)],
    so a linear (ridge) regressor can target it without the 180/0 degree
    wrap-around discontinuity a raw angle would have."""
    theta2 = np.deg2rad(np.asarray(angle_deg)) * 2.0
    return np.stack([np.cos(theta2), np.sin(theta2)], axis=1)


def doubled_unit_vector_to_angle(vec: np.ndarray) -> np.ndarray:
    """Inverse of angle_to_doubled_unit_vector: (N, 2) [cos(2*theta), sin(2*theta)]
    (need not be unit-norm -- e.g. a raw ridge-regression prediction) -> angle in
    degrees, mod 180."""
    vec = np.asarray(vec)
    theta2 = np.arctan2(vec[:, 1], vec[:, 0])
    return np.degrees(theta2 / 2.0) % 180.0


def circular_angle_error_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Smallest-magnitude difference between two mod-180 angles, in [0, 90] degrees
    (same convention as src/python/ot/centering_correction.py's `_angle_diff_deg`)."""
    d = np.abs(np.asarray(a_deg) - np.asarray(b_deg)) % 180.0
    return np.minimum(d, 180.0 - d)
