"""Centering correction between Xenium nucleus polygons and CellViT nucleus polygons.

Xenium nucleus centers, projected into H&E pixel space via the registered
Xenium<->H&E transform, are off from the true nucleus position on the H&E image by
up to ~60px with no consistent direction (see notebooks/diagnose_patch_coordinates.ipynb).
CellViT detects nuclei directly on the H&E image, so a CellViT nucleus close in
shape/position to a given Xenium nucleus is a good estimate of that nucleus's *true*
H&E position.

Pipeline (see `run_centering_correction` for the orchestrator):

1. `merge_touching_polygons` -- nuclei whose polygons touch/overlap are segmentation
   fragments of what is conceptually a single nucleus; union them before matching.
   Applied separately to the Xenium set and the CellViT set.
2. `match_merged_cells` -- for every merged Xenium nucleus, look at merged CellViT
   nuclei within `match_radius_px` and pick the best match by a weighted combination
   of three criteria (weights are all plain fields on `MatchingWeights` so they can be
   tuned by hand): centroid-aligned IoU (shape similarity with the position offset
   factored out -- the two polygons are compared after sliding one onto the other's
   centroid, since raw-position IoU would conflate "same nucleus" with "small
   registration error", which is exactly the unknown we're solving for), centroid
   distance, and shape-orientation agreement, down-weighted for round nuclei and
   up-weighted for spindle-shaped ones (see `_orientation_and_elongation`) since a
   round nucleus's long axis is essentially noise. The offset of a match is
   `cellvit_centroid - xenium_centroid` -- the vector to *add* to a Xenium-derived
   H&E coordinate to move it onto the true nucleus.
3. `average_offsets` -- per merged Xenium cell, average the raw offsets (from step 2)
   of every merged Xenium cell within `averaging_radius_px`, smoothing out per-cell
   matching noise while keeping the correction local.
4. `finalize_offsets` -- broadcast each merged cell's averaged offset back onto every
   original (pre-merge) Xenium `cell_id`.

Step 2 is the wall-clock bottleneck (per-pair polygon IoU + orientation) and is
parallelized across `n_workers` processes. Steps 1 and 3 are single vectorized
STRtree/GEOS calls (a bulk self-join plus a groupby-mean) -- already sub-second even
for ~10^5 cells, so chunked multiprocessing would only add process-spawn overhead
there and is intentionally not used.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely import STRtree, affinity
from shapely.geometry import MultiPolygon, Point, Polygon, shape as shapely_shape
from shapely.validation import make_valid

from src.python.cellvit_outputs_processing.fix_geojson import fix_geojson
from src.python.code_configs import paths

# --- default data roots, read from environment variables (see
# src/python/code_configs/paths.py and .env.example); matching
# notebooks/diagnose_patch_coordinates.ipynb ---
XENIUM_ROOT = paths.XENIUM_OUTPUT_ROOT
CELLVIT_OUTPUT_ROOT = paths.CELLVIT_GEOJSON_OUTPUT_ROOT


# ------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------

@dataclass
class MatchingWeights:
    """Weights for the merged-cell matching score in `match_merged_cells`.

    score = iou * centroid_aligned_iou
          - distance * clip(centroid_dist / distance_norm_px, 0, 1)
          - orientation * orientation_confidence * clip(angle_diff_deg / orientation_norm_deg, 0, 1)

    `centroid_aligned_iou` is the IoU of the two polygons after translating the
    CellViT one onto the Xenium centroid -- i.e. shape similarity with the (unknown)
    position offset factored out, rather than overlap at the current, misaligned
    positions.

    `orientation_confidence = min(elongation(xenium_geom), elongation(cellvit_geom))`
    in [0, 1) -- how much the orientation criterion actually counts for this pair.
    `elongation` (see `_orientation_and_elongation`) is ~0 for round/blob-shaped
    nuclei (whose long axis is essentially noise) and approaches 1 for spindle-shaped
    ones (where the orientation is a reliable, discriminating signal); using the min
    of both shapes means the pair only gets full orientation weight when *both* are
    clearly elongated.

    All fields are plain floats meant to be tuned by hand (e.g. from a notebook or
    the CLI below) -- set a weight to 0 to disable that criterion entirely.
    """
    iou: float = 1.0
    distance: float = 1.0
    orientation: float = 1.0
    distance_norm_px: float = 100.0
    orientation_norm_deg: float = 90.0


@dataclass
class CenteringCorrectionConfig:
    match_radius_px: float = 100.0
    averaging_radius_px: float = 112.0
    weights: MatchingWeights = field(default_factory=MatchingWeights)
    n_workers: int = 32
    chunk_size: int = 200


@dataclass
class CenteringCorrectionResult:
    xenium_gdf: gpd.GeoDataFrame
    cellvit_gdf: gpd.GeoDataFrame
    xenium_merge_map: dict
    xenium_merged_gdf: gpd.GeoDataFrame
    cellvit_merge_map: dict
    cellvit_merged_gdf: gpd.GeoDataFrame
    match_df: pd.DataFrame
    averaged_df: pd.DataFrame
    final_df: pd.DataFrame


# ------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------

def resolve_mpp(sample: str, xenium_root: os.PathLike = XENIUM_ROOT) -> float:
    """Read the Xenium morphology-image pixel size (microns/px) for `sample`."""
    exp_path = xenium_root / sample / 'experiment.xenium'
    with open(exp_path) as f:
        exp = json.load(f)
    for key in ('pixel_size', 'pixel_size_um', 'morphology_pixel_size'):
        if key in exp:
            return float(exp[key])
    raise KeyError(f'no known pixel-size key found in {exp_path}: {list(exp.keys())}')


def load_xenium_nuclei(sample: str, mpp: Optional[float] = None,
                        xenium_root: os.PathLike = XENIUM_ROOT) -> gpd.GeoDataFrame:
    """Xenium nucleus polygons for `sample`, in H&E pixel space (`geometry`), keyed by
    the native Xenium `cell_id`. Assumes the Xenium morphology image and the
    registered H&E already share a pixel grid (post-registration), so only the
    micron -> pixel scale (`mpp`) needs to be applied -- no additional affine
    transform (see notebooks/diagnose_patch_coordinates.ipynb, section 4).
    """
    if mpp is None:
        mpp = resolve_mpp(sample, xenium_root)

    nuc_bounds = pd.read_parquet(xenium_root / sample / 'nucleus_boundaries.parquet')
    xcol = next(c for c in nuc_bounds.columns if 'vertex_x' in c or c == 'x')
    ycol = next(c for c in nuc_bounds.columns if 'vertex_y' in c or c == 'y')

    cell_ids = []
    polygons = []
    for cell_id, group in nuc_bounds.groupby('cell_id', sort=False):
        xs = group[xcol].to_numpy() / mpp
        ys = group[ycol].to_numpy() / mpp
        if len(xs) < 3:
            continue
        poly = Polygon(np.column_stack([xs, ys]))
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            continue
        cell_ids.append(cell_id)
        polygons.append(poly)

    return gpd.GeoDataFrame({'cell_id': cell_ids, 'geometry': polygons})


def _explode_cellvit_feature(feat: dict) -> list[dict]:
    """One CellViT geojson Feature holds a class-level MultiPolygon unioning every
    nucleus of that PanNuke class slide-wide (see
    notebooks/diagnose_patch_coordinates.ipynb, section 9); explode it into one row
    per individual nucleus.
    """
    label = feat.get('properties', {}).get('classification', {}).get('name', '')
    geom = shapely_shape(feat['geometry'])
    sub_geoms = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    rows = []
    for sub in sub_geoms:
        if sub.is_empty or not isinstance(sub, Polygon):
            continue
        rows.append({'label': label, 'geometry': sub})
    return rows


def load_cellvit_nuclei(sample: str, cellvit_output_root: os.PathLike = CELLVIT_OUTPUT_ROOT,
                         force_refix: bool = False) -> gpd.GeoDataFrame:
    """CellViT-detected nucleus polygons for `sample`, already in H&E pixel space (no
    unit conversion needed). `cellvit_id` is assigned sequentially since CellViT's
    geojson has no native per-nucleus id.
    """
    out_dir = cellvit_output_root / f'{sample}_registered_HE.ome'
    raw_path = out_dir / 'cells.geojson'
    fixed_path = out_dir / 'cells.fixed.geojson'
    if force_refix or not fixed_path.exists():
        fix_geojson(str(raw_path), str(fixed_path))
    with open(fixed_path) as f:
        features = json.load(f)

    rows = []
    for feat in features:
        rows.extend(_explode_cellvit_feature(feat))

    gdf = gpd.GeoDataFrame(rows)
    gdf.insert(0, 'cellvit_id', [f'{sample}_cvit_{i}' for i in range(len(gdf))])
    return gdf


# ------------------------------------------------------------------------------------
# Step 1 -- merge touching/overlapping nuclei
# ------------------------------------------------------------------------------------

def merge_touching_polygons(gdf: gpd.GeoDataFrame, id_col: str,
                             merged_id_prefix: str) -> tuple[dict, gpd.GeoDataFrame]:
    """Union nuclei whose polygons touch or overlap into a single merged nucleus.

    Returns
    -------
    id_to_merged : maps every value of `gdf[id_col]` to a merged id.
    merged_gdf : one row per merged nucleus, columns [merged_id, geometry, n_members,
        member_ids].
    """
    n = len(gdf)
    geoms = gdf.geometry.to_numpy()
    ids = gdf[id_col].to_numpy()

    tree = STRtree(geoms)
    left, right = tree.query(geoms, predicate='intersects')
    # `left == right` is every geometry intersecting itself; connected_components
    # doesn't need the diagonal, and the STRtree self-join already returns each
    # off-diagonal edge in both directions, so the graph is already symmetric.
    keep = left != right
    edges_i, edges_j = left[keep], right[keep]
    adjacency = coo_matrix((np.ones(len(edges_i), dtype=bool), (edges_i, edges_j)), shape=(n, n))
    n_components, labels = connected_components(adjacency, directed=False)

    id_to_merged = {}
    merged_rows = []
    for label in range(n_components):
        member_mask = labels == label
        member_ids = ids[member_mask].tolist()
        merged_id = f'{merged_id_prefix}{label}'
        for mid in member_ids:
            id_to_merged[mid] = merged_id
        merged_geom = shapely.union_all(geoms[member_mask]) if member_mask.sum() > 1 else geoms[member_mask][0]
        merged_rows.append({
            'merged_id': merged_id,
            'geometry': merged_geom,
            'n_members': int(member_mask.sum()),
            'member_ids': member_ids,
        })

    merged_gdf = gpd.GeoDataFrame(merged_rows).set_index('merged_id', drop=False)
    return id_to_merged, merged_gdf


# ------------------------------------------------------------------------------------
# Step 2 -- weighted IoU / distance / orientation matching (parallel)
# ------------------------------------------------------------------------------------

def _orientation_and_elongation(geom) -> tuple[float, float]:
    """(long-axis angle in degrees mod 180 -- a nucleus has no front/back, elongation
    in [0, 1)) of `geom`'s minimum rotated rectangle. `elongation = 1 - short_side /
    long_side`: ~0 for round/square (blob-shaped) nuclei, where the long-axis angle is
    essentially arbitrary/noise; approaches 1 for spindle-shaped nuclei, where it's a
    reliable, discriminating signal. Works for Polygon and MultiPolygon alike.
    """
    rect = geom.minimum_rotated_rectangle
    coords = np.asarray(rect.exterior.coords)
    if len(coords) < 4:
        return 0.0, 0.0
    edges = np.diff(coords[:4], axis=0)
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    long_side, short_side = float(lengths.max()), float(lengths.min())
    longest = edges[np.argmax(lengths)]
    angle = float(np.degrees(np.arctan2(longest[1], longest[0])) % 180.0)
    elongation = (1.0 - short_side / long_side) if long_side > 0 else 0.0
    return angle, elongation


def _angle_diff_deg(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


_WORKER_STATE: dict = {}


def _init_matching_worker(cellvit_ids, cellvit_geoms, cellvit_centroids, cellvit_orientations,
                           cellvit_elongations):
    global _WORKER_STATE
    _WORKER_STATE = {
        'ids': cellvit_ids,
        'geoms': cellvit_geoms,
        'centroid_tree': STRtree([Point(xy) for xy in cellvit_centroids]),
        'centroids': cellvit_centroids,
        'orientations': cellvit_orientations,
        'elongations': cellvit_elongations,
    }


def _match_chunk(args) -> list[dict]:
    (xenium_ids, xenium_geoms, xenium_centroids, xenium_orientations, xenium_elongations,
     match_radius_px, weights) = args

    cellvit_ids = _WORKER_STATE['ids']
    cellvit_geoms = _WORKER_STATE['geoms']
    cellvit_centroids = _WORKER_STATE['centroids']
    cellvit_orientations = _WORKER_STATE['orientations']
    cellvit_elongations = _WORKER_STATE['elongations']
    tree = _WORKER_STATE['centroid_tree']

    rows = []
    for i in range(len(xenium_ids)):
        xen_geom = xenium_geoms[i]
        xen_centroid = xenium_centroids[i]
        candidate_idx = tree.query(Point(xen_centroid), predicate='dwithin', distance=match_radius_px)

        if len(candidate_idx) == 0:
            rows.append({
                'merged_id': xenium_ids[i], 'matched_cellvit_id': None,
                'iou': np.nan, 'distance': np.nan, 'orientation_diff': np.nan,
                'orientation_confidence': np.nan, 'score': np.nan,
                'offset_dx': np.nan, 'offset_dy': np.nan,
            })
            continue

        cand_geoms = cellvit_geoms[candidate_idx]
        cand_centroids = cellvit_centroids[candidate_idx]

        dxy = cand_centroids - xen_centroid
        dist = np.hypot(dxy[:, 0], dxy[:, 1])

        # centroid-aligned IoU: slide each candidate onto the Xenium centroid before
        # comparing shapes, so IoU measures shape similarity independent of the
        # (unknown) registration offset we're trying to correct -- IoU at the raw,
        # misaligned positions would otherwise punish a correct match just for being
        # far from the Xenium cell, which is exactly what `distance` already covers.
        aligned_cands = [affinity.translate(g, xoff=float(-d[0]), yoff=float(-d[1]))
                          for g, d in zip(cand_geoms, dxy)]
        inter_area = np.array([xen_geom.intersection(g).area for g in aligned_cands])
        union_area = np.array([xen_geom.union(g).area for g in aligned_cands])
        iou = np.divide(inter_area, union_area, out=np.zeros_like(inter_area), where=union_area > 0)

        angle_diff = _angle_diff_deg(cellvit_orientations[candidate_idx], xenium_orientations[i])
        # only trust the orientation criterion when both the Xenium and candidate
        # CellViT nucleus are visibly elongated -- a round shape's long axis is noise,
        # so a mismatch there shouldn't count against an otherwise good candidate.
        orientation_confidence = np.minimum(cellvit_elongations[candidate_idx], xenium_elongations[i])

        score = (
            weights.iou * iou
            - weights.distance * np.clip(dist / weights.distance_norm_px, 0.0, 1.0)
            - weights.orientation * orientation_confidence
              * np.clip(angle_diff / weights.orientation_norm_deg, 0.0, 1.0)
        )

        best = int(np.argmax(score))
        rows.append({
            'merged_id': xenium_ids[i],
            'matched_cellvit_id': cellvit_ids[candidate_idx[best]],
            'iou': float(iou[best]),
            'distance': float(dist[best]),
            'orientation_diff': float(angle_diff[best]),
            'orientation_confidence': float(orientation_confidence[best]),
            'score': float(score[best]),
            'offset_dx': float(dxy[best, 0]),
            'offset_dy': float(dxy[best, 1]),
        })
    return rows


def match_merged_cells(xenium_merged_gdf: gpd.GeoDataFrame, cellvit_merged_gdf: gpd.GeoDataFrame,
                        config: CenteringCorrectionConfig) -> pd.DataFrame:
    """For every row of `xenium_merged_gdf`, pick the best-matching row of
    `cellvit_merged_gdf` within `config.match_radius_px` (centroid distance) by the
    weighted centroid-aligned-IoU/distance/elongation-scaled-orientation score in
    `config.weights` (see `MatchingWeights`). Returns one row per input merged Xenium
    cell (unmatched cells get NaN score/offset columns).
    """
    xenium_ids = xenium_merged_gdf['merged_id'].to_numpy()
    xenium_geoms = xenium_merged_gdf.geometry.to_numpy()
    xenium_centroids = np.array([[g.centroid.x, g.centroid.y] for g in xenium_geoms])
    xenium_orientations, xenium_elongations = np.array(
        [_orientation_and_elongation(g) for g in xenium_geoms]).T

    cellvit_ids = cellvit_merged_gdf['merged_id'].to_numpy()
    cellvit_geoms = cellvit_merged_gdf.geometry.to_numpy()
    cellvit_centroids = np.array([[g.centroid.x, g.centroid.y] for g in cellvit_geoms])
    cellvit_orientations, cellvit_elongations = np.array(
        [_orientation_and_elongation(g) for g in cellvit_geoms]).T

    n = len(xenium_ids)
    chunk_size = max(1, config.chunk_size)
    chunks = [
        (
            xenium_ids[s:s + chunk_size], xenium_geoms[s:s + chunk_size],
            xenium_centroids[s:s + chunk_size], xenium_orientations[s:s + chunk_size],
            xenium_elongations[s:s + chunk_size], config.match_radius_px, config.weights,
        )
        for s in range(0, n, chunk_size)
    ]

    n_workers = max(1, min(config.n_workers, len(chunks), os.cpu_count() or 1))
    if n_workers == 1:
        _init_matching_worker(cellvit_ids, cellvit_geoms, cellvit_centroids, cellvit_orientations,
                               cellvit_elongations)
        results = [row for chunk in chunks for row in _match_chunk(chunk)]
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_matching_worker,
            initargs=(cellvit_ids, cellvit_geoms, cellvit_centroids, cellvit_orientations,
                      cellvit_elongations),
        ) as executor:
            results = [row for chunk_rows in executor.map(_match_chunk, chunks) for row in chunk_rows]

    return pd.DataFrame(results)


# ------------------------------------------------------------------------------------
# Step 3 -- average offsets over nearby merged cells
# ------------------------------------------------------------------------------------

def average_offsets(xenium_merged_gdf: gpd.GeoDataFrame, match_df: pd.DataFrame,
                     config: CenteringCorrectionConfig) -> pd.DataFrame:
    """For every merged Xenium cell, average the raw `offset_dx/offset_dy` (from
    `match_merged_cells`) of every merged Xenium cell -- including itself -- within
    `config.averaging_radius_px`, smoothing per-cell matching noise while keeping the
    correction local. Cells with zero matched neighbors in range get NaN averaged
    offsets (i.e. no correction applied downstream).

    A single vectorized STRtree self-join plus a groupby-mean already runs in well
    under a second for realistic per-WSI cell counts, so this step deliberately does
    not use the chunked multiprocessing that `match_merged_cells` needs.
    """
    merged_ids = xenium_merged_gdf['merged_id'].to_numpy()
    centroids = np.array([[g.centroid.x, g.centroid.y] for g in xenium_merged_gdf.geometry])

    match = match_df.set_index('merged_id').reindex(merged_ids)
    has_offset = match['offset_dx'].notna().to_numpy()

    tree = STRtree([Point(xy) for xy in centroids])
    anchor_idx, neighbor_idx = tree.query(
        [Point(xy) for xy in centroids], predicate='dwithin', distance=config.averaging_radius_px,
    )
    keep = has_offset[neighbor_idx]
    anchor_idx, neighbor_idx = anchor_idx[keep], neighbor_idx[keep]

    dx = match['offset_dx'].to_numpy()
    dy = match['offset_dy'].to_numpy()

    n = len(merged_ids)
    sum_dx = np.bincount(anchor_idx, weights=dx[neighbor_idx], minlength=n)
    sum_dy = np.bincount(anchor_idx, weights=dy[neighbor_idx], minlength=n)
    count = np.bincount(anchor_idx, minlength=n)

    with np.errstate(invalid='ignore'):
        avg_dx = np.where(count > 0, sum_dx / count, np.nan)
        avg_dy = np.where(count > 0, sum_dy / count, np.nan)

    return pd.DataFrame({
        'merged_id': merged_ids,
        'avg_offset_dx': avg_dx,
        'avg_offset_dy': avg_dy,
        'n_neighbors_used': count,
    })


# ------------------------------------------------------------------------------------
# Step 4 -- broadcast merged-cell offsets back to original cell_ids
# ------------------------------------------------------------------------------------

def finalize_offsets(xenium_gdf: gpd.GeoDataFrame, xenium_merge_map: dict,
                      averaged_df: pd.DataFrame) -> pd.DataFrame:
    final = xenium_gdf[['cell_id']].copy()
    final['merged_id'] = final['cell_id'].map(xenium_merge_map)
    final = final.merge(averaged_df, on='merged_id', how='left')
    final = final.rename(columns={'avg_offset_dx': 'offset_dx', 'avg_offset_dy': 'offset_dy'})
    final['matched'] = final['offset_dx'].notna()
    return final


# ------------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------------

def run_centering_correction(sample: str, config: Optional[CenteringCorrectionConfig] = None,
                              xenium_gdf: Optional[gpd.GeoDataFrame] = None,
                              cellvit_gdf: Optional[gpd.GeoDataFrame] = None) -> CenteringCorrectionResult:
    """Run the full 4-step pipeline for `sample`.

    `xenium_gdf`/`cellvit_gdf` can be passed in pre-loaded (e.g. from a notebook that
    already loaded them once) to skip re-reading from disk.
    """
    config = config or CenteringCorrectionConfig()

    if xenium_gdf is None:
        xenium_gdf = load_xenium_nuclei(sample)
    if cellvit_gdf is None:
        cellvit_gdf = load_cellvit_nuclei(sample)

    xenium_merge_map, xenium_merged_gdf = merge_touching_polygons(xenium_gdf, 'cell_id', 'xen_m')
    cellvit_merge_map, cellvit_merged_gdf = merge_touching_polygons(cellvit_gdf, 'cellvit_id', 'cvit_m')

    match_df = match_merged_cells(xenium_merged_gdf, cellvit_merged_gdf, config)
    averaged_df = average_offsets(xenium_merged_gdf, match_df, config)
    final_df = finalize_offsets(xenium_gdf, xenium_merge_map, averaged_df)

    return CenteringCorrectionResult(
        xenium_gdf=xenium_gdf, cellvit_gdf=cellvit_gdf,
        xenium_merge_map=xenium_merge_map, xenium_merged_gdf=xenium_merged_gdf,
        cellvit_merge_map=cellvit_merge_map, cellvit_merged_gdf=cellvit_merged_gdf,
        match_df=match_df, averaged_df=averaged_df, final_df=final_df,
    )


def save_result(result: CenteringCorrectionResult, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.final_df.to_parquet(out / 'final_offsets.parquet')
    result.match_df.to_parquet(out / 'match_results.parquet')
    result.averaged_df.to_parquet(out / 'averaged_offsets.parquet')
    result.xenium_merged_gdf.drop(columns='member_ids').to_parquet(out / 'xenium_merged.parquet')
    result.cellvit_merged_gdf.drop(columns='member_ids').to_parquet(out / 'cellvit_merged.parquet')
    with open(out / 'xenium_merge_map.json', 'w') as f:
        json.dump(result.xenium_merge_map, f)


# ------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=str, required=True, help='GSM sample id, e.g. GSM7990540')
    parser.add_argument('--match-radius', type=float, default=100.0)
    parser.add_argument('--averaging-radius', type=float, default=112.0)
    parser.add_argument('--w-iou', type=float, default=1.0)
    parser.add_argument('--w-distance', type=float, default=1.0)
    parser.add_argument('--w-orientation', type=float, default=1.0)
    parser.add_argument('--distance-norm-px', type=float, default=100.0)
    parser.add_argument('--orientation-norm-deg', type=float, default=90.0)
    parser.add_argument('--n-workers', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=200)
    parser.add_argument('--output', type=str, required=True, help='directory to write results to')
    args = parser.parse_args()

    cfg = CenteringCorrectionConfig(
        match_radius_px=args.match_radius,
        averaging_radius_px=args.averaging_radius,
        weights=MatchingWeights(
            iou=args.w_iou, distance=args.w_distance, orientation=args.w_orientation,
            distance_norm_px=args.distance_norm_px, orientation_norm_deg=args.orientation_norm_deg,
        ),
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
    )

    print(f'--- Running centering correction for {args.sample} ---')
    result = run_centering_correction(args.sample, cfg)
    print(f"{result.final_df['matched'].mean():.1%} of {len(result.final_df)} Xenium cells got a correction "
          f"(median |offset| = {np.hypot(result.final_df['offset_dx'], result.final_df['offset_dy']).median():.1f} px)")
    save_result(result, args.output)
    print(f'--- Wrote results to {args.output} ---')
