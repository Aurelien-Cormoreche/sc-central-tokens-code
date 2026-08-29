"""CellViT -> Xenium per-cell matching scores.

For every CellViT-detected nucleus, look at Xenium nuclei within `radius_px` of its
centroid (H&E pixel space) and score each candidate on three independent criteria:

- **distance_score** = 1 / (1 + (centroid distance in px) / 15) -- close to 1 for
  near-coincident centroids, decaying towards 0 as distance grows; the `/15` sets how
  many px it takes to meaningfully discount the score (e.g. dist=15px -> score=0.5),
  and the `+1` keeps the score finite (and <= 1) at distance == 0, since the caller
  asked for "small distance -> large score, i.e. 1/distance".
- **orientation_score** = 1 - angle_diff_deg / 90, where `angle_diff_deg` is the
  (0-90 deg, a nucleus has no front/back) difference between the two shapes'
  minimum-rotated-rectangle long-axis angles (`_orientation_and_elongation` from
  `centering_correction.py`). 1 when the long axes agree, 0 when perpendicular.
- **iou_score** = centroid-aligned IoU -- the Xenium candidate is translated so its
  centroid lands exactly on the CellViT cell's centroid before intersection/union,
  so this measures shape similarity alone, independent of the (possibly large)
  registration offset between the two datasets.

All three are saved unweighted for the winning pair, plus `weighted_score`, a
weighted sum of the three under a caller-supplied `MatchWeights`. The winning Xenium
candidate for a given CellViT cell is whichever maximizes `weighted_score`, so it can
differ across weightings -- `match_cellvit_to_xenium` takes a dict of named
`MatchWeights` and re-picks the argmax per config from the same computed candidate
scores, without recomputing geometry per weighting.

Deliberately does not merge touching/fragmented nuclei first (unlike
`centering_correction.py`, which needs that for its offset-averaging step) -- this
module scores CellViT's raw per-nucleus detections directly against raw Xenium nuclei.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree, affinity
from shapely.geometry import Point

from src.python.ot.centering_correction import (
    CELLVIT_OUTPUT_ROOT,
    XENIUM_ROOT,
    _angle_diff_deg,
    _orientation_and_elongation,
    load_cellvit_nuclei,
    load_xenium_nuclei,
)


# ------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------

@dataclass
class MatchWeights:
    """Weights for `weighted_score = distance * distance_score + orientation *
    orientation_score + iou * iou_score`. Set a weight to 0 to disable that criterion.
    """
    distance: float = 1.0
    orientation: float = 1.0
    iou: float = 1.0


@dataclass
class MatchingConfig:
    radius_px: float = 80.0
    n_workers: int = 32
    chunk_size: int = 200


# ------------------------------------------------------------------------------------
# Parallel scoring
# ------------------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _init_worker(xenium_ids, xenium_geoms, xenium_centroids, xenium_orientations):
    global _WORKER_STATE
    _WORKER_STATE = {
        'ids': xenium_ids,
        'geoms': xenium_geoms,
        'centroids': xenium_centroids,
        'orientations': xenium_orientations,
        'centroid_tree': STRtree([Point(xy) for xy in xenium_centroids]),
    }


def _score_chunk(args) -> dict[str, list[dict]]:
    (cellvit_ids, cellvit_geoms, cellvit_centroids, cellvit_orientations,
     radius_px, weight_configs) = args

    xenium_ids = _WORKER_STATE['ids']
    xenium_geoms = _WORKER_STATE['geoms']
    xenium_centroids = _WORKER_STATE['centroids']
    xenium_orientations = _WORKER_STATE['orientations']
    tree = _WORKER_STATE['centroid_tree']

    rows: dict[str, list[dict]] = {name: [] for name in weight_configs}

    for i in range(len(cellvit_ids)):
        cvit_geom = cellvit_geoms[i]
        cvit_centroid = cellvit_centroids[i]
        candidate_idx = tree.query(Point(cvit_centroid), predicate='dwithin', distance=radius_px)

        if len(candidate_idx) == 0:
            for name in weight_configs:
                rows[name].append({
                    'cellvit_id': cellvit_ids[i], 'matched_xenium_cell_id': None,
                    'distance_score': np.nan, 'orientation_score': np.nan, 'iou_score': np.nan,
                    'weighted_score': np.nan,
                })
            continue

        cand_geoms = xenium_geoms[candidate_idx]
        cand_centroids = xenium_centroids[candidate_idx]

        dxy = cand_centroids - cvit_centroid
        dist = np.hypot(dxy[:, 0], dxy[:, 1])
        distance_score = 1.0 / (1.0 + dist / 15.0)

        # centroid-aligned IoU: slide each Xenium candidate onto the CellViT centroid
        # before comparing shapes, so IoU measures shape similarity independent of the
        # (unknown) registration offset between the two datasets.
        aligned_cands = [affinity.translate(g, xoff=float(-d[0]), yoff=float(-d[1]))
                          for g, d in zip(cand_geoms, dxy)]
        inter_area = np.array([cvit_geom.intersection(g).area for g in aligned_cands])
        union_area = np.array([cvit_geom.union(g).area for g in aligned_cands])
        iou_score = np.divide(inter_area, union_area, out=np.zeros_like(inter_area), where=union_area > 0)

        angle_diff = _angle_diff_deg(xenium_orientations[candidate_idx], cellvit_orientations[i])
        orientation_score = 1.0 - angle_diff / 90.0

        for name, w in weight_configs.items():
            weighted = (w.distance * distance_score + w.orientation * orientation_score
                        + w.iou * iou_score)
            best = int(np.argmax(weighted))
            rows[name].append({
                'cellvit_id': cellvit_ids[i],
                'matched_xenium_cell_id': xenium_ids[candidate_idx[best]],
                'distance_score': float(distance_score[best]),
                'orientation_score': float(orientation_score[best]),
                'iou_score': float(iou_score[best]),
                'weighted_score': float(weighted[best]),
            })

    return rows


def match_cellvit_to_xenium(cellvit_gdf: gpd.GeoDataFrame, xenium_gdf: gpd.GeoDataFrame,
                             weight_configs: dict[str, MatchWeights],
                             config: Optional[MatchingConfig] = None) -> dict[str, pd.DataFrame]:
    """For every row of `cellvit_gdf`, find the best-matching row of `xenium_gdf`
    within `config.radius_px` (centroid distance) for each entry of `weight_configs`.
    Returns one DataFrame per weighting, each with one row per input CellViT cell
    (unmatched cells -- no Xenium candidate within radius -- get NaN score columns).
    """
    config = config or MatchingConfig()

    cellvit_ids = cellvit_gdf['cellvit_id'].to_numpy()
    cellvit_geoms = cellvit_gdf.geometry.to_numpy()
    cellvit_centroids = np.array([[g.centroid.x, g.centroid.y] for g in cellvit_geoms])
    cellvit_orientations = np.array([_orientation_and_elongation(g)[0] for g in cellvit_geoms])

    xenium_ids = xenium_gdf['cell_id'].to_numpy()
    xenium_geoms = xenium_gdf.geometry.to_numpy()
    xenium_centroids = np.array([[g.centroid.x, g.centroid.y] for g in xenium_geoms])
    xenium_orientations = np.array([_orientation_and_elongation(g)[0] for g in xenium_geoms])

    n = len(cellvit_ids)
    chunk_size = max(1, config.chunk_size)
    chunks = [
        (
            cellvit_ids[s:s + chunk_size], cellvit_geoms[s:s + chunk_size],
            cellvit_centroids[s:s + chunk_size], cellvit_orientations[s:s + chunk_size],
            config.radius_px, weight_configs,
        )
        for s in range(0, n, chunk_size)
    ]

    n_workers = max(1, min(config.n_workers, len(chunks), os.cpu_count() or 1))
    if n_workers == 1:
        _init_worker(xenium_ids, xenium_geoms, xenium_centroids, xenium_orientations)
        chunk_results = [_score_chunk(c) for c in chunks]
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker,
            initargs=(xenium_ids, xenium_geoms, xenium_centroids, xenium_orientations),
        ) as executor:
            chunk_results = list(executor.map(_score_chunk, chunks))

    result = {}
    for name in weight_configs:
        rows = [row for chunk in chunk_results for row in chunk[name]]
        result[name] = pd.DataFrame(rows)
    return result


# ------------------------------------------------------------------------------------
# Per-sample orchestrator
# ------------------------------------------------------------------------------------

def run_sample(sample: str, weight_configs: dict[str, MatchWeights],
               config: Optional[MatchingConfig] = None,
               xenium_root: os.PathLike = XENIUM_ROOT, cellvit_output_root: os.PathLike = CELLVIT_OUTPUT_ROOT,
               xenium_gdf: Optional[gpd.GeoDataFrame] = None,
               cellvit_gdf: Optional[gpd.GeoDataFrame] = None) -> dict[str, pd.DataFrame]:
    """Run `match_cellvit_to_xenium` for one GSM sample, loading nuclei from disk
    unless already-loaded GeoDataFrames are passed in. Each result DataFrame gets a
    leading `sample` column so per-weighting results from multiple samples can be
    concatenated directly.
    """
    config = config or MatchingConfig()
    if xenium_gdf is None:
        xenium_gdf = load_xenium_nuclei(sample, xenium_root=xenium_root)
    if cellvit_gdf is None:
        cellvit_gdf = load_cellvit_nuclei(sample, cellvit_output_root=cellvit_output_root)

    result = match_cellvit_to_xenium(cellvit_gdf, xenium_gdf, weight_configs, config)
    for df in result.values():
        df.insert(0, 'sample', sample)
    return result


def save_results(results_by_weighting: dict[str, pd.DataFrame], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results_by_weighting.items():
        df.to_parquet(out / f'{name}.parquet')
