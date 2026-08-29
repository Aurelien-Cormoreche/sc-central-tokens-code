"""Export CellViT->Xenium matches as a `patch_coordinates.h5` per GSM sample.

For every GSM sample: match every CellViT-detected nucleus to its best Xenium
candidate (`cellvit_xenium_matching.match_cellvit_to_xenium`, one weighting,
`EXPORT_WEIGHTS`), keep only matches with `weighted_score > SCORE_THRESHOLD`, and for
every ground-truth Xenium cell_id (`GROUND_TRUTH_ROOT/{sample}/patch_coordinates.h5`,
Marc's pipeline output -- same schema this script writes, see
`src/python/extract_embeddings/data/patch_dataset.py`) that has a CellViT cell matched
to it, copy that ground-truth row's `cell_id`/`cell_type` ("cell_labels") unchanged
but replace `x_start`/`y_start` with the matched CellViT cell's own centroid-based
patch corner. Written out as four same-length top-level datasets `x_start`, `y_start`,
`cell_id`, `cell_type` -- the schema `PatchDataset`/`ResizedCellDataset` already read.

- `x_start`/`y_start` = `round(cellvit_centroid) - 112`, the top-left corner of the
  CellViT cell's 224x224-centered patch (nucleus center = `x_start + 112`,
  `y_start + 112`), same convention as the ground-truth file.
- `cell_id`/`cell_type` are copied verbatim from the ground-truth row for that Xenium
  cell -- i.e. not recomputed -- so `cell_id` keeps its existing sample-prefixed form
  (`f'{sample}_{xenium_cell_id}'`, see `notebooks/diagnose_patch_coordinates.ipynb`
  section 3) and `cell_type` keeps whatever cell-type label the ground truth already
  assigned. Since `match_cellvit_to_xenium` matches independently per CellViT cell (no
  1-to-1 constraint), more than one CellViT cell can claim the same Xenium cell_id --
  only the single highest-`weighted_score` CellViT cell per Xenium cell_id is kept, so
  the output stays one row per ground-truth cell.

Output written under $POSITIONS_CONVERTED_ROOT/{sample}/patch_coordinates.h5 (see
src/python/code_configs/paths.py and .env.example).

Requires XENIUM_PROCESSED_OUTPUT_ROOT, POSITIONS_CONVERTED_ROOT, XENIUM_OUTPUT_ROOT and
CELLVIT_GEOJSON_OUTPUT_ROOT to be set -- run on the cluster:
    python3 -m src.python.ot.export_matched_patch_coordinates
    python3 -m src.python.ot.export_matched_patch_coordinates --sample GSM7990540
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.python.ot.centering_correction import load_cellvit_nuclei, load_xenium_nuclei
from src.python.ot.cellvit_xenium_matching import MatchWeights, MatchingConfig, run_sample
from src.python.code_configs import paths

GSM_SAMPLES = [
    'GSM7990532', 'GSM7990533', 'GSM7990534', 'GSM7990537', 'GSM7990540',
    'GSM7990541', 'GSM7990543', 'GSM7990549', 'GSM7990550', 'GSM7990551',
    'GSM7990554', 'GSM7990556', 'GSM7990557', 'GSM7990558', 'GSM7990559',
]

RADIUS_PX = 80.0
EXPORT_WEIGHTS = MatchWeights(distance=1.0, orientation=1.2, iou=5.0)
SCORE_THRESHOLD = 4.5
PATCH_HALF = 112  # PatchDataset/MultiCellPatchDataset's 224x224-centered-patch convention

# Marc's pipeline output -- same root as extract_embeddings.py's _CELLS_INFO_ROOT.
GROUND_TRUTH_ROOT = paths.XENIUM_PROCESSED_OUTPUT_ROOT
OUTPUT_ROOT = paths.POSITIONS_CONVERTED_ROOT


def _decode_bytes_col(s: pd.Series) -> pd.Series:
    # h5py reads variable-length HDF5 strings back as Python bytes objects; same
    # helper as notebooks/diagnose_patch_coordinates.ipynb.
    return s.apply(lambda x: x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else x)


def load_ground_truth_patch_coordinates(sample: str, root: os.PathLike = GROUND_TRUTH_ROOT) -> pd.DataFrame:
    """Read the ground-truth `patch_coordinates.h5` for `sample` (`cell_id`, `x_start`,
    `y_start`, `cell_type`). `cell_id` there is Xenium's native id prefixed with the
    sample name -- `native_cell_id` strips that prefix (same convention as
    `notebooks/diagnose_patch_coordinates.ipynb` section 3) so it can be joined
    against `xenium_gdf['cell_id']` / `match_cellvit_to_xenium`'s
    `matched_xenium_cell_id`, which are both Xenium-native/unprefixed.
    """
    path = root / sample / 'patch_coordinates.h5'
    with h5py.File(path, 'r') as f:
        df = pd.DataFrame({k: f[k][:] for k in ('cell_id', 'x_start', 'y_start', 'cell_type')})
    df['cell_id'] = _decode_bytes_col(df['cell_id'])
    df['cell_type'] = _decode_bytes_col(df['cell_type'])

    prefix = f'{sample}_'
    df['native_cell_id'] = (df['cell_id'].str.removeprefix(prefix)
                             if df['cell_id'].str.startswith(prefix).any() else df['cell_id'])
    return df


def build_patch_coordinates(sample: str, matched_df: pd.DataFrame, cellvit_gdf,
                             ground_truth_df: pd.DataFrame, score_threshold: float = SCORE_THRESHOLD,
                             patch_half: int = PATCH_HALF) -> dict[str, np.ndarray]:
    """Pure data-shaping step (no I/O): `matched_df` is one weighting's output of
    `match_cellvit_to_xenium`/`run_sample` (columns `cellvit_id`,
    `matched_xenium_cell_id`, `weighted_score`, ...) for `sample`; `cellvit_gdf` is
    that sample's CellViT nuclei (for centroid lookup, indexed or not by
    `cellvit_id`); `ground_truth_df` is `load_ground_truth_patch_coordinates(sample)`'s
    output.

    Returns the four arrays ready to write into `patch_coordinates.h5`, one row per
    ground-truth Xenium cell_id that has a CellViT cell matched to it with
    `weighted_score > score_threshold` (picking the single best-scoring CellViT cell
    when more than one matched the same Xenium cell_id). `cell_id`/`cell_type` are
    copied verbatim from the ground truth; `x_start`/`y_start` are recomputed from the
    matched CellViT cell's own centroid.
    """
    matched = matched_df[matched_df['weighted_score'] > score_threshold].copy()
    if matched.empty:
        return {'x_start': np.array([], dtype=np.int32), 'y_start': np.array([], dtype=np.int32),
                'cell_id': np.array([]), 'cell_type': np.array([])}
    matched['matched_xenium_cell_id'] = matched['matched_xenium_cell_id'].astype(str)

    # keep only the single best-scoring CellViT cell per Xenium cell_id, since matching
    # is independent per CellViT cell (no 1-to-1 constraint) and the ground-truth join
    # below must stay one row per Xenium cell
    best_idx = matched.groupby('matched_xenium_cell_id')['weighted_score'].idxmax()
    best = matched.loc[best_idx, ['cellvit_id', 'matched_xenium_cell_id', 'weighted_score']]

    merged = ground_truth_df.merge(best, left_on='native_cell_id', right_on='matched_xenium_cell_id', how='inner')
    n_no_ground_truth = len(best) - len(merged)
    if n_no_ground_truth:
        print(f'  {n_no_ground_truth}/{len(best)} matched Xenium cells have no row in the '
              f'ground-truth patch_coordinates.h5, dropping them')

    if cellvit_gdf.index.name != 'cellvit_id':
        cellvit_gdf = cellvit_gdf.set_index('cellvit_id')
    matched_geoms = cellvit_gdf.loc[merged['cellvit_id'], 'geometry']
    x_start = np.round(matched_geoms.centroid.x.to_numpy() - patch_half).astype(np.int32)
    y_start = np.round(matched_geoms.centroid.y.to_numpy() - patch_half).astype(np.int32)

    return {
        'x_start': x_start,
        'y_start': y_start,
        'cell_id': merged['cell_id'].to_numpy(),      # ground truth's own cell_id, copied as-is
        'cell_type': merged['cell_type'].to_numpy(),   # ground truth's own cell_type, copied as-is
    }


def write_patch_coordinates(sample: str, data: dict[str, np.ndarray],
                             output_root: os.PathLike = OUTPUT_ROOT) -> Path:
    n = len(data['x_start'])
    assert all(len(v) == n for v in data.values()), 'x_start/y_start/cell_id/cell_type must be same length'

    out_dir = output_root / sample
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'patch_coordinates.h5'
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('x_start', data=data['x_start'])
        f.create_dataset('y_start', data=data['y_start'])
        f.create_dataset('cell_id', data=data['cell_id'], dtype=h5py.string_dtype(encoding='utf-8'))
        f.create_dataset('cell_type', data=data['cell_type'], dtype=h5py.string_dtype(encoding='utf-8'))
    return out_path


def export_sample(sample: str, n_workers: int = 32) -> None:
    print(f'--- {sample} ---')
    cellvit_gdf = load_cellvit_nuclei(sample)
    xenium_gdf = load_xenium_nuclei(sample)

    config = MatchingConfig(radius_px=RADIUS_PX, n_workers=n_workers)
    matched_df = run_sample(sample, {'export': EXPORT_WEIGHTS}, config,
                             xenium_gdf=xenium_gdf, cellvit_gdf=cellvit_gdf)['export']

    n_matched = int(matched_df['matched_xenium_cell_id'].notna().sum())
    n_above_threshold = int((matched_df['weighted_score'] > SCORE_THRESHOLD).sum())
    print(f'  {n_matched}/{len(matched_df)} CellViT cells matched within {RADIUS_PX:.0f}px, '
          f'{n_above_threshold} above weighted_score > {SCORE_THRESHOLD}')
    if n_above_threshold == 0:
        print(f'  nothing to write for {sample}, skipping')
        return

    ground_truth_df = load_ground_truth_patch_coordinates(sample)
    data = build_patch_coordinates(sample, matched_df, cellvit_gdf, ground_truth_df)
    if len(data['x_start']) == 0:
        print(f'  no matched cells found in ground-truth patch_coordinates.h5 for {sample}, skipping')
        return

    out_path = write_patch_coordinates(sample, data)
    print(f'  wrote {len(data["x_start"])} cells to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=str, default=None,
                         help='run a single GSM sample instead of every sample in GSM_SAMPLES')
    parser.add_argument('--n-workers', type=int, default=32,
                         help='parallel workers for the CellViT<->Xenium matching step (per sample)')
    args = parser.parse_args()

    for sample in ([args.sample] if args.sample else GSM_SAMPLES):
        export_sample(sample, n_workers=args.n_workers)
