"""Find cells whose embedding comes from a black placeholder patch instead of the slide.

A failed WSI read (a corrupt tile) makes the extraction feed an all-black patch. Before
2fd4c62 the slide was not reopened after a failure, so every later patch read by that
DataLoader worker was black too. Every black patch gives the same embedding, so this looks
for a group of (near-)identical rows in each slide's embeddings_dataset.h5: real cells
never coincide. CPU-only (h5py + numpy), reads in chunks.

The WSI side (which tiles are unreadable, which cells a crop size would hit) is
src/python/extract_embeddings/check_wsi_regions.py.

Usage (repo root, .env sourced; slides = every WSI in a classifier config's splits):
    python scripts/check_black_patches.py UNI2 configs/train_within_cohort_split.yaml
    python scripts/check_black_patches.py UNI2_fov896 configs/paper/uni2_readout_fov_within_cohort_split.yaml
    python scripts/check_black_patches.py UNI2 configs/train_within_cohort_split.yaml --csv black_cells.csv
Exit code 1 if any slide has black-patch cells.
"""
import argparse
import csv
import os
import sys
from collections import Counter

import h5py
import numpy as np
import yaml

CHUNK = 16384   # rows per read: 100 MB at 1536-d float32


def _slides(config_path: str) -> list[str]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return sorted({w for sp in cfg.get('splits') or [] for side in ('train', 'val', 'test')
                   for w, v in (sp.get(side) or {}).items() if not isinstance(v, dict)})


def _dataset(f: h5py.File, key: str | None) -> h5py.Dataset:
    if key:
        return f[f'embeddings_{key}']
    for k in ('top_left', 'cls', 'nucleus', 'cell_token'):
        if f'embeddings_{k}' in f:
            return f[f'embeddings_{k}']
    return f[next(k for k in f if k.startswith('embeddings_'))]


def black_rows(dset: h5py.Dataset, rtol: float, min_group: int) -> np.ndarray:
    """Row indices of the largest group of near-identical rows (empty if < min_group)."""
    n = dset.shape[0]
    # pass 1: coarse key per row (first 32 dims rounded) -> the most common key
    counts: Counter = Counter()
    for s in range(0, n, CHUNK):
        x = dset[s:s + CHUNK, :32]
        counts.update(map(bytes, np.round(x, 2).astype(np.float32)))
    if not counts:
        return np.empty(0, dtype=np.int64)
    key, count = counts.most_common(1)[0]
    if count < min_group:
        return np.empty(0, dtype=np.int64)
    # pass 2: the group's mean vector, then every row within rtol of it (full dims)
    ref, m = None, 0
    for s in range(0, n, CHUNK):
        x = dset[s:s + CHUNK]
        sel = np.array([bytes(r) == key for r in np.round(x[:, :32], 2).astype(np.float32)])
        if sel.any():
            ref = x[sel].sum(0) + (ref if ref is not None else 0)
            m += int(sel.sum())
    ref = ref / m
    scale = np.linalg.norm(ref) + 1e-12
    rows = []
    for s in range(0, n, CHUNK):
        d = np.linalg.norm(dset[s:s + CHUNK] - ref, axis=1) / scale
        rows.append(s + np.flatnonzero(d < rtol))
    rows = np.concatenate(rows)
    return rows if len(rows) >= min_group else np.empty(0, dtype=np.int64)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('model_folder', help="embeddings folder without _h5, e.g. UNI2 or UNI2_fov896")
    p.add_argument('config', help="classifier YAML whose splits list the slides (or a .txt, one slide per line)")
    p.add_argument('--correction', default='raw')
    p.add_argument('--dataset', help="embedding key to test (default: top_left, else cls/nucleus/...)")
    p.add_argument('--rtol', type=float, default=5e-3, help="max relative L2 distance to the black embedding")
    p.add_argument('--min-group', type=int, default=3, help="smallest identical group reported")
    p.add_argument('--csv', help="write wsi, row, cell_id of every black-patch cell here")
    args = p.parse_args(argv)

    root = os.path.join(os.environ['DATASETS_ROOT'], f'{args.model_folder}_h5', args.correction)
    slides = (open(args.config).read().split() if args.config.endswith('.txt') else _slides(args.config))
    print(f"{len(slides)} slides under {root}", flush=True)
    found, out_rows = 0, []
    for w in slides:
        path = os.path.join(root, w, 'embeddings_dataset.h5')
        if not os.path.exists(path):
            print(f"MISSING  {w}", flush=True)
            continue
        with h5py.File(path, 'r', locking=False) as f:
            dset = _dataset(f, args.dataset)
            rows = black_rows(dset, args.rtol, args.min_group)
            n, key = dset.shape[0], dset.name.lstrip('/')
            if len(rows):
                ids = f['cell_ids'][rows]
                out_rows += [(w, int(r), c.decode() if isinstance(c, bytes) else c) for r, c in zip(rows, ids)]
        if not len(rows):
            print(f"ok       {w} ({n:,} cells)", flush=True)
            continue
        found += 1
        print(f"BLACK    {w}: {len(rows):,} / {n:,} cells ({100 * len(rows) / n:.2f}%), "
              f"rows {rows[0]:,}-{rows[-1]:,} ({key})", flush=True)
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            csv.writer(f).writerows([('wsi', 'row', 'cell_id'), *out_rows])
        print(f"wrote {len(out_rows):,} cells to {args.csv}")
    print(f"\n{found} slide(s) with black-patch cells out of {len(slides)}")
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
