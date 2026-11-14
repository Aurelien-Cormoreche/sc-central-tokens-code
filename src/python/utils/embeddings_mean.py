"""Precomputed mean of a slide's embedding datasets (embeddings_mean.h5).

The classifier averages several embedding datasets per cell (e.g. the 4 central tokens
bottom_left/bottom_right/top_left/top_right). Reading 4 datasets from ceph is what bounds
loading; this file holds their mean once, next to embeddings_dataset.h5, so the loader reads
a quarter of the bytes. In float32 (default) its rows are bit-identical to the mean the loader
computes on the fly (see mean_block); float16 halves the bytes again but rounds each value
(~5e-4 relative), so it changes results slightly.

The loader (utils/loading_functions.load_embedding_rows) uses the file only when it matches the
embeddings it would otherwise average -- same datasets in the same order, same cell ids, and a
few probe rows recomputed from embeddings_dataset.h5 equal to the stored ones -- and otherwise
falls back to averaging the datasets itself. A new extraction (create_output_file) and a
sync_embeddings merge delete the slide's file; re-run this script afterwards.

Usage (repo root, .env sourced; CPU only -- sbatch scripts/precompute_embeddings_mean.sh ...):
    # the slides and datasets a classifier config uses
    python -m src.python.utils.embeddings_mean --config configs/train_cross_cohort.yaml \\
        --model UNI2 --correction Base
    # every slide of an embeddings folder, explicit datasets
    python -m src.python.utils.embeddings_mean --embeddings-dir $DATASETS_ROOT/UNI2_h5/Base \\
        --datasets bottom_left bottom_right top_left top_right
    # --dtype float16, --overwrite, --shard 0/4 (slides 0, 4, 8, ...), --dry-run
"""
import argparse
import json
import os
import sys
import time
import zlib
from typing import Optional

import h5py
import numpy as np

MEAN_FILE = 'embeddings_mean.h5'
MEAN_DATASET = 'embeddings_mean'
SOURCE_FILE = 'embeddings_dataset.h5'
BLOCK_ROWS = 8192


def embedding_keys(f: h5py.File, embeddings_datasets) -> list[str]:
    """The dataset keys (without 'embeddings_') averaged for this file: embeddings_datasets,
    or every embeddings_* dataset when it is empty."""
    if len(embeddings_datasets) == 0:
        return [key.split('embeddings_')[1] for key in f.keys() if key.startswith('embeddings_')]
    return list(embeddings_datasets)


def mean_block(dsets, start: int, stop: int, local: Optional[np.ndarray] = None) -> np.ndarray:
    """Mean over dsets of rows start:stop (then [local]), accumulated in place in dataset order
    and divided once: the same operations as np.mean over axis 0 of the stacked datasets, so the
    result is bit-identical to it. Other dtypes (e.g. float16, which np.mean accumulates in
    float32) go through np.mean itself."""
    def take(d):
        block = d[start:stop]
        return block if local is None else block[local]
    if dsets[0].dtype in (np.float32, np.float64):
        acc = take(dsets[0])
        for d in dsets[1:]:
            acc += take(d)
        acc /= len(dsets)
        return acc
    return np.mean([take(d) for d in dsets], axis=0)


def ids_crc32(cell_ids: np.ndarray) -> int:
    return zlib.crc32(b'\n'.join(c if isinstance(c, bytes) else str(c).encode() for c in cell_ids))


def _probe_rows(n: int) -> list[int]:
    return sorted({0, n // 2, n - 1}) if n else []


def remove_mean_file(slide_dir) -> None:
    """Delete slide_dir's embeddings_mean.h5 (called whenever embeddings_dataset.h5 is rewritten)."""
    try:
        os.remove(os.path.join(slide_dir, MEAN_FILE))
    except FileNotFoundError:
        pass


def open_mean_file(slide_dir, src: h5py.File, keys: list[str], cell_ids: np.ndarray):
    """(open h5py.File, '') if slide_dir's embeddings_mean.h5 holds the mean of `keys` for
    exactly the rows of `src` (its open embeddings_dataset.h5, whose cell_ids are given), else
    (None, why not). The caller closes the returned file."""
    path = os.path.join(slide_dir, MEAN_FILE)
    if not os.path.isfile(path):
        return None, 'missing'
    try:
        g = h5py.File(path, 'r')
    except OSError:
        return None, 'unreadable'
    try:
        dsets = [src[f'embeddings_{k}'] for k in keys]
        ds = g.get(MEAN_DATASET)
        if ds is None or json.loads(g.attrs.get('sources', '[]')) != keys:
            why = 'other datasets'
        elif ds.shape != dsets[0].shape or ds.shape[0] != len(cell_ids):
            why = 'stale (rows)'
        elif int(g.attrs.get('cell_ids_crc32', -1)) != ids_crc32(cell_ids):
            why = 'stale (cell ids)'
        elif not all(np.array_equal(ds[r:r + 1], mean_block(dsets, r, r + 1).astype(ds.dtype), equal_nan=True)
                     for r in _probe_rows(ds.shape[0])):
            why = 'stale (values)'
        else:
            return g, ''
    except Exception as e:  # noqa: BLE001 -- any unreadable/odd file means "don't use it"
        why = f'unreadable ({type(e).__name__})'
    g.close()
    return None, why


def write_mean_file(slide_dir, embeddings_datasets, dtype=np.float32, overwrite: bool = False) -> str:
    """Write slide_dir/embeddings_mean.h5 from slide_dir/embeddings_dataset.h5. Written to a
    temporary name and renamed when complete, so a reader never sees a partial file. Returns
    what was done ('written', 'up to date', or why it was skipped)."""
    src_path = os.path.join(slide_dir, SOURCE_FILE)
    with h5py.File(src_path, 'r', locking=False) as src:
        keys = embedding_keys(src, embeddings_datasets)
        missing = [k for k in keys if f'embeddings_{k}' not in src]
        if missing:
            return f"skipped: no embeddings_{', embeddings_'.join(missing)}"
        if len(keys) < 2:
            return 'skipped: a single dataset (nothing to average)'
        dsets = [src[f'embeddings_{k}'] for k in keys]
        cell_ids = src['cell_ids'][()]
        if not overwrite:
            g, why = open_mean_file(slide_dir, src, keys, cell_ids)
            if g is not None:
                same_dtype = g[MEAN_DATASET].dtype == np.dtype(dtype)
                g.close()
                if same_dtype:
                    return 'up to date'
        n, dim = dsets[0].shape
        tmp = os.path.join(slide_dir, f'{MEAN_FILE}.tmp-{os.getpid()}')
        try:
            with h5py.File(tmp, 'w') as g:
                out = g.create_dataset(MEAN_DATASET, shape=(n, dim), dtype=dtype)
                for start in range(0, n, BLOCK_ROWS):
                    stop = min(start + BLOCK_ROWS, n)
                    out[start:stop] = mean_block(dsets, start, stop).astype(dtype, copy=False)
                g.attrs['sources'] = json.dumps(keys)
                g.attrs['cell_ids_crc32'] = ids_crc32(cell_ids)
            os.replace(tmp, os.path.join(slide_dir, MEAN_FILE))
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
    return 'written'


# ── CLI ──────────────────────────────────────────────────────────────────────────

def _config_slides_and_datasets(config_path: str) -> tuple[list[str], list[str]]:
    """WSIs named in a classifier config's splits (train/val/test, in first-seen order) and its
    data.embeddings_datasets."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    slides: dict[str, None] = {}
    for sp in cfg.get('splits') or []:
        for side in ('train', 'val', 'test'):
            for wsi in (sp.get(side) or {}):
                if wsi != 'cell_type_proportions':
                    slides[wsi] = None
    return list(slides), list((cfg.get('data') or {}).get('embeddings_datasets') or [])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument('--config', help='classifier config: its splits\' slides and data.embeddings_datasets')
    where.add_argument('--embeddings-dir', help='folder of <slide>/embeddings_dataset.h5 (every slide in it)')
    ap.add_argument('--model', help='with --config: data.model_name (folder <base-dir>/<model>_h5)')
    ap.add_argument('--correction', help='with --config: data.correction_name')
    ap.add_argument('--base-dir', default=os.environ.get('DATASETS_ROOT'), help='default: $DATASETS_ROOT')
    ap.add_argument('--datasets', nargs='*', help='keys to average (overrides the config; default: all embeddings_*)')
    ap.add_argument('--slides', nargs='*', help='only these slides')
    ap.add_argument('--dtype', choices=['float32', 'float16'], default='float32',
                    help='float32 is bit-identical to averaging while loading; float16 halves the bytes')
    ap.add_argument('--overwrite', action='store_true', help='rewrite files that are up to date')
    ap.add_argument('--shard', default='0/1', help='i/n: process slides i, i+n, i+2n, ... (array jobs)')
    ap.add_argument('--dry-run', action='store_true', help='only report which slides would be written')
    args = ap.parse_args(argv)

    datasets: list[str] = []
    if args.config:
        if not (args.model and args.correction and args.base_dir):
            ap.error('--config needs --model, --correction and --base-dir (or $DATASETS_ROOT)')
        base = os.path.join(args.base_dir, f'{args.model}_h5', args.correction)
        slides, datasets = _config_slides_and_datasets(args.config)
    else:
        base = args.embeddings_dir
        slides = sorted(d for d in os.listdir(base) if os.path.isfile(os.path.join(base, d, SOURCE_FILE)))
    if args.datasets is not None:
        datasets = args.datasets
    if args.slides:
        slides = [s for s in slides if s in set(args.slides)] if args.config else list(args.slides)
    i, n = (int(x) for x in args.shard.split('/'))
    slides = slides[i::n]

    print(f"{base}: {len(slides)} slide(s), mean of {datasets or 'every embeddings_* dataset'} "
          f"as {args.dtype}{' (dry run)' if args.dry_run else ''}", flush=True)
    n_failed = 0
    for k, slide in enumerate(slides, 1):
        slide_dir = os.path.join(base, slide)
        t = time.perf_counter()
        try:
            if not os.path.isfile(os.path.join(slide_dir, SOURCE_FILE)):
                status = f'skipped: no {SOURCE_FILE}'
            elif args.dry_run:
                with h5py.File(os.path.join(slide_dir, SOURCE_FILE), 'r', locking=False) as src:
                    keys = embedding_keys(src, datasets)
                    g, why = open_mean_file(slide_dir, src, keys, src['cell_ids'][()])
                    if g is not None:
                        why = f"up to date ({g[MEAN_DATASET].dtype})"
                        g.close()
                status = f'would write ({why})' if not why.startswith('up to date') else why
            else:
                status = write_mean_file(slide_dir, datasets, np.dtype(args.dtype), args.overwrite)
        except Exception as e:  # noqa: BLE001 -- report and go on with the other slides
            status = f'FAILED: {type(e).__name__}: {e}'
            n_failed += 1
        print(f"[{k}/{len(slides)}] {slide}: {status} ({time.perf_counter() - t:.0f}s)", flush=True)
    print(f"done: {len(slides) - n_failed} ok, {n_failed} failed")
    return 1 if n_failed else 0


if __name__ == '__main__':
    sys.exit(main())
