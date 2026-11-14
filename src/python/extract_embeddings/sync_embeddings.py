"""Bring finished embeddings_dataset.h5 files in line with a regenerated patch_coordinates.h5
without re-extracting whole slides.

For when the processing behind patch_coordinates.h5 was re-run with the cell ids kept (see
scripts/check_cell_id_drift.py). Each slide of an extraction JSON config is rewritten to what a
fresh extraction would produce: one row per row of the current patch_coordinates.h5, in its
order, labelled with its cell_type:

  kept cells     embeddings copied from the existing file -- assumes those cells did not move
                 (embeddings_dataset.h5 stores no x_start/y_start to check against)
  added cells    extracted now with the config's own settings (model, mode, size, save_cls /
                 save_cell / save_nucleus, builder_kwargs), from a temporary patch_coordinates.h5
                 holding only those cells, in {output_path}/_sync_tmp/
  removed cells  dropped
  labels         cell_labels rewritten from the current cell_type

Per slide: unchanged -> nothing to do; only labels differ -> cell_labels rewritten in place;
otherwise the new file is written next to the old one (embeddings_dataset.h5.sync_tmp), checked
(every row's cell_id against the row it was copied from, and a sample of rows' embeddings,
read back one by one, against their source) and moved over embeddings_dataset.h5. With
--keep-old the previous file is kept as embeddings_dataset.before_sync.h5.

Other per-cell datasets that the extraction does not write and that were added to the file later
(e.g. 'matching', 'x_centroid', 'y_centroid') can't be filled for the added cells: a merged file
leaves them out (the plan and the merge log name them). Left alone and reported: slides whose ids
were not conserved (under half the embedded cells still present -- re-extract those from
scratch), unfinished embeddings files, and duplicate ids.

Re-running is safe: a synced slide comes out unchanged, and a finished temporary extraction is
reused (is_extracted), so a killed job picks up where it stopped.

Per-cell modes only ('patch', 'resized'): in 'multicell' a cell's embedding depends on the tile
it falls in. Not for providers that fit a PCA on the slide's cells (Dummy): the added cells
would be projected on a different basis.

Usage (repo root, .env sourced; a GPU unless --dry-run, which never imports torch so it also runs
on a login node that can't load torch's CUDA libraries):
    python -m src.python.extract_embeddings.sync_embeddings configs/extract_embeddings/uni2_224.json --dry-run
    python -m src.python.extract_embeddings.sync_embeddings configs/extract_embeddings/uni2_224.json --samples GSM7990540
    sbatch scripts/sync_embeddings.sh configs/extract_embeddings/uni2_224.json
"""
import argparse
import contextlib
import copy
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from src.python.code_configs import paths
from .run_configs import _expand   # torch-free; extract_embeddings.py (torch) is imported only to extract

TMP_DIR = '_sync_tmp'
MERGED_SUFFIX = '.sync_tmp'
BACKUP_NAME = 'embeddings_dataset.before_sync.h5'
IDS_CHANGED_BELOW = 0.5    # kept / embedded cells below this -> ids not conserved, don't patch
CHUNK_ROWS = 16384         # rows per read/write when merging (16384 x 1536 float32 = 100 MB)
N_SPOT_CHECK = 512         # kept rows and added rows whose embeddings are re-read and compared
_TAKE_SLACK = 4096         # _take reads one slice if it spans at most 2 x len(idx) + this many rows


class SyncError(Exception):
    pass


# ── ids ──────────────────────────────────────────────────────────────────────────

def _key(v) -> bytes:
    return bytes(v) if isinstance(v, bytes) else str(v).encode('utf-8')


def _decode(v) -> str:
    return bytes(v).decode('utf-8') if isinstance(v, bytes) else str(v)


def _hash(values) -> np.ndarray:
    """int64 hashes of ids: ~1M decoded 73-char ids as a fixed-width array are ~300 MB."""
    return np.fromiter((hash(_key(v)) for v in values), dtype=np.int64, count=len(values))


def _same_ids(a, b) -> bool:
    return len(a) == len(b) and all(_key(x) == _key(y) for x, y in zip(a, b))


def _read_coords(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, 'r', locking=False) as f:
        return f['cell_id'][()], f['cell_type'][()]


# ── planning ─────────────────────────────────────────────────────────────────────

def plan_slide(run: dict) -> dict:
    """What to do for one inference run: action in unchanged / relabel / merge / skip.

    For a merge, old_idx[j] is the row of the existing file holding current row j's cell
    (-1: added cell) and added_rows the current rows to extract."""
    out = Path(run['output_path'])
    emb_path = out / 'embeddings_dataset.h5'
    coords_path = Path(run['dataset_configs']['cells_info_path'])
    plan = {'name': out.name, 'out': out, 'emb_path': emb_path, 'coords_path': coords_path, 'run': run}
    merged = out / ('embeddings_dataset.h5' + MERGED_SUFFIX)
    if not emb_path.exists() and merged.exists() and (out / BACKUP_NAME).exists():
        # killed between the two renames of a --keep-old swap: the merged file was complete and checked
        os.replace(merged, emb_path)
    if not coords_path.exists():
        return {**plan, 'action': 'skip', 'why': f"no {coords_path}"}
    if not emb_path.exists():
        return {**plan, 'action': 'skip', 'why': "not extracted yet -- run the normal extraction"}

    new_ids, new_labels = _read_coords(coords_path)
    with h5py.File(emb_path, 'r', locking=False) as f:
        old_ids, old_labels = f['cell_ids'][()], f['cell_labels'][()]
        extra = sorted(k for k, d in f.items()
                       if isinstance(d, h5py.Dataset) and d.shape[:1] == (len(old_ids),)
                       and not k.startswith('embeddings_') and k not in ('cell_ids', 'cell_labels'))
    n_old, n_new = len(old_ids), len(new_ids)
    plan.update(n_old=n_old, n_new=n_new)
    if any(len(v) == 0 for v in old_ids):
        return {**plan, 'action': 'skip', 'why': "unfinished extraction -- the normal extraction re-runs it"}
    oh, nh = _hash(old_ids), _hash(new_ids)
    if len(np.unique(oh)) != n_old or len(np.unique(nh)) != n_new:
        return {**plan, 'action': 'skip', 'why': "duplicate cell ids"}

    old_idx = np.full(n_new, -1, dtype=np.int64)
    if n_old and n_new:
        sorter = np.argsort(oh)
        pos = np.minimum(np.searchsorted(oh, nh, sorter=sorter), n_old - 1)
        hit = oh[sorter[pos]] == nh
        old_idx[hit] = sorter[pos][hit]
    kept_rows = np.flatnonzero(old_idx >= 0)
    if not _same_ids(old_ids[old_idx[kept_rows]], new_ids[kept_rows]):
        return {**plan, 'action': 'skip', 'why': "id hash collision -- compare the ids by hand"}
    added_rows = np.flatnonzero(old_idx < 0)
    n_kept, n_removed = len(kept_rows), n_old - len(kept_rows)
    n_relabel = sum(_key(a) != _key(b) for a, b in zip(old_labels[old_idx[kept_rows]], new_labels[kept_rows]))
    plan.update(n_kept=n_kept, n_removed=n_removed, n_added=len(added_rows), n_relabel=int(n_relabel),
                extra=extra)

    if n_kept < IDS_CHANGED_BELOW * n_old:
        return {**plan, 'action': 'skip', 'why': "ids not conserved -- delete the file and re-extract the slide"}
    if not len(added_rows) and not n_removed and np.array_equal(old_idx, np.arange(n_old)):
        return {**plan, 'action': 'relabel' if n_relabel else 'unchanged'}
    # per-cell datasets the extraction does not write (`extra`) are left out of the merged file
    return {**plan, 'action': 'merge', 'old_idx': old_idx, 'added_rows': added_rows}


# ── extraction of the added cells ────────────────────────────────────────────────

def prepare_tmp_extraction(plan: dict) -> dict:
    """Write {out}/_sync_tmp/patch_coordinates.h5 with only the added cells (every per-cell
    dataset of the current file, subset; the rest copied) and return the run extracting them.
    A finished temporary extraction of a different set of cells is deleted."""
    tmp = plan['out'] / TMP_DIR
    tmp.mkdir(exist_ok=True)
    rows = plan['added_rows']
    with h5py.File(plan['coords_path'], 'r', locking=False) as src:
        added_ids = src['cell_id'][()][rows]
        tmp_emb = tmp / 'embeddings_dataset.h5'
        if tmp_emb.exists():
            try:
                with h5py.File(tmp_emb, 'r', locking=False) as f:
                    stale = not _same_ids(f['cell_ids'][()], added_ids)
            except (OSError, KeyError):
                stale = True
            if stale:
                print(f"[sync] {plan['name']}: discarding a temporary extraction of other cells: {tmp_emb}")
                tmp_emb.unlink()
        with h5py.File(tmp / 'patch_coordinates.h5', 'w') as dst:
            n = src['cell_id'].shape[0]
            dst.attrs.update(src.attrs)
            for k, d in src.items():
                if not isinstance(d, h5py.Dataset):
                    continue
                data = d[()][rows] if d.shape[:1] == (n,) else d[()]
                dst.create_dataset(k, data=data, dtype=d.dtype)
                dst[k].attrs.update(d.attrs)
    run = copy.deepcopy(plan['run'])
    run['dataset_configs']['cells_info_path'] = str(tmp / 'patch_coordinates.h5')
    run['output_path'] = str(tmp)
    return run


# ── merge ────────────────────────────────────────────────────────────────────────

def _take(ds: h5py.Dataset, idx: np.ndarray) -> np.ndarray:
    """ds[idx] for distinct indices in any order; one slice read when idx is nearly contiguous."""
    if len(idx) == 0:
        return np.empty((0, *ds.shape[1:]), dtype=ds.dtype)
    lo, hi = int(idx.min()), int(idx.max()) + 1
    if hi - lo <= 2 * len(idx) + _TAKE_SLACK:
        return ds[lo:hi][idx - lo]
    order = np.argsort(idx)
    out = np.empty((len(idx), *ds.shape[1:]), dtype=ds.dtype)
    out[order] = ds[idx[order]]
    return out


def _update_stats(out: Path, labels: list[str]) -> None:
    """Refresh dataset_stats.json's cell counts (written by the extraction) if there is one."""
    path = out / 'dataset_stats.json'
    if not path.exists():
        return
    stats = json.loads(path.read_text())
    counts = Counter(labels)
    stats.update(num_samples=len(labels), num_classes=len(counts), class_distribution=dict(counts))
    path.write_text(json.dumps(stats))


def relabel_slide(plan: dict) -> None:
    _, new_labels = _read_coords(plan['coords_path'])
    labels = [_decode(v) for v in new_labels]
    with h5py.File(plan['emb_path'], 'a') as f:
        if f['cell_labels'].shape[0] != len(labels):
            raise SyncError("embeddings_dataset.h5 changed since planning")
        f['cell_labels'][:] = labels
    _update_stats(plan['out'], labels)


def merge_slide(plan: dict, keep_old: bool) -> None:
    out, emb_path = plan['out'], plan['emb_path']
    old_idx, added = plan['old_idx'], plan['added_rows']
    n_new = len(old_idx)
    tmp_idx = np.full(n_new, -1, dtype=np.int64)
    tmp_idx[added] = np.arange(len(added))
    kept_rows = np.flatnonzero(old_idx >= 0)

    new_ids, new_labels = _read_coords(plan['coords_path'])
    if len(new_ids) != n_new:
        raise SyncError("patch_coordinates.h5 changed since planning")
    merged = out / ('embeddings_dataset.h5' + MERGED_SUFFIX)
    tmp_emb = out / TMP_DIR / 'embeddings_dataset.h5'
    try:
        with contextlib.ExitStack() as stack:
            old = stack.enter_context(h5py.File(emb_path, 'r', locking=False))
            tmp = stack.enter_context(h5py.File(tmp_emb, 'r', locking=False)) if len(added) else None
            new = stack.enter_context(h5py.File(merged, 'w'))

            # every current row must come from a source row holding the same cell id
            if not _same_ids(old['cell_ids'][()][old_idx[kept_rows]], new_ids[kept_rows]):
                raise SyncError("existing embeddings_dataset.h5 changed since planning")
            emb_keys = sorted(k for k in old if k.startswith('embeddings_'))
            if tmp is not None:
                if not _same_ids(tmp['cell_ids'][()], new_ids[added]):
                    raise SyncError(f"{tmp_emb} does not hold the added cells -- delete {out / TMP_DIR} and re-run")
                tmp_keys = sorted(k for k in tmp if k.startswith('embeddings_'))
                if tmp_keys != emb_keys:
                    raise SyncError(f"existing file has {emb_keys} but the new extraction wrote {tmp_keys}: the "
                                    f"config's settings differ from the ones the slide was extracted with")
                for k in emb_keys:
                    if (old[k].shape[1:], old[k].dtype) != (tmp[k].shape[1:], tmp[k].dtype):
                        raise SyncError(f"{k}: existing {old[k].shape[1:]} {old[k].dtype}, "
                                        f"new extraction {tmp[k].shape[1:]} {tmp[k].dtype}")
                # files written before the attribute existed have none: those can't be checked
                old_prec, tmp_prec = old.attrs.get('precision'), tmp.attrs.get('precision')
                if old_prec is not None and old_prec != tmp_prec:
                    raise SyncError(f"existing embeddings were extracted at precision {old_prec}, the added "
                                    f"cells at {tmp_prec}: set \"precision\": \"{old_prec}\" in the config")

            new.attrs.update(old.attrs)
            for k in emb_keys:
                src = old[k]
                dst = new.create_dataset(k, shape=(n_new, *src.shape[1:]), dtype=src.dtype)
                dst.attrs.update(src.attrs)
                for j0 in range(0, n_new, CHUNK_ROWS):
                    j1 = min(j0 + CHUNK_ROWS, n_new)
                    sel = old_idx[j0:j1]
                    from_old = sel >= 0
                    buf = np.empty((j1 - j0, *src.shape[1:]), dtype=src.dtype)
                    buf[from_old] = _take(src, sel[from_old])
                    if not from_old.all():
                        buf[~from_old] = _take(tmp[k], tmp_idx[j0:j1][~from_old])
                    dst[j0:j1] = buf
            str_dt = h5py.string_dtype(encoding='utf-8')
            labels = [_decode(v) for v in new_labels]
            for k, values in (('cell_ids', [_decode(v) for v in new_ids]), ('cell_labels', labels)):
                new.create_dataset(k, data=values, dtype=str_dt)
                new[k].attrs.update(old[k].attrs)

            # spot check, reading rows one by one (independent of _take)
            rng = np.random.default_rng(0)
            rows = np.unique(np.concatenate([
                [0, n_new - 1],
                rng.choice(kept_rows, min(N_SPOT_CHECK, len(kept_rows)), replace=False),
                rng.choice(added, min(N_SPOT_CHECK, len(added)), replace=False),
            ]).astype(np.int64)) if n_new else np.empty(0, dtype=np.int64)
            for j in rows:
                if _decode(new['cell_ids'][j]) != _decode(new_ids[j]):
                    raise SyncError(f"row {j}: cell_id mismatch after writing")
                for k in emb_keys:
                    expected = old[k][old_idx[j]] if old_idx[j] >= 0 else tmp[k][tmp_idx[j]]
                    if not np.array_equal(new[k][j], expected):
                        raise SyncError(f"row {j} ({_decode(new_ids[j])}), {k}: embedding differs from its source")
    except BaseException:
        merged.unlink(missing_ok=True)
        raise

    if keep_old:
        os.replace(emb_path, out / BACKUP_NAME)
    os.replace(merged, emb_path)
    _update_stats(out, labels)
    shutil.rmtree(out / TMP_DIR, ignore_errors=True)


# ── entry point ──────────────────────────────────────────────────────────────────

def _print_plan(plans: list[dict]) -> None:
    print(f"{'action':10} {'sample':64} {'embedded':>9} {'current':>9} {'kept':>9} {'removed':>8} "
          f"{'added':>8} {'relabel':>8}")
    for p in plans:
        if 'n_kept' in p:
            nums = (f"{p['n_old']:9,} {p['n_new']:9,} {p['n_kept']:9,} {p['n_removed']:8,} "
                    f"{p['n_added']:8,} {p['n_relabel']:8,}")
        else:
            nums = ''
        note = (f"  ({p['why']})" if p['action'] == 'skip'
                else f"  (drops {', '.join(p['extra'])})" if p['action'] == 'merge' and p['extra'] else '')
        print(f"{p['action']:10} {p['name']:64} {nums}{note}")
    actionable = [p for p in plans if p['action'] in ('merge', 'relabel')]
    print(f"-> {sum(p['action'] == 'merge' for p in plans)} to merge, {sum(p['action'] == 'relabel' for p in plans)} "
          f"to relabel, {sum(p['action'] == 'unchanged' for p in plans)} unchanged, "
          f"{sum(p['action'] == 'skip' for p in plans)} skipped | cells to extract "
          f"{sum(p['n_added'] for p in actionable):,}, to drop {sum(p['n_removed'] for p in actionable):,}, "
          f"to relabel {sum(p['n_relabel'] for p in actionable):,}", flush=True)


_CELLS_INFO_ROOTS = {'xenium_processed': paths.XENIUM_PROCESSED_OUTPUT_ROOT,
                     'positions_converted': paths.POSITIONS_CONVERTED_ROOT}
_DEFAULT_OUTPUT_SUFFIX = {'patch': '_h5', 'resized': '_resized_h5', 'multicell': '_multicell_h5'}


def _light_runs(config_path: Path) -> tuple[dict, list[dict]]:
    """(config, [{output_path, dataset_configs: {cells_info_path}}]) for --dry-run, without
    importing extract_embeddings.py (torch). Mirrors its _resolve_samples + build_run_configs
    path logic (as check_wsi_regions does) -- keep in sync; the real run uses build_run_configs."""
    with open(config_path) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith('_')}
    samples = cfg.get('samples')
    if samples is None:
        with open((Path(config_path).parent / cfg['samples_manifest']).resolve(), newline='') as f:
            infos = {row['dataset']: {} for row in csv.DictReader(f) if row.get('status', 'ok') == 'ok'}
    elif isinstance(samples, list):
        infos = {name: {} for name in samples}
    else:
        infos = {name: dict(info or {}) for name, info in samples.items()}
    for name in cfg.get('exclude_samples', []):
        infos.pop(name, None)
    out_root = (Path(paths.DATASETS_ROOT)
                / (cfg.get('model_output_dir', cfg['model_name']) + cfg.get('output_suffix', _DEFAULT_OUTPUT_SUFFIX[cfg['mode']]))
                / cfg.get('correction_name', 'raw'))
    default_root = cfg.get('cells_info_root', 'xenium_processed')
    by_prefix = cfg.get('cells_info_root_by_prefix', {})
    runs = []
    for name, info in infos.items():
        root = info.get('cells_info_root') or next(
            (r for prefix, r in by_prefix.items() if name.startswith(prefix)), default_root)
        coords = Path(_CELLS_INFO_ROOTS.get(root, root)) / name / 'patch_coordinates.h5'
        runs.append({'output_path': str(out_root / name), 'dataset_configs': {'cells_info_path': str(coords)}})
    return cfg, runs


def sync_config(config_path: Path, samples: list[str] | None, dry_run: bool, keep_old: bool,
                exclude: list[str] | None = None) -> list[str]:
    """Sync every slide of one JSON run config. Returns '<sample>: <error>' for each failure."""
    if dry_run:
        cfg, runs = _light_runs(config_path)
    else:
        from .extract_embeddings import build_run_configs, load_run_config
        cfg = load_run_config(config_path)
    print(f"\n=== {config_path} ===", flush=True)
    if cfg['mode'] == 'multicell':
        print("[sync] mode 'multicell' is not supported (a cell's embedding depends on its tile); skipping")
        return []
    if cfg['model_name'] == 'Dummy':
        print("[sync] the Dummy provider fits a PCA on the slide's cells; re-extract instead; skipping")
        return []
    if not dry_run:
        (model_name, model_cfg), = build_run_configs(cfg).items()
        runs = model_cfg['inference_runs']
    runs = [r for r in runs if (not samples or Path(r['output_path']).name in samples)
            and Path(r['output_path']).name not in (exclude or [])]

    plans = []
    for run in runs:
        plan = plan_slide(run)
        if plan['action'] == 'unchanged':
            shutil.rmtree(plan['out'] / TMP_DIR, ignore_errors=True)   # leftover of a killed run
        plans.append(plan)
    _print_plan(plans)
    if dry_run:
        return []

    from .data.patch_dataset import PatchDataset
    from .data.resized_cell_dataset import ResizedCellDataset
    from .extract_embeddings import _input_paths, extract_embeddings, is_extracted

    failures = []
    for p in (p for p in plans if p['action'] == 'relabel'):
        try:
            relabel_slide(p)
            print(f"[sync] relabelled {p['name']} ({p['n_relabel']:,} cells)", flush=True)
        except Exception as e:
            failures.append(f"{p['name']}: {type(e).__name__}: {e}")

    merges = [p for p in plans if p['action'] == 'merge']
    tmp_runs = {}
    for p in (p for p in merges if p['n_added']):
        run = prepare_tmp_extraction(p)
        absent = [x for x in _input_paths(run) if not os.path.exists(x)]
        if absent:
            failures.append(f"{p['name']}: missing inputs {absent}")
            continue
        tmp_runs[p['name']] = run
    if tmp_runs:
        mode = cfg['mode']
        failed = extract_embeddings(
            {model_name: {'inference_runs': list(tmp_runs.values())}},
            batch_size=cfg.get('batch_size', 16), save_cls=cfg.get('save_cls', True),
            save_cell=cfg.get('save_cell', False), save_nucleus=cfg.get('save_nucleus', False),
            dataset_cls=PatchDataset if mode == 'patch' else ResizedCellDataset,
            skip_existing=True, continue_on_error=True, precision=cfg.get('precision'),
        )
        failures += [f"{Path(f).parent.name}: extraction of the added cells failed" for f in failed]

    for p in merges:
        if p['n_added'] and (p['name'] not in tmp_runs or not is_extracted(p['out'] / TMP_DIR)):
            continue    # missing inputs / failed extraction, reported above
        try:
            merge_slide(p, keep_old)
            print(f"[sync] merged {p['name']}: {p['n_new']:,} cells ({p['n_added']:,} added, "
                  f"{p['n_removed']:,} removed, {p['n_relabel']:,} relabelled)"
                  + (f", dropped {', '.join(p['extra'])}" if p['extra'] else ''), flush=True)
        except Exception as e:
            failures.append(f"{p['name']}: {type(e).__name__}: {e}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configs', nargs='+', help="extraction JSON configs and/or directories of them")
    parser.add_argument('--samples', nargs='*', help="only these samples (default: every sample in the configs)")
    parser.add_argument('--exclude-samples', nargs='*', default=[],
                        help="leave these samples alone (e.g. one being re-extracted in full at the same time)")
    parser.add_argument('--dry-run', action='store_true', help="only print what would be done (no model, no writes)")
    parser.add_argument('--keep-old', action='store_true',
                        help=f"keep each replaced file as {BACKUP_NAME} (default: overwritten)")
    args = parser.parse_args(argv)

    failures = []
    for config in _expand(args.configs):
        failures += sync_config(config, args.samples, args.dry_run, args.keep_old, args.exclude_samples)
    print(f"\n=== done: {len(failures)} failure(s) ===")
    for f in failures:
        print(f"  FAILED {f}")
    return 3 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
