"""Check that each slide's extracted cells still match its (regenerated) patch_coordinates.h5,
and estimate how many cells need re-extracting.

For every slide the given extraction JSON configs resolve to (samples / samples_manifest /
exclude_samples, cells_info_root[_by_prefix], {DATASETS_ROOT}/{model_output_dir}{output_suffix}/
{correction_name}/{sample} -- the same resolution as extract_embeddings.build_run_configs),
compares the cell_ids already in embeddings_dataset.h5 with the cell_id column of the current
patch_coordinates.h5:

  kept     ids in both files: their embeddings can be reused, provided the cell did not move
           (embeddings_dataset.h5 stores no coordinates, so that can't be checked here)
  removed  embedded but no longer in patch_coordinates -> drop those rows, nothing to extract
  added    in patch_coordinates but never embedded     -> the cells to extract
  label    kept cells whose cell_labels (embeddings) != cell_type (patch_coordinates): no
           re-extraction needed, but the embeddings file's cell_labels are stale
  order    whether the kept ids appear in the same relative order in both files

Only ids and labels are read (no embeddings, no WSI); CPU-only; files are opened with
locking=False so this can run while an extraction job is writing.

Usage (repo root, after `set -a; source .env; set +a`):
    python scripts/check_cell_id_drift.py configs/extract_embeddings/uni2_224.json
    python scripts/check_cell_id_drift.py configs/extract_embeddings/uni2_224_batches_2/ \
        --samples fetal_d54 Yu_DRG_area_1 --ids-csv changed_cells.csv
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

# Run as a plain script (`python scripts/...`), Python only puts scripts/ on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.python.code_configs import paths  # noqa: E402
from src.python.extract_embeddings.run_configs import _expand  # noqa: E402  (no torch import)

_CELLS_INFO_ROOTS = {'xenium_processed': paths.XENIUM_PROCESSED_OUTPUT_ROOT,
                     'positions_converted': paths.POSITIONS_CONVERTED_ROOT}
_DEFAULT_OUTPUT_SUFFIX = {'patch': '_h5', 'resized': '_resized_h5', 'multicell': '_multicell_h5'}

# kept / embedded cells below this -> ids were not conserved, re-extract the whole slide
IDS_CHANGED_BELOW = 0.5


def resolve_slides(config_path: Path) -> list[dict]:
    """[{sample, emb_path, coords_path}] one extraction JSON config reads/writes.

    Mirrors extract_embeddings._resolve_samples + build_run_configs without importing that
    module (it pulls in torch) -- keep in sync if the config format changes."""
    with open(config_path) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith('_')}
    samples = cfg.get('samples')
    if samples is None:
        with open((config_path.parent / cfg['samples_manifest']).resolve(), newline='') as f:
            infos = {row['dataset']: {} for row in csv.DictReader(f) if row.get('status', 'ok') == 'ok'}
    elif isinstance(samples, list):
        infos = {name: {} for name in samples}
    else:
        infos = {name: dict(info or {}) for name, info in samples.items()}
    for name in cfg.get('exclude_samples', []):
        infos.pop(name, None)

    model_dir = cfg.get('model_output_dir', cfg['model_name']) + cfg.get(
        'output_suffix', _DEFAULT_OUTPUT_SUFFIX[cfg['mode']])
    out_root = Path(paths.DATASETS_ROOT) / model_dir / cfg.get('correction_name', 'raw')
    default_root = cfg.get('cells_info_root', 'xenium_processed')
    by_prefix = cfg.get('cells_info_root_by_prefix', {})
    slides = []
    for name, info in infos.items():
        root = info.get('cells_info_root') or next(
            (r for prefix, r in by_prefix.items() if name.startswith(prefix)), default_root)
        slides.append({
            'sample': name,
            'out_root': out_root,
            'emb_path': out_root / name / 'embeddings_dataset.h5',
            'coords_path': Path(_CELLS_INFO_ROOTS.get(root, root)) / name / 'patch_coordinates.h5',
        })
    return slides


def _key(v, strip_prefix: bool = False) -> bytes:
    b = bytes(v) if isinstance(v, bytes) else str(v).encode('utf-8')
    return b.rsplit(b'_', 1)[-1] if strip_prefix else b


def _hash(values: np.ndarray, strip_prefix: bool = False) -> np.ndarray:
    """int64 hash of every id/label. Set operations run on these instead of the strings: a
    fixed-width unicode array of ~1M 73-char ids is ~300 MB and np.intersect1d copies it several
    times (ran out of memory on a login node). 64-bit hashes of ~1M ids: collision odds ~1e-7."""
    return np.fromiter((hash(_key(v, strip_prefix)) for v in values), dtype=np.int64, count=len(values))


def _decode(v) -> str:
    return bytes(v).decode('utf-8') if isinstance(v, bytes) else str(v)


def _read(path: Path, id_key: str, label_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Raw ids and labels as h5py returns them (bytes objects / fixed-width bytes), not decoded."""
    with h5py.File(path, 'r', locking=False) as f:
        return f[id_key][()], f[label_key][()]


def _unique_idx(h: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Indices where mask is set, one per distinct hash, in file order."""
    idx = np.flatnonzero(mask)
    _, first = np.unique(h[idx], return_index=True)
    return idx[np.sort(first)]


def compare(emb_ids, emb_labels, new_ids, new_labels) -> dict:
    written = np.fromiter((len(v) > 0 for v in emb_ids), dtype=bool, count=len(emb_ids))
    n_unwritten = int((~written).sum())
    emb_ids, emb_labels = emb_ids[written], emb_labels[written]

    stripped = False
    eh, nh = _hash(emb_ids), _hash(new_ids)
    common, i_emb, i_new = np.intersect1d(eh, nh, return_indices=True)
    if len(common) < IDS_CHANGED_BELOW * min(len(eh), len(nh)):
        # maybe only the id formatting changed (e.g. a '{sample}_' prefix added/dropped)
        eh2, nh2 = _hash(emb_ids, strip_prefix=True), _hash(new_ids, strip_prefix=True)
        common2, i_emb2, i_new2 = np.intersect1d(eh2, nh2, return_indices=True)
        if len(common2) > len(common):
            stripped, eh, nh = True, eh2, nh2
            common, i_emb, i_new = common2, i_emb2, i_new2

    changed = _hash(emb_labels[i_emb]) != _hash(new_labels[i_new])
    old_l, new_l = emb_labels[i_emb[changed]], new_labels[i_new[changed]]
    n_new_unique = len(np.unique(nh))
    return {
        'n_emb': len(eh) + n_unwritten, 'n_written': len(eh), 'n_unwritten': n_unwritten,
        'n_new': len(nh), 'n_new_unique': n_new_unique,
        'dup_emb': len(eh) - len(np.unique(eh)), 'dup_new': len(nh) - n_new_unique,
        'kept': len(common), 'stripped': stripped,
        'removed': emb_ids[_unique_idx(eh, ~np.isin(eh, nh))],
        'added': new_ids[_unique_idx(nh, ~np.isin(nh, eh))],
        'changed_ids': new_ids[i_new[changed]], 'old_labels': old_l, 'new_labels': new_l,
        'transitions': Counter(zip(map(_decode, old_l), map(_decode, new_l))),
        'order_kept': len(common) < 2 or bool(np.all(np.diff(i_new[np.argsort(i_emb)]) > 0)),
    }


def verdict(r: dict) -> tuple[str, int]:
    """(status, cells to extract)."""
    if r['n_unwritten']:
        return 'INCOMPLETE', r['n_new_unique']      # skip_existing re-extracts the whole slide anyway
    if r['kept'] < IDS_CHANGED_BELOW * r['n_written']:
        return 'IDS_CHANGED', r['n_new_unique']     # most embedded cells are gone: ids not conserved
    if len(r['added']) or len(r['removed']):
        return 'PATCH', len(r['added'])
    if r['stripped']:
        return 'ID_FORMAT', 0                        # same cells, but the id strings differ
    if len(r['changed_ids']):
        return 'LABELS_ONLY', 0
    return 'unchanged', 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configs', nargs='+', help="extraction JSON configs and/or directories of them")
    parser.add_argument('--samples', nargs='*', help="only these samples (default: every sample in the configs)")
    parser.add_argument('--ids-csv', help="write (sample, cell_id, change, old_label, new_label) of every "
                                          "added / removed / label-changed cell here")
    args = parser.parse_args(argv)

    slides = {}
    for config in _expand(args.configs):
        for s in resolve_slides(config):
            if args.samples and s['sample'] not in args.samples:
                continue
            slides.setdefault((s['emb_path'], s['coords_path']), s)
    print(f"{len(slides)} slide(s) to check")
    header = (f"{'status':13} {'sample':64} {'embedded':>9} {'current':>9} {'kept':>9} {'removed':>8} "
              f"{'added':>8} {'label Δ':>8} {'order':>5} {'to extract':>10}")

    # written slide by slide: across all slides this can be millions of rows
    ids_file = open(args.ids_csv, 'w', newline='') if args.ids_csv else None
    writer = csv.writer(ids_file) if ids_file else None
    if writer:
        writer.writerow(['sample', 'cell_id', 'change', 'old_label', 'new_label'])
    n_rows = 0

    statuses = Counter()
    to_extract = Counter()
    n_removed = n_relabel = 0
    details = []
    out_root = None
    for s in slides.values():
        name = s['sample']
        if s['out_root'] != out_root:
            out_root = s['out_root']
            print(f"\n### {out_root}\n{header}", flush=True)
        if not s['coords_path'].exists():
            has_emb = 'embeddings exist' if s['emb_path'].exists() else 'no embeddings either'
            print(f"{'NO_COORDS':13} {name:64} ({has_emb}) {s['coords_path']}", flush=True)
            statuses['NO_COORDS'] += 1
            continue
        new_ids, new_labels = _read(s['coords_path'], 'cell_id', 'cell_type')
        if not s['emb_path'].exists():
            n = len(np.unique(_hash(new_ids)))
            print(f"{'NOT_EXTRACTED':13} {name:64} {'-':>9} {len(new_ids):9,} {'':>9} {'':>8} {'':>8} {'':>8} "
                  f"{'':>5} {n:10,}", flush=True)
            statuses['NOT_EXTRACTED'] += 1
            to_extract['NOT_EXTRACTED'] += n
            continue
        try:
            emb_ids, emb_labels = _read(s['emb_path'], 'cell_ids', 'cell_labels')
        except (OSError, KeyError) as e:
            print(f"{'UNREADABLE':13} {name:64} {type(e).__name__}: {e}", flush=True)
            statuses['UNREADABLE'] += 1
            continue

        r = compare(emb_ids, emb_labels, new_ids, new_labels)
        del emb_ids, emb_labels, new_ids, new_labels
        status, n_extract = verdict(r)
        statuses[status] += 1
        to_extract[status] += n_extract
        n_removed += len(r['removed'])
        n_relabel += len(r['changed_ids'])
        order = '-' if r['kept'] < 2 else 'yes' if r['order_kept'] else 'NO'
        print(f"{status:13} {name:64} {r['n_emb']:9,} {r['n_new']:9,} {r['kept']:9,} {len(r['removed']):8,} "
              f"{len(r['added']):8,} {len(r['changed_ids']):8,} {order:>5} {n_extract:10,}", flush=True)

        notes = []
        if r['n_unwritten']:
            notes.append(f"{r['n_unwritten']:,} unwritten rows in embeddings_dataset.h5")
        if r['stripped']:
            notes.append("ids only match after stripping everything up to the last '_' (id format changed)")
        if r['dup_emb'] or r['dup_new']:
            notes.append(f"duplicate ids: {r['dup_emb']:,} embedded, {r['dup_new']:,} current")
        if len(r['added']):
            notes.append("added e.g. " + ", ".join(map(_decode, r['added'][:3])))
        if len(r['removed']):
            notes.append("removed e.g. " + ", ".join(map(_decode, r['removed'][:3])))
        if r['transitions']:
            notes.append("labels " + ", ".join(f"{a}->{b} ({n:,})" for (a, b), n in r['transitions'].most_common(5)))
        if notes:
            details.append((f"{name}  ({s['emb_path'].parent.parent.parent.name})", notes))

        if writer:
            writer.writerows((name, _decode(c), 'added', '', '') for c in r['added'])
            writer.writerows((name, _decode(c), 'removed', '', '') for c in r['removed'])
            writer.writerows((name, _decode(c), 'label_changed', _decode(a), _decode(b))
                             for c, a, b in zip(r['changed_ids'], r['old_labels'], r['new_labels']))
            n_rows += len(r['added']) + len(r['removed']) + len(r['changed_ids'])
            ids_file.flush()

    if details:
        print("\n## details")
        for name, notes in details:
            print(f"{name}\n    " + "\n    ".join(notes))

    print("\n## summary")
    print(f"{len(slides)} slides: " + ", ".join(f"{n} {st}" for st, n in statuses.most_common()))
    print(f"cells to extract: {sum(to_extract.values()):,}")
    if to_extract['PATCH']:
        print(f"    {to_extract['PATCH']:>12,} new cells on {statuses['PATCH']} PATCH slide(s)")
    for st in ('IDS_CHANGED', 'INCOMPLETE', 'NOT_EXTRACTED'):
        if to_extract[st]:
            print(f"    {to_extract[st]:>12,} whole-slide re-extraction of {statuses[st]} {st} slide(s)")
    print(f"embedded cells to drop (no longer in patch_coordinates): {n_removed:,}")
    print(f"kept cells with a stale cell_labels entry: {n_relabel:,}")
    print("(kept cells are assumed not to have moved: embeddings_dataset.h5 stores no coordinates)")

    if ids_file:
        ids_file.close()
        print(f"wrote {n_rows:,} row(s) to {args.ids_csv}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
