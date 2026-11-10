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

# kept / current cells below this -> ids were not conserved, re-extract the whole slide
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


def _as_str(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == 'U':
        return values
    if values.dtype.kind == 'S':
        return np.char.decode(values, 'utf-8')
    return np.array([v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in values], dtype=str)


def _read(path: Path, id_key: str, label_key: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, 'r', locking=False) as f:
        return _as_str(f[id_key][()]), _as_str(f[label_key][()])


def compare(emb_ids, emb_labels, new_ids, new_labels) -> dict:
    written = emb_ids != ''
    n_unwritten = int((~written).sum())
    emb_ids, emb_labels = emb_ids[written], emb_labels[written]

    stripped = False
    common, i_emb, i_new = np.intersect1d(emb_ids, new_ids, return_indices=True)
    if len(common) < IDS_CHANGED_BELOW * min(len(emb_ids), len(new_ids)):
        # maybe only the id formatting changed (e.g. a '{sample}_' prefix added/dropped)
        e2 = np.array([c.rsplit('_', 1)[-1] for c in emb_ids], dtype=str)
        n2 = np.array([c.rsplit('_', 1)[-1] for c in new_ids], dtype=str)
        common2, i_emb2, i_new2 = np.intersect1d(e2, n2, return_indices=True)
        if len(common2) > len(common):
            stripped = True
            emb_ids, new_ids = e2, n2
            common, i_emb, i_new = common2, i_emb2, i_new2

    removed = np.setdiff1d(emb_ids, new_ids)
    added = np.setdiff1d(new_ids, emb_ids)
    changed = emb_labels[i_emb] != new_labels[i_new]
    order_kept = len(common) < 2 or bool(np.all(np.diff(i_new[np.argsort(i_emb)]) > 0))
    return {
        'n_emb': len(emb_ids) + n_unwritten, 'n_unwritten': n_unwritten, 'n_new': len(new_ids),
        'n_new_unique': len(np.unique(new_ids)),
        'dup_emb': len(emb_ids) - len(np.unique(emb_ids)), 'dup_new': len(new_ids) - len(np.unique(new_ids)),
        'kept': len(common), 'removed': removed, 'added': added, 'stripped': stripped,
        'label_changed': list(zip(common[changed], emb_labels[i_emb][changed], new_labels[i_new][changed])),
        'transitions': Counter(zip(emb_labels[i_emb][changed], new_labels[i_new][changed])),
        'order_kept': order_kept,
    }


def verdict(r: dict) -> tuple[str, int]:
    """(status, cells to extract)."""
    if r['n_unwritten']:
        return 'INCOMPLETE', r['n_new_unique']      # skip_existing re-extracts the whole slide anyway
    if r['kept'] < IDS_CHANGED_BELOW * r['n_new_unique']:
        return 'IDS_CHANGED', r['n_new_unique']
    if len(r['added']) or len(r['removed']):
        return 'PATCH', len(r['added'])
    if r['stripped']:
        return 'ID_FORMAT', 0                        # same cells, but the id strings differ
    if r['label_changed']:
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

    statuses = Counter()
    to_extract = Counter()
    n_removed = n_relabel = 0
    details, rows = [], []
    out_root = None
    for s in slides.values():
        name = s['sample']
        if s['out_root'] != out_root:
            out_root = s['out_root']
            print(f"\n### {out_root}\n{header}")
        if not s['coords_path'].exists():
            print(f"{'NO_COORDS':13} {name:64} {s['coords_path']}")
            statuses['NO_COORDS'] += 1
            continue
        new_ids, new_labels = _read(s['coords_path'], 'cell_id', 'cell_type')
        if not s['emb_path'].exists():
            n = len(np.unique(new_ids))
            print(f"{'NOT_EXTRACTED':13} {name:64} {'-':>9} {len(new_ids):9,} {'':>9} {'':>8} {'':>8} {'':>8} "
                  f"{'':>5} {n:10,}")
            statuses['NOT_EXTRACTED'] += 1
            to_extract['NOT_EXTRACTED'] += n
            continue
        try:
            emb_ids, emb_labels = _read(s['emb_path'], 'cell_ids', 'cell_labels')
        except (OSError, KeyError) as e:
            print(f"{'UNREADABLE':13} {name:64} {type(e).__name__}: {e}")
            statuses['UNREADABLE'] += 1
            continue

        r = compare(emb_ids, emb_labels, new_ids, new_labels)
        status, n_extract = verdict(r)
        statuses[status] += 1
        to_extract[status] += n_extract
        n_removed += len(r['removed'])
        n_relabel += len(r['label_changed'])
        order = '-' if r['kept'] < 2 else 'yes' if r['order_kept'] else 'NO'
        print(f"{status:13} {name:64} {r['n_emb']:9,} {r['n_new']:9,} {r['kept']:9,} {len(r['removed']):8,} "
              f"{len(r['added']):8,} {len(r['label_changed']):8,} {order:>5} {n_extract:10,}")

        notes = []
        if r['n_unwritten']:
            notes.append(f"{r['n_unwritten']:,} unwritten rows in embeddings_dataset.h5")
        if r['stripped']:
            notes.append("ids only match after stripping everything up to the last '_' (id format changed)")
        if r['dup_emb'] or r['dup_new']:
            notes.append(f"duplicate ids: {r['dup_emb']:,} embedded, {r['dup_new']:,} current")
        if len(r['added']):
            notes.append("added e.g. " + ", ".join(r['added'][:3]))
        if len(r['removed']):
            notes.append("removed e.g. " + ", ".join(r['removed'][:3]))
        if r['transitions']:
            notes.append("labels " + ", ".join(f"{a}->{b} ({n})" for (a, b), n in r['transitions'].most_common(3)))
        if notes:
            details.append((f"{name}  ({s['emb_path'].parent.parent.parent.name})", notes))

        rows += [(name, c, 'added', '', '') for c in r['added']]
        rows += [(name, c, 'removed', '', '') for c in r['removed']]
        rows += [(name, c, 'label_changed', a, b) for c, a, b in r['label_changed']]

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

    if args.ids_csv:
        with open(args.ids_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample', 'cell_id', 'change', 'old_label', 'new_label'])
            writer.writerows(rows)
        print(f"wrote {len(rows):,} row(s) to {args.ids_csv}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
