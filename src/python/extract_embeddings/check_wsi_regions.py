"""Find unreadable regions in the WSIs an extraction config would read, and the cells they hit.

A corrupt compressed tile makes every read_region touching it fail (TIFFRGBAImageGet failed),
so instead of reading every cell's crop this reads each of the slide's storage tiles once,
then flags the cells whose crop overlaps a bad tile. Slides/cells are resolved like
extraction resolves them (samples / samples_manifest / exclude_samples, wsi_filename,
converted/raw root, cells_info_root[_by_prefix], size, builder_kwargs offsets).

CPU-only: deliberately does not import extract_embeddings.py (which pulls in torch), so
resolve_slides() mirrors its _resolve_samples + build_run_configs path logic -- keep the
two in sync if the config format changes.

Usage (from the repo root, with .env sourced):
    python -m src.python.extract_embeddings.check_wsi_regions configs/extract_embeddings/uni2_224.json
    python -m src.python.extract_embeddings.check_wsi_regions a.json dir/ --bad-cells-csv bad_cells.csv
"""
import argparse
import contextlib
import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide as Openslide

from src.python.code_configs import paths
from .run_configs import _expand

_CELLS_INFO_ROOTS = {'xenium_processed': paths.XENIUM_PROCESSED_OUTPUT_ROOT,
                     'positions_converted': paths.POSITIONS_CONVERTED_ROOT}


def _read_manifest(path: Path) -> dict[str, dict]:
    with open(path, newline='') as f:
        return {row['dataset']: {'converted': row['converted'].strip().lower() in ('true', '1'),
                                 'wsi_filename': row['wsi_filename']}
                for row in csv.DictReader(f) if row.get('status', 'ok') == 'ok'}


def resolve_slides(config_path: Path) -> list[dict]:
    """[{wsi_path, cells_info_path, x_size, y_size, offset_x, offset_y}] one JSON run config reads."""
    with open(config_path) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith('_')}
    manifest = (_read_manifest((config_path.parent / cfg['samples_manifest']).resolve())
                if 'samples_manifest' in cfg else None)
    samples = cfg.get('samples')
    if samples is None:
        infos = dict(manifest or {})
    elif isinstance(samples, list):
        infos = {name: (manifest or {}).get(name, {'converted': False}) for name in samples}
    else:
        infos = {name: {'converted': False, **(manifest or {}).get(name, {}), **(info or {})}
                 for name, info in samples.items()}
    for name in cfg.get('exclude_samples', []):
        infos.pop(name, None)

    size = cfg.get('size', 224)
    builder_kwargs = cfg.get('builder_kwargs', {})
    default_root = cfg.get('cells_info_root', 'xenium_processed')
    by_prefix = cfg.get('cells_info_root_by_prefix', {})
    slides = []
    for name, info in infos.items():
        root = info.get('cells_info_root') or next(
            (r for prefix, r in by_prefix.items() if name.startswith(prefix)), default_root)
        wsi_root = paths.WSI_CONVERTED_ROOT if info.get('converted') else paths.WSI_RAW_ROOT
        slides.append({
            'wsi_path': os.path.join(wsi_root, info.get('wsi_filename', f'{name}_he_image.ome.tif')),
            'cells_info_path': os.path.join(_CELLS_INFO_ROOTS.get(root, root), name, 'patch_coordinates.h5'),
            'x_size': size, 'y_size': size,
            'offset_x': builder_kwargs.get('offset_x', 0), 'offset_y': builder_kwargs.get('offset_y', 0),
        })
    return slides


@contextlib.contextmanager
def _silence_stderr():
    """libtiff prints a ZIPDecode line to fd 2 for every failed tile; hide that noise."""
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def scan_bad_tiles(wsi_path: str) -> tuple[list[tuple[int, int]], int, tuple[int, int]]:
    """(bad tile (col, row) indices, tile size, (W, H)) for level 0 of wsi_path."""
    wsi = Openslide.OpenSlide(wsi_path)
    W, H = wsi.dimensions
    tile = int(wsi.properties.get('openslide.level[0].tile-width', 1024))
    bad = []
    with _silence_stderr():
        for ty in range(0, H, tile):
            for tx in range(0, W, tile):
                try:
                    wsi.read_region((tx, ty), 0, (min(tile, W - tx), min(tile, H - ty)))
                except Openslide.OpenSlideError:
                    bad.append((tx // tile, ty // tile))
                    # a failed read leaves the handle in an error state -- reopen
                    wsi = Openslide.OpenSlide(wsi_path)
    return bad, tile, (W, H)


def cells_on_bad_tiles(dataset_configs: dict, bad: list, tile: int, dims: tuple[int, int]):
    """(cell_ids, n_cells) whose crop (clamped like PatchDataset._clamped_origin) overlaps a bad tile."""
    size_x, size_y = dataset_configs.get('x_size', 224), dataset_configs.get('y_size', 224)
    W, H = dims
    with h5py.File(dataset_configs['cells_info_path'], 'r') as f:
        x = f['x_start'][:].astype(np.int64) + dataset_configs.get('offset_x', 0)
        y = f['y_start'][:].astype(np.int64) + dataset_configs.get('offset_y', 0)
        cell_ids = f['cell_id'][:]
    x = np.clip(x, 0, W - size_x)
    y = np.clip(y, 0, H - size_y)
    hit = np.zeros(len(x), dtype=bool)
    for col, row in bad:
        hit |= ((x // tile <= col) & ((x + size_x - 1) // tile >= col)
                & (y // tile <= row) & ((y + size_y - 1) // tile >= row))
    return cell_ids[hit], len(x)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find unreadable WSI tiles and the cells they affect.")
    parser.add_argument('configs', nargs='+', help="extraction JSON configs and/or directories of them")
    parser.add_argument('--bad-cells-csv', help="write (wsi_path, cell_id) of every affected cell here")
    args = parser.parse_args(argv)

    # every distinct (slide, coordinates, geometry) the configs would read
    runs = {}
    for config in _expand(args.configs):
        for dc in resolve_slides(config):
            key = (dc['wsi_path'], dc['cells_info_path'], dc['x_size'], dc['y_size'],
                   dc['offset_x'], dc['offset_y'])
            runs.setdefault(key, dc)
    print(f"{len(runs)} slide/coordinate combination(s) to check", flush=True)

    scanned = {}
    bad_rows = []
    damaged = 0
    for (wsi_path, *_), dc in runs.items():
        name = os.path.basename(wsi_path)
        if not os.path.exists(wsi_path):
            print(f"MISSING WSI      {name}: {wsi_path}", flush=True)
            continue
        if wsi_path not in scanned:
            try:
                scanned[wsi_path] = scan_bad_tiles(wsi_path)
            except Openslide.OpenSlideError as e:
                print(f"CANNOT OPEN      {name}: {e}", flush=True)
                continue
        bad, tile, dims = scanned[wsi_path]
        if not bad:
            print(f"ok               {name}", flush=True)
            continue
        damaged += 1
        line = f"DAMAGED          {name} | bad {tile}px tiles: {len(bad)}"
        cells_info_path = dc.get('cells_info_path')
        if cells_info_path and os.path.exists(cells_info_path):
            ids, n = cells_on_bad_tiles(dc, bad, tile, dims)
            line += f" | cells affected: {len(ids)} / {n}"
            bad_rows += [(wsi_path, i.decode() if isinstance(i, bytes) else i) for i in ids]
        xs, ys = [c * tile for c, _ in bad], [r * tile for _, r in bad]
        line += f" | region x {min(xs)}-{max(xs) + tile}, y {min(ys)}-{max(ys) + tile}"
        print(line, flush=True)

    if args.bad_cells_csv:
        with open(args.bad_cells_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['wsi_path', 'cell_id'])
            writer.writerows(bad_rows)
        print(f"wrote {len(bad_rows)} affected cell(s) to {args.bad_cells_csv}")
    print(f"\n{damaged} damaged slide(s) out of {len(scanned)} scanned")
    return 1 if damaged else 0


if __name__ == '__main__':
    sys.exit(main())
