"""Find unreadable or black regions in the WSIs an extraction config would read, and the cells they hit.

Two kinds of bad region, both found by reading each of the slide's storage tiles once:
- unreadable tiles: a corrupt compressed tile makes every read_region touching it fail
  (TIFFRGBAImageGet failed), and extraction feeds a black placeholder patch instead;
- black areas: reads succeed but the pixels are black. A converted slide (see
  notebooks/check_wsi_openslide.ipynb) is black wherever its raw OME-TIFF had tiles with no
  data, because tifffile fills those with 0 -- extraction logs no warning for these cells.
Every cell's crop is then checked against both (by BLOCK-px blocks for black). For a
converted slide with black areas, the raw file's tiles without data are counted too, to
tell whether the conversion is where the black came from. Slides/cells are resolved like
extraction resolves them (samples / samples_manifest / exclude_samples, wsi_filename,
converted/raw root, cells_info_root[_by_prefix], size -- or mode resized's size_side crop
around the cell -- and builder_kwargs offsets).

CPU-only: deliberately does not import extract_embeddings.py (which pulls in torch), so
resolve_slides() mirrors its _resolve_samples + build_run_configs path logic -- keep the
two in sync if the config format changes.

Usage (from the repo root, with .env sourced):
    python -m src.python.extract_embeddings.check_wsi_regions configs/extract_embeddings/uni2_224.json
    python -m src.python.extract_embeddings.check_wsi_regions a.json dir/ --bad-cells-csv bad_cells.csv
    python -m src.python.extract_embeddings.check_wsi_regions configs/extract_embeddings/paper_uni2_fov  # every crop size
Exit code 1 if any cell's crop touches an unreadable tile or a black area.
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
BLOCK = 16       # px: black pixels are counted per BLOCK x BLOCK block
BLACK_MAX = 8    # a pixel is black when none of its RGB channels exceeds this
FULL = 0.9       # a crop at least this black is reported as a black patch


def _read_manifest(path: Path) -> dict[str, dict]:
    with open(path, newline='') as f:
        return {row['dataset']: {'converted': row['converted'].strip().lower() in ('true', '1'),
                                 'wsi_filename': row['wsi_filename']}
                for row in csv.DictReader(f) if row.get('status', 'ok') == 'ok'}


def resolve_slides(config_path: Path) -> list[dict]:
    """[{wsi_path, raw_wsi_paths, cells_info_path, x_size, y_size, offset_x, offset_y}] one JSON
    run config reads; raw_wsi_paths: where the raw file of a converted slide may be (else [])."""
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
    offset_x, offset_y = builder_kwargs.get('offset_x', 0), builder_kwargs.get('offset_y', 0)
    if cfg.get('mode') == 'resized' and cfg.get('size_side'):
        # ResizedCellDataset._fixed_box: a size_side crop centred on x_start + offset (112)
        size = int(cfg['size_side'])
        offset_x = offset_y = builder_kwargs.get('offset', 112) - size // 2
    default_root = cfg.get('cells_info_root', 'xenium_processed')
    by_prefix = cfg.get('cells_info_root_by_prefix', {})
    slides = []
    for name, info in infos.items():
        root = info.get('cells_info_root') or next(
            (r for prefix, r in by_prefix.items() if name.startswith(prefix)), default_root)
        wsi_root = paths.WSI_CONVERTED_ROOT if info.get('converted') else paths.WSI_RAW_ROOT
        wsi_filename = info.get('wsi_filename', f'{name}_he_image.ome.tif')
        slides.append({
            'wsi_path': os.path.join(wsi_root, wsi_filename),
            # the conversion keeps .ome.tif names and renames a raw .tif to .ome.tif
            'raw_wsi_paths': ([os.path.join(paths.WSI_RAW_ROOT, f) for f in
                               dict.fromkeys([wsi_filename, wsi_filename.removesuffix('.ome.tif') + '.tif'])]
                              if info.get('converted') else []),
            'cells_info_path': os.path.join(_CELLS_INFO_ROOTS.get(root, root), name, 'patch_coordinates.h5'),
            'x_size': size, 'y_size': size, 'offset_x': offset_x, 'offset_y': offset_y,
        })
    return slides


def _str(cell_id) -> str:
    return cell_id.decode() if isinstance(cell_id, bytes) else str(cell_id)


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


def scan_slide(wsi_path: str) -> dict:
    """Read level 0 tile by tile, once. Returns {'bad': unreadable tile (col, row) indices,
    'tile': tile size, 'dims': (W, H), 'black': black-pixel count per BLOCK x BLOCK block,
    'block': BLOCK}."""
    wsi = Openslide.OpenSlide(wsi_path)
    W, H = wsi.dimensions
    tile = int(wsi.properties.get('openslide.level[0].tile-width', 1024))
    blk = BLOCK if tile % BLOCK == 0 else tile
    black = np.zeros((-(-H // blk), -(-W // blk)), dtype=np.uint32)
    bad = []
    with _silence_stderr():
        for ty in range(0, H, tile):
            for tx in range(0, W, tile):
                w, h = min(tile, W - tx), min(tile, H - ty)
                try:
                    a = np.asarray(wsi.read_region((tx, ty), 0, (w, h)))
                except Openslide.OpenSlideError:
                    bad.append((tx // tile, ty // tile))
                    # a failed read leaves the handle in an error state -- reopen
                    wsi = Openslide.OpenSlide(wsi_path)
                    continue
                # fully transparent pixels have RGB 0 too, and .convert('RGB') makes them black
                dark = a[..., :3].max(axis=-1) <= BLACK_MAX
                if not dark.any():
                    continue
                bh, bw = -(-h // blk), -(-w // blk)
                padded = np.zeros((bh * blk, bw * blk), dtype=np.uint32)
                padded[:h, :w] = dark
                black[ty // blk:ty // blk + bh, tx // blk:tx // blk + bw] = \
                    padded.reshape(bh, blk, bw, blk).sum(axis=(1, 3))
    return {'bad': bad, 'tile': tile, 'dims': (W, H), 'black': black, 'block': blk}


def cell_crops(dataset_configs: dict, dims: tuple[int, int]):
    """(cell_ids, x0, y0, size_x, size_y): every cell's crop, clamped like
    PatchDataset._clamped_origin and ResizedCellDataset._crop_box."""
    size_x, size_y = dataset_configs.get('x_size', 224), dataset_configs.get('y_size', 224)
    W, H = dims
    with h5py.File(dataset_configs['cells_info_path'], 'r') as f:
        x = f['x_start'][:].astype(np.int64) + dataset_configs.get('offset_x', 0)
        y = f['y_start'][:].astype(np.int64) + dataset_configs.get('offset_y', 0)
        cell_ids = f['cell_id'][:]
    x = np.maximum(0, np.minimum(x, W - size_x))
    y = np.maximum(0, np.minimum(y, H - size_y))
    return cell_ids, x, y, size_x, size_y


def on_bad_tiles(x, y, size_x, size_y, bad: list, tile: int) -> np.ndarray:
    """Per cell: does its crop overlap an unreadable tile?"""
    hit = np.zeros(len(x), dtype=bool)
    for col, row in bad:
        hit |= ((x // tile <= col) & ((x + size_x - 1) // tile >= col)
                & (y // tile <= row) & ((y + size_y - 1) // tile >= row))
    return hit


def black_fraction(x, y, size_x, size_y, black: np.ndarray, blk: int) -> np.ndarray:
    """Per cell: the fraction of its crop's pixels that are black, from the per-block black
    pixel counts. A block the crop only partly covers contributes in proportion to the
    overlap (as if its black pixels were spread evenly), which is exact for black areas
    aligned to the block grid -- such as whole missing tiles."""
    S = np.zeros((black.shape[0] + 1, black.shape[1] + 1), dtype=np.int64)
    S[1:, 1:] = black.cumsum(0, dtype=np.int64).cumsum(1)

    def rect(r0, r1, c0, c1):  # black pixels in blocks rows r0..r1, cols c0..c1 (inclusive)
        return S[r1 + 1, c1 + 1] - S[r0, c1 + 1] - S[r1 + 1, c0] + S[r0, c0]

    c0, c1 = x // blk, (x + size_x - 1) // blk
    r0, r1 = y // blk, (y + size_y - 1) // blk
    # the parts of the first/last block column/row the crop does not cover
    a0, a1 = (x - c0 * blk) / blk, ((c1 + 1) * blk - x - size_x) / blk
    b0, b1 = (y - r0 * blk) / blk, ((r1 + 1) * blk - y - size_y) / blk
    n_black = (rect(r0, r1, c0, c1)
               - a0 * rect(r0, r1, c0, c0) - a1 * rect(r0, r1, c1, c1)
               - b0 * rect(r0, r0, c0, c1) - b1 * rect(r1, r1, c0, c1)
               + a0 * b0 * rect(r0, r0, c0, c0) + a0 * b1 * rect(r1, r1, c0, c0)
               + a1 * b0 * rect(r0, r0, c1, c1) + a1 * b1 * rect(r1, r1, c1, c1))
    return np.clip(n_black / (size_x * size_y), 0.0, 1.0)


def _black_area(scan: dict) -> str:
    black, blk, (W, H) = scan['black'], scan['block'], scan['dims']
    rows, cols = np.nonzero(black)
    return (f"black area {100 * black.sum() / (W * H):.2f}% of the slide, "
            f"x {cols.min() * blk}-{min((cols.max() + 1) * blk, W)}, y {rows.min() * blk}-{min((rows.max() + 1) * blk, H)}")


def raw_tiles_without_data(raw_paths: list[str]) -> str:
    """Where a converted slide's black came from: how many level-0 tiles of its raw file have
    no data (offset or byte count 0), which tifffile -- and so the conversion -- fills with 0."""
    path = next((p for p in raw_paths if os.path.exists(p)), None)
    if path is None:
        return "raw file not found"
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            pages = [p for p in tf.series[0].levels[0].pages if p is not None]
            offsets = np.concatenate([np.asarray(p.dataoffsets) for p in pages])
            counts = np.concatenate([np.asarray(p.databytecounts) for p in pages])
    except Exception as e:
        return f"raw file unreadable ({type(e).__name__}: {e})"[:200]
    missing = int(((offsets == 0) | (counts == 0)).sum())
    return f"raw file: {missing:,} / {len(counts):,} tiles without data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find unreadable or black WSI regions and the cells they affect.")
    parser.add_argument('configs', nargs='+', help="extraction JSON configs and/or directories of them")
    parser.add_argument('--bad-cells-csv', help="write (wsi_path, crop_px, cell_id, reason, black_frac) "
                                                "of every affected cell here")
    args = parser.parse_args(argv)

    # every distinct (slide, coordinates, geometry) the configs would read
    runs = {}
    for config in _expand(args.configs):
        for dc in resolve_slides(config):
            key = (dc['wsi_path'], dc['cells_info_path'], dc['x_size'], dc['y_size'],
                   dc['offset_x'], dc['offset_y'])
            runs.setdefault(key, dc)
    print(f"{len(runs)} slide/coordinate combination(s) to check", flush=True)

    # slide by slide (all of one slide's crop sizes in a row): one slide's scan in memory
    bad_rows = []
    damaged, blackened, scanned = set(), set(), set()
    current, scan = None, None
    for key in sorted(runs, key=lambda k: (k[0], k[2], k[3])):
        dc = runs[key]
        wsi_path = dc['wsi_path']
        name = f"{os.path.basename(wsi_path)} [{dc['x_size']}px crop]"
        if not os.path.exists(wsi_path):
            print(f"MISSING WSI      {name}: {wsi_path}", flush=True)
            continue
        if wsi_path != current:
            current, scan = wsi_path, None
            try:
                scan = scan_slide(wsi_path)
                scanned.add(wsi_path)
            except Openslide.OpenSlideError as e:
                print(f"CANNOT OPEN      {name}: {e}", flush=True)
        if scan is None:
            continue
        bad, tile, black, blk = scan['bad'], scan['tile'], scan['black'], scan['block']
        cells_info_path = dc.get('cells_info_path')
        crops = (cell_crops(dc, scan['dims'])
                 if cells_info_path and os.path.exists(cells_info_path) else None)
        details = []
        if bad:
            damaged.add(wsi_path)
            xs, ys = [c * tile for c, _ in bad], [r * tile for _, r in bad]
            details.append(f"unreadable {tile}px tiles: {len(bad)}, region x {min(xs)}-{max(xs) + tile}, "
                           f"y {min(ys)}-{max(ys) + tile}")
            if crops is not None:
                ids, x, y, sx, sy = crops
                hit = on_bad_tiles(x, y, sx, sy, bad, tile)
                details.append(f"cells on them: {hit.sum():,} / {len(x):,}")
                bad_rows += [(wsi_path, dc['x_size'], _str(i), 'unreadable_tile', '') for i in ids[hit]]
        if crops is not None and black.any():
            ids, x, y, sx, sy = crops
            frac = black_fraction(x, y, sx, sy, black, blk)
            if (frac > 0).any():
                blackened.add(wsi_path)
                n_full, n_part = int((frac >= FULL).sum()), int(((frac > 0) & (frac < FULL)).sum())
                details.append(_black_area(scan))
                details.append(f"cells with a black crop: {n_full:,} >={FULL:.0%} black, "
                               f"{n_part:,} partly / {len(x):,}")
                if dc.get('raw_wsi_paths'):
                    if 'raw' not in scan:
                        scan['raw'] = raw_tiles_without_data(dc['raw_wsi_paths'])
                    details.append(scan['raw'])
                bad_rows += [(wsi_path, dc['x_size'], _str(i), 'black', f"{f:.3f}")
                             for i, f in zip(ids[frac > 0], frac[frac > 0])]
        if not details:
            note = f" ({_black_area(scan)}, no cell's crop on it)" if black.any() else ""
            print(f"ok               {name}{note}", flush=True)
            continue
        status = 'DAMAGED' if bad else 'BLACK'
        print(f"{status:<16} {name} | " + " | ".join(details), flush=True)

    if args.bad_cells_csv:
        with open(args.bad_cells_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['wsi_path', 'crop_px', 'cell_id', 'reason', 'black_frac'])
            writer.writerows(bad_rows)
        print(f"wrote {len(bad_rows)} affected cell(s) to {args.bad_cells_csv}")
    print(f"\n{len(scanned)} slide(s) scanned: {len(damaged)} with unreadable tiles, "
          f"{len(blackened)} with cells on black areas")
    return 1 if damaged or blackened else 0


if __name__ == '__main__':
    sys.exit(main())
