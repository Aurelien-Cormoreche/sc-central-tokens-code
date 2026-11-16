"""Who wrote each raw H&E TIFF, and which of its level-0 tiles have no data.

A tile with no data (offset or byte count 0) is legal TIFF and most readers show it as
empty, but OpenSlide fails on it (TIFFRGBAImageGet failed), and tifffile -- so the
conversion in notebooks/check_wsi_openslide.ipynb -- fills it with 0, i.e. black.
Reads only the file headers, so it is quick and fine on a login node.

Usage (repo root, .env sourced; a sample name is looked up as
$WSI_RAW_ROOT/{name}_he_image.ome.tif):
    python scripts/raw_tiff_info.py GSM9509144_NSCLC_chuvio_L4_L4 GSM9509145_NSCLC_lung_L1_L1
    python scripts/raw_tiff_info.py /path/to/slide.ome.tif
"""
import math
import os
import re
import sys

import numpy as np
import tifffile


def describe(path: str) -> str:
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        pages = [p for p in series.levels[0].pages if p is not None]
        p = pages[0]
        creator = re.search(r'Creator="([^"]*)"', tf.ome_metadata or '')
        empty_xy, n_empty, n_total = set(), 0, 0
        for q in pages:
            counts, offsets = np.asarray(q.databytecounts), np.asarray(q.dataoffsets)
            missing = np.flatnonzero((counts == 0) | (offsets == 0))
            n_empty += len(missing)
            n_total += len(counts)
            if q.is_tiled and len(missing):
                ntx = math.ceil(q.imagewidth / q.tilewidth)
                nty = math.ceil(q.imagelength / q.tilelength)
                i = missing % (ntx * nty)   # planar-separate pages repeat the grid per sample
                empty_xy |= {(int(c) * q.tilewidth, int(r) * q.tilelength) for r, c in zip(i // ntx, i % ntx)}
        layout = f"tile {p.tilewidth}x{p.tilelength}" if p.is_tiled else "strips"
        line = (f"software: {p.software or '-'} | OME creator: {creator.group(1) if creator else '-'} | "
                f"{series.axes} {series.shape} | {layout} {p.compression.name} | "
                f"level-0 tiles without data: {n_empty:,} / {n_total:,}")
        if empty_xy:
            xs, ys = [x for x, _ in empty_xy], [y for _, y in empty_xy]
            area = len(empty_xy) * p.tilewidth * p.tilelength / (p.imagewidth * p.imagelength)
            line += (f" = {100 * area:.2f}% of the image, x {min(xs)}-{min(max(xs) + p.tilewidth, p.imagewidth)}, "
                     f"y {min(ys)}-{min(max(ys) + p.tilelength, p.imagelength)}")
    return line


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        path = arg if os.path.exists(arg) else os.path.join(os.environ['WSI_RAW_ROOT'], f'{arg}_he_image.ome.tif')
        name = os.path.basename(path)
        try:
            print(f"{name} | {describe(path)}", flush=True)
        except Exception as e:
            print(f"{name} | ERROR {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
