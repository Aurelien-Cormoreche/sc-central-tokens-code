#!/usr/bin/env python3
"""Re-convert a slide whose raw TIFF has 32-bit offsets that wrapped past 4 GiB (see check_tiff_offsets.py),
reading the raw file with the wrap undone.

Same recipe as notebooks/check_wsi_openslide.ipynb (256 x 256 tiles, uncompressed, single level, OME BigTIFF,
the raw file's resolution tags), but streamed band by band (~60 MB in memory instead of the whole slide).
Steps, each refusing to go on when something is off:
  1. raw: uncompressed chunky RGB(A), offsets that decrease (wrapped), and a plain wrap: every segment lands
     inside the file once 2**32 is added after each decrease
  2. write <converted>.partial.ome.tif
  3. verify the new file with OpenSlide: equal to the raw pixels (wrap undone) on random regions and on bands
     across every wrapped row, and equal to the old converted copy above the first wrapped row
  4. swap: the old copy becomes <name>.wrapped (a backup to delete once you have checked the slide), the new one
     takes its name, so bench's paths do not change. The file's content fingerprint changes with it, so bench
     no longer counts the slide's dark-spot map, manifest row, embeddings or probe results as done.

  python scripts/fix_wrapped_tiff.py GSM9509150_NSCLC_lung_L4_L4
Run it on a compute node (reads and writes ~2 x the slide's size; a few minutes).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_tiff_offsets as cto  # noqa: E402  (unwrap, read_region, WRAP)

TILE = 256
N_RANDOM = 64


def wrapped_rows(page, raw_off, fixed_off) -> list:
    """First image row of every segment where the added multiple of 2**32 changes."""
    add = fixed_off - raw_off
    ks = [k for k in range(len(add)) if k == 0 and add[0] or k > 0 and add[k] != add[k - 1]]
    if page.is_tiled:
        ntx = math.ceil(int(page.imagewidth) / int(page.tilewidth))
        return sorted({(k // ntx) * int(page.tilelength) for k in ks})
    rps = int(page.rowsperstrip) if page.rowsperstrip else int(page.imagelength)
    return sorted({k * rps for k in ks})


def tiles(fh, page, offsets, W: int, H: int):
    """256 x 256 RGB tiles in row-major order (edge tiles zero-padded, as tifffile pads them)."""
    for y0 in range(0, H, TILE):
        h = min(TILE, H - y0)
        band = cto.read_region(fh, page, offsets, 0, y0, W, h)[..., :3]
        for x0 in range(0, W, TILE):
            t = np.zeros((TILE, TILE, 3), np.uint8)
            blk = band[:, x0:x0 + TILE]
            t[:h, :blk.shape[1]] = blk
            yield t


def fix(raw: str, conv: str, wrap: int = cto.WRAP, seed: int = 0) -> str:
    import openslide
    import tifffile
    from bench.utils import file_fingerprint

    t0 = time.time()
    fp_old = file_fingerprint(conv)
    size = os.path.getsize(raw)
    with tifffile.TiffFile(raw) as tf:
        page = tf.series[0].levels[0].keyframe
        comp = getattr(page.compression, "name", str(page.compression))
        planar = getattr(page.planarconfig, "name", str(page.planarconfig))
        if comp != "NONE" or planar != "CONTIG" or int(page.samplesperpixel) not in (3, 4):
            raise SystemExit(f"raw is {comp} / {planar} / {page.samplesperpixel} samples: only uncompressed chunky "
                             f"RGB(A) can be read with fixed offsets")
        if len(tf.series[0].pages) != 1:
            raise SystemExit(f"series 0 has {len(tf.series[0].pages)} pages: expected one RGB page")
        W, H = int(page.imagewidth), int(page.imagelength)
        off = np.asarray(page.dataoffsets, dtype=np.int64)
        cnt = np.asarray(page.databytecounts, dtype=np.int64)
        try:
            fixed = cto.unwrap(off, cnt, size, wrap)
        except ValueError as e:
            raise SystemExit(f"not a plain wrap, refusing: {e}")
        if np.array_equal(fixed, off):
            raise SystemExit("the raw offsets do not wrap: nothing to fix")
        rows = wrapped_rows(page, off, fixed)
        res = getattr(page, "resolution", None)
        unit = getattr(page, "resolutionunit", None)
        extra = {"resolution": res, "resolutionunit": unit} if res and all(res) and unit is not None else {}
        print(f"raw  {raw}: {W} x {H}, {'BigTIFF' if tf.is_bigtiff else 'classic TIFF'}, {size / 2**30:.2f} GiB, "
              f"{len(off)} segments; wrapped from rows {rows}")
        with openslide.OpenSlide(conv) as sl:
            if sl.dimensions != (W, H):
                raise SystemExit(f"converted copy is {sl.dimensions}, raw {W} x {H}: not the same image")

        tmp = conv[: -len(".ome.tif")] + ".partial.ome.tif" if conv.endswith(".ome.tif") else conv + ".partial"
        print(f"writing {tmp} ...")
        with open(raw, "rb") as fh:
            tifffile.imwrite(tmp, tiles(fh, page, fixed, W, H), shape=(H, W, 3), dtype=np.uint8, bigtiff=True,
                             tile=(TILE, TILE), photometric="rgb", planarconfig="contig", compression=None,
                             ome=True, metadata={"axes": "YXS"}, **extra)
        print(f"  {os.path.getsize(tmp) / 2**30:.2f} GiB in {time.time() - t0:.0f} s")

        # verify against the raw pixels (wrap undone) and, above the first wrap, against the old copy
        rng = np.random.default_rng(seed)
        S = 512
        boxes = []
        for _ in range(N_RANDOM):
            x, y = int(rng.integers(0, max(1, W - S + 1))), int(rng.integers(0, max(1, H - S + 1)))
            boxes.append((x, y, min(S, W - x), min(S, H - y)))
        xs = [int(x) for x in np.linspace(0, max(0, W - S), 6)]
        for r in rows:                                          # across every wrapped row
            y = int(min(max(0, r - S // 2), max(0, H - S)))
            boxes += [(x, y, min(S, W - x), min(S, H - y)) for x in xs]
        y = max(0, rows[0] - S)                                 # just above the first one
        boxes += [(x, y, min(S, W - x), rows[0] - y) for x in xs]
        n_new, n_old = 0, 0
        with open(raw, "rb") as fh, openslide.OpenSlide(tmp) as new, openslide.OpenSlide(conv) as old:
            if new.dimensions != (W, H):
                raise SystemExit(f"new file is {new.dimensions}, expected {W} x {H}")
            for x, y, w, h in boxes:
                want = cto.read_region(fh, page, fixed, x, y, w, h)[..., :3]
                got = np.asarray(new.read_region((x, y), 0, (w, h)))[..., :3]
                if not np.array_equal(want, got):
                    raise SystemExit(f"verification failed at ({x}, {y}): new file != raw pixels; {tmp} kept for a look")
                n_new += 1
                if y + h <= rows[0]:
                    prev = np.asarray(old.read_region((x, y), 0, (w, h)))[..., :3]
                    if not np.array_equal(prev, got):
                        raise SystemExit(f"verification failed at ({x}, {y}): above the first wrap the new file "
                                         f"differs from the old copy; {tmp} kept for a look")
                    n_old += 1
            band = rows[0]
            y = int(min(max(0, band - S), max(0, H - 2 * S)))
            a = np.asarray(new.read_region((W // 2 - S, y), 0, (2 * S, 2 * S)))[..., :3]
            b = np.asarray(old.read_region((W // 2 - S, y), 0, (2 * S, 2 * S)))[..., :3]
        cut = band - y
        print(f"verified: {n_new} regions equal to the raw pixels, {n_old} of them (above row {rows[0]}) equal to the "
              f"old copy")
        print(f"  tissue colour around row {band}:  old copy above {cto.tissue_rgb(b[:cut])} / below "
              f"{cto.tissue_rgb(b[cut:])}   new above {cto.tissue_rgb(a[:cut])} / below {cto.tissue_rgb(a[cut:])}")

    backup = conv + ".wrapped"
    os.replace(conv, backup)
    os.replace(tmp, conv)
    print(f"swapped: {conv} is the fixed file; the old one is {backup} (delete it once the slide looks right)")
    fp_new = file_fingerprint(conv)
    print(f"content fingerprint {fp_old} -> {fp_new}")
    if fp_new == fp_old:
        print("  WARNING: same fingerprint, so bench will not notice the change: delete the slide's "
              "{bench_root}/darkspots/<slide>.npz by hand before rerunning darkspots and the manifest")
    return conv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("slide")
    ap.add_argument("--config", help="bench YAML (default: $BENCH_CONFIG or configs/bench.yaml)")
    a = ap.parse_args(argv)
    from bench.config import load_config
    from bench.data.inspect import load_inspect
    cfg = load_config(a.config)
    ins = load_inspect(cfg)
    if a.slide not in ins.index:
        raise SystemExit(f"{a.slide} is not in the inspect table")
    row = ins.loc[a.slide].to_dict()
    raw, conv = str(row.get("raw_path") or ""), str(row["wsi_path"])
    if not raw or not os.path.isfile(raw):
        raise SystemExit(f"no raw original for {a.slide} (inspect raw_path = {raw!r})")
    if not row.get("converted"):
        raise SystemExit(f"{a.slide}: bench reads the raw file itself ({conv}); nothing to re-convert")
    fix(raw, conv)
    d = a.slide
    print(f"\nnext (the slide's old dark-spot map, manifest row and embeddings no longer count as done):\n"
          f"  bench darkspots --datasets {d}\n  bench manifest --datasets {d}\n  bench overview --datasets {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
