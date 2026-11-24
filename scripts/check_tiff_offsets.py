#!/usr/bin/env python3
"""Check whether a slide's raw TIFF has 32-bit offsets that wrapped past 4 GiB, and whether undoing the wrap
recovers the image.

A classic (non-Big) TIFF stores where each strip / tile starts as a 32-bit number. When the pixel data passes
4 GiB, those numbers wrap round, and every reader then takes each later strip from 2**32 bytes earlier in the
file. 2**32 is not a multiple of 3, so the colours come out rotated (R,G,B -> B,R,G or G,B,R) and the rows
shifted sideways. Converting such a file (notebooks/check_wsi_openslide.ipynb: tifffile asarray -> BigTIFF)
copies the wrong pixels faithfully. If the file was written in order, the data is still all there, and
adding 2**32 to every offset after the wrap reads it correctly.

For each slide (default: the two SPLIT slides with > 4 GiB of pixels, and the largest one below the limit as a
control) it prints the raw file's layout and offset wraps, and writes {bench_root}/qc/tiff_offsets/<slide>.png:
a band of the image around the first wrapped row, read three ways --
  1. the converted copy bench reads (OpenSlide)
  2. the raw file at its stored offsets (what every reader, and the conversion, gets)
  3. the raw file with the wrap undone
with the mean tissue colour above / below the boundary. Panel 3 purple on both sides = fixable.
Light: reads the TIFF headers and a few MB per slide.

  python scripts/check_tiff_offsets.py
  python scripts/check_tiff_offsets.py GSM9509150_NSCLC_lung_L4_L4
  python scripts/check_tiff_offsets.py --scan      # headers only: every slide's file (and raw original) that is a
                                                   # classic TIFF over 4 GiB, i.e. could have wrapped offsets
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

WRAP = 2 ** 32
DEFAULT = ["GSM9509144_NSCLC_chuvio_L4_L4", "GSM9509150_NSCLC_lung_L4_L4", "GSM9509143_NSCLC_chuvio_L3_L3"]
BAND_H, BAND_W, DOWN = 1536, 3072, 3


def unwrap(offsets, bytecounts, filesize: int, wrap: int = WRAP) -> np.ndarray:
    """Offsets of segments written in order, with the 32-bit wrap undone: each offset gets the smallest
    multiple of `wrap` that puts it at or after the end of the previous segment."""
    off = np.asarray(offsets, dtype=np.int64)
    cnt = np.asarray(bytecounts, dtype=np.int64)
    out = off.copy()
    add = 0
    for k in range(1, len(off)):
        while off[k] + add < out[k - 1] + cnt[k - 1]:
            add += wrap
        out[k] = off[k] + add
        if out[k] + cnt[k] > filesize:
            raise ValueError(f"segment {k} would end at {out[k] + cnt[k]} > file size {filesize}: not a plain wrap")
    return out


def read_region(fh, page, offsets, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """(h, w, spp) uint8 from an uncompressed, chunky (contig) page, at the given segment offsets."""
    W, spp = int(page.imagewidth), int(page.samplesperpixel)
    out = np.zeros((h, w, spp), np.uint8)
    if page.is_tiled:
        th, tw = int(page.tilelength), int(page.tilewidth)
        ntx = math.ceil(W / tw)
        for ty in range(y0 // th, (y0 + h - 1) // th + 1):
            for tx in range(x0 // tw, (x0 + w - 1) // tw + 1):
                fh.seek(int(offsets[ty * ntx + tx]))
                tile = np.frombuffer(fh.read(th * tw * spp), np.uint8)
                tile = np.pad(tile, (0, th * tw * spp - tile.size)).reshape(th, tw, spp)
                ya, yb = max(y0, ty * th), min(y0 + h, (ty + 1) * th)
                xa, xb = max(x0, tx * tw), min(x0 + w, (tx + 1) * tw)
                out[ya - y0:yb - y0, xa - x0:xb - x0] = tile[ya - ty * th:yb - ty * th, xa - tx * tw:xb - tx * tw]
    else:
        rps = int(page.rowsperstrip) if page.rowsperstrip else int(page.imagelength)
        for y in range(y0, y0 + h):
            k = y // rps
            fh.seek(int(offsets[k]) + ((y - k * rps) * W + x0) * spp)
            row = np.frombuffer(fh.read(w * spp), np.uint8)
            out[y - y0, :row.size // spp] = row[: row.size // spp * spp].reshape(-1, spp)
    return out


def tissue_rgb(a: np.ndarray) -> str:
    rgb = a[..., :3].reshape(-1, 3).astype(int)
    t = rgb[(rgb.min(1) < 200) & (rgb.max(1) > 30)]
    return "no tissue" if len(t) < 100 else "RGB " + " ".join(f"{v:.0f}" for v in t.mean(0))


def first_wrapped_row(page, raw_off, fixed_off) -> int | None:
    k = np.flatnonzero(raw_off != fixed_off)
    if not len(k):
        return None
    k = int(k[0])
    if page.is_tiled:
        ntx = math.ceil(int(page.imagewidth) / int(page.tilewidth))
        return (k // ntx) * int(page.tilelength)
    return k * (int(page.rowsperstrip) if page.rowsperstrip else int(page.imagelength))


def check(cfg, row: dict, out_dir) -> None:
    import openslide
    import tifffile
    from PIL import Image, ImageDraw
    from bench.data.crop_sheet import _font
    from bench.data.darkspots import load_coords

    slide, raw, conv = row["slide"], str(row.get("raw_path") or ""), row["wsi_path"]
    W, H = int(row["width"]), int(row["height"])
    print(f"\n== {slide}: {W} x {H} px, {W * H * 3 / 2**30:.2f} GiB of RGB "
          f"(the 4 GiB mark is at row {WRAP / (W * 3):.0f} if the pixels start at the file's beginning)")
    if not raw or not os.path.isfile(raw):
        print(f"  no raw original found (inspect raw_path = {raw!r})")
        return
    size = os.path.getsize(raw)
    with tifffile.TiffFile(raw) as tf:
        page = tf.series[0].levels[0].keyframe
        comp = getattr(page.compression, "name", str(page.compression))
        planar = getattr(page.planarconfig, "name", str(page.planarconfig))
        off = np.asarray(page.dataoffsets, dtype=np.int64)
        cnt = np.asarray(page.databytecounts, dtype=np.int64)
        layout = (f"tiles {page.tilewidth}x{page.tilelength}" if page.is_tiled
                  else f"strips of {page.rowsperstrip} rows")
        print(f"  raw  {raw}\n       {size / 2**30:.2f} GiB, {'BigTIFF' if tf.is_bigtiff else 'classic TIFF (32-bit offsets)'}, "
              f"compression {comp}, {planar}, {page.samplesperpixel} samples, {layout}, {len(off)} segments, "
              f"{len(tf.series[0].pages)} page(s) in series 0")
        print(f"       offsets: first {off[0]:,}, max {off.max():,}; decreases: {int((np.diff(off) < 0).sum())}; "
              f"data end (sum of segments) {off[0] + cnt.sum():,} vs file size {size:,}")
        try:
            fixed = unwrap(off, cnt, size, WRAP)
        except ValueError as e:
            fixed = None
            print(f"       cannot undo a wrap: {e}")
        wrapped_at = first_wrapped_row(page, off, fixed) if fixed is not None else None
        if fixed is not None:
            print(f"       {'no wrap' if wrapped_at is None else f'wrap undone: the first wrapped segment starts at row {wrapped_at}'}")
        with openslide.OpenSlide(conv) as sl:
            cw, ch = sl.dimensions
        print(f"  conv {conv}\n       {os.path.getsize(conv) / 2**30:.2f} GiB, {cw} x {ch} px")
        if comp != "NONE" or planar != "CONTIG" or page.samplesperpixel not in (3, 4):
            print("  raw is not uncompressed chunky RGB: this script reads only that layout (no panels)")
            return

        yb = wrapped_at if wrapped_at is not None else int(round(WRAP / (W * 3)))
        yb = min(max(yb, BAND_H // 2), H - BAND_H // 2)
        y0 = yb - BAND_H // 2
        try:                                                    # x where there are cells near the boundary
            cx, cy, _, _ = load_coords(row["coords_path"], cfg.geometry.centre_offset_px)
            near = np.abs(cy - yb) < BAND_H
            xc = float(np.median(cx[near])) if near.any() else W / 2
        except Exception:
            xc = W / 2
        x0 = int(np.clip(xc - BAND_W // 2, 0, max(0, W - BAND_W)))
        w, h = min(BAND_W, W - x0), BAND_H
        panels = []
        with openslide.OpenSlide(conv) as sl:
            a = np.asarray(sl.read_region((x0, y0), 0, (w, h)))[..., :3]
        panels.append(("1. converted copy (what bench reads)", a))
        with open(raw, "rb") as fh:
            panels.append(("2. raw, stored offsets (what every reader gets)", read_region(fh, page, off, x0, y0, w, h)[..., :3]))
            if fixed is not None and wrapped_at is not None:
                panels.append(("3. raw, wrap undone", read_region(fh, page, fixed, x0, y0, w, h)[..., :3]))
    f_big, f_small = _font(16), _font(13)
    pw, ph = w // DOWN, h // DOWN
    sheet = Image.new("RGB", (len(panels) * (pw + 12) + 10, ph + 110), "white")
    d = ImageDraw.Draw(sheet)
    d.text((10, 6), f"{slide}: rows {y0}-{y0 + h}, columns {x0}-{x0 + w}; yellow = row {yb} "
                    f"({'first wrapped segment' if wrapped_at is not None else 'predicted 4 GiB mark'})",
           fill="black", font=f_big)
    print(f"  band rows {y0}-{y0 + h}, columns {x0}-{x0 + w}; tissue colour above / below row {yb}:")
    for j, (title, a) in enumerate(panels):
        top, bot = tissue_rgb(a[: yb - y0]), tissue_rgb(a[yb - y0:])
        print(f"    {title:48s} above: {top:16s} below: {bot}")
        im = Image.fromarray(a.reshape(ph, DOWN, pw, DOWN, 3).mean(axis=(1, 3)).astype(np.uint8))
        ImageDraw.Draw(im).line([(0, (yb - y0) / DOWN), (pw, (yb - y0) / DOWN)], fill=(255, 215, 0), width=2)
        X = 10 + j * (pw + 12)
        d.text((X, 30), title, fill="black", font=f_small)
        d.text((X, 48), f"above {top}   below {bot}", fill=(90, 90, 90), font=f_small)
        sheet.paste(im, (X, 70))
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{slide}.png"
    sheet.save(png)
    print(f"  {png}")


def scan(ins) -> None:
    """Every WSI / raw original in the inspect table that is a classic TIFF over 4 GiB."""
    import tifffile
    rows = []
    for r in ins.itertuples():
        for kind, path in (("bench reads", r.wsi_path), ("raw original", getattr(r, "raw_path", "") or "")):
            if not isinstance(path, str) or not path or not os.path.isfile(path):
                continue
            size = os.path.getsize(path)
            try:
                with tifffile.TiffFile(path) as tf:
                    big = tf.is_bigtiff
                    off = np.asarray(tf.series[0].levels[0].keyframe.dataoffsets, dtype=np.int64)
                    dec = int((np.diff(off) < 0).sum()) if len(off) > 1 else 0
            except Exception as e:
                rows.append((r.slide, kind, size, f"unreadable header: {type(e).__name__}", ""))
                continue
            if not big and size > WRAP:
                rows.append((r.slide, kind, size, "CLASSIC TIFF > 4 GiB", f"{dec} offset decrease(s)"))
    print(f"{len(ins)} slides scanned (headers only)")
    for slide, kind, size, what, extra in rows:
        print(f"  {slide:45s} {kind:13s} {size / 2**30:6.2f} GiB  {what}  {extra}")
    if not rows:
        print("  no classic TIFF over 4 GiB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("slides", nargs="*", default=DEFAULT)
    ap.add_argument("--scan", action="store_true", help="only list classic TIFFs over 4 GiB among all slides' files")
    ap.add_argument("--config", help="bench YAML (default: $BENCH_CONFIG or configs/bench.yaml)")
    a = ap.parse_args(argv)
    from bench.config import load_config
    from bench.data.inspect import load_inspect
    cfg = load_config(a.config)
    ins = load_inspect(cfg)
    if a.scan:
        scan(ins)
        return 0
    for s in a.slides:
        if s not in ins.index:
            print(f"{s}: not in the inspect table")
            continue
        try:
            check(cfg, ins.loc[s].to_dict(), cfg.root / "qc" / "tiff_offsets")
        except Exception as e:
            print(f"  {s}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
