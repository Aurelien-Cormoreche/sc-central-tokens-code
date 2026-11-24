#!/usr/bin/env python3
"""Compare a converted slide (the copy bench reads) with its raw original: where do they stop matching, and is
the raw itself right there?

GSM9509150's converted copy turns teal below some row, with the rows shifted sideways: from there its bytes
are read one position off (R,G,B -> G,B,R). Either step can be at fault:
  - the conversion (notebooks/check_wsi_openslide.ipynb, or the older extract_embs_parallel.py: tifffile reads
    the whole slide, then writes it as an uncompressed BigTIFF). Then the raw is fine below the line and
    re-converting it fixes the copy: scripts/reconvert_slide.py.
  - the raw file itself: its pixels are already wrong there. If it is a classic TIFF over 4 GiB whose 32-bit
    offsets wrapped, the data is still in the file and reconvert_slide.py reads it with the wrap undone;
    otherwise only another source (the original scan) can fix it.
For each slide (default: the two SPLIT slides with > 4 GiB of pixels, and the largest one below as a control):
  1. the raw file's layout (compression, tiles, BigTIFF / classic, offset wraps) and the other raw files of the
     same sample
  2. a scan down the slide: every 512 rows, boxes across the width read from both files (raw tiles decoded with
     tifffile, as the conversion did), then a row-by-row look where they start to differ
  3. {bench_root}/qc/conversion/<slide>.png: a band around that row read as (1) the converted copy, (2) the raw
     file, (3) the raw file with an offset wrap undone (only when its offsets wrap), with the mean tissue
     colour above / below the line. Panel 2 purple below the line: the raw is fine, re-convert.
Reads the headers and some tens of MB per slide; seconds to a minute (login node is fine).

  python scripts/check_converted_slide.py
  python scripts/check_converted_slide.py GSM9509150_NSCLC_lung_L4_L4
  python scripts/check_converted_slide.py --scan    # headers only: slide files that are classic TIFFs > 4 GiB
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

WRAP = 2 ** 32
DEFAULT = ["GSM9509144_NSCLC_chuvio_L4_L4", "GSM9509150_NSCLC_lung_L4_L4", "GSM9509143_NSCLC_chuvio_L3_L3"]
BAND_H, BAND_W, DOWN = 1536, 3072, 3
DIFF_BAD = 8.0          # mean |difference| (0-255) above which two reads of the same box differ (JPEG decoders
                        # of different versions stay well below)
SCAN_STEP, SCAN_BOX, SCAN_NX = 512, 256, 8


# ── reading the raw file segment by segment, at chosen offsets ──

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


def _segment(fh, page, offsets, k: int, sh: int, sw: int, spp: int) -> np.ndarray:
    """Segment k (tile or strip) as (rows, sw, spp) uint8: raw bytes when uncompressed, else decoded."""
    fh.seek(int(offsets[k]))
    data = fh.read(int(page.databytecounts[k]))
    if int(page.compression) == 1:
        a = np.frombuffer(data, np.uint8)
        n = sh * sw * spp
        return np.pad(a[:n], (0, max(0, n - a.size))).reshape(sh, sw, spp)
    seg = page.decode(data, k, jpegtables=getattr(page, "jpegtables", None))[0]
    if seg is None:
        raise ValueError(f"segment {k} could not be decoded")
    seg = np.asarray(seg)
    return seg.reshape(seg.shape[-3:])


def read_region(fh, page, offsets, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """(h, w, spp) uint8 of a chunky (contig) page, reading its segments at the given offsets."""
    W, H, spp = int(page.imagewidth), int(page.imagelength), int(page.samplesperpixel)
    out = np.zeros((h, w, spp), np.uint8)
    if page.is_tiled:
        th, tw = int(page.tilelength), int(page.tilewidth)
        ntx = math.ceil(W / tw)
        for ty in range(y0 // th, (y0 + h - 1) // th + 1):
            for tx in range(x0 // tw, (x0 + w - 1) // tw + 1):
                tile = _segment(fh, page, offsets, ty * ntx + tx, th, tw, spp)
                ya, yb = max(y0, ty * th), min(y0 + h, (ty + 1) * th)
                xa, xb = max(x0, tx * tw), min(x0 + w, (tx + 1) * tw)
                out[ya - y0:yb - y0, xa - x0:xb - x0] = tile[ya - ty * th:yb - ty * th, xa - tx * tw:xb - tx * tw]
        return out
    rps = min(int(page.rowsperstrip or H), H)
    for k in range(y0 // rps, (y0 + h - 1) // rps + 1):
        ya, yb = max(y0, k * rps), min(y0 + h, (k + 1) * rps, H)
        if int(page.compression) == 1:                         # only the bytes needed
            for y in range(ya, yb):
                fh.seek(int(offsets[k]) + ((y - k * rps) * W + x0) * spp)
                row = np.frombuffer(fh.read(w * spp), np.uint8)
                n = row.size // spp
                out[y - y0, :n] = row[: n * spp].reshape(-1, spp)
            continue
        strip = _segment(fh, page, offsets, k, min(rps, H - k * rps), W, spp)
        out[ya - y0:yb - y0] = strip[ya - k * rps:yb - k * rps, x0:x0 + w]
    return out


def first_wrapped_row(page, raw_off, fixed_off) -> int | None:
    k = np.flatnonzero(raw_off != fixed_off)
    if not len(k):
        return None
    k = int(k[0])
    if page.is_tiled:
        ntx = math.ceil(int(page.imagewidth) / int(page.tilewidth))
        return (k // ntx) * int(page.tilelength)
    return k * min(int(page.rowsperstrip or page.imagelength), int(page.imagelength))


def raw_layout(tf, size: int, wrap: int = WRAP) -> dict:
    """Level 0 of series 0: page, stored offsets, offsets to read with (wrap undone when the offsets wrap),
    first wrapped row, and a one-line description."""
    page = tf.series[0].levels[0].keyframe
    off = np.asarray(page.dataoffsets, dtype=np.int64)
    cnt = np.asarray(page.databytecounts, dtype=np.int64)
    try:
        fixed = unwrap(off, cnt, size, wrap)
        bad_wrap = ""
    except ValueError as e:
        fixed, bad_wrap = None, str(e)
    wrapped_at = first_wrapped_row(page, off, fixed) if fixed is not None else None
    comp = getattr(page.compression, "name", str(page.compression))
    planar = getattr(page.planarconfig, "name", str(page.planarconfig))
    layout = (f"tiles {page.tilewidth}x{page.tilelength}" if page.is_tiled
              else f"strips of {min(int(page.rowsperstrip or page.imagelength), int(page.imagelength))} rows")
    desc = (f"{size / 2**30:.2f} GiB, {'BigTIFF' if tf.is_bigtiff else 'classic TIFF (32-bit offsets)'}, "
            f"{page.imagewidth} x {page.imagelength}, compression {comp}, {planar}, {page.samplesperpixel} samples, "
            f"{page.dtype}, {layout}, {len(off)} segments, {len(tf.series[0].pages)} page(s) in series 0; "
            f"offsets decrease {int((np.diff(off) < 0).sum())} time(s)")
    return {"page": page, "stored": off, "fixed": fixed, "wrapped_at": wrapped_at, "bad_wrap": bad_wrap,
            "desc": desc, "readable": planar == "CONTIG" and int(page.samplesperpixel) in (3, 4)
            and str(page.dtype) == "uint8" and len(tf.series[0].pages) == 1}


# ── comparing two reads of the same slide ──

def _xs(W: int, nx: int, bw: int):
    return sorted({int(min(max(0, (i + 0.5) * W / nx - bw / 2), max(0, W - bw))) for i in range(nx)})


def _diff(a, b) -> np.ndarray:
    return np.abs(a[..., :3].astype(np.int16) - b[..., :3].astype(np.int16))


def find_boundary(read_a, read_b, W: int, H: int, step: int = SCAN_STEP, box: int = SCAN_BOX,
                  nx: int = SCAN_NX):
    """(first row where the two reads differ or None, coarse profile [(y, worst mean |diff|)]).
    read_a / read_b: (x, y, w, h) -> (h, w, >=3) uint8."""
    bw, bh = min(box, W), min(box, H)
    xs = _xs(W, nx, bw)
    ys = list(range(0, H - bh + 1, step))
    if ys[-1] != H - bh:                                    # down to the last rows
        ys.append(H - bh)
    prof = []
    for y in ys:
        prof.append((y, max(float(_diff(read_a(x, y, bw, bh), read_b(x, y, bw, bh)).mean()) for x in xs)))
    bad = [y for y, d in prof if d > DIFF_BAD]
    if not bad:
        return None, prof
    y0 = max(0, bad[0] - step)
    h = min(H, bad[0] + bh) - y0
    rows = np.zeros(h)
    for x in xs:                                            # row by row, worst over the x positions
        rows = np.maximum(rows, _diff(read_a(x, y0, bw, h), read_b(x, y0, bw, h)).mean(axis=(1, 2)))
    k = np.flatnonzero(rows > DIFF_BAD)
    return (y0 + int(k[0]) if len(k) else bad[0]), prof


def tissue_rgb(a: np.ndarray) -> str:
    rgb = a[..., :3].reshape(-1, 3).astype(int)
    t = rgb[(rgb.min(1) < 200) & (rgb.max(1) > 30)]
    return "no tissue" if len(t) < 100 else "RGB " + " ".join(f"{v:.0f}" for v in t.mean(0))


def _profile_text(prof, boundary) -> str:
    if boundary is None:
        return f"the copy matches the raw file at all {len(prof)} sampled rows (worst mean |diff| " \
               f"{max(d for _, d in prof):.1f})"
    above = [d for y, d in prof if y + SCAN_BOX <= boundary]
    below = [d for y, d in prof if y >= boundary]
    return (f"the copy matches the raw file above row {boundary} (worst mean |diff| "
            f"{max(above) if above else 0:.1f} over {len(above)} sampled rows) and differs from there "
            f"({sum(d > DIFF_BAD for d in below)} of {len(below)} sampled rows below, up to {max(below or [0]):.0f})")


# ── the check ──

def check(cfg, row: dict, out_dir) -> None:
    import openslide
    import tifffile
    from PIL import Image, ImageDraw
    from bench.data.crop_sheet import _font
    from bench.data.darkspots import load_coords

    slide, raw, conv = row["slide"], str(row.get("raw_path") or ""), row["wsi_path"]
    W, H = int(row["width"]), int(row["height"])
    print(f"\n== {slide}: {W} x {H} px, {W * H * 3 / 2**30:.2f} GiB of RGB (the 4 GiB mark of uncompressed RGB "
          f"is at row {WRAP / (W * 3):.0f}, 2 GiB at row {WRAP / 2 / (W * 3):.0f})")
    if not raw or not os.path.isfile(raw):
        print(f"  no raw original found (inspect raw_path = {raw!r})")
        return
    gsm = slide.split("_")[0]
    sib = sorted(p for p in Path(raw).parent.glob(f"{gsm}*") if p.is_file())
    print(f"  raw  {raw}")
    print("       other files of this sample there: " + (", ".join(f"{p.name} ({p.stat().st_size / 2**30:.2f} GiB)"
                                                             for p in sib if str(p) != raw) or "none"))
    print(f"  conv {conv} ({os.path.getsize(conv) / 2**30:.2f} GiB)")
    if not row.get("converted"):
        print("  bench reads the raw file itself: nothing to compare")
        return
    size = os.path.getsize(raw)
    with tifffile.TiffFile(raw) as tf, open(raw, "rb") as fh, openslide.OpenSlide(conv) as sl:
        L = raw_layout(tf, size)
        page = L["page"]
        print(f"       raw: {L['desc']}")
        if L["bad_wrap"]:
            print(f"       offsets go back but not as a plain 32-bit wrap: {L['bad_wrap']}")
        elif L["wrapped_at"] is not None:
            print(f"       offsets wrap past 4 GiB: the first wrapped segment starts at row {L['wrapped_at']}")
        if (int(page.imagewidth), int(page.imagelength)) != sl.dimensions:
            print(f"  raw is {page.imagewidth} x {page.imagelength}, the copy {sl.dimensions}: not the same image")
            return
        if not L["readable"]:
            print("  raw is not a single chunky uint8 RGB(A) page: this script cannot read it")
            return

        def read_conv(x, y, w, h):
            return np.asarray(sl.read_region((x, y), 0, (w, h)))[..., :3]

        def read_raw(x, y, w, h):
            return read_region(fh, page, L["stored"], x, y, w, h)[..., :3]

        boundary, prof = find_boundary(read_conv, read_raw, W, H)
        print(f"  scan: {_profile_text(prof, boundary)}")
        yb = boundary if boundary is not None else (L["wrapped_at"] or int(round(WRAP / (W * 3))))
        yb = min(max(yb, 1), H - 1)
        y0 = int(min(max(0, yb - BAND_H // 2), max(0, H - BAND_H)))
        h = min(BAND_H, H - y0)
        try:                                                    # x where there are cells near the line
            cx, cy, _, _ = load_coords(row["coords_path"], cfg.geometry.centre_offset_px)
            near = np.abs(cy - yb) < BAND_H
            xc = float(np.median(cx[near])) if near.any() else W / 2
        except Exception:
            xc = W / 2
        x0 = int(np.clip(xc - BAND_W // 2, 0, max(0, W - BAND_W)))
        w = min(BAND_W, W - x0)
        panels = [("1. converted copy (what bench reads)", read_conv(x0, y0, w, h)),
                  ("2. raw file (tiles decoded as the conversion did)", read_raw(x0, y0, w, h))]
        if L["fixed"] is not None and L["wrapped_at"] is not None:
            panels.append(("3. raw file, offset wrap undone",
                           read_region(fh, page, L["fixed"], x0, y0, w, h)[..., :3]))
    what = "copy and raw start to differ" if boundary is not None else (
        "first wrapped segment" if L["wrapped_at"] is not None else "4 GiB mark")
    f_big, f_small = _font(16), _font(13)
    pw, ph = w // DOWN, h // DOWN
    sheet = Image.new("RGB", (len(panels) * (pw + 12) + 10, ph + 110), "white")
    d = ImageDraw.Draw(sheet)
    d.text((10, 6), f"{slide}: rows {y0}-{y0 + h}, columns {x0}-{x0 + w}; yellow = row {yb} ({what})",
           fill="black", font=f_big)
    print(f"  band rows {y0}-{y0 + h}, columns {x0}-{x0 + w}; tissue colour above / below row {yb} ({what}):")
    cols = {}
    for j, (title, a) in enumerate(panels):
        top, bot = tissue_rgb(a[: yb - y0]), tissue_rgb(a[yb - y0:])
        cols[j] = (top, bot)
        print(f"    {title:50s} above: {top:16s} below: {bot}")
        a = a[: ph * DOWN, : pw * DOWN]
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
    if boundary is None:
        print("  -> the converted copy is a faithful copy of the raw file"
              + (" (but the raw's offsets wrap: see panel 3; scripts/reconvert_slide.py reads them unwrapped)"
                 if L["wrapped_at"] is not None else "; if it looks wrong, the raw file itself is wrong"))
    else:
        print(f"  -> the conversion changed the image from row {boundary}: if panel 2 looks right below the line, "
              f"`python scripts/reconvert_slide.py {slide}` rebuilds the copy from the raw file")


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
            check(cfg, ins.loc[s].to_dict(), cfg.root / "qc" / "conversion")
        except Exception as e:
            print(f"  {s}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
