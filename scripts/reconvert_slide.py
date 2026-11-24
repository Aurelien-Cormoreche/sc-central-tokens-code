#!/usr/bin/env python3
"""Rebuild a slide's converted copy from its raw original when the copy is wrong
(scripts/check_converted_slide.py shows where it stops matching the raw file).

Same recipe as notebooks/check_wsi_openslide.ipynb (256 x 256 tiles, uncompressed, single level, OME BigTIFF,
the raw file's resolution tags), but streamed: the raw tiles are decoded a band of rows at a time (a few
hundred MB in memory, never the whole slide in one array), and a classic TIFF whose 32-bit offsets wrapped
past 4 GiB is read at the unwrapped offsets. Steps, each refusing to go on when something is off:
  1. raw: one chunky uint8 RGB(A) page of the copy's size; offsets that go back must be a plain 32-bit wrap
  2. the old copy must differ from the raw somewhere; otherwise re-converting changes nothing (if the slide
     looks wrong, the raw file itself is wrong and needs another source)
  3. write <converted>.partial.ome.tif
  4. verify with OpenSlide: equal to the raw pixels on random boxes and across the row where the old copy went
     wrong, and equal to the old copy above that row
  5. swap: the old copy becomes <name>.old (delete it once the slide looks right), the new one takes its name,
     so bench's paths do not change. Its content fingerprint changes too, so bench redoes the slide's
     dark-spot map, manifest row, embeddings and probe results.

  python scripts/reconvert_slide.py GSM9509150_NSCLC_lung_L4_L4
Run it on a compute node: reads the raw file, writes width x height x 3 bytes (8 GB for that slide); minutes.
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
import check_converted_slide as cc  # noqa: E402

TILE = 256
N_RANDOM = 64
OLD_TOL = 1.0           # mean |diff| allowed between the new and the old copy above the bad row (JPEG decoders
                        # of other versions may differ by a unit here and there)


def tiles(fh, page, offsets, W: int, H: int):
    """256 x 256 RGB tiles in row-major order (edge tiles zero-padded, as tifffile pads them), decoding the
    raw file one band of rows at a time (each raw tile once)."""
    if page.is_tiled:
        seg_h = int(page.tilelength)
    else:
        seg_h = 1 if int(page.compression) == 1 else min(int(page.rowsperstrip or H), H)
    band = min(4096, TILE * math.ceil(seg_h / TILE))
    for y0 in range(0, H, band):
        bh = min(band, H - y0)
        arr = cc.read_region(fh, page, offsets, 0, y0, W, bh)[..., :3]
        for sy in range(0, bh, TILE):
            h = min(TILE, bh - sy)
            for x0 in range(0, W, TILE):
                t = np.zeros((TILE, TILE, 3), np.uint8)
                blk = arr[sy:sy + h, x0:x0 + TILE]
                t[:h, :blk.shape[1]] = blk
                yield t


def reconvert(raw: str, conv: str, wrap: int = cc.WRAP, seed: int = 0) -> str:
    import openslide
    import tifffile
    from bench.utils import file_fingerprint

    t0 = time.time()
    backup = conv + ".old"
    if os.path.exists(backup):
        raise SystemExit(f"{backup} exists (an earlier run's backup): check it and delete it first")
    tmp = conv[: -len(".ome.tif")] + ".partial.ome.tif" if conv.endswith(".ome.tif") else conv + ".partial"
    fp_old = file_fingerprint(conv)
    with tifffile.TiffFile(raw) as tf, open(raw, "rb") as fh:
        L = cc.raw_layout(tf, os.path.getsize(raw), wrap)
        page = L["page"]
        W, H = int(page.imagewidth), int(page.imagelength)
        print(f"raw  {raw}\n     {L['desc']}")
        if not L["readable"]:
            raise SystemExit("the raw file is not a single chunky uint8 RGB(A) page: cannot rebuild from it")
        if L["bad_wrap"]:
            raise SystemExit(f"the raw offsets go back, but not as a plain 32-bit wrap: {L['bad_wrap']}")
        offsets = L["fixed"]                                    # the stored ones when nothing wraps
        if L["wrapped_at"] is not None:
            print(f"     offsets wrap past 4 GiB from row {L['wrapped_at']}: read unwrapped")

        def read_src(x, y, w, h):
            return cc.read_region(fh, page, offsets, x, y, w, h)[..., :3]

        with openslide.OpenSlide(conv) as old:
            if old.dimensions != (W, H):
                raise SystemExit(f"the copy is {old.dimensions}, the raw {W} x {H}: not the same image")
            boundary, prof = cc.find_boundary(lambda x, y, w, h: np.asarray(old.read_region((x, y), 0, (w, h)))[..., :3],
                                              read_src, W, H)
        print(f"old copy vs raw: {cc._profile_text(prof, boundary)}")
        if boundary is None:
            raise SystemExit("the old copy already equals the raw file wherever sampled: re-converting would change "
                             "nothing. If the slide looks wrong, the raw file itself is wrong")

        res = getattr(page, "resolution", None)
        unit = getattr(page, "resolutionunit", None)
        extra = {"resolution": res, "resolutionunit": unit} if res and all(res) and unit is not None else {}
        print(f"writing {tmp} ...")
        tifffile.imwrite(tmp, tiles(fh, page, offsets, W, H), shape=(H, W, 3), dtype=np.uint8, bigtiff=True,
                         tile=(TILE, TILE), photometric="rgb", planarconfig="contig", compression=None,
                         ome=True, metadata={"axes": "YXS"}, **extra)
        print(f"  {os.path.getsize(tmp) / 2**30:.2f} GiB, {time.time() - t0:.0f} s so far")

        rng = np.random.default_rng(seed)
        S = 512
        boxes = []
        for _ in range(N_RANDOM):
            x, y = int(rng.integers(0, max(1, W - S + 1))), int(rng.integers(0, max(1, H - S + 1)))
            boxes.append((x, y, min(S, W - x), min(S, H - y)))
        xs = cc._xs(W, 6, min(S, W))
        yc = int(min(max(0, boundary - S // 2), max(0, H - S)))      # across the bad row
        boxes += [(x, yc, min(S, W - x), min(S, H - yc)) for x in xs]
        ya = max(0, boundary - S)                                    # just above it
        if boundary > ya:
            boxes += [(x, ya, min(S, W - x), boundary - ya) for x in xs]
        n_old = 0
        with openslide.OpenSlide(tmp) as new, openslide.OpenSlide(conv) as old:
            if new.dimensions != (W, H):
                raise SystemExit(f"the new file is {new.dimensions}, expected {W} x {H}; {tmp} kept for a look")
            for x, y, w, h in boxes:
                got = np.asarray(new.read_region((x, y), 0, (w, h)))[..., :3]
                if not np.array_equal(got, read_src(x, y, w, h)):
                    raise SystemExit(f"verification failed at ({x}, {y}): the new file differs from the raw pixels; "
                                     f"{tmp} kept for a look")
                if y + h <= boundary:
                    prev = np.asarray(old.read_region((x, y), 0, (w, h)))[..., :3]
                    dif = float(cc._diff(prev, got).mean())
                    if dif > OLD_TOL:
                        raise SystemExit(f"verification failed at ({x}, {y}): above row {boundary} the new file differs "
                                         f"from the old copy (mean |diff| {dif:.2f}); {tmp} kept for a look")
                    n_old += 1
            y = int(min(max(0, boundary - S), max(0, H - 2 * S)))
            xm = int(min(max(0, W // 2 - S), max(0, W - 2 * S)))
            a = np.asarray(new.read_region((xm, y), 0, (min(2 * S, W), min(2 * S, H))))[..., :3]
            b = np.asarray(old.read_region((xm, y), 0, (min(2 * S, W), min(2 * S, H))))[..., :3]
        cut = boundary - y
        print(f"verified: {len(boxes)} boxes equal to the raw pixels, {n_old} of them (above row {boundary}) equal "
              f"to the old copy")
        print(f"  tissue colour around row {boundary}: old copy above {cc.tissue_rgb(b[:cut])} / below "
              f"{cc.tissue_rgb(b[cut:])};  new above {cc.tissue_rgb(a[:cut])} / below {cc.tissue_rgb(a[cut:])}")

    os.replace(conv, backup)
    os.replace(tmp, conv)
    fp_new = file_fingerprint(conv)
    print(f"swapped: {conv} is the new file; the old one is {backup} (delete it once the slide looks right)")
    print(f"content fingerprint {fp_old} -> {fp_new}  ({time.time() - t0:.0f} s)")
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
    reconvert(raw, conv)
    d = a.slide
    print(f"\nnext (the slide's old dark-spot map, manifest row and embeddings no longer count as done):\n"
          f"  bench darkspots --datasets {d} --local\n  bench manifest --datasets {d}\n"
          f"  bench overview --datasets {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
