"""`bench overview`: one PDF page per slide showing the whole slide, to look through every slide quickly.

Each page: the slide downscaled (left) and the same with every cell of patch_coordinates.h5 as a cyan dot
(right), read from the file bench extracts from (inspect table), plus per-slide numbers that point at
problems (red when above a threshold):
  outside image   cells whose centroid falls outside the image             (> 0.5%: coordinates / wrong file)
  on glass        cells on background (overview pixel min RGB >= 235)      (> 5%: registration off)
  on black        cells on black / no-data (overview pixel max RGB <= 20)  (> 0.5%: missing tiles)
  unreadable      share of the image in tiles that cannot be decoded (blue; > 0: corrupt tiles)
  black area      share of the image that is black / no-data
The overview is block-averaged from the best pyramid level; single-level files (the converted copies) are
read in full, in strips (minutes for the largest), so run it on a compute node with --workers.
Writes {bench_root}/qc/overview/overview_all.pdf, one PNG per slide and overview_stats.csv; with --datasets
overview_subset.pdf / overview_stats_subset.csv instead, so overview_all.pdf stays complete. Torch-free.
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

CELL_RGB = (0, 200, 255)
STRIP_BYTES = 128 * 2**20
LIMITS = {"outside_pct": 0.5, "on_glass_pct": 5.0, "on_black_pct": 0.5, "unreadable_pct": 0.0}
BLACK_MAX = 20                          # overview pixel max RGB <= this: black / no-data
PAGE_HEAD, PAGE_LEFT = 86, 10           # where the plain overview sits on a page
BAD_RGB = (30, 60, 220)                 # unreadable, as in the dark-spot review images
SUB_PX = 512                            # a failed strip is re-read in blocks of about this size


def thumb_geometry(sl, max_side: int):
    """(W, H, level, downsample, level width, level height, block) of the overview, from the header only."""
    W, H = sl.dimensions
    level = sl.get_best_level_for_downsample(max(1.0, max(W, H) / max_side))
    ds = float(sl.level_downsamples[level])
    lw, lh = sl.level_dimensions[level]
    g = max(1, math.ceil(max(lw, lh) / max_side))          # integer block size at that level
    return W, H, level, ds, lw, lh, g


def downscaled(path: str, max_side: int):
    """(H x W x 3 uint8 overview, level-0 (W, H), level-0 px -> overview px factor, level, block, unreadable
    mask). A strip that fails to decode is re-read in ~SUB_PX blocks after reopening the slide (OpenSlide
    stays in an error state after a failed read); blocks that still fail are painted BAD_RGB."""
    import openslide
    state = {"sl": openslide.OpenSlide(str(path))}

    def reopen():
        try:
            state["sl"].close()
        except Exception:
            pass
        state["sl"] = openslide.OpenSlide(str(path))

    try:
        W, H, level, ds, lw, lh, g = thumb_geometry(state["sl"], max_side)
        tw, th = lw // g, lh // g
        out = np.empty((th, tw, 3), np.uint8)
        bad = np.zeros((th, tw), bool)

        def fill(x0, y0, w, h):                                # level px, all multiples of g
            a = np.asarray(state["sl"].read_region((int(round(x0 * ds)), int(round(y0 * ds))), level, (w, h)))
            rgb = a[..., :3].copy()
            rgb[a[..., 3] == 0] = 255                          # transparent = not scanned: white, as extraction
            blk = rgb.reshape(h // g, g, w // g, g, 3).sum(axis=(1, 3), dtype=np.uint32)
            out[y0 // g: (y0 + h) // g, x0 // g: (x0 + w) // g] = (blk // (g * g)).astype(np.uint8)

        rows = max(g, (STRIP_BYTES // max(lw * 4, 1)) // g * g)
        sb = max(g, (SUB_PX // g) * g)
        for y0 in range(0, th * g, rows):
            h = min(rows, th * g - y0)
            try:
                fill(0, y0, tw * g, h)
                continue
            except openslide.OpenSlideError:
                reopen()
            for yy in range(y0, y0 + h, sb):
                hh = min(sb, y0 + h - yy)
                for xx in range(0, tw * g, sb):
                    ww = min(sb, tw * g - xx)
                    try:
                        fill(xx, yy, ww, hh)
                    except openslide.OpenSlideError:
                        reopen()
                        out[yy // g: (yy + hh) // g, xx // g: (xx + ww) // g] = BAD_RGB
                        bad[yy // g: (yy + hh) // g, xx // g: (xx + ww) // g] = True
        return out, (W, H), 1.0 / (ds * g), level, g, bad
    finally:
        state["sl"].close()


def slide_page(cfg, row: dict, max_side: int):
    """(PIL page, stats dict) for one slide."""
    from PIL import Image, ImageDraw
    from bench.data.crop_sheet import _font, _num
    from bench.data.darkspots import load_coords

    thumb, (W, H), k, level, g, bad = downscaled(row["wsi_path"], max_side)
    th, tw = thumb.shape[:2]
    stats = {"slide": row["slide"], "cohort": row.get("cohort", ""), "width": W, "height": H}
    right = thumb.copy()
    try:
        cx, cy, _, _ = load_coords(row["coords_path"], cfg.geometry.centre_offset_px)
    except Exception as e:
        cx = cy = None
        stats["error"] = f"no coordinates ({type(e).__name__})"
    if cx is not None:
        n = len(cx)
        inside = (cx >= 0) & (cy >= 0) & (cx < W) & (cy < H)
        tx, ty = np.floor(cx * k).astype(np.int64), np.floor(cy * k).astype(np.int64)
        ok = inside & (tx >= 0) & (ty >= 0) & (tx < tw) & (ty < th)
        px = thumb[ty[ok], tx[ok]].astype(np.int16)
        stats.update(n_cells=n, outside_pct=100.0 * (1 - inside.mean()) if n else 0.0,
                     on_glass_pct=100.0 * float((px.min(1) >= 235).mean()) if len(px) else 0.0,
                     on_black_pct=100.0 * float((px.max(1) <= BLACK_MAX).mean()) if len(px) else 0.0)
        right[ty[ok], tx[ok]] = CELL_RGB
    stats["unreadable_pct"] = 100.0 * float(bad.mean())
    stats["black_area_pct"] = 100.0 * float((thumb.max(2) <= BLACK_MAX).mean())

    gap, head, foot = 16, PAGE_HEAD, 22
    page = Image.new("RGB", (2 * tw + gap + 20, th + head + foot), "white")
    d = ImageDraw.Draw(page)
    f_big, f_small = _font(18), _font(13)
    mpp = _num(row.get("mpp"))
    mpp_txt = f"{mpp:.4f} um/px ({row.get('mpp_source')})" if mpp else "pixel size unknown"
    d.text((10, 6), f"{row['slide']}   [{stats['cohort']}]", fill="black", font=f_big)
    d.text((10, 32), f"{W} x {H} px   {mpp_txt}   converted={row.get('converted')}   "
                     f"overview: level {level}, 1 px = {1 / k:.0f} slide px", fill="black", font=f_small)
    x = 10
    items = [("cells", f"{stats.get('n_cells', 0):,}", False)]
    for key, label in (("outside_pct", "outside image"), ("on_glass_pct", "on glass"), ("on_black_pct", "on black"),
                       ("unreadable_pct", "unreadable (blue)")):
        if key in stats:
            items.append((label, f"{stats[key]:.2f}%", stats[key] > LIMITS[key]))
    items.append(("black area", f"{stats['black_area_pct']:.2f}%", False))
    if "error" in stats:
        items.append(("", stats["error"], True))
    for label, val, bad in items:
        txt = f"{label} {val}".strip()
        d.text((x, 54), txt, fill=(210, 0, 0) if bad else "black", font=f_small)
        x += int(d.textlength(txt, font=f_small)) + 28
    page.paste(Image.fromarray(thumb), (PAGE_LEFT, head))
    page.paste(Image.fromarray(right), (10 + tw + gap, head))
    d.text((10, head + th + 4), "image", fill=(90, 90, 90), font=f_small)
    d.text((10 + tw + gap, head + th + 4), "image + cells (cyan)", fill=(90, 90, 90), font=f_small)
    return page, stats


def _job(args):
    cfg, row, max_side = args
    try:
        page, stats = slide_page(cfg, row, max_side)
        return row["slide"], page, stats, None
    except Exception as e:
        return row["slide"], None, None, f"{type(e).__name__}: {e}"


def run_overview(cfg, slides, workers: int = 4, max_side: int = 2000, name: str = "all") -> Optional[Path]:
    """name: the PDF is overview_{name}.pdf (a subset run must not replace overview_all.pdf)."""
    import pandas as pd
    from bench.data.inspect import load_inspect
    from bench.utils import LOG, print_table
    ins = load_inspect(cfg)
    out = cfg.root / "qc" / "overview"
    out.mkdir(parents=True, exist_ok=True)
    tasks, problems = [], []
    for s in slides:
        if s.slide not in ins.index:
            problems.append(f"{s.slide}: not in the inspect table")
            continue
        row = ins.loc[s.slide].to_dict()
        if not row.get("wsi_path") or not Path(str(row["wsi_path"])).is_file():
            problems.append(f"{s.slide}: no WSI file ({row.get('status')})")
            continue
        tasks.append((cfg, row, max_side))
    pdf = out / f"overview_{name}.pdf"
    tmp = pdf.with_name(pdf.name + ".tmp")
    tmp.unlink(missing_ok=True)
    rows, n_pages = [], 0
    ex = ProcessPoolExecutor(max_workers=workers) if workers > 1 and len(tasks) > 1 else None
    try:
        results = ex.map(_job, tasks) if ex else map(_job, tasks)
        for slide, page, stats, err in results:               # slide order kept
            if err:
                problems.append(f"{slide}: {err}")
                continue
            page.save(out / f"{slide}.png")
            page.save(tmp, "PDF", resolution=110, append=n_pages > 0)
            n_pages += 1
            rows.append(stats)
            LOG.info("%s: done (%d/%d)", slide, n_pages, len(tasks))
    finally:
        if ex:
            ex.shutdown()
    if n_pages:
        tmp.replace(pdf)
        df = pd.DataFrame(rows)
        df.to_csv(out / ("overview_stats.csv" if name == "all" else f"overview_stats_{name}.csv"), index=False)
        view = []
        for r in rows:
            flags = [k.replace("_pct", "") for k, lim in LIMITS.items() if r.get(k, 0) > lim]
            view.append({"slide": r["slide"], "cohort": r["cohort"], "cells": r.get("n_cells", ""),
                         **{k.replace("_pct", "%"): f"{r[k]:.2f}" for k in (*LIMITS, "black_area_pct") if k in r},
                         "check": " ".join(flags) + (f" {r['error']}" if "error" in r else "")})
        print_table(view)
        print(f"\n{n_pages} slides: {pdf}")
    for p in problems:
        print(f"  SKIPPED {p}")
    return pdf if n_pages else None
