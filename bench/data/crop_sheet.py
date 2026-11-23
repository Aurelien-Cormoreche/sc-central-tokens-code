"""`bench crops`: a visual check of the crops the models will see. For every slide, a few cells are cut at
every model input size with the extraction code itself (same centroid, scale and crop origin as `bench
extract`, read with WSIReader), and drawn side by side:

  red cross    the cell centroid (patch_coordinates x_start/y_start + centre_offset_px)
  yellow box   the next smaller input size, inside each larger crop (the crops are nested)
  scale bar    from the slide's pixel size (bench inspect); absent when it is unknown
  header       slide, cohort, pixel size and its source, scale, size; column = input px, models, FOV in um

One PNG per slide plus one PDF with all of them: {bench_root}/qc/crops/. Needs the inspect table only (not the
manifest); cells are drawn from patch_coordinates, far enough from the edge for the largest crop. Torch-free.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

TILE = 256           # display px per crop
HEAD = 56            # page header height
LABEL_W = 210        # row label width
COL_HEAD = 34


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                           # Pillow < 10.1: fixed-size bitmap font
        return ImageFont.load_default()


def _num(v) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _sizes(cfg, models: Optional[List[str]]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for m in cfg.enabled_models(models):
        out.setdefault(int(cfg.models[m].input_px), []).append(m)
    return dict(sorted(out.items()))


def pick_cells(cx: np.ndarray, cy: np.ndarray, types: np.ndarray, width_ext: int, height_ext: int, scale: float,
               margin_ext: float, k: int, seed: int) -> np.ndarray:
    """k cells whose largest crop lies inside the image, of different cell types where possible."""
    ex, ey = cx * scale, cy * scale
    ok = np.flatnonzero((ex >= margin_ext) & (ey >= margin_ext) & (ex < width_ext - margin_ext)
                        & (ey < height_ext - margin_ext))
    if len(ok) == 0:
        return ok
    rng = np.random.default_rng(seed)
    order = rng.permutation(ok)
    chosen, seen = [], set()
    for i in order:                                 # first pass: one cell per type
        if types[i] not in seen:
            chosen.append(i)
            seen.add(types[i])
        if len(chosen) == k:
            break
    for i in order:                                 # fill up if there are fewer types than k
        if len(chosen) == k:
            break
        if i not in chosen:
            chosen.append(i)
    return np.array(chosen, dtype=np.int64)


def _bar_um(fov_um: float) -> float:
    for v in (100, 50, 20, 10, 5, 2, 1, 0.5):
        if v <= 0.4 * fov_um:
            return v
    return 0.5


def slide_sheet(cfg, row: dict, sizes: Dict[int, List[str]], k: int, seed: int):
    """(PIL image, list of chosen cell ids) for one slide."""
    from PIL import Image, ImageDraw
    from bench.data.darkspots import load_coords
    from bench.data.reader import WSIReader, scale_for
    from bench.data.regions import crop_origins

    mpp = _num(row.get("mpp"))
    scale = _num(row.get("scale")) or scale_for(mpp, cfg.resolution.resample_below_mpp, cfg.resolution.target_mpp)
    cx, cy, ids, types = load_coords(row["coords_path"], cfg.geometry.centre_offset_px)
    types = np.array([t.decode() if isinstance(t, bytes) else str(t) for t in types], dtype=object)
    rd = WSIReader(row["wsi_path"], scale=scale)
    try:
        largest = max(sizes)
        sel = pick_cells(cx, cy, types, rd.ext_width, rd.ext_height, scale, largest / 2 + 2, k, seed)
        ncol = len(sizes)
        page = Image.new("RGB", (LABEL_W + ncol * TILE, HEAD + COL_HEAD + max(len(sel), 1) * TILE), "white")
        d = ImageDraw.Draw(page)
        f_big, f_small = _font(16), _font(12)
        src = row.get("mpp_source") or "none"
        mpp_txt = f"{mpp:.4f} um/px ({src})" if mpp else "pixel size UNKNOWN (no scale bar)"
        d.text((8, 6), f"{row['slide']}   [{row.get('cohort', '')}]", fill="black", font=f_big)
        d.text((8, 30), f"{mpp_txt}   scale {scale:g}   {rd.width}x{rd.height} px   "
                        f"converted={row.get('converted')}   inspect: {row.get('status', '')}",
               fill="black", font=f_small)
        for j, (n, models) in enumerate(sizes.items()):
            fov = f"{n * mpp / scale:.1f} um" if mpp else "? um"
            d.text((LABEL_W + j * TILE + 6, HEAD + 2), f"{n} px = {fov}", fill="black", font=f_small)
            d.text((LABEL_W + j * TILE + 6, HEAD + 17), ", ".join(models)[:40], fill=(90, 90, 90), font=f_small)
        if len(sel) == 0:
            d.text((8, HEAD + COL_HEAD + 8), f"no cell at least {largest // 2} px from the image edge",
                   fill=(200, 0, 0), font=f_big)
        size_list = list(sizes)
        for r, i in enumerate(sel):
            y0 = HEAD + COL_HEAD + r * TILE
            cid = ids[i].decode() if isinstance(ids[i], bytes) else str(ids[i])
            d.text((8, y0 + 8), cid[-28:], fill="black", font=f_small)
            d.text((8, y0 + 26), types[i][:28], fill="black", font=f_small)
            d.text((8, y0 + 44), f"x={cx[i]:.0f} y={cy[i]:.0f} px", fill=(90, 90, 90), font=f_small)
            for j, n in enumerate(size_list):
                ex, ey = cx[i] * scale, cy[i] * scale
                ox, oy = crop_origins(np.array([ex]), np.array([ey]), n)
                crop = rd.read_ext(int(ox[0]), int(oy[0]), n, n)
                z = TILE / n
                im = Image.fromarray(crop).resize((TILE, TILE), Image.NEAREST if n < TILE else Image.BOX)
                t = ImageDraw.Draw(im)
                px, py = (ex - ox[0]) * z, (ey - oy[0]) * z          # centroid in display px
                if j > 0:                                            # next smaller crop, nested
                    m = size_list[j - 1]
                    sx, sy = crop_origins(np.array([ex]), np.array([ey]), m)
                    bx, by = (sx[0] - ox[0]) * z, (sy[0] - oy[0]) * z
                    t.rectangle([bx, by, bx + m * z - 1, by + m * z - 1], outline=(255, 210, 0), width=2)
                t.line([px - 8, py, px + 8, py], fill=(255, 0, 0), width=2)
                t.line([px, py - 8, px, py + 8], fill=(255, 0, 0), width=2)
                if mpp:
                    fov_um = n * mpp / scale
                    bar = _bar_um(fov_um)
                    L = bar / fov_um * TILE
                    t.rectangle([8, TILE - 16, 8 + L, TILE - 10], fill="black", outline="white")
                    t.text((8, TILE - 32), f"{bar:g} um", fill="black", font=f_small,
                           stroke_width=2, stroke_fill="white")
                page.paste(im, (LABEL_W + j * TILE, y0))
                d.rectangle([LABEL_W + j * TILE, y0, LABEL_W + (j + 1) * TILE - 1, y0 + TILE - 1], outline=(180, 180, 180))
        chosen = [ids[i].decode() if isinstance(ids[i], bytes) else str(ids[i]) for i in sel]
    finally:
        rd.close()
    return page, chosen


def run_crop_sheets(cfg, slides, models: Optional[List[str]] = None, cells: int = 3, seed: int = 0) -> Path:
    from bench.data.inspect import load_inspect
    from bench.utils import LOG
    ins = load_inspect(cfg)
    sizes = _sizes(cfg, models)
    if any(n % 2 for n in sizes):
        raise SystemExit(f"input sizes must be even: {list(sizes)}")
    out = cfg.root / "qc" / "crops"
    out.mkdir(parents=True, exist_ok=True)
    pages, problems = [], []
    for s in slides:
        if s.slide not in ins.index:
            problems.append(f"{s.slide}: not in the inspect table")
            continue
        row = {k: v for k, v in ins.loc[s.slide].to_dict().items()}
        if not row.get("wsi_path") or not Path(str(row["wsi_path"])).is_file():
            problems.append(f"{s.slide}: no WSI file ({row.get('status')})")
            continue
        try:
            page, chosen = slide_sheet(cfg, row, sizes, cells, seed)
        except Exception as e:
            problems.append(f"{s.slide}: {type(e).__name__}: {e}")
            continue
        page.save(out / f"{s.slide}.png")
        pages.append(page)
        LOG.info("%s: cells %s", s.slide, ", ".join(chosen))
    if pages:
        pdf = out / "crops_all.pdf"
        tmp = pdf.with_name(pdf.name + ".tmp")
        from bench.data.overview import PDF_JPEG
        pages[0].save(tmp, "PDF", save_all=True, append_images=pages[1:], resolution=96, **PDF_JPEG)
        tmp.replace(pdf)
        print(f"{len(pages)} slide sheets: {out}/<slide>.png and {pdf}")
    for p in problems:
        print(f"  SKIPPED {p}")
    return out
