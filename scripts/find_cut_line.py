#!/usr/bin/env python3
"""Find a horizontal cut line on a slide -- the first y, from the top, where cells start lying on a black
background -- and draw it for a visual check (keep what is above).

A cell is on black when its pixel in the whole-slide overview has max RGB <= 20: the same test as `bench
overview` (each overview pixel is the mean of a block of ~38 x 38 slide px for a 75k-px-wide slide). Cells are
binned in horizontal bands of one overview row. The line is the top of the first band in which at least
--frac of the band's cells are on black, and the same holds for the --window slide px below it taken together
(at least --min-cells cells on black there), so an isolated black hole in the upper image does not trigger it.

Writes {bench_root}/qc/cut/<slide>_cutline.png:
  top     the overview with every cell (cyan), cells on black (red) and the line (yellow), and next to it the
          share of cells on black per band (the dashed line is --frac)
  bottom  four close-ups (2048 x 2048 slide px at 1/4 resolution) centred on the line, where cells are
Prints the line and the cells above / below it. Light enough for the login node: it reuses the overview page
that `bench overview` wrote (without one it reads the whole slide: minutes, use a compute node), the cell
coordinates and four small regions.

  python scripts/find_cut_line.py GSM9509144_NSCLC_chuvio_L4_L4
  python scripts/find_cut_line.py GSM9509144_NSCLC_chuvio_L4_L4 --y 21500     # draw a line of your own
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

import numpy as np

ZOOM_PX, ZOOM_DOWN = 2048, 4
PROFILE_W = 300
CYAN, RED, YELLOW = (0, 200, 255), (255, 40, 40), (255, 215, 0)


def find_line(cy: np.ndarray, onb: np.ndarray, band: float, frac: float = 0.03, min_cells: int = 20,
              window: float = 1000.0):
    """(line y in slide px or None, cells per band, cells on black per band). Band i covers
    [i * band, (i + 1) * band)."""
    nb = int(max(cy.max(initial=0.0), 0) // band) + 1
    b = np.clip((np.maximum(cy, 0) // band).astype(np.int64), 0, nb - 1)
    n = np.bincount(b, minlength=nb).astype(np.int64)
    k = np.bincount(b, weights=onb.astype(np.float64), minlength=nb).round().astype(np.int64)
    w = max(1, int(round(window / band)))
    cn, ck = np.concatenate([[0], np.cumsum(n)]), np.concatenate([[0], np.cumsum(k)])
    for i in range(nb):
        if k[i] == 0 or k[i] < frac * n[i]:
            continue
        j = min(nb, i + w)
        kw, nw = ck[j] - ck[i], cn[j] - cn[i]
        if kw >= min_cells and kw >= frac * nw:
            return i * band, n, k
    return None, n, k


def read_rgb(sl, x: int, y: int, w: int, h: int) -> np.ndarray:
    a = np.asarray(sl.read_region((int(x), int(y)), 0, (int(w), int(h))))
    rgb = a[..., :3].copy()
    rgb[a[..., 3] == 0] = 255
    return rgb


def zoom_xs(cx: np.ndarray, cy: np.ndarray, y: float, width: int, n: int = 4) -> list:
    near = np.abs(cy - y) < ZOOM_PX / 2
    qs = (np.arange(n) + 0.5) / n
    xs = np.quantile(cx[near], qs) if near.sum() >= n else qs * width
    return [int(v) for v in xs]


def cut_sheet(slide: str, sl, thumb: np.ndarray, kf: float, cx: np.ndarray, cy: np.ndarray, onb: np.ndarray,
              y_line: float, detected, n: np.ndarray, k: np.ndarray, band: float, frac: float, note: str):
    """The PIL sheet. kf = overview px per slide px."""
    from PIL import Image, ImageDraw
    from bench.data.crop_sheet import _font
    W, H = sl.dimensions
    th, tw = thumb.shape[:2]
    f_big, f_small = _font(18), _font(13)

    over = thumb.copy()
    tx, ty = np.floor(cx * kf).astype(np.int64), np.floor(cy * kf).astype(np.int64)
    ok = (tx >= 0) & (ty >= 0) & (tx < tw) & (ty < th)
    over[ty[ok & ~onb], tx[ok & ~onb]] = CYAN
    over[ty[ok & onb], tx[ok & onb]] = RED
    ov = Image.fromarray(over)
    d = ImageDraw.Draw(ov)
    yl = y_line * kf
    d.line([(0, yl), (tw, yl)], fill=YELLOW, width=3)
    d.text((8, max(0, yl - 22)), f"y = {y_line:.0f}", fill=YELLOW, font=f_big, stroke_width=2, stroke_fill="black")

    # share of cells on black per band, aligned with the overview rows (band = one overview row)
    prof = Image.new("RGB", (PROFILE_W, th), "white")
    dp = ImageDraw.Draw(prof)
    share = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    xmax = min(1.0, max(0.1, float(np.ceil(share.max(initial=0) * 10) / 10)))
    L, R = 52, PROFILE_W - 10
    for i, v in enumerate(share[:th]):
        if v > 0:
            dp.line([(L, i), (L + (R - L) * v / xmax, i)], fill=RED)
    xt = L + (R - L) * frac / xmax
    for yy in range(0, th, 8):
        dp.line([(xt, yy), (xt, yy + 3)], fill=(120, 120, 120))
    dp.line([(L, 0), (L, th)], fill="black")
    for t in np.linspace(0, xmax, 3):
        xx = L + (R - L) * t / xmax
        dp.text((min(xx - 12, PROFILE_W - 36), th - 16), f"{t:.0%}", fill="black", font=f_small)
    step = 5000 if H > 20000 else 1000
    for yv in range(step, H, step):
        dp.line([(L - 5, yv * kf), (L, yv * kf)], fill="black")
        dp.text((2, yv * kf - 7), f"{yv // 1000}k", fill="black", font=f_small)
    dp.line([(L, yl), (R, yl)], fill=YELLOW, width=3)
    dp.text((L + 4, 4), "cells on black", fill="black", font=f_small)

    # close-ups centred on the line
    zs = ZOOM_PX // ZOOM_DOWN
    zooms = []
    for x in zoom_xs(cx, cy, y_line, W):
        x0 = int(np.clip(x - ZOOM_PX // 2, 0, max(0, W - ZOOM_PX)))
        y0 = int(np.clip(y_line - ZOOM_PX // 2, 0, max(0, H - ZOOM_PX)))
        w, h = min(ZOOM_PX, W - x0), min(ZOOM_PX, H - y0)
        w, h = w // ZOOM_DOWN * ZOOM_DOWN, h // ZOOM_DOWN * ZOOM_DOWN
        try:
            rgb = read_rgb(sl, x0, y0, w, h)
            small = rgb.reshape(h // ZOOM_DOWN, ZOOM_DOWN, w // ZOOM_DOWN, ZOOM_DOWN, 3).mean(axis=(1, 3))
            im = Image.fromarray(small.astype(np.uint8))
        except Exception as e:
            im = Image.new("RGB", (w // ZOOM_DOWN, h // ZOOM_DOWN), (200, 200, 200))
            ImageDraw.Draw(im).text((10, 10), f"unreadable: {type(e).__name__}", fill="black", font=f_small)
        dz = ImageDraw.Draw(im)
        sel = (cx >= x0) & (cx < x0 + w) & (cy >= y0) & (cy < y0 + h)
        for xx, yy, b in zip((cx[sel] - x0) / ZOOM_DOWN, (cy[sel] - y0) / ZOOM_DOWN, onb[sel]):
            dz.ellipse([xx - 2, yy - 2, xx + 2, yy + 2], outline=RED if b else CYAN)
        dz.line([(0, (y_line - y0) / ZOOM_DOWN), (im.width, (y_line - y0) / ZOOM_DOWN)], fill=YELLOW, width=2)
        zooms.append((x0, y0, im))
        ov_d = ImageDraw.Draw(ov)
        ov_d.rectangle([x0 * kf, y0 * kf, (x0 + w) * kf, (y0 + h) * kf], outline=(255, 0, 255), width=2)

    head, gap = 96, 12
    row1_w = 10 + tw + gap + PROFILE_W + 10
    row2_w = 10 + len(zooms) * (zs + gap)
    page = Image.new("RGB", (max(row1_w, row2_w), head + th + 40 + zs + 30), "white")
    d = ImageDraw.Draw(page)
    above, below = cy < y_line, cy >= y_line
    pct = lambda m: f"{100 * onb[m].mean():.2f}%" if m.any() else "-"
    d.text((10, 6), f"{slide}: keep above y = {y_line:.0f} px   ({W} x {H} px)", fill="black", font=f_big)
    det = "none found" if detected is None else f"{detected:.0f}"
    d.text((10, 32), f"detected line: {det}   rule: first band of {band:.0f} px with >= {frac:.0%} of its cells on "
                     f"black, sustained below   {note}", fill="black", font=f_small)
    d.text((10, 52), f"above: {int(above.sum()):,} cells, {pct(above)} on black      below: {int(below.sum()):,} cells, "
                     f"{pct(below)} on black", fill="black", font=f_small)
    d.text((10, 72), "cyan = cell, red = cell on black (overview pixel max RGB <= 20), yellow = line, "
                     "magenta = close-ups below (1/4 resolution)", fill=(90, 90, 90), font=f_small)
    page.paste(ov, (10, head))
    page.paste(prof, (10 + tw + gap, head))
    for j, (x0, y0, im) in enumerate(zooms):
        X = 10 + j * (zs + gap)
        d.text((X, head + th + 14), f"x {x0}-{x0 + ZOOM_PX}, y {y0}-{y0 + ZOOM_PX}", fill="black", font=f_small)
        page.paste(im, (X, head + th + 32))
    return page


def overview_thumb(cfg, slide: str, sl, wsi_path: str, max_side: int):
    """(thumb, overview px per slide px, where it came from)."""
    from PIL import Image
    from bench.data.overview import PAGE_HEAD, PAGE_LEFT, downscaled, thumb_geometry
    W, H, level, ds, lw, lh, g = thumb_geometry(sl, max_side)
    tw, th = lw // g, lh // g
    page = cfg.root / "qc" / "overview" / f"{slide}.png"
    if page.exists():
        im = Image.open(page).convert("RGB")
        if im.width == 2 * tw + 16 + 20 and im.height == th + PAGE_HEAD + 22:
            return np.asarray(im)[PAGE_HEAD:PAGE_HEAD + th, PAGE_LEFT:PAGE_LEFT + tw].copy(), 1.0 / (ds * g), str(page)
        print(f"{page} has another size (made with another --max-side?): reading the slide instead", file=sys.stderr)
    print("reading the whole slide for the overview (minutes)...", file=sys.stderr)
    thumb, _, kf, _, _, _ = downscaled(wsi_path, max_side)
    return thumb, kf, "read from the slide"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slide")
    ap.add_argument("--y", type=float, help="draw this line (slide px) instead of the detected one")
    ap.add_argument("--frac", type=float, default=0.03, help="share of a band's cells on black that starts the cut")
    ap.add_argument("--min-cells", type=int, default=20, help="cells on black needed in the window below the line")
    ap.add_argument("--window", type=float, default=1000.0, help="slide px below the line that must stay black")
    ap.add_argument("--max-side", type=int, default=2000, help="as used by bench overview")
    ap.add_argument("--config", help="bench YAML (default: $BENCH_CONFIG or configs/bench.yaml)")
    a = ap.parse_args(argv)

    import openslide
    from bench.config import load_config
    from bench.data.darkspots import load_coords
    from bench.data.inspect import load_inspect
    from bench.data.overview import BLACK_MAX

    cfg = load_config(a.config)
    ins = load_inspect(cfg)
    if a.slide not in ins.index:
        close = difflib.get_close_matches(a.slide, list(ins.index), n=3, cutoff=0.4)
        raise SystemExit(f"{a.slide} is not in the inspect table" + (f"; did you mean: {', '.join(close)}" if close else ""))
    row = ins.loc[a.slide].to_dict()
    cx, cy, _, _ = load_coords(row["coords_path"], cfg.geometry.centre_offset_px)
    sl = openslide.OpenSlide(row["wsi_path"])
    try:
        thumb, kf, src = overview_thumb(cfg, a.slide, sl, row["wsi_path"], a.max_side)
        th, tw = thumb.shape[:2]
        tx, ty = np.floor(cx * kf).astype(np.int64), np.floor(cy * kf).astype(np.int64)
        inside = (tx >= 0) & (ty >= 0) & (tx < tw) & (ty < th)
        onb = np.zeros(len(cx), bool)
        onb[inside] = thumb[ty[inside], tx[inside]].max(axis=1) <= BLACK_MAX
        band = 1.0 / kf                                      # one overview row
        detected, n, k = find_line(cy, onb, band, a.frac, a.min_cells, a.window)
        y_line = a.y if a.y is not None else detected
        if y_line is None:
            raise SystemExit(f"no band reaches {a.frac:.0%} of cells on black: try a lower --frac, or give --y")
        note = f"(drawn: your --y {a.y:.0f})" if a.y is not None else ""
        page = cut_sheet(a.slide, sl, thumb, kf, cx, cy, onb, y_line, detected, n, k, band, a.frac, note)
    finally:
        sl.close()
    out = cfg.root / "qc" / "cut"
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{a.slide}_cutline.png"
    page.save(png)
    above = cy < y_line
    print(f"{a.slide}: line y = {y_line:.0f} px (detected: {'none' if detected is None else f'{detected:.0f}'}; "
          f"overview: {src})")
    print(f"  above: {int(above.sum()):,} cells ({100 * onb[above].mean() if above.any() else 0:.2f}% on black)   "
          f"below: {int((~above).sum()):,} cells ({100 * onb[~above].mean() if (~above).any() else 0:.2f}% on black)")
    print(f"  {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
