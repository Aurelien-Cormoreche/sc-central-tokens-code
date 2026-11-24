#!/usr/bin/env python3
"""Find the offset between the purple image and its green copy in a slide whose raw file repeats itself past
4 GiB (GSM9509150, GSM9509144), and map pixel locations in the green part onto the purple one.

Model (tested here, not assumed): the file was written from an uncompressed BGR buffer whose byte index
wrapped at 2**32. Then every pixel from index n = floor(2**32 / 3) on (reading order, p = y * W + x) shows
pixel p - n: the purple image again, shifted by n pixels in reading order (dy, dx = divmod(n, W); the part
pushed past the right edge reappears at the left, one row lower), with the bytes one position off, so its
colours come out (R,G,B) -> (G,B,R). A second wrap (2**33, GSM9509144) repeats it once more.

What it does, for one slide (the converted copy bench reads, identical to the raw file):
  1. overview: the shift between the green and the purple tissue by phase correlation (independent of the model)
  2. full resolution: refines it on boxes of green tissue, finds the channel order that undoes the colours,
     and compares with the model's (dy, dx)
  3. cells: maps those in the green part to the purple one and reports where they land: on tissue or not, and
     on top of a purple-part cell or not (a cell of the same tissue imaged twice)
  4. writes {bench_root}/qc/conversion/<slide>_copy_offset.png (green box with its cells | the purple region it
     was copied from with the mapped cells in cyan, the purple part's own cells in yellow) and
     <slide>_copy_offset.json (the mapping)

  python scripts/find_copy_offset.py GSM9509150_NSCLC_lung_L4_L4
Reads the overview (bench overview PNG if present, else the slide: minutes) and a few MB at full resolution.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

WRAP = 2 ** 32
BOX, MARGIN = 512, 96


# ── the model ──

def copy_shift(k: int, wrap: int = WRAP) -> int:
    """Pixels between a pixel of the k-th copy and the pixel it shows (reading order)."""
    return (k * wrap) // 3


def to_original(x, y, W: int, wrap: int = WRAP):
    """File pixel (x, y) -> (x, y) of the pixel it shows, and the copy number k (0 = not a copy)."""
    x = np.asarray(np.rint(x), dtype=np.int64)
    y = np.asarray(np.rint(y), dtype=np.int64)
    p = y * W + x
    k = (3 * p + 1) // wrap                                   # wrapped once its last two bytes are past the mark
    q = p - (k * wrap) // 3
    return q % W, q // W, k


# ── measuring ──

def _density(a: np.ndarray) -> np.ndarray:
    return 255.0 - a[..., :3].astype(np.float32).mean(axis=2)   # the same for any channel order


def tissue(a: np.ndarray) -> np.ndarray:
    return (a[..., :3].min(axis=2) < 215) & (a[..., :3].max(axis=2) > 30)


def green(a: np.ndarray) -> np.ndarray:
    """Tissue whose colours are rotated (G > R; H&E purple and pink have R > G)."""
    return tissue(a) & (a[..., 1].astype(np.int16) > a[..., 0].astype(np.int16) + 5)


def thumb_shift(thumb: np.ndarray, n_peaks: int = 3):
    """[(dy, dx, strength)] in overview px: shifts s with green(p) ~ purple(p - s), by phase correlation of the
    green and the purple tissue (zero-padded, so shifts are not taken modulo the image size)."""
    d = _density(thumb)
    g = green(thumb)
    a = np.where(tissue(thumb) & ~g, d, 0.0)
    b = np.where(g, d, 0.0)
    H, W = d.shape
    s = (2 * H, 2 * W)
    R = np.fft.rfft2(b, s) * np.conj(np.fft.rfft2(a, s))
    R /= np.abs(R) + 1e-9
    r = np.fft.irfft2(R, s)
    out = []
    for _ in range(n_peaks):
        i, j = np.unravel_index(int(np.argmax(r)), r.shape)
        out.append((i if i < H else i - 2 * H, j if j < W else j - 2 * W, float(r[i, j])))
        r[max(0, i - 3):i + 4, max(0, j - 3):j + 4] = 0
    return out


def locate(small: np.ndarray, big: np.ndarray):
    """(oy, ox, ncc): where `small` sits in `big` (normalized cross-correlation of tissue density, via FFT)."""
    s, b = _density(small), _density(big)
    h, w = s.shape
    Hb, Wb = b.shape
    s = s - s.mean()
    fb = np.fft.rfft2(b)
    corr = lambda t: np.fft.irfft2(fb * np.conj(np.fft.rfft2(t, b.shape)), b.shape)  # noqa: E731
    num = corr(s)[: Hb - h + 1, : Wb - w + 1]
    ones = np.ones_like(s)
    fb2 = np.fft.rfft2(b * b)
    s1 = np.fft.irfft2(fb * np.conj(np.fft.rfft2(ones, b.shape)), b.shape)[: Hb - h + 1, : Wb - w + 1]
    s2 = np.fft.irfft2(fb2 * np.conj(np.fft.rfft2(ones, b.shape)), b.shape)[: Hb - h + 1, : Wb - w + 1]
    var_b = np.maximum(s2 - s1 * s1 / (h * w), 1e-6)
    ncc = num / np.sqrt(var_b * float((s * s).sum()) + 1e-9)
    oy, ox = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    return int(oy), int(ox), float(ncc[oy, ox])


def best_order(stored: np.ndarray, original: np.ndarray):
    """(channel order of `stored` that gives `original`, mean |diff| with it, mean |diff| as stored)."""
    res = {p: float(np.abs(stored[..., list(p)].astype(np.int16) - original.astype(np.int16)).mean())
           for p in itertools.permutations(range(3))}
    p = min(res, key=res.get)
    return p, res[p], res[(0, 1, 2)]


def refine(read, W: int, H: int, guess, thumb: np.ndarray, kf: float, n_boxes: int = 6, seed: int = 0,
           box: int = BOX, margin: int = MARGIN):
    """Full-resolution shift from boxes of green tissue near the overview's guess (dy, dx):
    [(x, y, dy, dx, ncc, order, mad, mad_as_stored)]."""
    gy, gx = guess
    BOX, MARGIN = box, margin  # noqa: N806
    gm = green(thumb)
    ys, xs = np.nonzero(gm)
    X = np.floor((xs + 0.5) / kf - BOX / 2).astype(np.int64)
    Y = np.floor((ys + 0.5) / kf - BOX / 2).astype(np.int64)
    ok = ((X >= 0) & (Y >= 0) & (X + BOX <= W) & (Y + BOX <= H)
          & (X - gx - MARGIN >= 0) & (Y - gy - MARGIN >= 0)
          & (X - gx + BOX + MARGIN <= W) & (Y - gy + BOX + MARGIN <= H))
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(n_boxes * 4, len(idx)), replace=False)
    out = []
    for i in pick:
        x, y = int(X[i]), int(Y[i])
        g = read(x, y, BOX, BOX)
        if green(g).mean() < 0.3:
            continue
        sx, sy = x - gx - MARGIN, y - gy - MARGIN
        src = read(sx, sy, BOX + 2 * MARGIN, BOX + 2 * MARGIN)
        oy, ox, ncc = locate(g, src)
        dy, dx = y - (sy + oy), x - (sx + ox)
        order, mad, mad0 = best_order(g, src[oy:oy + BOX, ox:ox + BOX])
        out.append((x, y, dy, dx, ncc, order, mad, mad0))
        if len(out) >= n_boxes:
            break
    return out


def nuclear_contrast(read, cx, cy, W: int, H: int, n: int = 400, seed: int = 0) -> float:
    """Median darkness at the cell centroids minus 10-18 px around them: clearly > 0 when the centroids sit on
    the nuclei of the image they are placed on, ~0 when they do not."""
    if not len(cx):
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cx), size=min(n, len(cx)), replace=False)
    R = 20
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(yy, xx)
    centre, ring = rr <= 3, (rr >= 10) & (rr <= 18)
    out = []
    for i in idx:
        x, y = int(round(cx[i])), int(round(cy[i]))
        if x - R < 0 or y - R < 0 or x + R + 1 > W or y + R + 1 > H:
            continue
        d = _density(read(x - R, y - R, 2 * R + 1, 2 * R + 1))
        if tissue(read(x - R, y - R, 2 * R + 1, 2 * R + 1)).mean() < 0.5:
            continue
        out.append(float(d[centre].mean() - d[ring].mean()))
    return float(np.median(out)) if out else float("nan")


def cell_report(read, cx, cy, W: int, H: int, wrap: int = WRAP) -> dict:
    """Do the green-part cells belong to the green tissue (their centroids on its nuclei, as the purple-part
    cells are on the purple tissue's), and would mapping put them on top of purple-part cells?"""
    from scipy.spatial import cKDTree
    mx, my, k = to_original(cx, cy, W, wrap)
    top, cp = k == 0, k > 0
    r = {"n_cells": int(len(cx)), "n_purple_part": int(top.sum()), "n_green_part": int(cp.sum()),
         "nuclear_contrast_purple_part": nuclear_contrast(read, cx[top], cy[top], W, H),
         "nuclear_contrast_purple_part_moved_9px": nuclear_contrast(read, cx[top] + 9, cy[top] + 5, W, H)}
    if cp.any():
        r["nuclear_contrast_green_part_as_placed"] = nuclear_contrast(read, cx[cp], cy[cp], W, H)
        tree = cKDTree(np.c_[cx[top], cy[top]])
        d, _ = tree.query(np.c_[mx[cp], my[cp]])
        dj, _ = tree.query(np.c_[mx[cp] + 37, my[cp] + 23])             # baseline: same points, moved
        r["mapped_within_3px_of_a_purple_part_cell"] = float((d <= 3).mean())
        r["baseline_moved_37_23px"] = float((dj <= 3).mean())
    return r


# ── picture ──

def sheet(slide, read, W, H, cx, cy, mx, my, k, out_png, wrap: int = WRAP):
    from PIL import Image, ImageDraw
    from bench.data.crop_sheet import _font
    cp = np.flatnonzero(k == 1)
    if not len(cp):
        return None
    dy, dx = divmod(copy_shift(1, wrap), W)
    D = 2
    S = int(min(1024, (H - dy - 1) // (2 * D) * (2 * D), (W - dx) // (2 * D) * (2 * D)))
    ok = cp[(cx[cp] >= dx + S / 2) & (cx[cp] + S / 2 < W) & (cy[cp] >= dy + 1 + S / 2) & (cy[cp] + S / 2 < H)]
    if not len(ok):
        ok = cp
    c = ok[np.argsort(np.hypot(cx[ok] - np.median(cx[ok]), cy[ok] - np.median(cy[ok])))[0]]
    x0 = int(np.clip(cx[c] - S / 2, dx, W - S))                 # inside the copy, right of the wrapped strip
    y0 = int(np.clip(cy[c] - S / 2, dy + 1, H - S))
    sx, sy = x0 - dx, y0 - dy                                   # the region it was copied from
    a, b = read(x0, y0, S, S), read(sx, sy, S, S)

    def small(img):
        return Image.fromarray(img.reshape(S // D, D, S // D, D, 3).mean(axis=(1, 3)).astype(np.uint8))

    A, B = small(a), small(b)
    da, db = ImageDraw.Draw(A), ImageDraw.Draw(B)
    inA = (k == 1) & (cx >= x0) & (cx < x0 + S) & (cy >= y0) & (cy < y0 + S)
    for x, y in zip(cx[inA], cy[inA]):
        u, v = (x - x0) / D, (y - y0) / D
        da.ellipse([u - 3, v - 3, u + 3, v + 3], outline=(0, 230, 255), width=2)
    inB_top = (k == 0) & (cx >= sx) & (cx < sx + S) & (cy >= sy) & (cy < sy + S)
    for x, y in zip(cx[inB_top], cy[inB_top]):
        u, v = (x - sx) / D, (y - sy) / D
        db.ellipse([u - 3, v - 3, u + 3, v + 3], outline=(255, 215, 0), width=2)
    for x, y in zip(mx[inA], my[inA]):
        u, v = (x - sx) / D, (y - sy) / D
        db.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(0, 230, 255))
    f, fs = _font(16), _font(13)
    page = Image.new("RGB", (max(2 * (S // D) + 30, 900), S // D + 70), "white")
    d = ImageDraw.Draw(page)
    d.text((10, 6), f"{slide}: green-part box at ({x0}, {y0}) and the purple region it was copied from ({sx}, {sy})",
           fill="black", font=f)
    d.text((10, 32), f"as stored + its cells (cyan rings, {int(inA.sum())})", fill="black", font=fs)
    d.text((20 + S // D, 32), f"copied from: those cells mapped (cyan dots), own cells (yellow, {int(inB_top.sum())})",
           fill="black", font=fs)
    page.paste(A, (10, 60))
    page.paste(B, (20 + S // D, 60))
    page.save(out_png)
    return out_png


# ── main ──

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("slide")
    ap.add_argument("--max-side", type=int, default=2000, help="overview size, as used by bench overview")
    ap.add_argument("--config", help="bench YAML (default: $BENCH_CONFIG or configs/bench.yaml)")
    a = ap.parse_args(argv)
    import openslide
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from find_cut_line import overview_thumb
    from bench.config import load_config
    from bench.data.darkspots import load_coords
    from bench.data.inspect import load_inspect
    cfg = load_config(a.config)
    ins = load_inspect(cfg)
    if a.slide not in ins.index:
        raise SystemExit(f"{a.slide} is not in the inspect table")
    row = ins.loc[a.slide].to_dict()
    path = row["wsi_path"]
    out = cfg.root / "qc" / "conversion"
    out.mkdir(parents=True, exist_ok=True)
    with openslide.OpenSlide(path) as sl:
        W, H = sl.dimensions
        thumb, kf, src = overview_thumb(cfg, a.slide, sl, path, a.max_side)

        def read(x, y, w, h):
            return np.asarray(sl.read_region((int(x), int(y)), 0, (int(w), int(h))))[..., :3]

        n1 = copy_shift(1)
        mdy, mdx = divmod(n1, W)
        print(f"{a.slide}: {W} x {H} px; overview {thumb.shape[1]} x {thumb.shape[0]} ({src})")
        print(f"model (2**32-byte wrap): copies start at pixel {n1:,} (row {mdy}, column {mdx}); shift "
              f"(dy, dx) = ({mdy}, {mdx}), and ({mdy + 1}, {mdx - W}) for columns < {mdx}"
              + (f"; second copy from row {copy_shift(2) // W}" if copy_shift(2) < W * H else ""))
        g = green(thumb)
        rows = np.flatnonzero(g.sum(1) > 0.5 * np.maximum(tissue(thumb).sum(1), 1) * (tissue(thumb).sum(1) > 5))
        if len(rows):
            print(f"overview: green tissue from about row {rows[0] / kf:.0f} ({g.sum() / max(tissue(thumb).sum(), 1):.0%} "
                  f"of the tissue is green)")
        peaks = thumb_shift(thumb)
        print("overview shift green -> purple (dy, dx, strength), in slide px: "
              + "; ".join(f"({py / kf:.0f}, {px / kf:.0f}, {st:.3f})" for py, px, st in peaks))
        boxes = []
        for py, px, st in peaks[:2]:
            if st >= 0.3 * peaks[0][2]:
                boxes += refine(read, W, H, (int(round(py / kf)), int(round(px / kf))), thumb, kf, n_boxes=4)
        if not boxes:
            print("no box of green tissue to refine on")
            return 1
        print("full resolution, per box: (x, y) -> shift (dy, dx), match (NCC), channel order, mean |diff| "
              "after / before re-ordering")
        for x, y, dy, dx, ncc, order, mad, mad0 in boxes:
            print(f"  ({x}, {y}) -> ({dy}, {dx})  NCC {ncc:.3f}  stored channels {order} = original RGB  "
                  f"|diff| {mad:.1f} / {mad0:.1f}")
        good = [b for b in boxes if b[4] > 0.5]
        shifts = sorted({(b[2], b[3]) for b in good})
        model = [(mdy, mdx), (mdy + 1, mdx - W)]
        match = bool(good) and all(any(abs(b[2] - u) <= 1 and abs(b[3] - v) <= 1 for u, v in model) for b in good)
        print(f"measured shift(s): {shifts}; " + ("MATCHES the 2**32-byte model" if match else
                                                  "does NOT match the 2**32-byte model: send me this output"))

    cx, cy, _, _ = load_coords(row["coords_path"], cfg.geometry.centre_offset_px)
    mx, my, k = to_original(cx, cy, W)
    with openslide.OpenSlide(path) as sl:
        def read(x, y, w, h):
            return np.asarray(sl.read_region((int(x), int(y)), 0, (int(w), int(h))))[..., :3]
        rep = cell_report(read, cx, cy, W, H)
        png = sheet(a.slide, read, W, H, cx, cy, mx, my, k, out / f"{a.slide}_copy_offset.png")
    print("cells:")
    for key, v in rep.items():
        print(f"  {key:45s} {v:.3f}" if isinstance(v, float) else f"  {key:45s} {v:,}")
    c_top = rep["nuclear_contrast_purple_part"]
    c_green = rep.get("nuclear_contrast_green_part_as_placed", float("nan"))
    if c_green == c_green and c_top == c_top:
        print("  -> the green-part cells " + ("DO sit on the green tissue's nuclei: the green copy is their image"
                                             if c_green > 0.5 * c_top else
                                             "do NOT sit on the green tissue's nuclei (the purple-part cells do on "
                                             "theirs): the green copy is not their tissue"))
    res = {"slide": a.slide, "width": W, "height": H, "model": "2**32-byte wrap of a BGR buffer",
           "model_matches": match, "measured_shifts": [list(s) for s in shifts],
           "copy_starts_at_pixel": n1, "shift_rows": mdy, "shift_cols": mdx,
           "mapping": "p = y*W + x; k = (3p + 1) // 2**32; p' = p - (k * 2**32) // 3; x' = p' % W; y' = p' // W",
           "cells": rep}
    jp = out / f"{a.slide}_copy_offset.json"
    jp.write_text(json.dumps(res, indent=1))
    print(f"{png}\n{jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
