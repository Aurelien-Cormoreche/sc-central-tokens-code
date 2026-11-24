#!/usr/bin/env python3
"""Measure the offset between the purple image and its green copy in a slide whose raw file repeats itself
(GSM9509150; GSM9509144), and map pixel locations in the green part onto the purple one.

Everything is measured on the slide, nothing assumed:
  1. where the green part starts: the row where the tissue colour flips (overview, refined at full resolution
     on a few columns)
  2. the shift: on the overview, every shift (dy, dx) is tried: the green tissue moved back by it and compared
     pixel by pixel with the purple part (mean squared difference; all shifts at once with FFTs); the best is then
     refined to the pixel on full-resolution boxes of green tissue; and the channel order that turns the green
     back into the purple
  3. how much of the green part that one shift explains: a grid of green-tissue boxes, each compared with the
     purple box the shift points to
  4. the cells in the green part, mapped by the shift: do their centroids sit on the nuclei of the green tissue
     as placed (as the purple part's cells do on theirs), and do the mapped positions fall on purple-part cells
  5. {bench_root}/qc/conversion/<slide>_copy_offset.png: green boxes as stored | colours restored | the purple
     box the shift points to | their difference; <slide>_copy_cells.png: a green box with its cells | the purple
     box with those cells mapped (cyan) and its own cells (yellow); <slide>_copy_offset.json: green part = rows
     >= green_from_row; there (x, y) -> (x - dx, y - dy)

  python scripts/find_copy_offset.py GSM9509150_NSCLC_lung_L4_L4
  python scripts/find_copy_offset.py GSM9509150_NSCLC_lung_L4_L4 --landmarks XG YG XP YP
      # use your own shift: the same feature at (XG, YG) in the green part and (XP, YP) in the purple part
      # (QuPath pixel coordinates); it is checked on the green patches and drawn like a measured one
Reads the overview (bench overview PNG if present, else the whole slide: minutes) and some MB at full res.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

WRAP = 2 ** 32
BOX, MARGIN = 512, 96


# ── a guess, only compared with the measurement ──

def copy_shift(k: int, wrap: int = WRAP) -> int:
    """If a BGR buffer's byte index wrapped at `wrap`: pixels (reading order) between a copied pixel and its
    original."""
    return (k * wrap) // 3


def to_original(x, y, W: int, wrap: int = WRAP):
    """That guess as a mapping: file pixel (x, y) -> (x, y) of its original, and the copy number k."""
    x = np.asarray(np.rint(x), dtype=np.int64)
    y = np.asarray(np.rint(y), dtype=np.int64)
    p = y * W + x
    k = (3 * p + 1) // wrap
    q = p - (k * wrap) // 3
    return q % W, q // W, k


# ── colour classes ──

def _density(a: np.ndarray) -> np.ndarray:
    return 255.0 - a[..., :3].astype(np.float32).mean(axis=2)   # the same for any channel order


def tissue(a: np.ndarray) -> np.ndarray:
    return (a[..., :3].min(axis=2) < 215) & (a[..., :3].max(axis=2) > 30)


def green(a: np.ndarray) -> np.ndarray:
    """Tissue whose colours are swapped (G > R; H&E purple and pink have R > G)."""
    return tissue(a) & (a[..., 1].astype(np.int16) > a[..., 0].astype(np.int16) + 5)


# ── where the green part starts ──

def green_start(read, W: int, H: int, thumb: np.ndarray, kf: float, n_cols: int = 6):
    """(first row of the green part, [(column, row)] measured at full resolution), or (None, [])."""
    g, t = green(thumb), tissue(thumb)
    nt = t.sum(1)
    frac = g.sum(1) / np.maximum(nt, 1)
    rows = np.flatnonzero(nt >= 5)
    if not len(rows):
        return None, []
    isg = frac[rows] > 0.5
    tail = np.cumsum(isg[::-1])[::-1] / np.arange(len(rows), 0, -1)     # share of green rows from here down
    i = np.flatnonzero(isg & (tail >= 0.9))
    if not len(i):
        return None, []
    r0 = int(rows[i[0]])
    y_est = (r0 + 0.5) / kf
    # columns with purple tissue just above and green just below
    above, below = t[max(0, r0 - 3):r0] & ~g[max(0, r0 - 3):r0], g[r0:r0 + 3]
    cols = np.flatnonzero(above.any(0) & below.any(0)) if len(above) and len(below) else np.array([], int)
    if not len(cols):
        return int(round(y_est)), []
    cols = cols[np.linspace(0, len(cols) - 1, min(n_cols, len(cols))).astype(int)]
    half = int(4 / kf)
    y0, y1 = max(0, int(y_est) - half), min(H, int(y_est) + half)
    meas = []
    for j in cols:
        x0 = int(min(max(0, (j + 0.5) / kf - 64), W - 128))
        a = read(x0, y0, 128, y1 - y0)
        ta, ga = tissue(a), green(a)
        f = ga.sum(1) / np.maximum(ta.sum(1), 1)
        ok = (ta.sum(1) >= 16) & (f > 0.5)
        k = np.flatnonzero(ok)
        if len(k):
            meas.append((x0 + 64, y0 + int(k[0])))
    if not meas:
        return int(round(y_est)), []
    return int(np.median([m[1] for m in meas])), meas


# ── the shift ──

def _xcorr(A: np.ndarray, B: np.ndarray, shape) -> np.ndarray:
    """C[s] = sum_p A[p] * B[p - s] for every shift s at once (zero-padded to `shape`; index = s mod shape)."""
    return np.fft.irfft2(np.fft.rfft2(A, shape) * np.conj(np.fft.rfft2(B, shape)), shape)


def search_shift(thumb: np.ndarray, start_t: int, min_overlap: float = 0.3, n_best: int = 3):
    """Every shift (dy, dx) of the overview tried at once: the green part's tissue moved back by it and compared
    with the purple part, pixel by pixel (mean squared difference of tissue density, which ignores the colour
    swap). [(dy, dx, rms difference, share of the green tissue compared)], best first, in overview px; shifts
    that compare under `min_overlap` of the green tissue are not considered."""
    d = _density(thumb).astype(np.float64)
    H, W = d.shape
    rows = np.arange(H)[:, None]
    Mg = (tissue(thumb) & (rows >= start_t)).astype(np.float64)       # green tissue
    Mp = np.broadcast_to(rows < start_t, d.shape).astype(np.float64)  # anything in the purple part
    shape = (2 * H, 2 * W)
    n = _xcorr(Mg, Mp, shape)
    sq = _xcorr(Mg * d * d, Mp, shape) - 2 * _xcorr(Mg * d, Mp * d, shape) + _xcorr(Mg, Mp * d * d, shape)
    tot = max(Mg.sum(), 1.0)
    dy = np.where(np.arange(2 * H) < H, np.arange(2 * H), np.arange(2 * H) - 2 * H)[:, None]
    dx = np.where(np.arange(2 * W) < W, np.arange(2 * W), np.arange(2 * W) - 2 * W)[None, :]
    ok = (n >= min_overlap * tot) & (dy > 0)
    mse = np.where(ok, np.maximum(sq, 0) / np.maximum(n, 1.0), np.inf)
    out = []
    for _ in range(n_best):
        i, j = np.unravel_index(int(np.argmin(mse)), mse.shape)
        if not np.isfinite(mse[i, j]):
            break
        out.append((int(dy[i, 0]), int(dx[0, j]), float(np.sqrt(mse[i, j])), float(n[i, j] / tot)))
        mse[max(0, i - 5):i + 6, max(0, j - 5):j + 6] = np.inf
    return out


def locate(small: np.ndarray, big: np.ndarray):
    """(oy, ox, ncc): where `small` sits in `big` (normalized cross-correlation of tissue density, via FFT)."""
    s, b = _density(small), _density(big)
    h, w = s.shape
    Hb, Wb = b.shape
    s = s - s.mean()
    fb = np.fft.rfft2(b)
    num = np.fft.irfft2(fb * np.conj(np.fft.rfft2(s, b.shape)), b.shape)[: Hb - h + 1, : Wb - w + 1]
    fo = np.conj(np.fft.rfft2(np.ones_like(s), b.shape))
    s1 = np.fft.irfft2(fb * fo, b.shape)[: Hb - h + 1, : Wb - w + 1]
    s2 = np.fft.irfft2(np.fft.rfft2(b * b) * fo, b.shape)[: Hb - h + 1, : Wb - w + 1]
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


def _green_boxes(thumb, kf, W, H, start, shift, box, margin, n, seed):
    """Top-left corners of boxes centred on green tissue below `start` whose source (shifted by `shift`, with
    `margin`) lies in the image."""
    gy, gx = shift
    ys, xs = np.nonzero(green(thumb))
    X = np.floor((xs + 0.5) / kf - box / 2).astype(np.int64)
    Y = np.floor((ys + 0.5) / kf - box / 2).astype(np.int64)
    ok = ((X >= 0) & (Y >= (start or 0)) & (X + box <= W) & (Y + box <= H)
          & (X - gx - margin >= 0) & (Y - gy - margin >= 0)
          & (X - gx + box + margin <= W) & (Y - gy + box + margin <= H))
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    rng = np.random.default_rng(seed)
    return [(int(X[i]), int(Y[i])) for i in rng.choice(idx, size=min(n, len(idx)), replace=False)]


def refine(read, W: int, H: int, guess, thumb: np.ndarray, kf: float, n_boxes: int = 6, seed: int = 0,
           box: int = BOX, margin: int = MARGIN, start=None):
    """Full-resolution shift from boxes of green tissue near the overview's guess (dy, dx):
    [(x, y, dy, dx, ncc, order, mad, mad_as_stored)]."""
    out = []
    for x, y in _green_boxes(thumb, kf, W, H, start, guess, box, margin, n_boxes * 4, seed):
        g = read(x, y, box, box)
        if green(g).mean() < 0.3:
            continue
        sx, sy = x - guess[1] - margin, y - guess[0] - margin
        src = read(sx, sy, box + 2 * margin, box + 2 * margin)
        oy, ox, ncc = locate(g, src)
        order, mad, mad0 = best_order(g, src[oy:oy + box, ox:ox + box])
        out.append((x, y, y - (sy + oy), x - (sx + ox), ncc, order, mad, mad0))
        if len(out) >= n_boxes:
            break
    return out


def coverage(read, W: int, H: int, thumb, kf, start, shift, n: int = 40, box: int = 256, tol: int = 8,
             seed: int = 1):
    """[(x, y, ncc, explained)] for a spread of green-tissue boxes: explained = the purple box `shift` points
    to matches it (to within 2 px)."""
    out = []
    for x, y in _green_boxes(thumb, kf, W, H, start, shift, box, tol, n * 3, seed):
        g = read(x, y, box, box)
        if green(g).mean() < 0.3:
            continue
        src = read(x - shift[1] - tol, y - shift[0] - tol, box + 2 * tol, box + 2 * tol)
        oy, ox, ncc = locate(g, src)
        out.append((x, y, ncc, ncc > 0.5 and abs(oy - tol) <= 2 and abs(ox - tol) <= 2))
        if len(out) >= n:
            break
    return out


def measure(read, W: int, H: int, thumb: np.ndarray, kf: float, start=None, box: int = BOX,
            margin: int = MARGIN):
    """Candidate shifts, best first: [{"dy", "dx", "boxes", "ncc", "order", "mad", "mad_as_stored",
    "overview"}]."""
    cands = []
    start_t = int(round((start or 0) * kf))
    best = search_shift(thumb, start_t)
    margin = max(margin, int(2 / kf) + 8)                  # the overview's shift is good to about 1 / kf px
    for py, px, rms, ov in best:
        if rms > 2 * best[0][2]:
            continue
        bx = [b for b in refine(read, W, H, (int(round(py / kf)), int(round(px / kf))), thumb, kf,
                                box=box, margin=margin, start=start) if b[4] > 0.5]
        if not bx:
            continue
        dy, dx = Counter((b[2], b[3]) for b in bx).most_common(1)[0][0]
        same = [b for b in bx if abs(b[2] - dy) <= 1 and abs(b[3] - dx) <= 1]
        if any(abs(c["dy"] - dy) <= 1 and abs(c["dx"] - dx) <= 1 for c in cands):
            continue
        cands.append({"dy": int(dy), "dx": int(dx), "boxes": len(same), "ncc": float(np.median([b[4] for b in same])),
                      "order": Counter(b[5] for b in same).most_common(1)[0][0],
                      "mad": float(np.median([b[6] for b in same])),
                      "mad_as_stored": float(np.median([b[7] for b in same])),
                      "overview": (round(py / kf), round(px / kf), round(rms, 1), round(ov, 2)),
                      "where": [(b[0], b[1]) for b in same]})
    return sorted(cands, key=lambda c: (-c["boxes"], -c["ncc"]))


# ── cells ──

def map_cells(cx, cy, start: int, dy: int, dx: int, max_steps: int = 3):
    """(x', y', steps): cells at rows >= start moved by (-dx, -dy), again while they are still in the green
    part (steps = how many times; 0 = purple part)."""
    mx, my = np.asarray(cx, float).copy(), np.asarray(cy, float).copy()
    steps = np.zeros(len(mx), int)
    for _ in range(max_steps):
        m = my >= start
        if not m.any():
            break
        mx[m] -= dx
        my[m] -= dy
        steps[m] += 1
    return mx, my, steps


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
        a = read(x - R, y - R, 2 * R + 1, 2 * R + 1)
        if tissue(a).mean() < 0.5:
            continue
        d = _density(a)
        out.append(float(d[centre].mean() - d[ring].mean()))
    return float(np.median(out)) if out else float("nan")


def cell_report(read, cx, cy, mx, my, steps, W: int, H: int) -> dict:
    from scipy.spatial import cKDTree
    top, cp = steps == 0, steps > 0
    r = {"n_cells": int(len(cx)), "n_purple_part": int(top.sum()), "n_green_part": int(cp.sum()),
         "n_green_part_mapped_twice_or_more": int((steps > 1).sum()),
         "n_green_part_mapped_outside_image": int((cp & ((mx < 0) | (my < 0) | (mx >= W) | (my >= H))).sum()),
         "nuclear_contrast_purple_part": nuclear_contrast(read, cx[top], cy[top], W, H),
         "nuclear_contrast_purple_part_moved_9px": nuclear_contrast(read, cx[top] + 9, cy[top] + 5, W, H)}
    if cp.any() and top.any():
        r["nuclear_contrast_green_part_as_placed"] = nuclear_contrast(read, cx[cp], cy[cp], W, H)
        tree = cKDTree(np.c_[cx[top], cy[top]])
        d, _ = tree.query(np.c_[mx[cp], my[cp]])
        dj, _ = tree.query(np.c_[mx[cp] + 37, my[cp] + 23])
        r["mapped_within_3px_of_a_purple_part_cell"] = float((d <= 3).mean())
        r["baseline_moved_37_23px"] = float((dj <= 3).mean())
    return r


# ── pictures ──

def compare_sheet(slide, read, where, dy: int, dx: int, order, out_png, S: int = 512, D: int = 2):
    """Per green box: as stored | colours restored | the purple box the shift points to | |difference| x 3."""
    from PIL import Image, ImageDraw
    from bench.data.crop_sheet import _font
    where = where[:3]
    if not where:
        return None
    w = S // D
    page = Image.new("RGB", (4 * (w + 10) + 10, len(where) * (w + 30) + 40), "white")
    d = ImageDraw.Draw(page)
    f, fs = _font(16), _font(13)
    d.text((10, 6), f"{slide}: shift (dy, dx) = ({dy}, {dx}); green (x, y) = purple (x - {dx}, y - {dy})",
           fill="black", font=f)
    for r, (x, y) in enumerate(where):
        g = read(x, y, S, S)
        fixed = g[..., list(order)]
        pu = read(x - dx, y - dy, S, S)
        diff = np.clip(np.abs(fixed.astype(np.int16) - pu.astype(np.int16)) * 3, 0, 255).astype(np.uint8)
        mad = float(np.abs(fixed.astype(np.int16) - pu.astype(np.int16)).mean())
        Y = 34 + r * (w + 30)
        for c, (title, img) in enumerate((("green, as stored", g), ("colours restored", fixed),
                                          (f"purple at ({x - dx}, {y - dy})", pu),
                                          (f"|difference| x3 (mean {mad:.1f})", 255 - diff))):
            X = 10 + c * (w + 10)
            d.text((X, Y), f"{title}" + (f"  ({x}, {y})" if c == 0 else ""), fill="black", font=fs)
            page.paste(Image.fromarray(img.reshape(w, D, w, D, 3).mean(axis=(1, 3)).astype(np.uint8)), (X, Y + 18))
    page.save(out_png)
    return out_png


def sheet(slide, read, W, H, cx, cy, mx, my, steps, start, dy, dx, out_png, S: int = 1024, D: int = 2):
    from PIL import Image, ImageDraw
    from bench.data.crop_sheet import _font
    one = np.flatnonzero(steps == 1)
    if not len(one):
        return None
    S = int(min(S, (H - start) // (2 * D) * (2 * D), W // (2 * D) * (2 * D)))
    ok = one[(cx[one] - S / 2 >= max(0, dx)) & (cx[one] + S / 2 <= min(W, W + dx))
             & (cy[one] - S / 2 >= start) & (cy[one] + S / 2 <= H)]
    if not len(ok):
        ok = one
    c = ok[np.argsort(np.hypot(cx[ok] - np.median(cx[ok]), cy[ok] - np.median(cy[ok])))[0]]
    x0 = int(np.clip(cx[c] - S / 2, max(0, dx), min(W, W + dx) - S))
    y0 = int(np.clip(cy[c] - S / 2, start, H - S))
    sx, sy = x0 - dx, y0 - dy
    a, b = read(x0, y0, S, S), read(sx, sy, S, S)

    def small(img):
        return Image.fromarray(img.reshape(S // D, D, S // D, D, 3).mean(axis=(1, 3)).astype(np.uint8))

    A, B = small(a), small(b)
    da, db = ImageDraw.Draw(A), ImageDraw.Draw(B)
    inA = (steps == 1) & (cx >= x0) & (cx < x0 + S) & (cy >= y0) & (cy < y0 + S)
    for x, y in zip(cx[inA], cy[inA]):
        u, v = (x - x0) / D, (y - y0) / D
        da.ellipse([u - 3, v - 3, u + 3, v + 3], outline=(0, 230, 255), width=2)
    inB = (steps == 0) & (cx >= sx) & (cx < sx + S) & (cy >= sy) & (cy < sy + S)
    for x, y in zip(cx[inB], cy[inB]):
        u, v = (x - sx) / D, (y - sy) / D
        db.ellipse([u - 3, v - 3, u + 3, v + 3], outline=(255, 215, 0), width=2)
    for x, y in zip(mx[inA], my[inA]):
        u, v = (x - sx) / D, (y - sy) / D
        db.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(0, 230, 255))
    f, fs = _font(16), _font(13)
    page = Image.new("RGB", (max(2 * (S // D) + 30, 900), S // D + 70), "white")
    d = ImageDraw.Draw(page)
    d.text((10, 6), f"{slide}: green box at ({x0}, {y0}); shift (dy, dx) = ({dy}, {dx}) -> purple box at ({sx}, {sy})",
           fill="black", font=f)
    d.text((10, 32), f"green, as stored + its cells (cyan rings, {int(inA.sum())})", fill="black", font=fs)
    d.text((20 + S // D, 32), f"purple: those cells mapped (cyan dots), its own cells (yellow, {int(inB.sum())})",
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
    ap.add_argument("--landmarks", nargs=4, type=float, metavar=("XG", "YG", "XP", "YP"),
                    help="a feature in the green part (XG, YG) and the same in the purple part (XP, YP): use that shift")
    ap.add_argument("--start", type=int, help="first row of the green part (default: measured)")
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
    cx, cy, _, _ = load_coords(row["coords_path"], cfg.geometry.centre_offset_px)
    with openslide.OpenSlide(path) as sl:
        W, H = sl.dimensions
        thumb, kf, src = overview_thumb(cfg, a.slide, sl, path, a.max_side)

        def read(x, y, w, h):
            return np.asarray(sl.read_region((int(x), int(y)), 0, (int(w), int(h))))[..., :3]

        print(f"{a.slide}: {W} x {H} px; overview {thumb.shape[1]} x {thumb.shape[0]} ({src})")
        start, cols = green_start(read, W, H, thumb, kf)
        if start is None and a.start is None:
            print("no green part found")
            return 1
        print(f"green part from row {start}" + (f" (measured at columns {', '.join(f'{x}: row {y}' for x, y in cols)})"
                                                if cols else " (overview estimate)"))
        if a.start is not None:
            start = a.start
            print(f"  using row {start} (--start)")
        if a.landmarks:
            xg, yg, xp, yp = a.landmarks
            ldy, ldx = int(round(yg - yp)), int(round(xg - xp))
            print(f"landmarks: green ({xg:.0f}, {yg:.0f}) = purple ({xp:.0f}, {yp:.0f}) -> shift (dy, dx) = ({ldy}, {ldx})")
            g0 = read(int(xg) - 128, int(yg) - 128, 256, 256)
            p0 = read(int(xp) - 128 - 32, int(yp) - 128 - 32, 256 + 64, 256 + 64)
            oy, ox, ncc = locate(g0, p0)
            order, mad, mad0 = best_order(g0, p0[oy:oy + 256, ox:ox + 256])
            print(f"  around the landmark: best match {oy - 32:+d} / {ox - 32:+d} px from the purple point, NCC {ncc:.3f}, "
                  f"stored channels {order} give the purple (mean |diff| {mad:.1f}, {mad0:.1f} as stored)")
            cands = [{"dy": ldy, "dx": ldx, "boxes": 0, "ncc": ncc, "order": order, "mad": mad, "mad_as_stored": mad0,
                      "overview": "landmarks"}]
        else:
            cands = measure(read, W, H, thumb, kf, start)
        if not cands:
            print("no shift found: no box of green tissue matched the purple tissue")
            return 1
        print("overview, every shift compared (dy, dx, rms difference, share of green tissue compared): "
              + "; ".join(f"({v[0] / kf:.0f}, {v[1] / kf:.0f}, {v[2]:.1f}, {v[3]:.2f})"
                          for v in search_shift(thumb, int(round(start * kf)))))
        for i, c in enumerate(cands):
            print(f"{'shift' if i == 0 else 'other shift'} (dy, dx) = ({c['dy']}, {c['dx']}): green (x, y) = purple "
                  f"(x - {c['dx']}, y - {c['dy']}); {c['boxes']} boxes, NCC {c['ncc']:.3f}, colours: stored channels "
                  f"{c['order']} give the purple (mean |diff| {c['mad']:.1f}, {c['mad_as_stored']:.1f} as stored); "
                  f"overview {c['overview']}")
        best = cands[0]
        dy, dx = best["dy"], best["dx"]
        cov = coverage(read, W, H, thumb, kf, start, (dy, dx))
        n_ok = sum(c[3] for c in cov)
        print(f"this shift explains {n_ok} of {len(cov)} green-tissue boxes spread over the green part")
        bad = [c for c in cov if not c[3]]
        if bad:
            print("  not explained (x, y, NCC): " + ", ".join(f"({x}, {y}, {v:.2f})" for x, y, v, _ in bad[:12]))
        gdy, gdx = divmod(copy_shift(1), W)
        print(f"(for comparison, the 2**32-byte guess was ({gdy}, {gdx}) from row {gdy})")

        cmp_png = compare_sheet(a.slide, read, best.get("where") or [(x, y) for x, y, _, ok in cov if ok],
                                dy, dx, best["order"], out / f"{a.slide}_copy_offset.png")
        mx, my, steps = map_cells(cx, cy, start, dy, dx)
        rep = cell_report(read, cx, cy, mx, my, steps, W, H)
        png = sheet(a.slide, read, W, H, cx, cy, mx, my, steps, start, dy, dx, out / f"{a.slide}_copy_cells.png")
    print("cells:")
    for key, v in rep.items():
        print(f"  {key:45s} {v:.3f}" if isinstance(v, float) else f"  {key:45s} {v:,}")
    res = {"slide": a.slide, "width": W, "height": H, "green_from_row": start,
           "green_from_row_by_column": [list(c) for c in cols], "dy": dy, "dx": dx,
           "mapping": "rows >= green_from_row: (x, y) -> (x - dx, y - dy)",
           "colour_order": list(best["order"]), "boxes_explained": [n_ok, len(cov)],
           "other_shifts": [[c["dy"], c["dx"], c["boxes"]] for c in cands[1:]], "cells": rep}
    jp = out / f"{a.slide}_copy_offset.json"
    jp.write_text(json.dumps(res, indent=1))
    print(f"{cmp_png}\n{png}\n{jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
