"""Look at the cells check_black_patches.py flagged: what their crops contain and where they are.

check_black_patches.py finds groups of identical embeddings, i.e. identical crops. Those
are black for a failed read, but saturated white glass or a flat fill colour give identical
crops too. This reads the flagged cells' crops from the WSI the extraction read (resolved
from an extraction JSON config, like check_wsi_regions) and, per slide:
  - prints what a random sample of flagged crops contains (black / white / flat / tissue,
    mean colour), next to the same for unflagged cells;
  - saves {out}/{slide}_crops.png: a grid of flagged crops;
  - saves {out}/{slide}_overview.png: the slide downscaled to <= MAX_SIDE px; left as is,
    right with every cell in cyan and the flagged cells in red.
The overview reads the whole slide once (a few minutes for a large one), so use a compute node.
CPU-only (openslide, h5py, numpy, Pillow).

Usage (repo root, .env sourced; CSV from check_black_patches.py --csv):
    python scripts/show_flagged_cells.py split_black_cells.csv \
        configs/extract_embeddings/paper_uni2_fov/uni2_fov224_split.json --out flagged_cells
    ... --slides GSM9509144_NSCLC_chuvio_L4_L4      # only these slides
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import openslide
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.python.extract_embeddings.check_wsi_regions import cell_crops, resolve_slides  # noqa: E402  (no torch)
from src.python.extract_embeddings.run_configs import _expand  # noqa: E402

MAX_SIDE = 2500   # px: longest side of the overview image
GRID_COLS = 6


def classify(a: np.ndarray) -> str:
    if (a.max(axis=-1) <= 8).mean() >= 0.99:
        return 'black'
    if (a.min(axis=-1) >= 247).mean() >= 0.99:
        return 'white'
    if a.reshape(-1, 3).std(axis=0).max() < 3:
        return 'flat'
    return 'tissue'


def read_crop(wsi, x, y, sx, sy) -> np.ndarray:
    try:
        return np.asarray(wsi.read_region((int(x), int(y)), 0, (int(sx), int(sy))).convert('RGB'))
    except openslide.OpenSlideError:
        return None


def describe(wsi, idx, x, y, sx, sy) -> str:
    """What the crops of cells idx contain: count and mean RGB per kind."""
    colours, failed = defaultdict(list), 0
    for i in idx:
        a = read_crop(wsi, x[i], y[i], sx, sy)
        if a is None:
            failed += 1
            continue
        colours[classify(a)].append(a.reshape(-1, 3).mean(axis=0))
    parts = [f"{k} {len(v)} (RGB {np.mean(v, axis=0).round().astype(int).tolist()})"
             for k, v in sorted(colours.items(), key=lambda kv: -len(kv[1]))]
    return f"{len(idx)} sampled: " + ", ".join(parts + ([f"read failed {failed}"] if failed else []))


def overview(wsi, f: int) -> np.ndarray:
    """Level 0 averaged over f x f blocks, read in ~4096 px regions."""
    W, H = wsi.dimensions
    out = np.zeros((H // f, W // f, 3), dtype=np.uint8)
    step = f * max(1, 4096 // f)
    for y in range(0, H - f + 1, step):
        for x in range(0, W - f + 1, step):
            bw, bh = min(step, W - x) // f, min(step, H - y) // f
            a = read_crop(wsi, x, y, bw * f, bh * f)
            if a is not None:
                out[y // f:y // f + bh, x // f:x // f + bw] = a.reshape(bh, f, bw, f, 3).mean(axis=(1, 3))
    return out


def mark(img: np.ndarray, x, y, f: int, colour, r: int = 1) -> None:
    cx = np.clip(x // f, 0, img.shape[1] - 1)
    cy = np.clip(y // f, 0, img.shape[0] - 1)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            img[np.clip(cy + dy, 0, img.shape[0] - 1), np.clip(cx + dx, 0, img.shape[1] - 1)] = colour


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('csv', help="check_black_patches.py --csv output (wsi, row, cell_id)")
    p.add_argument('configs', nargs='+', help="extraction JSON config(s) / dirs that list the slides")
    p.add_argument('--out', default='flagged_cells')
    p.add_argument('--slides', nargs='*', help="only these slides (default: every slide in the CSV)")
    p.add_argument('--n', type=int, default=24, help="crops in the grid")
    p.add_argument('--sample', type=int, default=300, help="crops read for the content summary")
    p.add_argument('--no-overview', action='store_true', help="skip the whole-slide overview")
    args = p.parse_args(argv)

    flagged = defaultdict(set)
    with open(args.csv, newline='') as f:
        for row in csv.DictReader(f):
            flagged[row['wsi']].add(row['cell_id'])
    slides = {Path(dc['cells_info_path']).parent.name: dc
              for config in _expand(args.configs) for dc in resolve_slides(config)}
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    for name in (args.slides or sorted(flagged)):
        if name not in slides:
            print(f"{name}: not in the given configs")
            continue
        dc = slides[name]
        wsi = openslide.OpenSlide(dc['wsi_path'])
        ids, x, y, sx, sy = cell_crops(dc, wsi.dimensions)
        ids = np.array([i.decode() if isinstance(i, bytes) else str(i) for i in ids])
        is_flagged = np.isin(ids, list(flagged.get(name, ())))
        fl, other = np.flatnonzero(is_flagged), np.flatnonzero(~is_flagged)
        print(f"\n{name}: {len(fl):,} flagged / {len(ids):,} cells "
              f"({len(flagged.get(name, ())) - len(fl):,} flagged ids not found) | {dc['wsi_path']}")
        if not len(fl):
            continue
        print(f"  flagged crops   : {describe(wsi, rng.choice(fl, min(args.sample, len(fl)), replace=False), x, y, sx, sy)}")
        print(f"  unflagged crops : {describe(wsi, rng.choice(other, min(args.sample, len(other)), replace=False), x, y, sx, sy)}")
        fx, fy = x[fl] + sx // 2, y[fl] + sy // 2
        print(f"  flagged cell centres: x {fx.min():,}-{fx.max():,}, y {fy.min():,}-{fy.max():,} "
              f"(slide {wsi.dimensions[0]:,} x {wsi.dimensions[1]:,})")

        show = rng.choice(fl, min(args.n, len(fl)), replace=False)
        rows = -(-len(show) // GRID_COLS)
        grid = Image.new('RGB', (GRID_COLS * (sx + 4), rows * (sy + 4)), (255, 0, 255))
        for k, i in enumerate(show):
            a = read_crop(wsi, x[i], y[i], sx, sy)
            if a is not None:
                grid.paste(Image.fromarray(a), ((k % GRID_COLS) * (sx + 4), (k // GRID_COLS) * (sy + 4)))
        grid.save(os.path.join(args.out, f"{name}_crops.png"))

        if not args.no_overview:
            f = max(1, -(-max(wsi.dimensions) // MAX_SIDE))
            left = overview(wsi, f)
            right = left.copy()
            mark(right, x[other] + sx // 2, y[other] + sy // 2, f, (0, 200, 255), r=0)
            mark(right, fx, fy, f, (255, 0, 0), r=1)
            gap = np.full((left.shape[0], 8, 3), 255, dtype=np.uint8)
            Image.fromarray(np.concatenate([left, gap, right], axis=1)).save(
                os.path.join(args.out, f"{name}_overview.png"))
        print(f"  saved {args.out}/{name}_crops.png" + ("" if args.no_overview else f" and {name}_overview.png"))
    return 0


if __name__ == '__main__':
    sys.exit(main())
