"""Look at the cells check_black_patches.py flagged: what their crops contain and where they are.

check_black_patches.py finds groups of identical embeddings, i.e. identical crops. Those
are black for a failed read, but saturated white glass or a flat fill colour give identical
crops too. This reads crops from the WSI the extraction read (resolved from an extraction
JSON config, like check_wsi_regions) and, for every slide of the configs:
  - prints what a random sample of flagged crops contains (black / white / flat / tissue,
    with each kind's mean colour), next to the same for unflagged cells;
  - saves {out}/{slide}_crops.png: a grid of flagged crops (slides with flagged cells);
  - saves {out}/{slide}_overview.png: the slide downscaled to <= MAX_SIDE px; left as is,
    right with every cell in cyan and the flagged cells in red;
then a table of all slides. The overview reads each slide once in full (minutes for a large
one), so use a compute node and --workers. CPU-only (openslide, h5py, numpy, Pillow).

Usage (repo root, .env sourced; CSV from check_black_patches.py --csv):
    python scripts/show_flagged_cells.py split_black_cells.csv \
        configs/extract_embeddings/paper_uni2_fov/uni2_fov224_split.json --out flagged_cells --workers 8
    ... --slides GSM9509144_NSCLC_chuvio_L4_L4      # only these slides
    ... --flagged-only                              # only slides with flagged cells
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def read_crop(wsi, x, y, sx, sy) -> np.ndarray | None:
    try:
        return np.asarray(wsi.read_region((int(x), int(y)), 0, (int(sx), int(sy))).convert('RGB'))
    except openslide.OpenSlideError:
        return None


def describe(wsi, idx, x, y, sx, sy) -> tuple[str, dict]:
    """What the crops of cells idx contain: (text with count and mean RGB per kind, {kind: count})."""
    colours, failed = defaultdict(list), 0
    for i in idx:
        a = read_crop(wsi, x[i], y[i], sx, sy)
        if a is None:
            failed += 1
            continue
        colours[classify(a)].append(a.reshape(-1, 3).mean(axis=0))
    kinds = {k: len(v) for k, v in sorted(colours.items(), key=lambda kv: -len(kv[1]))}
    if failed:
        kinds['read failed'] = failed
    parts = [f"{k} {len(colours[k])} (RGB {np.mean(colours[k], axis=0).round().astype(int).tolist()})"
             if k in colours else f"{k} {n}" for k, n in kinds.items()]
    return f"{len(idx)} sampled: " + ", ".join(parts), kinds


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


def process(name: str, dc: dict, flagged_ids: set, out: str, n: int, sample: int, do_overview: bool):
    """One slide: (lines to print, summary row)."""
    wsi = openslide.OpenSlide(dc['wsi_path'])
    ids, x, y, sx, sy = cell_crops(dc, wsi.dimensions)
    ids = np.array([i.decode() if isinstance(i, bytes) else str(i) for i in ids])
    is_flagged = np.isin(ids, list(flagged_ids))
    fl, other = np.flatnonzero(is_flagged), np.flatnonzero(~is_flagged)
    rng = np.random.default_rng(0)
    not_found = len(flagged_ids) - len(fl)
    lines = [f"{name}: {len(fl):,} flagged / {len(ids):,} cells"
             + (f" ({not_found:,} flagged ids not in the slide's cells)" if not_found else "")
             + f" | {dc['wsi_path']}"]
    kinds = {}
    if len(fl):
        text, kinds = describe(wsi, rng.choice(fl, min(sample, len(fl)), replace=False), x, y, sx, sy)
        lines.append(f"  flagged crops   : {text}")
    if len(other):
        text, _ = describe(wsi, rng.choice(other, min(sample, len(other)), replace=False), x, y, sx, sy)
        lines.append(f"  unflagged crops : {text}")
    fx, fy = x[fl] + sx // 2, y[fl] + sy // 2
    saved = []
    if len(fl):
        lines.append(f"  flagged cell centres: x {fx.min():,}-{fx.max():,}, y {fy.min():,}-{fy.max():,} "
                     f"(slide {wsi.dimensions[0]:,} x {wsi.dimensions[1]:,})")
        show = rng.choice(fl, min(n, len(fl)), replace=False)
        rows = -(-len(show) // GRID_COLS)
        grid = Image.new('RGB', (GRID_COLS * (sx + 4), rows * (sy + 4)), (255, 0, 255))
        for k, i in enumerate(show):
            a = read_crop(wsi, x[i], y[i], sx, sy)
            if a is not None:
                grid.paste(Image.fromarray(a), ((k % GRID_COLS) * (sx + 4), (k // GRID_COLS) * (sy + 4)))
        grid.save(os.path.join(out, f"{name}_crops.png"))
        saved.append(f"{name}_crops.png")
    if do_overview:
        f = max(1, -(-max(wsi.dimensions) // MAX_SIDE))
        left = overview(wsi, f)
        right = left.copy()
        mark(right, x[other] + sx // 2, y[other] + sy // 2, f, (0, 200, 255), r=0)
        if len(fl):
            mark(right, fx, fy, f, (255, 0, 0), r=1)
        gap = np.full((left.shape[0], 8, 3), 255, dtype=np.uint8)
        Image.fromarray(np.concatenate([left, gap, right], axis=1)).save(os.path.join(out, f"{name}_overview.png"))
        saved.append(f"{name}_overview.png")
    if saved:
        lines.append(f"  saved {', '.join(saved)}")
    kinds_text = ", ".join(f"{k} {v}" for k, v in kinds.items()) or '-'
    return lines, (name, len(ids), len(fl), kinds_text)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('csv', help="check_black_patches.py --csv output (wsi, row, cell_id)")
    p.add_argument('configs', nargs='+', help="extraction JSON config(s) / dirs that list the slides")
    p.add_argument('--out', default='flagged_cells')
    p.add_argument('--slides', nargs='*', help="only these slides (default: every slide in the configs)")
    p.add_argument('--flagged-only', action='store_true', help="only slides with flagged cells")
    p.add_argument('--n', type=int, default=24, help="crops in the grid")
    p.add_argument('--sample', type=int, default=300, help="crops read for each content summary")
    p.add_argument('--no-overview', action='store_true', help="skip the whole-slide overview")
    p.add_argument('--workers', type=int, default=1, help="slides processed in parallel")
    args = p.parse_args(argv)

    flagged = defaultdict(set)
    with open(args.csv, newline='') as f:
        for row in csv.DictReader(f):
            flagged[row['wsi']].add(row['cell_id'])
    slides = {Path(dc['cells_info_path']).parent.name: dc
              for config in _expand(args.configs) for dc in resolve_slides(config)}
    names = args.slides or sorted(slides)
    if args.flagged_only:
        names = [s for s in names if flagged.get(s)]
    for s in sorted(set(flagged) - set(slides)):
        print(f"{s}: has flagged cells but is not in the given configs")
    missing = [s for s in names if s not in slides]
    for s in missing:
        print(f"{s}: not in the given configs")
    names = [s for s in names if s in slides]
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(names)} slide(s), {args.workers} worker(s), output in {args.out}/", flush=True)

    summary = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, s, slides[s], flagged.get(s, set()), args.out, args.n, args.sample,
                               not args.no_overview): s for s in names}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                lines, row = fut.result()
            except Exception as e:
                lines, row = [f"{s}: ERROR {type(e).__name__}: {e}"], (s, 0, 0, 'error')
            print("\n" + "\n".join(lines), flush=True)
            summary.append(row)

    print(f"\n{'slide':<40} {'cells':>9} {'flagged':>9} {'%':>6}  flagged crops (sample)")
    for name, n_cells, n_flagged, kinds in sorted(summary):
        pct = f"{100 * n_flagged / n_cells:.2f}" if n_cells else '-'
        print(f"{name:<40} {n_cells:>9,} {n_flagged:>9,} {pct:>6}  {kinds}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
