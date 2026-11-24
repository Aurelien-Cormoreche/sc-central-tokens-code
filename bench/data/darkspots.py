"""`bench darkspots`: per-slide map of dark (no-data) and unreadable areas.

A block of filters.block_px x block_px native pixels is *dark* when at least block_full_frac of
its pixels are no-data (max RGB <= black_max, or fully transparent), and *bad* when it lies in a
storage tile that cannot be decoded. Only the scan tiles within L-inf linf_px (extraction px) of
some cell are read, which is exact for the manifest's filter (bench/data/manifest.py) and skips
empty glass. Reads use scan_tile_px areas; when one fails, that area is re-read storage tile by
storage tile so only the corrupt tiles are marked bad.

Output per slide: {bench_root}/darkspots/{slide}.npz (+ .png review image).
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from bench.config import BenchConfig
from bench.data.reader import ReadError, WSIReader
from bench.data.remap import read_positions, remap_for, remap_key
from bench.utils import LOG, atomic_path, file_fingerprint, fmt_hours


def darkspot_path(cfg: BenchConfig, slide: str) -> Path:
    return cfg.root / "darkspots" / f"{slide}.npz"


def load_coords(coords_path: str, centre_offset: float):
    import h5py
    with h5py.File(coords_path, "r") as f:
        cx = f["x_start"][()].astype(np.float64) + centre_offset
        cy = f["y_start"][()].astype(np.float64) + centre_offset
        ids = f["cell_id"][()]
        types = f["cell_type"][()]
    return cx, cy, ids, types


def smoke_rows(cx: np.ndarray, cy: np.ndarray, n: int, seed: int, width: int, height: int,
               margin: float) -> np.ndarray:
    """A spatially compact, deterministic set of n cells: the n nearest to a random cell that is
    at least `margin` native px from the image edge."""
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero((cx > margin) & (cy > margin) & (cx < width - margin) & (cy < height - margin))
    pool = ok if len(ok) else np.arange(len(cx))
    if len(pool) == 0:
        return np.zeros(0, dtype=np.int64)
    c = pool[rng.integers(len(pool))]
    d = np.maximum(np.abs(cx - cx[c]), np.abs(cy - cy[c]))
    k = min(n, len(cx))
    return np.sort(np.argpartition(d, k - 1)[:k]).astype(np.int64)


def needed_tiles(cx: np.ndarray, cy: np.ndarray, r_native: float, width: int, height: int,
                 tile: int) -> np.ndarray:
    """(th, tw) bool: scan tiles intersecting some cell's L-inf box of radius r_native."""
    th, tw = -(-height // tile), -(-width // tile)
    need = np.zeros((th, tw), dtype=bool)
    if len(cx) == 0:
        return need
    x0 = np.clip(np.floor((cx - r_native) / tile), 0, tw - 1).astype(np.int64)
    x1 = np.clip(np.floor((cx + r_native) / tile), 0, tw - 1).astype(np.int64)
    y0 = np.clip(np.floor((cy - r_native) / tile), 0, th - 1).astype(np.int64)
    y1 = np.clip(np.floor((cy + r_native) / tile), 0, th - 1).astype(np.int64)
    span = int(max((x1 - x0).max(), (y1 - y0).max()))
    for dy in range(span + 1):
        for dx in range(span + 1):
            yy, xx = y0 + dy, x0 + dx
            ok = (yy <= y1) & (xx <= x1)
            need[yy[ok], xx[ok]] = True
    return need


def _block_reduce_nodata(rgba: np.ndarray, block: int, black_max: int) -> np.ndarray:
    """Fraction of no-data pixels per block; rgba shape must be a multiple of block."""
    nod = rgba[..., :3].max(axis=-1) <= black_max
    if rgba.shape[-1] == 4:
        nod |= rgba[..., 3] == 0
    h, w = nod.shape
    return nod.reshape(h // block, block, w // block, block).mean(axis=(1, 3))


def scan_slide(reader: WSIReader, need: np.ndarray, tile: int, block: int, black_max: int,
               full_frac: float, storage_tile: int) -> dict:
    W, H = reader.width, reader.height
    bh, bw = -(-H // block), -(-W // block)
    frac = np.zeros((bh, bw), dtype=np.float32)
    bad = np.zeros((bh, bw), dtype=bool)
    scanned = np.zeros_like(need)
    n_bad_storage = 0
    for ty, tx in zip(*np.nonzero(need)):
        x, y = int(tx) * tile, int(ty) * tile
        w, h = min(tile, W - x), min(tile, H - y)
        # pad the read to whole blocks; pixels beyond the image read as transparent and are
        # not used (frac of blocks straddling the edge only counts in-image pixels below)
        wp, hp = -(-w // block) * block, -(-h // block) * block
        b0y, b0x = y // block, x // block
        try:
            rgba = reader.read_native_rgba(x, y, wp, hp)
            f = _block_reduce_nodata(rgba, block, black_max)
            frac[b0y:b0y + f.shape[0], b0x:b0x + f.shape[1]] = f
        except ReadError:
            # localise: storage tile by storage tile
            st = max(block, (storage_tile // block) * block)
            for sy in range(y, y + h, st):
                for sx in range(x, x + w, st):
                    sw, sh = min(st, x + w - sx), min(st, y + h - sy)
                    swp, shp = -(-sw // block) * block, -(-sh // block) * block
                    by, bx = sy // block, sx // block
                    try:
                        rgba = reader.read_native_rgba(sx, sy, swp, shp)
                        f = _block_reduce_nodata(rgba, block, black_max)
                        frac[by:by + f.shape[0], bx:bx + f.shape[1]] = f
                    except ReadError:
                        n_bad_storage += 1
                        bad[by:by + shp // block, bx:bx + swp // block] = True
        scanned[ty, tx] = True
    # the edge column/row of blocks may include out-of-image (transparent) pixels; the border
    # filter removes cells near the edge anyway, so edge blocks are not reported as dark
    dark = frac >= full_frac
    if W % block:
        dark[:, -1] = False
    if H % block:
        dark[-1, :] = False
    return {"dark": dark, "bad": bad, "scanned": scanned, "n_bad_storage_tiles": n_bad_storage}


def remove_small_blobs(mask: np.ndarray, min_blocks: int) -> np.ndarray:
    if min_blocks <= 1 or not mask.any():
        return mask
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_blocks
    keep[0] = False
    return keep[lab]


def save_png(path: Path, dark: np.ndarray, bad: np.ndarray, scanned: np.ndarray, block: int, tile: int,
             max_px: int = 2000) -> None:
    from PIL import Image
    bh, bw = dark.shape
    f = max(1, int(math.ceil(max(bh, bw) / max_px)))
    sh, sw = -(-bh // f), -(-bw // f)

    def pool(m: np.ndarray) -> np.ndarray:
        p = np.zeros((sh * f, sw * f), dtype=bool)
        p[:bh, :bw] = m
        return p.reshape(sh, f, sw, f).any(axis=(1, 3))

    rep = tile // block
    sc = np.repeat(np.repeat(scanned, rep, axis=0), rep, axis=1)[:bh, :bw]
    img = np.full((sh, sw, 3), 255, dtype=np.uint8)
    img[pool(sc)] = (225, 225, 225)
    img[pool(dark)] = (220, 30, 30)
    img[pool(bad)] = (30, 60, 220)
    with atomic_path(path) as tmp:
        Image.fromarray(img).save(tmp, format="PNG")


def darkspots_one(cfg: BenchConfig, row: dict, smoke: bool = False) -> dict:
    t0 = time.time()
    fl = cfg.filters
    s = float(row["scale"])
    cx, cy, _, _ = load_coords(row["coords_path"], cfg.geometry.centre_offset_px)
    r_native = fl.linf_px / s
    if smoke:
        rows = smoke_rows(cx, cy, cfg.smoke.max_cells_per_slide, cfg.smoke.seed, int(row["width"]),
                          int(row["height"]), margin=(fl.border_px + 2 * fl.linf_px) / s)
        cx, cy = cx[rows], cy[rows]
    rm = remap_for(cfg, row["slide"])
    cx, cy, _ = read_positions(rm, cx, cy)                      # scan where the cells are read
    reader = WSIReader(row["wsi_path"], scale=s)
    try:
        need = needed_tiles(cx, cy, r_native, reader.width, reader.height, fl.scan_tile_px)
        storage_tile = 1024
        try:
            import openslide
            sl = openslide.OpenSlide(row["wsi_path"])
            storage_tile = int(sl.properties.get("openslide.level[0].tile-width", 1024))
            sl.close()
        except Exception:
            pass
        res = scan_slide(reader, need, fl.scan_tile_px, fl.block_px, fl.black_max, fl.block_full_frac,
                         storage_tile)
    finally:
        reader.close()
    dark = remove_small_blobs(res["dark"], fl.min_blob_blocks)
    out = darkspot_path(cfg, row["slide"])
    meta = dict(slide=row["slide"], wsi_path=row["wsi_path"], wsi_fp=file_fingerprint(row["wsi_path"]),
                width=reader.width, height=reader.height,
                block=fl.block_px, tile=fl.scan_tile_px, black_max=fl.black_max, full_frac=fl.block_full_frac,
                min_blob_blocks=fl.min_blob_blocks, r_native=r_native, linf_px=fl.linf_px, scale=s,
                n_cells_considered=int(len(cx)), smoke=smoke, remap=remap_key(rm))
    with atomic_path(out) as tmp:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, dark=np.packbits(dark), bad=np.packbits(res["bad"]),
                                shape=np.array(dark.shape), scanned=res["scanned"],
                                meta=np.array([repr(meta)]))
    save_png(out.with_suffix(".png"), dark, res["bad"], res["scanned"], fl.block_px, fl.scan_tile_px)
    stats = dict(slide=row["slide"], tiles_scanned=int(res["scanned"].sum()), tiles_total=int(res["scanned"].size),
                 dark_blocks=int(dark.sum()), bad_blocks=int(res["bad"].sum()),
                 bad_storage_tiles=res["n_bad_storage_tiles"], seconds=round(time.time() - t0, 1))
    LOG.info("darkspots %s: %d/%d tiles scanned, %d dark blocks, %d bad blocks (%s)", row["slide"],
             stats["tiles_scanned"], stats["tiles_total"], stats["dark_blocks"], stats["bad_blocks"],
             fmt_hours(stats["seconds"] / 3600))
    return stats


def load_darkspots(cfg: BenchConfig, slide: str) -> Optional[dict]:
    p = darkspot_path(cfg, slide)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    shape = tuple(int(v) for v in z["shape"])
    n = shape[0] * shape[1]
    dark = np.unpackbits(z["dark"])[:n].reshape(shape).astype(bool)
    bad = np.unpackbits(z["bad"])[:n].reshape(shape).astype(bool)
    import ast
    meta = ast.literal_eval(str(z["meta"][0]))
    return {"dark": dark, "bad": bad, "scanned": z["scanned"].astype(bool), "meta": meta}


def plan_costs(rows: List[dict]) -> List[float]:
    """Relative scan cost per slide (for LPT packing of CPU tasks): level-0 area."""
    return [float(r.get("width", 0)) * float(r.get("height", 0)) for r in rows]


# ── jobs ──

def plan_jobs(cfg: BenchConfig, slides, inspect) -> List[dict]:
    """One CPU job per slide. Estimate: every 4096^2 scan tile ~0.6 s (an upper bound -- only tiles
    near cells are read)."""
    jobs = []
    for s in slides:
        r = inspect.loc[s.slide].to_dict()
        if r["status"] != "ok":
            raise SystemExit(f"{s.slide}: inspect status {r['status']} -- fix it before the dark-spot scan")
        row = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()
               if k in ("slide", "wsi_path", "coords_path", "scale", "width", "height")}
        tiles = (float(r["width"]) / cfg.filters.scan_tile_px + 1) * (float(r["height"]) / cfg.filters.scan_tile_px + 1)
        jobs.append({"kind": "darkspots", "key": f"darkspots/{s.slide}", "group": "darkspots", "slide": s.slide,
                     "row": row, "n_cells": int(r.get("n_cells", 0)), "est_s": 30 + 0.6 * tiles,
                     "out": str(darkspot_path(cfg, s.slide)), "out_gb": 0.0,
                     "params": cfg.filters.model_dump()})
    return jobs


def map_matches_image(meta: dict, map_path, wsi_path: str, fp: Optional[str] = None) -> bool:
    """The map was made from the image file as it is now: same content fingerprint, or, for maps made before
    fingerprints were recorded, the image has not been replaced since the map was written."""
    if "wsi_fp" in meta:
        return meta["wsi_fp"] == (fp or file_fingerprint(wsi_path))
    return os.path.getmtime(map_path) >= os.path.getmtime(wsi_path)


def job_done(cfg: BenchConfig, job: dict) -> bool:
    """Done = the map exists and was made with the current filter parameters, the slide's current scale, `remap`
    setting and image file."""
    p = Path(job["out"])
    if not p.exists():
        return False
    try:
        import ast
        meta = ast.literal_eval(str(np.load(p, allow_pickle=False)["meta"][0]))
    except Exception:
        return False
    fl = cfg.filters
    r_native = fl.linf_px / float(job["row"]["scale"])         # the scale changes the radius in native px
    return (abs(float(meta.get("r_native", -1.0)) - r_native) < 1e-6
            and meta.get("block") == fl.block_px and meta.get("tile") == fl.scan_tile_px
            and meta.get("black_max") == fl.black_max and meta.get("full_frac") == fl.block_full_frac
            and meta.get("min_blob_blocks") == fl.min_blob_blocks and meta.get("linf_px") == fl.linf_px
            and bool(meta.get("smoke")) == cfg.is_smoke
            and meta.get("remap") == remap_key(remap_for(cfg, job.get("slide") or job["row"].get("slide", "")))
            and map_matches_image(meta, p, job["row"]["wsi_path"]))


def _one(args):
    cfg, job = args
    try:
        return job["key"], "done", darkspots_one(cfg, job["row"], cfg.is_smoke), None
    except Exception as e:  # reported per slide by the parent
        import traceback
        return job["key"], "failed", {}, (f"{type(e).__name__}: {e}", traceback.format_exc()[-4000:])


def run_darkspots_bin(cfg: BenchConfig, jobs: List[dict], device: str = "cpu") -> int:
    """Slides of a bin in parallel processes (one WSI handle each)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    from bench.jobs import set_status
    by_key = {j["key"]: j for j in jobs}
    for j in jobs:
        set_status(cfg, j, "running")
    n = max(1, min(len(jobs), cfg.slurm.cpus, os.cpu_count() or 1))
    failed = 0
    with ProcessPoolExecutor(max_workers=n) as ex:
        for key, state, info, err in ex.map(_one, [(cfg, j) for j in jobs]):
            if state == "done":
                set_status(cfg, by_key[key], "done", **info)
            else:
                failed += 1
                LOG.error("%s FAILED: %s", key, err[0])
                set_status(cfg, by_key[key], "failed", error=err[0], traceback=err[1])
    return failed
