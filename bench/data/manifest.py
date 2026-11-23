"""`bench manifest`: the one global cell manifest every model, readout and stage uses.

Per slide, {bench_root}/manifest/cells/{slide}.parquet holds every row of the slide's
patch_coordinates.h5 (row i = coordinates row i) with:
  cell_id, label_raw, label (mapped; "Unknown" for unmapped types), cx, cy (native centroid),
  ex, ey (extraction-space centroid), drop ("" | border | region | darkspot | smoke), labelled, extract.
`extract` = kept by the image filter and (labelled, or on a gene-expression slide);
classification uses extract & labelled.

Image filter: drop a cell when its rounded extraction position is within filters.border_px of
the image edge, outside the slide's `regions` box or within filters.border_px of its edges, or when a
dark/unreadable block lies within L-inf filters.linf_px (extraction px).
{bench_root}/manifest/slides.parquet summarises counts and the per-slide hash that embeddings
are validated against; manifest/class_counts.json gives per-cohort class counts (fold label
spaces).
"""
from __future__ import annotations

import hashlib
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from bench.config import BenchConfig
from bench.data.darkspots import load_coords, load_darkspots, smoke_rows
from bench.registry import UNKNOWN, Slide, genexp_slides
from bench.utils import LOG, atomic_path, print_table, read_json, write_json


def cells_path(cfg: BenchConfig, slide: str) -> Path:
    return cfg.root / "manifest" / "cells" / f"{slide}.parquet"


def slides_manifest_path(cfg: BenchConfig) -> Path:
    return cfg.root / "manifest" / "slides.parquet"


def _sat(mask: np.ndarray) -> np.ndarray:
    s = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    s[1:, 1:] = np.cumsum(np.cumsum(mask, axis=0, dtype=np.int64), axis=1)
    return s


def _box_any(sat: np.ndarray, y0, y1, x0, x1) -> np.ndarray:
    """Per cell: any True in rows y0..y1, cols x0..x1 (inclusive)."""
    tot = sat[y1 + 1, x1 + 1] - sat[y0, x1 + 1] - sat[y1 + 1, x0] + sat[y0, x0]
    return tot > 0


def linf_hits(mask: np.ndarray, cell: int, cx: np.ndarray, cy: np.ndarray, r: float) -> np.ndarray:
    """Per cell, whether a True cell-sized (`cell` px) block of `mask` intersects the L-inf box
    of radius r around (cx, cy)."""
    h, w = mask.shape
    x0 = np.clip(np.floor((cx - r) / cell), 0, w - 1).astype(np.int64)
    x1 = np.clip(np.floor((cx + r) / cell), 0, w - 1).astype(np.int64)
    y0 = np.clip(np.floor((cy - r) / cell), 0, h - 1).astype(np.int64)
    y1 = np.clip(np.floor((cy + r) / cell), 0, h - 1).astype(np.int64)
    return _box_any(_sat(mask), y0, y1, x0, x1)


def manifest_hash(df: pd.DataFrame, mask=None) -> str:
    """Hash of the cells an extraction covers (rows, positions, ids, labels)."""
    ext = df[df["extract"] if mask is None else mask]
    h = hashlib.sha256()
    h.update(ext.index.to_numpy(np.int64).tobytes())
    h.update(np.round(ext[["ex", "ey"]].to_numpy(np.float64), 4).tobytes())
    h.update("\n".join(ext["cell_id"].astype(str)).encode())
    h.update("\n".join(ext["label"].astype(str)).encode())
    return h.hexdigest()[:16]


def build_slide_manifest(cfg: BenchConfig, row: dict, genexp: bool, smoke: bool = False) -> pd.DataFrame:
    from src.python.code_configs.mappings import mapping_factory
    mapping = mapping_factory(cfg.mapping)
    fl = cfg.filters
    s = float(row["scale"])
    cx, cy, ids, types = load_coords(row["coords_path"], cfg.geometry.centre_offset_px)
    n = len(cx)
    label = np.array([mapping.get(t if isinstance(t, bytes) else str(t).encode(), UNKNOWN) for t in types],
                     dtype=object)
    df = pd.DataFrame({
        "cell_id": [c.decode() if isinstance(c, bytes) else str(c) for c in ids],
        "label_raw": [t.decode() if isinstance(t, bytes) else str(t) for t in types],
        "label": label.astype(str),
        "cx": cx, "cy": cy, "ex": cx * s, "ey": cy * s,
    })
    drop = np.full(n, "", dtype=object)
    if smoke:
        keep = np.zeros(n, dtype=bool)
        keep[smoke_rows(cx, cy, cfg.smoke.max_cells_per_slide, cfg.smoke.seed, int(row["width"]),
                        int(row["height"]), margin=(fl.border_px + 2 * fl.linf_px) / s)] = True
        drop[~keep] = "smoke"
    ew, eh = int(row["ext_width"]), int(row["ext_height"])
    rx = np.floor(df["ex"].to_numpy() + 0.5)
    ry = np.floor(df["ey"].to_numpy() + 0.5)
    border = (rx - fl.border_px < 0) | (ry - fl.border_px < 0) | (rx + fl.border_px > ew) | (ry + fl.border_px > eh)
    drop[(drop == "") & border] = "border"
    reg = cfg.regions.get(row["slide"])
    if reg is not None:                       # the box edges act like the image edge
        b, out = fl.border_px, np.zeros(n, dtype=bool)
        if reg.x_min is not None:
            out |= rx - b < reg.x_min * s
        if reg.x_max is not None:
            out |= rx + b > reg.x_max * s
        if reg.y_min is not None:
            out |= ry - b < reg.y_min * s
        if reg.y_max is not None:
            out |= ry + b > reg.y_max * s
        drop[(drop == "") & out] = "region"

    ds = load_darkspots(cfg, row["slide"])
    if ds is None:
        raise SystemExit(f"no dark-spot map for {row['slide']}: run `bench darkspots` first")
    meta = ds["meta"]
    r_native = fl.linf_px / s
    if abs(meta["r_native"] - r_native) > 1e-6 or meta["block"] != fl.block_px:
        raise SystemExit(f"{row['slide']}: dark-spot map was made with other filter settings; rerun `bench darkspots`")
    cand = drop == ""
    if cand.any():
        bad = ds["dark"] | ds["bad"]
        hit = linf_hits(bad, fl.block_px, cx[cand], cy[cand], r_native)
        # every kept cell's box must lie in scanned tiles, otherwise the map is incomplete
        unscanned = linf_hits(~ds["scanned"], meta["tile"], cx[cand], cy[cand], r_native)
        if (unscanned & ~hit).any():
            raise SystemExit(f"{row['slide']}: dark-spot map does not cover {int((unscanned & ~hit).sum())} cells "
                             f"(made for another cell set?) -- rerun `bench darkspots`")
        idx = np.flatnonzero(cand)
        drop[idx[hit]] = "darkspot"
    df["drop"] = drop.astype(str)
    df["labelled"] = df["label"] != UNKNOWN
    df["extract"] = (df["drop"] == "") & (df["labelled"] | genexp)
    dup = df.loc[df["extract"], "cell_id"].duplicated()
    if dup.any():
        raise SystemExit(f"{row['slide']}: {int(dup.sum())} duplicate cell ids among extracted cells")
    df.index.name = "row"
    return df


def _manifest_job(args):
    cfg_dump, row, genexp, smoke = args
    cfg = BenchConfig.model_validate(cfg_dump)
    df = build_slide_manifest(cfg, row, genexp, smoke)
    out = cells_path(cfg, row["slide"])
    with atomic_path(out) as tmp:
        df.to_parquet(tmp)
    counts = df.loc[df["extract"] & df["labelled"], "label"].value_counts().to_dict()
    return {
        "slide": row["slide"], "cohort": row["cohort"], "n_total": int(len(df)),
        "n_labelled": int(df["labelled"].sum()),
        "n_border": int((df["drop"] == "border").sum()), "n_region": int((df["drop"] == "region").sum()),
        "n_darkspot": int((df["drop"] == "darkspot").sum()),
        "n_extract": int(df["extract"].sum()), "n_classify": int((df["extract"] & df["labelled"]).sum()),
        "genexp": genexp, "hash": manifest_hash(df),
        "hash_classify": manifest_hash(df, df["extract"] & df["labelled"]), "class_counts": counts,
    }


def check_regions(cfg: BenchConfig, known) -> None:
    """Every `regions` entry names a known slide and a non-empty box."""
    bad = sorted(set(cfg.regions) - set(known) - set(cfg.exclude))
    if bad:
        raise SystemExit(f"regions: unknown slide(s) {', '.join(bad)} (check the names in configs/bench.yaml)")
    for name, r in cfg.regions.items():
        for lo, hi in ((r.x_min, r.x_max), (r.y_min, r.y_max)):
            if lo is not None and hi is not None and lo >= hi:
                raise SystemExit(f"regions: {name} is empty ({lo} >= {hi})")


def run_manifest(cfg: BenchConfig, slides: List[Slide], inspect: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    from bench.registry import load_slides
    check_regions(cfg, {s.slide for s in load_slides(cfg)} | set(inspect.index))
    gx = {s.slide for ss in genexp_slides(cfg, slides).values() for s in ss}
    jobs = []
    for s in slides:
        r = inspect.loc[s.slide].to_dict()
        if r["status"] != "ok":
            raise SystemExit(f"{s.slide}: inspect status {r['status']} -- fix it before building the manifest")
        jobs.append((cfg.dump(), r, s.slide in gx, cfg.is_smoke))
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        rows = list(ex.map(_manifest_job, jobs))
    new = pd.DataFrame(rows)
    out = slides_manifest_path(cfg)
    if out.exists():
        old = pd.read_parquet(out)
        new = pd.concat([old[~old.slide.isin(new.slide)], new], ignore_index=True)
    new = new.sort_values(["cohort", "slide"]).reset_index(drop=True)
    new["class_counts"] = new["class_counts"].map(lambda d: {k: int(v) for k, v in d.items()} if isinstance(d, dict) else d)
    with atomic_path(out) as tmp:
        new.drop(columns=["class_counts"]).to_parquet(tmp, index=False)
    # per-cohort class counts (fold label spaces)
    cc: Dict[str, Dict[str, int]] = {}
    per_slide = read_json(cfg.root / "manifest" / "class_counts_by_slide.json", {}) or {}
    for r in new.itertuples():
        if isinstance(r.class_counts, dict):
            per_slide[r.slide] = r.class_counts
    for r in new.itertuples():
        for k, v in per_slide.get(r.slide, {}).items():
            cc.setdefault(r.cohort, {}).setdefault(k, 0)
            cc[r.cohort][k] += int(v)
    write_json(cfg.root / "manifest" / "class_counts_by_slide.json", per_slide)
    write_json(cfg.root / "manifest" / "class_counts.json", cc)
    shown = new[new.slide.isin([s.slide for s in slides])]
    cols = [c for c in ("n_total", "n_labelled", "n_border", "n_region", "n_darkspot", "n_extract", "n_classify")
            if c in shown]
    agg = shown.groupby("cohort")[cols].sum()
    agg.loc["TOTAL"] = agg.sum()
    print_table([{"cohort": k, **{c: f"{int(v):,}" for c, v in r.items()}} for k, r in agg.iterrows()])
    LOG.info("wrote %s", out)
    return new


# ── loaders ──

def load_slides_manifest(cfg: BenchConfig) -> pd.DataFrame:
    p = slides_manifest_path(cfg)
    if not p.exists():
        raise SystemExit(f"{p} not found: run `bench manifest` first")
    return pd.read_parquet(p).set_index("slide", drop=False)


def load_cells(cfg: BenchConfig, slide: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_parquet(cells_path(cfg, slide), columns=columns)


def load_class_counts(cfg: BenchConfig) -> Dict[str, Dict[str, int]]:
    cc = read_json(cfg.root / "manifest" / "class_counts.json")
    if cc is None:
        raise SystemExit("manifest/class_counts.json not found: run `bench manifest` first")
    return cc


def extraction_cells(cfg: BenchConfig, slide: str) -> pd.DataFrame:
    df = load_cells(cfg, slide, ["cell_id", "label", "ex", "ey", "extract", "labelled"])
    return df[df["extract"]]


def ext_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return int(math.floor(width * scale + 1e-9)), int(math.floor(height * scale + 1e-9))
