"""Extraction jobs: one per (stage, model, variant, slide), costed for LPT packing.

Variants
  stage 1 : native        -- N = model input_px, every extracted cell (incl. unlabelled cells on
                             gene-expression slides), readouts cls + central
  stage 2a: fov{F:03d}    -- N = F rounded to the patch size (stage2.round_to_patch), labelled cells
                             of stage2.cohorts; F == input_px reuses the stage-1 file
  stage 2b: mc{T}         -- multicell tiles of T px (stage2.multicell.tile_px), central readout only
Cost = cells / throughput(model, N) + per-slide overhead; throughput from calibration.json
(written by `bench smoke`), else the config prior scaled by (input_px / N)^2.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from bench.config import BenchConfig
from bench.extract.store import embeddings_path
from bench.models.encoders import round_to_patch
from bench.registry import Slide, stage2_slides
from bench.utils import read_json

# (standard embedding dim, token dim) per model -- for disk estimates only
DIMS = {"uni2": (1536, 1536), "conch15": (768, 1024), "virchow2": (2560, 1280), "hoptimus1": (1536, 1536),
        "lemon": (384, 384), "genbio": (4608, 4608), "tiny": (32, 32)}


def calibration(cfg: BenchConfig) -> dict:
    # written by `bench calibrate` / `bench smoke` to the main bench_root (smoke runs use it too)
    root = cfg.root.parent if cfg.is_smoke else cfg.root
    return read_json(root / "calibration.json", {}) or {}


def cells_per_s(cfg: BenchConfig, model: str, n: int) -> float:
    cal = calibration(cfg).get(model, {})
    if str(n) in cal:
        return float(cal[str(n)])
    m = cfg.models[model]
    factor = (m.input_px / n) ** 2
    return m.cells_per_s * min(max(factor, 1 / 64), 8.0)


def stage_variants(cfg: BenchConfig, stage: str, model: str) -> List[Tuple[str, int, Optional[Tuple[str, str]]]]:
    """[(variant, N, alias)] -- alias = (stage, variant) of an existing file to reuse instead."""
    m = cfg.models[model]
    if stage == "1":
        return [("native", m.input_px, None)]
    if stage == "2a":
        out = []
        for f in cfg.stage2.fov_px:
            n = round_to_patch(f, m.patch, cfg.stage2.round_to_patch)
            out.append((f"fov{f:03d}", n, ("1", "native") if n == m.input_px else None))
        return out
    if stage == "2b":
        t = cfg.stage2.multicell.tile_px
        if t is None:
            raise SystemExit("stage2.multicell.tile_px is not set (Gate 2): pick 224 or 448 from the 2a results "
                             "(`bench report`, Overview > Gate 2) and set it in the config")
        return [(f"mc{t}", t, None)]
    raise ValueError(f"unknown stage {stage!r}")


def resolve_path(cfg: BenchConfig, stage: str, model: str, variant: str, slide: str) -> Path:
    """Where the embeddings of (stage, model, variant) live for a slide, following 2a aliases."""
    if stage == "2a":
        for v, _, alias in stage_variants(cfg, "2a", model):
            if v == variant and alias:
                return embeddings_path(cfg.root, alias[0], model, alias[1], slide)
    return embeddings_path(cfg.root, stage, model, variant, slide)


def cell_set(stage: str) -> str:
    return "extract" if stage == "1" else "classify"


def readouts_for(cfg: BenchConfig, stage: str) -> Tuple[str, ...]:
    return ("central",) if stage == "2b" else tuple(cfg.readouts)


@functools.lru_cache(maxsize=4)
def _slides_manifest(root: str, mtime: float):
    import pandas as pd
    return pd.read_parquet(Path(root) / "manifest" / "slides.parquet").set_index("slide", drop=False)


def slides_manifest(cfg: BenchConfig):
    p = cfg.root / "manifest" / "slides.parquet"
    if not p.exists():
        raise SystemExit(f"{p} not found: run `bench manifest` first")
    return _slides_manifest(str(cfg.root), p.stat().st_mtime)


def expected(cfg: BenchConfig, stage: str, slide: str) -> Tuple[int, str]:
    """(rows, manifest hash) an embedding file of this stage must match (current manifest)."""
    r = slides_manifest(cfg).loc[slide]
    if cell_set(stage) == "extract":
        return int(r["n_extract"]), str(r["hash"])
    return int(r["n_classify"]), str(r["hash_classify"])


def _n_tiles(cfg: BenchConfig, slide: str, t: int) -> int:
    from bench.data.manifest import load_cells
    df = load_cells(cfg, slide, ["ex", "ey", "extract", "labelled"])
    df = df[df["extract"] & df["labelled"]]
    s = int(round(t * cfg.stage2.multicell.central_frac))
    return len({(int(a), int(b)) for a, b in zip(np.floor(df["ex"] / s), np.floor(df["ey"] / s))})


def plan_jobs(cfg: BenchConfig, stage: str, models: List[str], slides: List[Slide]) -> List[dict]:
    if stage in ("2a", "2b"):
        slides = stage2_slides(cfg, slides)
    jobs = []
    for model in models:
        for variant, n, alias in stage_variants(cfg, stage, model):
            if alias:
                continue  # reuses the stage-1 file (status checks it)
            cps = cells_per_s(cfg, model, n)
            for s in slides:
                rows, h = expected(cfg, stage, s.slide)
                if stage == "2b":
                    units = _n_tiles(cfg, s.slide, n)
                    est = units / cps + cfg.extract.slide_overhead_s
                else:
                    est = rows / cps + cfg.extract.slide_overhead_s
                d_std, d_tok = DIMS.get(model, (1024, 1024))
                width = d_tok if stage == "2b" else d_std + d_tok
                jobs.append({
                    "kind": "extract", "key": f"extract/{stage}/{model}/{variant}/{s.slide}",
                    "group": f"{model}/{variant}", "stage": stage, "model": model, "variant": variant,
                    "input_px": n, "slide": s.slide, "cohort": s.cohort, "n_cells": rows, "rows": rows, "hash": h,
                    "readouts": list(readouts_for(cfg, stage)), "multicell": stage == "2b",
                    "est_s": est, "out_gb": rows * width * 2 / 1e9,
                    "out": str(embeddings_path(cfg.root, stage, model, variant, s.slide)),
                })
    return jobs
