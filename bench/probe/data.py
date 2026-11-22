"""Load one readout of many slides into a single float16 host tensor (labelled cells only).

Rows are kept in each file's order; `slide_idx`, `mrow` (manifest row) and `label` identify every
row, so folds and Stage-3 subsets are defined on (slide, manifest row) -- independent of the
model, whose extraction order depends on its crop size.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

from bench.config import BenchConfig
from bench.extract import planner
from bench.extract.store import is_valid
from bench.registry import UNKNOWN, Slide
from bench.utils import LOG, Timer

_BLOCK = 65536


@dataclass
class Loaded:
    emb: torch.Tensor           # (N, D) float16, CPU
    slide_idx: np.ndarray       # (N,) index into `slides`
    mrow: np.ndarray            # (N,) manifest row
    label: np.ndarray           # (N,) global class index into `classes`
    slides: List[Slide]
    classes: List[str]

    @property
    def slide_names(self) -> List[str]:
        return [s.slide for s in self.slides]


def _paths(cfg: BenchConfig, stage: str, model: str, variant: str, readout: str, slides: List[Slide]):
    """(path, file stage) per slide, after 2a aliasing; raises if any file is missing/invalid."""
    out, bad = [], []
    for s in slides:
        p = planner.resolve_path(cfg, stage, model, variant, s.slide)
        file_stage = "1" if "stage1" in p.parts else stage
        rows, h = planner.expected(cfg, file_stage, s.slide)
        ok, why = is_valid(p, rows, h, (readout,))
        if not ok:
            bad.append(f"{s.slide}: {why}")
        out.append(p)
    if bad:
        raise SystemExit(f"{len(bad)} embedding file(s) for {model}/{variant}/{readout} (stage {stage}) are not "
                         f"ready -- run `bench extract` first:\n  " + "\n  ".join(bad[:20]))
    return out


def load(cfg: BenchConfig, stage: str, model: str, variant: str, readout: str, slides: List[Slide],
         classes: List[str]) -> Loaded:
    import h5py
    t = Timer()
    paths = _paths(cfg, stage, model, variant, readout, slides)
    code = {c: i for i, c in enumerate(classes)}
    key = f"embeddings_{readout}"

    def first(p: Path):
        with h5py.File(p, "r") as f:
            lab = np.array([x.decode() for x in f["labels"][()]])
            keep = np.array([l != UNKNOWN and l in code for l in lab])
            return keep, lab, f["manifest_row"][()], f[key].shape[1]

    with ThreadPoolExecutor(max_workers=8) as ex:
        meta = list(ex.map(first, paths))
    dims = {m[3] for m in meta}
    if len(dims) != 1:
        raise RuntimeError(f"inconsistent embedding dims across slides: {dims}")
    dim = dims.pop()
    counts = [int(m[0].sum()) for m in meta]
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    n = int(offsets[-1])
    emb = torch.empty((n, dim), dtype=torch.float16)
    out_np = emb.numpy()
    slide_idx = np.repeat(np.arange(len(slides)), counts).astype(np.int64)
    mrow = np.concatenate([m[2][m[0]] for m in meta]).astype(np.int64) if n else np.zeros(0, np.int64)
    label = np.concatenate([[code[l] for l in m[1][m[0]]] for m in meta]).astype(np.int64) if n else np.zeros(0, np.int64)

    def second(i: int):
        keep = meta[i][0]
        pos = offsets[i]
        with h5py.File(paths[i], "r") as f:
            d = f[key]
            for a in range(0, d.shape[0], _BLOCK):
                k = keep[a:a + _BLOCK]
                if not k.any():
                    continue
                block = d[a:a + _BLOCK][k]
                out_np[pos:pos + len(block)] = block
                pos += len(block)
        assert pos == offsets[i + 1]

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(second, range(len(paths))))
    LOG.info("loaded %s/%s/%s stage %s: %d cells x %d from %d slides (%.1f GB) in %.0fs", model, variant, readout,
             stage, n, dim, len(slides), emb.numel() * 2 / 1e9, t())
    return Loaded(emb, slide_idx, mrow, label, slides, classes)


def z_stats(emb: torch.Tensor, rows: np.ndarray, chunk: int = 262144):
    """Per-feature mean and std (+1e-8) of emb[rows], accumulated in float64 chunk by chunk."""
    d = emb.shape[1]
    s1 = np.zeros(d, dtype=np.float64)
    s2 = np.zeros(d, dtype=np.float64)
    rows = np.sort(np.asarray(rows))
    for a in range(0, len(rows), chunk):
        x = emb[torch.from_numpy(rows[a:a + chunk])].double().numpy()
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
    n = max(len(rows), 1)
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0.0)
    std = np.sqrt(var) + 1e-8
    dt = np.float16 if emb.dtype == torch.float16 else np.float32
    if dt == np.float16:
        # trainer.train_linear_jointly z-scores in the embeddings' dtype; a near-constant feature
        # (std < 1e-3) would otherwise overflow / divide by zero in float16
        std = np.maximum(std, 1e-3)
    return mean.astype(dt), std.astype(dt)
