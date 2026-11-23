"""Extraction worker: one bin = one model (loaded once) x many slides.

Per slide: cells -> crop origins -> tile-aligned regions (bench/data/regions.py). DataLoader
workers each own a WSI handle and read one region per item (resampled once if needed); the GPU
gathers the crops of a region by indexing, batches across regions, runs one bf16
inference_mode forward per batch and computes both readouts; a writer thread appends float16
rows to a temporary h5 that is renamed into place when complete. A failed read is retried once
after reopening the slide; a second failure fails the slide (no placeholder crops).
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bench.config import BenchConfig
from bench.data.reader import WSIReader
from bench.data.regions import crop_origins, emission_order, plan_regions
from bench.extract import planner
from bench.extract.store import H5Writer
from bench.models.encoders import Encoder, build_encoder
from bench.models.readouts import nearest_2x2_flat
from bench.provenance import git_state, package_versions
from bench.utils import LOG, fmt_hours


job_done = planner.job_done


# ── data loading ──

class _Regions(Dataset):
    def __init__(self, wsi_path: str, scale: float, boxes: List[tuple], pad_outside: bool):
        self.wsi_path, self.scale, self.boxes, self.pad = wsi_path, scale, boxes, pad_outside
        self._reader: Optional[WSIReader] = None

    def __len__(self) -> int:
        return len(self.boxes)

    def __getitem__(self, i: int):
        if self._reader is None:
            self._reader = WSIReader(self.wsi_path, self.scale)
        x0, y0, w, h = self.boxes[i]
        arr = self._reader.read_ext(x0, y0, w, h, pad_value=255, allow_outside=self.pad)
        return i, torch.from_numpy(np.ascontiguousarray(arr))


def _loader(ds: Dataset, cfg: BenchConfig, cuda: bool) -> DataLoader:
    nw = min(cfg.extract.num_workers, len(ds))
    kw = {"prefetch_factor": cfg.extract.prefetch_factor} if nw > 0 else {}
    return DataLoader(ds, batch_size=None, shuffle=False, num_workers=nw, pin_memory=cuda, **kw)


class _Writer:
    """Background h5 writes so the GPU never waits on disk."""

    def __init__(self, w: H5Writer):
        self.w = w
        self.q: "queue.Queue" = queue.Queue(maxsize=16)
        self.err: Optional[BaseException] = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if self.err is None:
                try:
                    self.w.write(*item)
                except BaseException as e:  # surfaced in put()/close()
                    self.err = e

    def put(self, start: int, arrays: Dict[str, np.ndarray]) -> None:
        if self.err:
            raise self.err
        self.q.put((start, arrays))

    def close(self) -> None:
        self.q.put(None)
        self.t.join()
        if self.err:
            raise self.err


def _to_np(out: Dict[str, torch.Tensor], dtype) -> Dict[str, np.ndarray]:
    return {k: v.to(torch.float16 if dtype == "float16" else torch.float32).cpu().numpy() for k, v in out.items()}


# ── single-cell crops ──

def _meta_arrays(df, order: np.ndarray) -> Dict[str, np.ndarray]:
    sub = df.iloc[order]
    return {
        "cell_ids": np.array(sub["cell_id"].astype(str).tolist(), dtype="S"),
        "cell_labels": np.array(sub["label_raw"].astype(str).tolist(), dtype="S"),
        "labels": np.array(sub["label"].astype(str).tolist(), dtype="S"),
        "manifest_row": sub.index.to_numpy(np.int64),
        "x": sub["cx"].to_numpy(np.float32), "y": sub["cy"].to_numpy(np.float32),
    }


def _cells(cfg: BenchConfig, job: dict):
    from bench.data.manifest import load_cells
    df = load_cells(cfg, job["slide"], ["cell_id", "label_raw", "label", "cx", "cy", "ex", "ey", "extract", "labelled"])
    mask = df["extract"] if planner.cell_set(job["stage"]) == "extract" else df["extract"] & df["labelled"]
    return df[mask]


def extract_slide(cfg: BenchConfig, enc: Encoder, job: dict, row: dict, max_cells: Optional[int] = None) -> dict:
    t0 = time.time()
    n_px = enc.input_px
    df = _cells(cfg, job)
    rows, mhash = planner.expected(cfg, job["stage"], job["slide"])
    if len(df) != rows:
        raise RuntimeError(f"manifest has {len(df)} cells, slides.parquet says {rows}: rerun `bench manifest`")
    ox, oy = crop_origins(df["ex"].to_numpy(), df["ey"].to_numpy(), n_px)
    regions = plan_regions(ox, oy, n_px, cfg.extract.region_px)
    order = emission_order(regions)
    dims = {k: v for k, v in enc.dims.items() if k in job["readouts"]}
    cuda = enc.device.type == "cuda"
    bs = max(1, int(cfg.models[job["model"]].batch_size * min(64.0, (cfg.models[job["model"]].input_px / n_px) ** 2)))
    writer = H5Writer(job["out"], len(df), dims, _meta_arrays(df, order), cfg.storage.dtype, tag=job["key"].replace("/", "_")[-60:])
    wt = _Writer(writer)
    try:
        ds = _Regions(row["wsi_path"], float(row["scale"]), [(r.x0, r.y0, r.w, r.h) for r in regions], pad_outside=False)
        ar = torch.arange(n_px, device=enc.device)
        pend: List[torch.Tensor] = []
        n_pend, written = 0, 0

        def flush(force: bool) -> None:
            nonlocal pend, n_pend, written
            while n_pend >= bs or (force and n_pend):
                batch = torch.cat(pend) if len(pend) > 1 else pend[0]
                take, rest = batch[:bs], batch[bs:]
                pend, n_pend = ([rest] if len(rest) else []), len(rest)
                out = enc.encode(take, central="central" in dims)
                out = {k: v for k, v in out.items() if k in dims}
                wt.put(written, _to_np(out, cfg.storage.dtype))
                written += len(take)

        for i, region in _loader(ds, cfg, cuda):
            r = regions[i]
            reg = region.to(enc.device, non_blocking=True)
            dy = torch.from_numpy(r.dy).to(enc.device)
            dx = torch.from_numpy(r.dx).to(enc.device)
            for lo in range(0, len(r.cells), bs):
                iy = (dy[lo:lo + bs, None] + ar)[:, :, None]
                ix = (dx[lo:lo + bs, None] + ar)[:, None, :]
                crops = reg[iy, ix]                                   # (k, N, N, 3) uint8
                pend.append(crops)
                n_pend += len(crops)
                flush(False)
        flush(True)
        wt.close()
        secs = time.time() - t0
        writer.finalize(_attrs(cfg, enc, job, row, mhash, secs, len(df)))
    except BaseException:
        try:
            wt.close()
        except BaseException:
            pass
        writer.abort()
        raise
    return {"cells": len(df), "seconds": round(secs, 1), "cells_per_s": round(len(df) / max(secs, 1e-9), 1),
            "regions": len(regions)}


def _num(v) -> float:
    """float, NaN for a missing value (unknown pixel size)."""
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _attrs(cfg, enc: Encoder, job, row, mhash, secs, n) -> dict:
    g = git_state()
    return {
        "model": job["model"], "revision": str(enc.revision or ""), "stage": job["stage"], "variant": job["variant"],
        "slide": job["slide"], "input_px": enc.input_px, "patch": enc.patch, "grid": enc.grid,
        "scale": float(row["scale"]), "native_mpp": _num(row.get("mpp")),
        "extraction_mpp": _num(row.get("mpp")) / float(row["scale"]), "mpp_source": str(row.get("mpp_source") or ""),
        "readouts": list(job["readouts"]), "precision": enc.precision,
        "readout_cls": "model-card standard embedding",
        "readout_central": ("mean of the 2x2 tokens nearest each cell in its multicell tile" if job["multicell"]
                            else "mean of the central 2x2 tokens (1 token for odd grids); crop centred on the cell"),
        "manifest_hash": mhash, "config_hash": cfg.section_hash("geometry", "resolution", "filters", "mapping"),
        "git_commit": g["commit"] or "", "git_dirty": g["dirty"], "torch": package_versions().get("torch") or "",
        "timm": package_versions().get("timm") or "", "seconds": round(secs, 1),
        "cells_per_s": round(n / max(secs, 1e-9), 1), "duplicates": 0,
    }


# ── multicell tiles (stage 2b) ──

def extract_multicell_slide(cfg: BenchConfig, enc: Encoder, job: dict, row: dict) -> dict:
    t0 = time.time()
    t_px = enc.input_px
    s_px = int(round(t_px * cfg.stage2.multicell.central_frac))
    margin = (t_px - s_px) // 2
    df = _cells(cfg, job)
    rows, mhash = planner.expected(cfg, job["stage"], job["slide"])
    if len(df) != rows:
        raise RuntimeError(f"manifest has {len(df)} cells, slides.parquet says {rows}: rerun `bench manifest`")
    ex, ey = df["ex"].to_numpy(), df["ey"].to_numpy()
    kx, ky = np.floor(ex / s_px).astype(np.int64), np.floor(ey / s_px).astype(np.int64)
    order = np.lexsort((ex, ey, kx, ky))
    keys = np.stack([ky[order], kx[order]], 1)
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1]
    tiles = [(int(kx[order[a]]) * s_px - margin, int(ky[order[a]]) * s_px - margin) for a in starts]
    bounds = list(zip(starts, np.r_[starts[1:], len(order)]))
    px, py = ex[order] - np.repeat([t[0] for t in tiles], np.diff(np.r_[starts, len(order)])), \
        ey[order] - np.repeat([t[1] for t in tiles], np.diff(np.r_[starts, len(order)]))
    tok_idx = torch.from_numpy(nearest_2x2_flat(px, py, enc.patch, enc.grid)).to(enc.device)
    dims = {"central": enc.dims["central"]}
    writer = H5Writer(job["out"], len(df), dims, _meta_arrays(df, order), cfg.storage.dtype, tag=job["key"].replace("/", "_")[-60:])
    wt = _Writer(writer)
    cuda = enc.device.type == "cuda"
    m = cfg.models[job["model"]]
    bt = max(1, int(m.batch_size * min(16.0, (m.input_px / t_px) ** 2)))
    try:
        ds = _Regions(row["wsi_path"], float(row["scale"]), [(x, y, t_px, t_px) for x, y in tiles], pad_outside=True)
        buf, buf_ids, written = [], [], 0

        def run(force: bool):
            nonlocal buf, buf_ids, written
            if not buf or (len(buf) < bt and not force):
                return
            toks = enc.tokens(torch.stack(buf))                      # (B, g*g, D)
            for b, ti in enumerate(buf_ids):
                a, z = bounds[ti]
                emb = toks[b][tok_idx[a:z]].mean(dim=1)              # (k, D)
                wt.put(written, _to_np({"central": emb}, cfg.storage.dtype))
                written += z - a
            buf, buf_ids = [], []

        for i, tile in _loader(ds, cfg, cuda):
            buf.append(tile.to(enc.device, non_blocking=True))
            buf_ids.append(i)
            run(False)
        run(True)
        wt.close()
        secs = time.time() - t0
        attrs = _attrs(cfg, enc, job, row, mhash, secs, len(df))
        attrs.update(tile_px=t_px, central_px=s_px, n_tiles=len(tiles), forward_passes=len(tiles))
        writer.finalize(attrs)
    except BaseException:
        try:
            wt.close()
        except BaseException:
            pass
        writer.abort()
        raise
    return {"cells": len(df), "tiles": len(tiles), "seconds": round(secs, 1),
            "cells_per_s": round(len(df) / max(secs, 1e-9), 1)}


# ── bin runner ──

def run_extract_bin(cfg: BenchConfig, jobs: List[dict], device: str) -> int:
    from bench.data.inspect import load_inspect
    from bench.jobs import set_status
    from bench.provenance import model_revisions
    insp = load_inspect(cfg)
    revs = model_revisions(cfg.paths.optional("bench_root"))
    groups: Dict[tuple, List[dict]] = {}
    for j in jobs:
        groups.setdefault((j["model"], j["input_px"]), []).append(j)
    failed = 0
    for (model, n_px), gj in groups.items():
        LOG.info("loading %s at %d px on %s", model, n_px, device)
        mcfg = cfg.models[model]
        if mcfg.revision is None and revs.get(model, {}).get("sha"):
            # pin to the snapshot `bench fetch-weights` downloaded (works offline on compute nodes)
            mcfg = mcfg.model_copy(update={"revision": revs[model]["sha"]})
        enc = build_encoder(model, mcfg, n_px, device, cfg.extract.precision)
        for j in gj:
            if job_done(cfg, j):
                continue
            set_status(cfg, j, "running")
            try:
                row = insp.loc[j["slide"]].to_dict()
                fn = extract_multicell_slide if j["multicell"] else extract_slide
                info = fn(cfg, enc, j, row)
                set_status(cfg, j, "done", model=model, input_px=n_px, stage=j["stage"], **info)
                LOG.info("%s: %d cells in %s (%.0f cells/s)", j["key"], info["cells"],
                         fmt_hours(info["seconds"] / 3600), info["cells_per_s"])
            except Exception as e:
                failed += 1
                LOG.error("%s FAILED: %s", j["key"], e)
                set_status(cfg, j, "failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
        del enc
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return failed
