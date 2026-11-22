"""Embedding store: one h5 per (stage, model, variant, slide).

    {bench_root}/embeddings/stage{1|2a|2b}/{model}/{variant}/{slide}/embeddings_dataset.h5

Datasets (rows in extraction order, all the same length):
    embeddings_cls, embeddings_central   (n, D) float16   -- key names as read by
                                          deepspot_training and src/python/utils/loading_functions
    cell_ids     (n,) bytes   patch_coordinates cell_id
    cell_labels  (n,) bytes   raw cell type (mapped by the reader, as in the existing loaders)
    labels       (n,) bytes   mapped label (config `mapping`)
    manifest_row (n,) int64   row in manifest/cells/{slide}.parquet (== patch_coordinates row)
    x, y         (n,) float32 native level-0 centroid
Attributes: model, revision, input_px, patch, grid, scale, readouts, readout definitions,
manifest_hash, rows, nonfinite, duplicates, config hash, git commit, versions, complete=True.

Writes go to a temporary file in the same directory that is fsync'ed and renamed into place
only once complete, so a file that exists is complete. is_valid() additionally checks the row
count and manifest hash (a regenerated manifest invalidates old embeddings).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from bench.utils import fsync_and_replace, tmp_path_for

SCHEMA_VERSION = 1
H5_NAME = "embeddings_dataset.h5"


def embeddings_dir(root: Path, stage: str, model: str, variant: str) -> Path:
    return root / "embeddings" / f"stage{stage}" / model / variant


def embeddings_path(root: Path, stage: str, model: str, variant: str, slide: str) -> Path:
    return embeddings_dir(root, stage, model, variant) / slide / H5_NAME


def is_valid(path: Path, rows: int, manifest_hash: str, readouts: Tuple[str, ...]) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        import h5py
        with h5py.File(path, "r") as f:
            a = f.attrs
            if not bool(a.get("complete", False)):
                return False, "incomplete"
            if int(a.get("schema_version", 0)) != SCHEMA_VERSION:
                return False, "schema"
            if int(a.get("rows", -1)) != rows:
                return False, f"rows {int(a.get('rows', -1))} != manifest {rows}"
            if str(a.get("manifest_hash", "")) != manifest_hash:
                return False, "manifest changed"
            for r in readouts:
                key = f"embeddings_{r}"
                if key not in f or f[key].shape[0] != rows:
                    return False, f"{key} missing/short"
            if int(a.get("nonfinite", 0)):
                return False, "non-finite values"
    except Exception as e:  # truncated/corrupt file
        return False, f"unreadable ({type(e).__name__})"
    return True, "ok"


class H5Writer:
    """Preallocated datasets filled sequentially (rows arrive in extraction order)."""

    def __init__(self, final: Path, n: int, dims: Dict[str, int], meta_arrays: Dict[str, np.ndarray],
                 dtype: str = "float16", tag: Optional[str] = None):
        import h5py
        self.final = Path(final)
        self.final.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = tmp_path_for(self.final, tag)
        self.n = n
        self.f = h5py.File(self.tmp, "w")
        self.dsets = {}
        for r, d in dims.items():
            chunk = (max(1, min(4096, n)), d)
            self.dsets[r] = self.f.create_dataset(f"embeddings_{r}", shape=(n, d), dtype=dtype,
                                                  chunks=chunk if n else None)
        for k, v in meta_arrays.items():
            self.f.create_dataset(k, data=v)
        self.nonfinite = 0
        self.written = 0

    def write(self, start: int, arrays: Dict[str, np.ndarray]) -> None:
        k = None
        for r, a in arrays.items():
            k = len(a)
            self.dsets[r][start:start + k] = a
            if not np.isfinite(a).all():
                self.nonfinite += int((~np.isfinite(a)).any(axis=1).sum())
        if k:
            self.written = max(self.written, start + k)

    def finalize(self, attrs: dict) -> None:
        if self.written != self.n:
            raise RuntimeError(f"only {self.written}/{self.n} rows written")
        a = self.f.attrs
        for k, v in attrs.items():
            a[k] = v if not isinstance(v, (list, tuple)) else np.array([str(x) for x in v], dtype="S")
        a["schema_version"] = SCHEMA_VERSION
        a["rows"] = self.n
        a["nonfinite"] = self.nonfinite
        a["complete"] = self.nonfinite == 0
        self.f.flush()
        self.f.close()
        if self.nonfinite:
            os.unlink(self.tmp)
            raise RuntimeError(f"{self.nonfinite} rows with non-finite embeddings; nothing written")
        fsync_and_replace(self.tmp, self.final)

    def abort(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass
        try:
            os.unlink(self.tmp)
        except FileNotFoundError:
            pass


def read_attrs(path: Path) -> dict:
    import h5py
    with h5py.File(path, "r") as f:
        return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.attrs.items()}
