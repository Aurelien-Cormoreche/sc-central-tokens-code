"""`bench inspect`: resolve every slide's WSI and cell-coordinate files, read the pixel size from
the file itself, cross-check it against xenium_metadata.csv, and derive the extraction scale.

Writes {bench_root}/manifest/slides_inspect.parquet (+ .csv). Stops (non-zero exit) when a slide
has no readable WSI, no coordinates, an unknown pixel size, or a pixel size that disagrees with
the metadata -- fix those with `mpp_overrides` in the config or by fixing the files.
"""
from __future__ import annotations

import csv
import math
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from bench import REPO
from bench.config import BenchConfig
from bench.data.reader import scale_for
from bench.registry import Slide
from bench.utils import LOG, atomic_path, print_table

WSI_SUFFIXES = ("_he_image.ome.tif", "_registered_HE.ome.tif", "_registered_HE.tif", "_he_image.tif")
_UNIT_TO_UM = {"m": 1e6, "mm": 1e3, "cm": 1e4, "µm": 1.0, "um": 1.0, "micron": 1.0, "microns": 1.0,
               "micrometer": 1.0, "nm": 1e-3, "\\u00B5m": 1.0}


def inspect_path(cfg: BenchConfig) -> Path:
    return cfg.root / "manifest" / "slides_inspect.parquet"


def load_inspect(cfg: BenchConfig) -> pd.DataFrame:
    p = inspect_path(cfg)
    if not p.exists():
        raise SystemExit(f"{p} not found: run `bench inspect` first")
    return pd.read_parquet(p).set_index("slide", drop=False)


def _wsi_manifest(cfg: BenchConfig) -> dict:
    p = cfg.paths.optional("wsi_manifest")
    if p is None:
        return {}
    out = {}
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("status", "ok") != "ok":
                continue
            out[r["dataset"]] = (str(r["converted"]).strip().lower() in ("true", "1"), r["wsi_filename"])
    return out


def resolve_wsi(cfg: BenchConfig, slide: str, manifest: dict) -> Tuple[Optional[Path], bool]:
    if slide in manifest:
        converted, fname = manifest[slide]
        root = cfg.paths.get("wsi_converted" if converted else "wsi_raw")
        return root / fname, converted
    # converted files exist only where the raw file was unreadable, so they win
    for converted, root_name in ((True, "wsi_converted"), (False, "wsi_raw")):
        root = cfg.paths.optional(root_name)
        if root is None:
            continue
        for suf in WSI_SUFFIXES:
            p = root / f"{slide}{suf}"
            if p.is_file():
                return p, converted
    return None, False


def _mpp_from_tiff(path: Path) -> Tuple[Optional[float], str]:
    import tifffile
    with tifffile.TiffFile(path) as tf:
        if tf.is_ome and tf.ome_metadata:
            try:
                root = ET.fromstring(tf.ome_metadata)
                pix = next(el for el in root.iter() if el.tag.endswith("Pixels"))
                v = pix.get("PhysicalSizeX")
                if v:
                    unit = pix.get("PhysicalSizeXUnit", "µm")
                    return float(v) * _UNIT_TO_UM.get(unit, 1.0), "ome"
            except (ET.ParseError, StopIteration, ValueError):
                pass
        page = tf.pages[0]
        tags = page.tags
        if "XResolution" in tags:
            num, den = tags["XResolution"].value
            px_per_unit = num / den if den else 0
            if px_per_unit > 0:
                if tf.is_imagej and tf.imagej_metadata:
                    unit = str(tf.imagej_metadata.get("unit", "")).strip()
                    if unit in _UNIT_TO_UM:
                        return _UNIT_TO_UM[unit] / px_per_unit, "imagej"
                ru = int(tags["ResolutionUnit"].value) if "ResolutionUnit" in tags else 2
                if ru == 3:
                    return 1e4 / px_per_unit, "tiff_cm"
                if ru == 2 and px_per_unit > 1000:   # pixels per inch; < 1000 dpi is a screen default
                    return 25400.0 / px_per_unit, "tiff_inch"
    return None, "none"


def read_mpp(path: Path) -> Tuple[Optional[float], str, dict]:
    """(mpp, source, info) from the file: OpenSlide properties first, then OME/ImageJ/TIFF tags."""
    import openslide
    info: dict = {}
    sl = openslide.OpenSlide(str(path))
    try:
        info["width"], info["height"] = sl.dimensions
        info["levels"] = sl.level_count
        p = sl.properties
        mx, my = p.get(openslide.PROPERTY_NAME_MPP_X), p.get(openslide.PROPERTY_NAME_MPP_Y)
        if mx:
            mpp = float(mx)
            if my and abs(float(my) - mpp) / mpp > 0.01:
                info["anisotropic"] = f"{mx}x{my}"
            return mpp, "openslide", info
    finally:
        sl.close()
    mpp, src = _mpp_from_tiff(path)
    return mpp, src, info


def _metadata_mpp() -> dict:
    p = REPO / "xenium_metadata.csv"
    out: dict = {}
    if not p.exists():
        return out
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        v = r.get("he_pixel_size_um", "")
        try:
            out[re.sub(r"_out$", "", r["xenium_folder"])] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def coords_path(cfg: BenchConfig, slide: str) -> Path:
    return cfg.paths.get("xenium_processed") / slide / "patch_coordinates.h5"


def _inspect_one(cfg: BenchConfig, s: Slide, manifest: dict, meta: dict) -> dict:
    import h5py
    row = {"slide": s.slide, "cohort": s.cohort, "donor": s.donor, "val": s.val, "status": "ok", "error": ""}
    wsi, converted = resolve_wsi(cfg, s.slide, manifest)
    row.update(wsi_path=str(wsi) if wsi else "", converted=converted)
    cp = coords_path(cfg, s.slide)
    row["coords_path"] = str(cp)
    try:
        with h5py.File(cp, "r") as f:
            row["n_cells"] = int(f["cell_id"].shape[0])
    except Exception as e:
        row.update(status="missing_coords", error=f"{type(e).__name__}: {e}", n_cells=0)
    if wsi is None:
        row.update(status="missing_wsi", error="no WSI file found (see WSI_SUFFIXES / paths.wsi_manifest)")
        return row
    try:
        mpp, src, info = read_mpp(wsi)
        row.update(info)
    except Exception as e:
        row.update(status="open_failed", error=f"{type(e).__name__}: {e}")
        return row
    meta_mpp = meta.get(s.slide)
    row["meta_mpp"] = meta_mpp
    row["file_mpp"] = mpp
    row["file_mpp_source"] = src
    if s.slide in cfg.mpp_overrides:
        mpp, src = float(cfg.mpp_overrides[s.slide]), "override"
    row.update(mpp=mpp, mpp_source=src)
    if mpp is None:
        if row["status"] == "ok":
            row.update(status="unknown_mpp", error="no pixel size in the file; set mpp_overrides")
        return row
    if src != "override" and meta_mpp and abs(mpp - meta_mpp) / meta_mpp > cfg.resolution.mpp_mismatch_tol:
        row.update(status="mpp_mismatch", error=f"file {mpp:.5f} vs xenium_metadata {meta_mpp:.5f} um/px")
    s_ = scale_for(mpp, cfg.resolution.resample_below_mpp, cfg.resolution.target_mpp)
    row.update(scale=s_, ext_width=int(math.floor(row["width"] * s_ + 1e-9)),
               ext_height=int(math.floor(row["height"] * s_ + 1e-9)),
               ext_mpp=mpp / s_)
    return row


def run_inspect(cfg: BenchConfig, slides: List[Slide], workers: int = 8, strict: bool = True) -> pd.DataFrame:
    manifest, meta = _wsi_manifest(cfg), _metadata_mpp()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(lambda s: _inspect_one(cfg, s, manifest, meta), slides))
    df = pd.DataFrame(rows)
    # previous inspect results for slides not in this run are kept
    out = inspect_path(cfg)
    if out.exists():
        old = pd.read_parquet(out)
        df = pd.concat([old[~old.slide.isin(df.slide)], df], ignore_index=True)
    df = df.sort_values(["cohort", "slide"]).reset_index(drop=True)
    for col in ("width", "height", "levels", "ext_width", "ext_height", "n_cells"):
        if col in df:
            df[col] = df[col].fillna(0).astype("int64")
    with atomic_path(out) as tmp:
        df.to_parquet(tmp, index=False)
    with atomic_path(out.with_suffix(".csv")) as tmp:
        df.to_csv(tmp, index=False)
    shown = df[df.slide.isin([s.slide for s in slides])]
    cols = [c for c in ("slide", "cohort", "status", "mpp", "mpp_source", "meta_mpp", "scale", "ext_mpp",
                        "width", "height", "n_cells", "converted") if c in shown]
    view = shown[cols].copy()
    for c in ("mpp", "meta_mpp", "scale", "ext_mpp"):
        if c in view:
            view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    print_table(view.to_dict("records"))
    bad = shown[shown.status != "ok"]
    LOG.info("wrote %s (%d slides, %d problems)", out, len(shown), len(bad))
    if len(bad):
        for r in bad.itertuples():
            print(f"  PROBLEM {r.slide}: {r.status} -- {r.error}")
        if strict:
            raise SystemExit(f"{len(bad)} slide(s) need attention (see above); fix them or use mpp_overrides")
    return df
