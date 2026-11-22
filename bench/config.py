"""configs/bench.yaml schema (pydantic) and loader.

String values may reference environment variables: `${env:NAME}` (required -- a clear error is
raised only when that path is actually used) or `${env:NAME,default}` (optional; an empty
default gives None). Nothing in the config or the code hardcodes a data location.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bench import REPO

_ENV_RE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")
_MISSING = "\x00MISSING_ENV:"


def _interpolate(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if not isinstance(value, str) or "${env:" not in value:
        return value

    full = _ENV_RE.fullmatch(value)
    if full:
        name, default = full.group(1), full.group(2)
        got = os.environ.get(name)
        if got:
            return got
        if default is not None:
            return default or None
        return _MISSING + name

    def sub(m: re.Match) -> str:
        got = os.environ.get(m.group(1))
        if got:
            return got
        if m.group(2) is not None:
            return m.group(2)
        raise EnvironmentError(f"environment variable {m.group(1)!r} is not set (used in {value!r})")

    return _ENV_RE.sub(sub, value)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Paths(_Strict):
    bench_root: Optional[str] = None
    wsi_raw: Optional[str] = None
    wsi_converted: Optional[str] = None
    xenium_processed: Optional[str] = None
    hf_home: Optional[str] = None
    # Optional CSV (dataset, converted, wsi_filename[, status]) as written by
    # notebooks/check_wsi_openslide.ipynb; without it `bench inspect` scans wsi_raw/wsi_converted.
    wsi_manifest: Optional[str] = None

    def get(self, name: str) -> Path:
        value = getattr(self, name)
        if value is None:
            raise EnvironmentError(f"paths.{name} is not set in the bench config")
        if value.startswith(_MISSING):
            env = value[len(_MISSING):]
            raise EnvironmentError(
                f"paths.{name} needs environment variable {env!r}, which is not set. "
                f"Export it (see README_bench.md, 'Environment') before running this command."
            )
        return Path(value)

    def optional(self, name: str) -> Optional[Path]:
        value = getattr(self, name)
        if value is None or value.startswith(_MISSING):
            return None
        return Path(value)


class Geometry(_Strict):
    # patch_coordinates.h5 stores the top-left corner of a 224-px crop centred on the cell:
    # centroid = (x_start, y_start) + centre_offset_px (native level-0 pixels).
    centre_offset_px: float = 112.0


class Resolution(_Strict):
    resample_below_mpp: float = 0.2
    target_mpp: float = 0.25
    mpp_mismatch_tol: float = 0.05     # relative tolerance file MPP vs xenium_metadata.csv


class Filters(_Strict):
    black_max: int = 8                 # a pixel is no-data if max(R,G,B) <= black_max (or alpha == 0)
    block_px: int = 16                 # dark-spot map resolution (native px)
    block_full_frac: float = 0.9       # a block is dark if >= this fraction of its pixels is no-data
    min_blob_blocks: int = 1           # dark blobs smaller than this (4-connected blocks) are ignored
    linf_px: int = 224                 # drop cells within this L-inf distance (extraction px) of a dark block
    border_px: int = 224               # drop cells within this distance (extraction px) of the image edge
    scan_tile_px: int = 4096           # dark-spot scan read size (native px); only tiles near cells are read


class ModelCfg(_Strict):
    input_px: int
    patch: int
    batch_size: int = 256
    cells_per_s: float = 300.0         # RTX 4090 bf16 throughput prior at input_px (calibration.json wins)
    revision: Optional[str] = None     # HF revision to pin (commit sha); null = whatever fetch-weights got
    arch: Optional[str] = None         # model-specific (LEMON: vits8 | vitb8)
    enabled: bool = True


class ExtractCfg(_Strict):
    region_px: int = 2048              # read-region tile size (extraction px), + crop margin
    num_workers: int = 12
    prefetch_factor: int = 4
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    slide_overhead_s: float = 60.0     # per-slide fixed cost for the planner (open, plan, write)
    bin_overhead_s: float = 300.0      # per-task fixed cost (model load, CUDA init)


class Storage(_Strict):
    dtype: Literal["float16", "float32"] = "float16"


class Slurm(_Strict):
    partition: Optional[str] = "gpu"
    gres: str = "gpu:rtx_4090:1"
    cpus: int = 16
    mem_gb: int = 64
    max_hours: float = 12.0
    max_array: int = 32                # max simultaneously running array tasks (%N)
    account: Optional[str] = None
    cpu_partition: Optional[str] = None   # CPU-only tasks (darkspots); null = cluster default
    cpu_mem_gb: int = 32
    extra: List[str] = Field(default_factory=list)    # extra "#SBATCH ..." option strings
    setup: str = ""                    # shell lines before python (module load / conda activate)
    python: str = "python"


class Grid(_Strict):
    lr: List[float] = Field(default_factory=lambda: [1e-3, 1e-4])
    weight_decay: List[float] = Field(default_factory=lambda: [1e-4, 1e-2])
    loss_weights: List[str] = Field(default_factory=lambda: ["none", "by_cell_type"])


class ClassifyCfg(_Strict):
    schemes: List[Literal["within", "cross_loco", "single_cohort"]] = Field(
        default_factory=lambda: ["within", "cross_loco", "single_cohort"])
    within_cohorts: List[str] = Field(default_factory=lambda: ["SPLIT", "VANNAN", "MCCORD", "JOHNSON"])
    selection_metric: Dict[str, str] = Field(default_factory=lambda: {
        "within": "slide_macro_f1", "cross_loco": "cohort_balanced_slide_macro_f1",
        "single_cohort": "slide_macro_f1"})
    min_class_cells: int = 1           # a class is "present" in a cohort with at least this many cells
    seed: int = 42
    epochs: int = 30
    patience: int = 5
    min_delta: float = 0.0
    batch_size: int = 1024
    zscore: bool = True
    grid: Grid = Field(default_factory=Grid)
    gpu_cells_per_s: float = 2.0e6     # linear-probe throughput prior (cell-epochs/s, all trials in lockstep)


class GenexpSet(_Strict):
    cohort: Optional[str] = None
    slides: List[str] = Field(default_factory=list)   # fnmatch patterns


class GenexpCfg(_Strict):
    sets: Dict[str, GenexpSet] = Field(default_factory=lambda: {
        "PRIME": GenexpSet(slides=["GSM9509*_5k_*", "Xenium_Prime_*"]),
        "VANNAN": GenexpSet(cohort="VANNAN")})
    template: str = "configs/deepspot_lopo_prime.yaml"   # training + hparams + data defaults
    num_genes: int = 300
    top: List[int] = Field(default_factory=lambda: [50, 100, 300])
    readout: str = "auto"              # auto | cls | central
    hvg_max_genes: int = 500
    gpu_hours_per_fold: float = 0.4
    mem_gb: int = 128


class GateRule(_Strict):
    metric: str = "macro_f1_present"
    over: Literal["within_cohort_tasks"] = "within_cohort_tasks"
    split: Literal["test", "val"] = "test"


class GateChoice(_Strict):
    model: Optional[str] = None
    readout: Optional[str] = None


class Gate1(_Strict):
    rule: GateRule = Field(default_factory=GateRule)
    chosen: GateChoice = Field(default_factory=GateChoice)


class Multicell(_Strict):
    tile_px: Optional[int] = None      # Gate 2: set from the 2a results (224 or 448)
    central_frac: float = 0.5          # a cell belongs to the tile whose central window (frac*T) holds it


class Stage2(_Strict):
    cohorts: List[str] = Field(default_factory=lambda: ["SPLIT", "VANNAN"])
    fov_px: List[int] = Field(default_factory=lambda: [56, 112, 224, 448])
    round_to_patch: Literal["up", "down", "nearest"] = "up"
    multicell: Multicell = Field(default_factory=Multicell)


class MlpCfg(_Strict):
    hidden: int = 256
    dropout: float = 0.2
    grid: Grid = Field(default_factory=lambda: Grid(lr=[1e-3, 1e-4], weight_decay=[1e-4],
                                                    loss_weights=["none", "by_cell_type"]))
    gpu_cells_per_s: float = 1.0e6


class Stage3(_Strict):
    fractions: List[float] = Field(default_factory=lambda: [0.005, 0.01, 0.05, 1.0])
    seeds: List[int] = Field(default_factory=lambda: [0, 1, 2])
    min_per_class: int = 1
    probes: Dict[str, List[str]] = Field(default_factory=lambda: {
        "linear": ["within", "cross_loco"], "mlp": ["cross_loco"]})
    mlp: MlpCfg = Field(default_factory=MlpCfg)


class SmokeCfg(_Strict):
    # >= 3 donors per cohort incl. each cohort's val donor, so every scheme and the PRIME
    # gene-expression set get a train/val/test fold: JOHNSON d54/d105/d136(val), SPLIT 5K L1/L3/L5/L7(val)
    slides: List[str] = Field(default_factory=lambda: [
        "fetal_d54", "fetal_d105", "fetal_d136", "GSM9509134_NSCLC_5k_L1_L1", "GSM9509135_NSCLC_5k_L3_L3",
        "GSM9509136_NSCLC_5k_L5_L5", "GSM9509138_NSCLC_5k_L7_L7"])
    max_cells_per_slide: int = 300
    epochs: int = 2
    max_trials: int = 2
    seed: int = 0


class BenchConfig(_Strict):
    paths: Paths = Field(default_factory=Paths)
    slides_csv: str = "configs/bench_slides.csv"
    exclude: List[str] = Field(default_factory=list)
    cohorts: List[str] = Field(default_factory=lambda: ["SPLIT", "VANNAN", "MCCORD", "JOHNSON", "10X"])
    mapping: str = "simplified_broad"
    geometry: Geometry = Field(default_factory=Geometry)
    resolution: Resolution = Field(default_factory=Resolution)
    mpp_overrides: Dict[str, float] = Field(default_factory=dict)
    filters: Filters = Field(default_factory=Filters)
    models: Dict[str, ModelCfg] = Field(default_factory=dict)
    readouts: List[Literal["cls", "central"]] = Field(default_factory=lambda: ["cls", "central"])
    extract: ExtractCfg = Field(default_factory=ExtractCfg)
    storage: Storage = Field(default_factory=Storage)
    slurm: Slurm = Field(default_factory=Slurm)
    classify: ClassifyCfg = Field(default_factory=ClassifyCfg)
    genexp: GenexpCfg = Field(default_factory=GenexpCfg)
    gate1: Gate1 = Field(default_factory=Gate1)
    stage2: Stage2 = Field(default_factory=Stage2)
    stage3: Stage3 = Field(default_factory=Stage3)
    smoke: SmokeCfg = Field(default_factory=SmokeCfg)
    # set at load time, not in YAML
    source: Optional[str] = None
    is_smoke: bool = False

    @field_validator("models")
    @classmethod
    def _patch_divides(cls, models: Dict[str, ModelCfg]) -> Dict[str, ModelCfg]:
        for name, m in models.items():
            if m.input_px % m.patch:
                raise ValueError(f"models.{name}: input_px {m.input_px} is not a multiple of patch {m.patch}")
        return models

    @model_validator(mode="after")
    def _gates_consistent(self) -> "BenchConfig":
        ch = self.gate1.chosen
        if (ch.model is None) != (ch.readout is None):
            raise ValueError("gate1.chosen: set both model and readout, or neither")
        if ch.model is not None and ch.model not in self.models:
            raise ValueError(f"gate1.chosen.model {ch.model!r} is not in models")
        if ch.readout is not None and ch.readout not in self.readouts:
            raise ValueError(f"gate1.chosen.readout {ch.readout!r} is not in readouts")
        t = self.stage2.multicell.tile_px
        if t is not None and t not in (224, 448):
            raise ValueError("stage2.multicell.tile_px must be 224 or 448 (or null until Gate 2)")
        return self

    # ── helpers ──
    @property
    def root(self) -> Path:
        return self.paths.get("bench_root")

    def repo_path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else REPO / p

    def enabled_models(self, only: Optional[List[str]] = None) -> List[str]:
        names = [n for n, m in self.models.items() if m.enabled]
        if only:
            unknown = sorted(set(only) - set(self.models))
            if unknown:
                raise SystemExit(f"unknown model(s): {', '.join(unknown)} (known: {', '.join(self.models)})")
            names = [n for n in names if n in only]
        return names

    def dump(self) -> dict:
        return self.model_dump(mode="json")

    def section_hash(self, *sections: str) -> str:
        d = self.dump()
        return stable_hash({s: d[s] for s in sections})


# planning priors / resources: changing them must not invalidate finished results
PLANNING_KEYS = {"gpu_cells_per_s", "gpu_hours_per_fold", "mem_gb", "cells_per_s", "batch_size_hint"}


def without_planning(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: without_planning(v) for k, v in d.items() if k not in PLANNING_KEYS}
    if isinstance(d, list):
        return [without_planning(v) for v in d]
    return d


def stable_hash(obj: Any, n: int = 16) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def default_config_path() -> Path:
    return Path(os.environ.get("BENCH_CONFIG", REPO / "configs" / "bench.yaml"))


def load_config(path: Optional[os.PathLike] = None, overrides: Optional[dict] = None) -> BenchConfig:
    path = Path(path) if path else default_config_path()
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if overrides:
        raw = deep_update(raw, overrides)
    cfg = BenchConfig.model_validate(_interpolate(raw))
    cfg.source = str(path)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def smoke_config(cfg: BenchConfig) -> BenchConfig:
    """The --smoke variant of a config: outputs under {bench_root}/smoke, only the smoke slides,
    at most smoke.max_cells_per_slide cells each (drawn in bench/data/manifest.py), short
    training, at most smoke.max_trials grid trials."""
    if cfg.is_smoke:
        return cfg
    d = cfg.dump()
    root = cfg.paths.optional("bench_root")
    if root is not None:
        d["paths"]["bench_root"] = str(root / "smoke")
    d["classify"]["epochs"] = cfg.smoke.epochs
    d["classify"]["patience"] = cfg.smoke.epochs
    g = d["classify"]["grid"]
    # keep the first max_trials combinations of the grid (lr varies fastest)
    g["weight_decay"] = g["weight_decay"][:1]
    g["loss_weights"] = g["loss_weights"][:max(1, cfg.smoke.max_trials // max(1, len(g["lr"][:2])))]
    g["lr"] = g["lr"][:min(2, cfg.smoke.max_trials)]
    d["stage3"]["mlp"]["grid"] = {"lr": g["lr"][:1], "weight_decay": g["weight_decay"][:1],
                                  "loss_weights": g["loss_weights"][:1]}
    d["stage3"]["seeds"] = d["stage3"]["seeds"][:2]
    d.pop("source", None)
    d["is_smoke"] = True
    out = BenchConfig.model_validate(d)
    out.source = cfg.source
    return out
