"""Provenance written next to every result: resolved config, git state, package versions,
hardware, HF model revisions. Collected once per process and cached."""
from __future__ import annotations

import functools
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from bench import REPO, __version__

_PACKAGES = ("torch", "timm", "transformers", "numpy", "pandas", "h5py", "scikit-learn", "scipy",
             "openslide-python", "tifffile", "pyarrow", "pydantic", "lightning", "deepspot", "scanpy",
             "anndata", "trident", "huggingface-hub", "einops", "pillow")


def _run(cmd: list[str]) -> Optional[str]:
    try:
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def git_state() -> dict[str, Any]:
    commit = (_run(["git", "rev-parse", "HEAD"]) or "").strip() or None
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"]) or ""
    diff = _run(["git", "diff", "HEAD"]) if status.strip() else ""
    return {"commit": commit, "dirty": bool(status.strip()), "diff": diff or "",
            "branch": (_run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "").strip() or None}


@functools.lru_cache(maxsize=1)
def package_versions() -> dict[str, Optional[str]]:
    from importlib import metadata
    out: dict[str, Optional[str]] = {}
    for p in _PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


@functools.lru_cache(maxsize=1)
def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"host": socket.gethostname(), "python": sys.version.split()[0],
                            "platform": platform.platform(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                            "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID")}
    try:
        import torch
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    smi = None
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        smi = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    info["driver"] = smi.strip().splitlines()[0] if smi else None
    return info


def model_revisions(root: Optional[Path]) -> dict:
    from bench.utils import read_json
    if root is None:
        return {}
    # `bench fetch-weights` writes to the main bench_root; smoke runs live in bench_root/smoke
    for d in (root, root.parent):
        r = read_json(d / "weights_revisions.json")
        if r:
            return r
    return {}


def collect(cfg=None, extra: Optional[dict] = None, with_diff: bool = False) -> dict[str, Any]:
    g = dict(git_state())
    if not with_diff:
        g["diff_lines"] = len(g.pop("diff").splitlines())
    rec: dict[str, Any] = {
        "bench_version": __version__,
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "git": g,
        "packages": package_versions(),
        "hardware": hardware(),
    }
    if cfg is not None:
        rec["config_source"] = cfg.source
        rec["config"] = cfg.dump()
        rec["model_revisions"] = model_revisions(cfg.paths.optional("bench_root"))
    if extra:
        rec.update(extra)
    return rec


def write_run_provenance(directory: Path, cfg, extra: Optional[dict] = None) -> None:
    """provenance.json (+ git.diff when the tree is dirty) in `directory`."""
    from bench.utils import atomic_path, write_json
    rec = collect(cfg, extra)
    write_json(directory / "provenance.json", rec)
    diff = git_state()["diff"]
    if diff:
        with atomic_path(directory / "git.diff") as tmp:
            tmp.write_text(diff)
