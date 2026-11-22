"""Torch-free model facts: Hugging Face repos, the weight files each loader reads, input-size rounding.

Imported by the planning / submission / reporting commands, which run on login nodes, where torch's CUDA
libraries do not load ("failed to map segment from shared object"). Keep torch (and anything importing it) out.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Tuple

from bench.config import ModelCfg

REPOS = {
    "uni2": "MahmoodLab/UNI2-h",
    "virchow2": "paige-ai/Virchow2",
    "hoptimus1": "bioptimus/H-optimus-1",
    "conch15": "MahmoodLab/conchv1_5",
    "lemon": "aliceblondel/LEMON",
    "genbio": "genbio-ai/genbio-pathfm",
}


def round_to_patch(px: int, patch: int, mode: str = "up") -> int:
    f = {"up": math.ceil, "down": math.floor, "nearest": round}[mode]
    return max(1, int(f(px / patch))) * patch


# ── weights in the local Hugging Face cache (`bench fetch-weights`) ──

def weight_spec(name: str, mcfg: ModelCfg) -> Tuple[Optional[list], list]:
    """(files to download, None = the whole repo; files the loaders above read, each a tuple of alternatives).
    timm models read config.json plus model.safetensors or pytorch_model.bin."""
    if name == "conch15":
        return ["pytorch_model_vision.bin"], [("pytorch_model_vision.bin",)]
    if name == "lemon":
        f = f"lemon_{mcfg.arch or 'vits8'}.pth.tar"
        return ["mean_std.json", f], [("mean_std.json",), (f,)]
    if name == "genbio":
        return None, [("config.json",), ("*.safetensors", "*.bin")]
    return None, [("config.json",), ("model.safetensors", "pytorch_model.bin")]


def hf_cache_dir() -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)          # $HF_HUB_CACHE, else $HF_HOME/hub


def cached_snapshot(repo: str, revision: Optional[str] = None) -> Optional[Path]:
    """Snapshot folder of `repo` in the local cache, without network: the commit `revision` if it is a sha,
    else the ref `revision` (default main), else -- no revision and no ref -- the only snapshot there is."""
    d = hf_cache_dir() / f"models--{repo.replace('/', '--')}"
    snaps = d / "snapshots"
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        return snaps / revision if (snaps / revision).is_dir() else None
    ref = d / "refs" / (revision or "main")
    if ref.is_file():
        p = snaps / ref.read_text().strip()
        return p if p.is_dir() else None
    found = [p for p in snaps.iterdir() if p.is_dir()] if snaps.is_dir() and not revision else []
    return found[0] if len(found) == 1 else None


def missing_weight_files(name: str, mcfg: ModelCfg, snap: Optional[Path]) -> list:
    """Required files absent from `snap` (a broken symlink, e.g. from an incomplete copy, counts as absent)."""
    if snap is None:
        return ["not in the cache (or several snapshots and no refs/main: pin models.<name>.revision)"]
    _, req = weight_spec(name, mcfg)
    return [" or ".join(alts) for alts in req if not any(f.exists() for a in alts for f in snap.glob(a))]


def hf_download_command(repo: str, revision: Optional[str], files: Optional[list]) -> str:
    return " ".join(["hf", "download", repo, *(files or [])] + (["--revision", revision] if revision else []))
