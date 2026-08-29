"""Filesystem roots for externally-managed data, read from environment variables.

Every path here points at data that lives outside this repo and differs machine-to-
machine / user-to-user (raw and converted WSIs, Xenium instrument output, extracted
embeddings, model checkpoints, ...) -- nothing here is hardcoded. Copy `.env.example`
(repo root) to `.env`, fill in the paths for your machine, and either
`export $(grep -v '^#' .env | xargs)` or `set -a; source .env; set +a` before running
any script that touches these. See the top-level README's "Environment variables"
section for what each one is used for.

Lookups are lazy (`_LazyPath`, resolved on first `os.fspath()`/`str()`/`/` use, not at
import time) so importing a module that has one of these as a *default* argument value
never fails just because some unrelated env var isn't set -- only the entry points that
actually touch a given root require that root's variable, and callers that pass an
explicit path (or pre-loaded data, bypassing disk I/O entirely) never need it set at all.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Environment variable {name!r} is not set. Copy .env.example to .env, fill "
            f"in the paths for your machine, and source it (or export {name} directly) "
            f"before running this script -- see the README's Environment variables section."
        )
    return Path(value)


class _LazyPath(os.PathLike):
    """A path whose backing environment variable is only looked up on first use."""

    def __init__(self, name: str):
        self._name = name

    def _resolve(self) -> Path:
        return _env_path(self._name)

    def __fspath__(self) -> str:
        return str(self._resolve())

    def __str__(self) -> str:
        return str(self._resolve())

    def __repr__(self) -> str:
        return f"_LazyPath({self._name!r})"

    def __truediv__(self, other) -> Path:
        return self._resolve() / other


# Root where extracted embeddings are written/read, and (as `data.base_dir` in
# configs/*.yaml) where classifier_training / deepspot_training load them from:
# {DATASETS_ROOT}/{model_name}_h5/{correction_name}/{wsi}/embeddings_dataset.h5
DATASETS_ROOT = _LazyPath("DATASETS_ROOT")

# Per-sample processed Xenium pipeline output: {root}/{sample}/patch_coordinates.h5
# (ground-truth cell positions/types, see extract_embeddings/data/patch_dataset.py) and
# {root}/{sample}/adata.h5ad (gene-expression AnnData, used by deepspot_training).
XENIUM_PROCESSED_OUTPUT_ROOT = _LazyPath("XENIUM_PROCESSED_OUTPUT_ROOT")

# Root of raw H&E WSIs: {root}/{wsi_filename}.
WSI_RAW_ROOT = _LazyPath("WSI_RAW_ROOT")

# Root of converted/registered H&E WSIs (see notebooks referenced in src/python/ot/).
WSI_CONVERTED_ROOT = _LazyPath("WSI_CONVERTED_ROOT")

# Root of raw per-sample Xenium instrument output: nucleus_boundaries.parquet,
# cell_boundaries.csv.gz, experiment.xenium.
XENIUM_OUTPUT_ROOT = _LazyPath("XENIUM_OUTPUT_ROOT")

# Root of per-sample Xenium<->H&E affine alignment matrices:
# {root}/{sample}_he_imagealignment.csv.
ALIGNMENT_MATRIX_ROOT = _LazyPath("ALIGNMENT_MATRIX_ROOT")

# CellViT-centroid-corrected patch_coordinates.h5 (same schema as
# XENIUM_PROCESSED_OUTPUT_ROOT's ground truth, but x_start/y_start recomputed from the
# matched CellViT nucleus's own centroid). Read by extract_embeddings.py, written by
# src/python/ot/export_matched_patch_coordinates.py.
POSITIONS_CONVERTED_ROOT = _LazyPath("POSITIONS_CONVERTED_ROOT")

# Root of CellViT nucleus-segmentation cells.geojson outputs (used only by
# src/python/ot/centering_correction.py).
CELLVIT_GEOJSON_OUTPUT_ROOT = _LazyPath("CELLVIT_GEOJSON_OUTPUT_ROOT")

# Directory containing CellViT checkpoint files (CellViT-SAM-H-x40-AMP.pth,
# CellViT-256-x40-AMP.pth). Used by
# extract_embeddings/inference_providers/cellvit_inference_provider.py.
CELLVIT_CACHE = _LazyPath("CELLVIT_CACHE")

# Full path to the CTransPath checkpoint file (ctranspath.pth). Used by
# extract_embeddings/inference_providers/ctranspath_inference_provider.py.
CTRANSPATH_WEIGHTS = _LazyPath("CTRANSPATH_WEIGHTS")
