from .inference_providers.inference_provider import InferenceProvider
from .data.patch_dataset import PatchDataset, MultiCellPatchDataset
from .data.resized_cell_dataset import ResizedCellDataset
from torch.utils.data import Dataset
import argparse
import csv
import h5py
import json
import os
import sys
from pathlib import Path

from src.python.code_configs import paths

# ── data roots (see src/python/code_configs/paths.py and .env.example) ─────────

_OUTPUT_ROOT           = paths.DATASETS_ROOT
_CELLS_INFO_ROOT       = paths.XENIUM_PROCESSED_OUTPUT_ROOT
_WSI_ROOT_RAW          = paths.WSI_RAW_ROOT
_WSI_ROOT_CONVERTED    = paths.WSI_CONVERTED_ROOT
_CELL_BOUNDARIES_ROOT  = paths.XENIUM_OUTPUT_ROOT
_ALIGNMENT_MATRIX_ROOT = paths.ALIGNMENT_MATRIX_ROOT
# CellViT-centroid-corrected patch_coordinates.h5 (src/python/ot/export_matched_patch_coordinates.py):
# same cell_id/cell_type labels as _CELLS_INFO_ROOT's ground truth, but x_start/y_start
# recomputed from the matched CellViT nucleus's own centroid instead of Marc's.
_POSITIONS_CONVERTED_ROOT = paths.POSITIONS_CONVERTED_ROOT

# ── skip already-extracted slides ─────────────────────────────────────────────

def is_extracted(output_path: str | os.PathLike) -> bool:
    """Whether {output_path}/embeddings_dataset.h5 holds a finished extraction.

    create_output_file preallocates every dataset at full length before inference
    starts and save_embeddings* fills it batch by batch in order, so a job killed
    mid-run leaves a full-shaped file with empty trailing rows. Only treat the file
    as done if it opens and every cell_ids entry has been written (not only the last
    one: two jobs writing the same file can leave a hole in the middle).

    Opened with locking=False so a dry run / the notebook checking a slide that is
    being extracted doesn't make the writer's next h5py.File(..., 'a') fail with
    BlockingIOError.
    """
    h5_path = os.path.join(output_path, 'embeddings_dataset.h5')
    if not os.path.isfile(h5_path):
        return False
    try:
        with h5py.File(h5_path, 'r', locking=False) as f:
            cell_ids = f['cell_ids'][:]
            if len(cell_ids) == 0:
                return False
            n_empty = sum(1 for c in cell_ids if len(c) == 0)
            if n_empty:
                print(f"[skip_existing] {h5_path}: {n_empty}/{len(cell_ids)} cell_ids not written; will re-extract")
            return n_empty == 0
    except (OSError, KeyError) as e:
        print(f"[skip_existing] unreadable {h5_path} ({e}); will re-extract")
        return False


def _drop_extracted_runs(model_name: str, runs: list[dict]) -> list[dict]:
    remaining = []
    for run in runs:
        if is_extracted(run['output_path']):
            print(f"[skip_existing] [{model_name}] already extracted, skipping: {run['output_path']}")
        else:
            remaining.append(run)
    print(f"[skip_existing] [{model_name}] {len(runs) - len(remaining)} already extracted, {len(remaining)} to run")
    return remaining


# ── inference runners ─────────────────────────────────────────────────────────

def run_attention_only(
    configs: dict,
    output_path: str,
    n_samples: int = 10,
    dataset_cls: type[Dataset] = PatchDataset,
):
    """Run attention-only forward passes and save per-head attention maps.

    dataset_cls: dataset class to instantiate per inference run, e.g. PatchDataset
        (default, fixed-size centre patch) or ResizedCellDataset (a size_side ×
        size_side crop around the cell centroid, resized up to `size`). Must
        expose the same interface PatchDataset uses (x_size/y_size, labels_dataset,
        get_raw_patch) and yield (patch, label, cell_id) triples.

    Output layout::
        {output_path}/{model_name}/{wsi_name}/{label}/{cell_id}/
            patch.png
            cls/weights.npy  cls/head_00.png  ...
            top_left/weights.npy  top_left/head_00.png  ...
    """
    for model_name, model_cfg in configs.items():
        provider = select_inference_provider(model_name)
        provider.load_model()
        for run in model_cfg['inference_runs']:
            dataset = dataset_cls(**run['dataset_configs'])
            wsi_name = os.path.basename(run['output_path'])
            attn_path = os.path.join(output_path, model_name, wsi_name)
            print(f"Running attention-only for {model_name} on {wsi_name}, cells: {len(dataset)}")
            print('-' * 20)
            provider.run_attention_only(dataset, attn_path, n_samples=n_samples)
            print('-' * 20)


def extract_embeddings(
    configs: dict,
    batch_size: int = 16,
    visualize_attention: bool = False,
    attention_output_path: str | None = None,
    n_attention_samples: int = 10,
    save_cls: bool = True,
    save_cell: bool = False,
    save_nucleus: bool = False,
    dataset_cls: type[Dataset] = PatchDataset,
    skip_existing: bool = True,
    continue_on_error: bool = False,
) -> list[str]:
    """Extract per-cell embeddings using the centre-patch approach (one patch per cell).

    dataset_cls: dataset class to instantiate per inference run, e.g. PatchDataset
        (default, reads fixed-size patches from an H5 coordinate file) or
        ResizedCellDataset (crops a per-cell or fixed-size box around Xenium cell
        boundaries and resizes it). Must yield (patch, label, cell_id) triples.

    save_cell / save_nucleus: in addition to the fixed four-central-token patches
        (self.patches_to_save on the inference provider), mean-pool the tokens whose
        pixel footprint overlaps that cell's/nucleus's Xenium boundary polygon and
        save them as embeddings_cell / embeddings_nucleus. Requires each run's
        dataset_configs to include cells_csv_path / nucleus_boundaries_path and
        alignment_matrix_path (see build_configs(..., with_boundaries=True) and
        PatchDataset).

    skip_existing: skip every run whose output_path already holds a finished
        embeddings_dataset.h5 (see is_extracted).

    continue_on_error: if True, a run that raises is logged and the remaining runs
        still go ahead (its half-written embeddings_dataset.h5 fails is_extracted, so
        the next skip_existing pass re-extracts it).

    Returns:
        output_path of every run that raised (always empty unless continue_on_error).
    """
    failed = []
    for model_name, model_cfg in configs.items():
        runs = model_cfg['inference_runs']
        if skip_existing:
            runs = _drop_extracted_runs(model_name, runs)
            if not runs:
                continue
        inference_provider = select_inference_provider(model_name)
        inference_provider.load_model()
        if getattr(inference_provider, 'use_pca', False):
            pca_datasets = [dataset_cls(**run['dataset_configs']) for run in runs]
            inference_provider.fit_pca(pca_datasets)
        for inference_run in runs:
            try:
                dataset = dataset_cls(**inference_run['dataset_configs'])
                print(f"Running inference [{model_name}] on {inference_run['dataset_configs']['wsi_path']}, cells: {len(dataset)}")
                print('-' * 20)
                run_attn_path = None
                if visualize_attention and attention_output_path is not None:
                    wsi_name = os.path.basename(inference_run['output_path'])
                    run_attn_path = os.path.join(attention_output_path, model_name, wsi_name)
                inference_provider.inference(
                    dataset, inference_run['output_path'],
                    batch_size=batch_size,
                    visualize_attention=visualize_attention,
                    attention_output_path=run_attn_path,
                    n_attention_samples=n_attention_samples,
                    save_cls=save_cls,
                    save_cell=save_cell,
                    save_nucleus=save_nucleus,
                )
            except Exception as e:
                if not continue_on_error:
                    raise
                _report_run_failure(model_name, inference_run, e)
                failed.append(inference_run['output_path'])
            print('-' * 20)
    return failed


def extract_embeddings_multicell(
    configs: dict,
    batch_size: int = 4,
    save_cell: bool = False,
    save_nucleus: bool = False,
    skip_existing: bool = True,
    continue_on_error: bool = False,
) -> list[str]:
    """Extract per-cell embeddings by tiling WSIs into patches and selecting each cell's token.

    save_cell / save_nucleus: forwarded to each provider's inference_multicell (see
        InferenceProvider.pool_boundary_tokens_multicell) -- requires each run's
        dataset_configs to include cells_csv_path / nucleus_boundaries_path /
        alignment_matrix_path (see build_multicell_configs(..., with_boundaries=True)).

    skip_existing: skip every run whose output_path already holds a finished
        embeddings_dataset.h5 (see is_extracted).

    continue_on_error: see extract_embeddings. Returns the output_path of every run
        that raised.
    """
    failed = []
    for model_name, model_cfg in configs.items():
        runs = model_cfg['inference_runs']
        if skip_existing:
            runs = _drop_extracted_runs(model_name, runs)
            if not runs:
                continue
        inference_provider = select_inference_provider(model_name)
        inference_provider.load_model()
        for inference_run in runs:
            try:
                dataset = MultiCellPatchDataset(**inference_run['dataset_configs'])
                print(f"Running multicell inference [{model_name}] on {inference_run['dataset_configs']['wsi_path']}, patches: {len(dataset)}, cells: {dataset.total_cells}")
                print('-' * 20)
                inference_provider.inference_multicell(
                    dataset, inference_run['output_path'], batch_size=batch_size,
                    save_cell=save_cell, save_nucleus=save_nucleus,
                )
            except Exception as e:
                if not continue_on_error:
                    raise
                _report_run_failure(model_name, inference_run, e)
                failed.append(inference_run['output_path'])
            print('-' * 20)
    return failed


def _report_run_failure(model_name: str, inference_run: dict, e: Exception) -> None:
    import traceback
    traceback.print_exc()
    print(f"[continue_on_error] [{model_name}] FAILED on {inference_run['dataset_configs']['wsi_path']} "
          f"-> {inference_run['output_path']}: {type(e).__name__}: {e}")


# ── provider factory ──────────────────────────────────────────────────────────

def select_inference_provider(model_name: str) -> InferenceProvider:
    match model_name:
        case 'VirchowV2':
            from .inference_providers.virchow_v2_inference_provider import VirchowV2InferenceProvider
            return VirchowV2InferenceProvider()
        case 'UNI2':
            from .inference_providers.uni2_inference_provider import UNI2InferenceProvider
            return UNI2InferenceProvider()
        case 'CellViT-SAM':
            from .inference_providers.cellvit_inference_provider import CellViTInferenceProvider
            return CellViTInferenceProvider('SAM')
        case 'CellViT-HIPT':
            from .inference_providers.cellvit_inference_provider import CellViTInferenceProvider
            return CellViTInferenceProvider('HIPT')
        case 'CTransPath':
            from .inference_providers.ctranspath_inference_provider import CTransPathInferenceProvider
            return CTransPathInferenceProvider()
        case 'CONCH':
            from .inference_providers.conch_inference_provider import CONCHInferenceProvider
            return CONCHInferenceProvider()
        case 'HOptimus1':
            from .inference_providers.hoptimus1_inference_provider import HOptimus1InferenceProvider
            return HOptimus1InferenceProvider()
        case 'PhikonV2':
            from .inference_providers.phikon_v2_inference_provider import PhikonV2InferenceProvider
            return PhikonV2InferenceProvider()
        case 'Dummy':
            from .inference_providers.dummy_inference_provider import DummyInferenceProvider
            return DummyInferenceProvider(use_pca=True)
        case _:
            raise ValueError(f"Model '{model_name}' is not supported.")


# ── config builders ───────────────────────────────────────────────────────────

def build_configs(
    model_name: str,
    infos: dict,
    x_size: int = 224,
    y_size: int = 224,
    output_suffix: str = '_h5',
    correction_name: str = 'raw',
    offset_x: int = 0,
    offset_y: int = 0,
    cells_info_root: str | os.PathLike = _CELLS_INFO_ROOT,
    with_boundaries: bool = False,
    cell_boundaries_root: str | os.PathLike = _CELL_BOUNDARIES_ROOT,
    alignment_matrix_root: str | os.PathLike = _ALIGNMENT_MATRIX_ROOT,
    mask: bool = False,
    mask_grid_size: int = 3,
    mask_token_size: int = 14,
    mask_value: tuple[int, int, int] = (127, 127, 127),
) -> dict:
    """Build a config dict for extract_embeddings / run_attention_only.

    Args:
        model_name: key used by select_inference_provider (e.g. 'UNI2').
        infos: {dataset_name: {'converted': bool, 'model_output_dir': str, 'wsi_filename': str (optional)}}
            'wsi_filename' overrides the default '{dataset_name}_he_image.ome.tif' naming
            (e.g. for GSM samples named '{dataset_name}_registered_HE.ome.tif').
        output_suffix: appended to model_output_dir to form the output folder name.
        correction_name: subfolder between the model folder and the dataset folder,
            i.e. output_path = {DATASETS_ROOT}/{model_output_dir}{output_suffix}/{correction_name}/{dataset_name},
            the layout classifier_training / deepspot_training read from. Defaults to
            'raw' (uncorrected embeddings).
        offset_x: pixel offset added to every cell x_start (passed to PatchDataset).
        offset_y: pixel offset added to every cell y_start (passed to PatchDataset).
        cells_info_root: root directory containing {dataset_name}/patch_coordinates.h5.
            Defaults to _CELLS_INFO_ROOT (Marc's ground-truth pipeline output); pass
            _POSITIONS_CONVERTED_ROOT to use the CellViT-centroid-corrected patch
            coordinates from src/python/OT/export_matched_patch_coordinates.py instead.
        with_boundaries: if True, also wire in each dataset_name's Xenium
            cell_boundaries.csv.gz / nucleus_boundaries.parquet and alignment matrix
            (same layout as build_resized_cell_configs), so PatchDataset can compute
            boundary_polygon(idx, 'cell'/'nucleus') for extract_embeddings(save_cell=...,
            save_nucleus=...). No-op unless those flags are also passed to
            extract_embeddings().
        cell_boundaries_root / alignment_matrix_root: roots for the boundary files
            (only used when with_boundaries=True).
        mask / mask_grid_size / mask_token_size / mask_value: forwarded to PatchDataset
            to mask the central mask_grid_size x mask_grid_size model tokens (see
            PatchDataset and data.central_mask). mask_token_size should match the
            target model's ViT patch size (default 14).

    Returns:
        {model_name: {'inference_runs': [{'dataset_configs': ..., 'output_path': ...}]}}
    """
    runs = []
    for dataset_name, info in infos.items():
        wsi_root = _WSI_ROOT_CONVERTED if info['converted'] else _WSI_ROOT_RAW
        wsi_filename = info.get('wsi_filename', f'{dataset_name}_he_image.ome.tif')
        dataset_configs = {
            'wsi_path': os.path.join(wsi_root, wsi_filename),
            'cells_info_path': os.path.join(cells_info_root, dataset_name, 'patch_coordinates.h5'),
            'x_size': x_size,
            'y_size': y_size,
            'transform': None,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'mask': mask,
            'mask_grid_size': mask_grid_size,
            'mask_token_size': mask_token_size,
            'mask_value': mask_value,
        }
        if with_boundaries:
            dataset_configs.update({
                'cells_csv_path': os.path.join(cell_boundaries_root, f'{dataset_name}_out', 'cell_boundaries.csv.gz'),
                'nucleus_boundaries_path': os.path.join(cell_boundaries_root, f'{dataset_name}_out', 'nucleus_boundaries.parquet'),
                'alignment_matrix_path': os.path.join(alignment_matrix_root, f'{dataset_name}_he_imagealignment.csv'),
            })
        runs.append({
            'output_path': os.path.join(_OUTPUT_ROOT, f"{info['model_output_dir']}{output_suffix}", correction_name, dataset_name),
            'dataset_configs': dataset_configs,
        })
    return {model_name: {'inference_runs': runs}}


def build_resized_cell_configs(
    model_name: str,
    infos: dict,
    size: int = 224,
    size_side: int | None = None,
    output_suffix: str = '_resized_h5',
    correction_name: str = 'raw',
    cells_info_root: str | os.PathLike = _CELLS_INFO_ROOT,
    cell_boundaries_root: str | os.PathLike = _CELL_BOUNDARIES_ROOT,
    alignment_matrix_root: str | os.PathLike = _ALIGNMENT_MATRIX_ROOT,
    offset: int = 112,
    mask: bool = False,
    mask_grid_size: int = 3,
    mask_token_size: int = 14,
    mask_value: tuple[int, int, int] = (127, 127, 127),
) -> dict:
    """Build a config dict wired for extract_embeddings(dataset_cls=ResizedCellDataset).

    Args:
        model_name: key used by select_inference_provider (e.g. 'UNI2').
        infos: {dataset_name: {'converted': bool, 'model_output_dir': str, 'wsi_filename': str (optional)}}
            'wsi_filename' overrides the default '{dataset_name}_he_image.ome.tif' naming
            (e.g. for GSM samples named '{dataset_name}_registered_HE.ome.tif').
        size: output patch size (patches are resized to size × size).
        correction_name: subfolder between the model folder and the dataset folder
            (see build_configs). Defaults to 'raw'.
        size_side: if None, crop side is derived per-cell from its Xenium
            boundary polygon; otherwise every cell uses a fixed size_side x
            size_side crop around its centroid (see ResizedCellDataset). E.g. for
            a 40x-scanned WSI, size_side=2*size crops at native (40x) pixel
            density over the same field of view a size x size crop would cover
            on a 20x scan, then resizes down to size x size.
        cells_info_root: root directory containing {dataset_name}/patch_coordinates.h5.
            Defaults to _CELLS_INFO_ROOT (Marc's ground-truth pipeline output); pass
            _POSITIONS_CONVERTED_ROOT to use the CellViT-centroid-corrected patch
            coordinates from src/python/OT/export_matched_patch_coordinates.py instead.
        cell_boundaries_root / alignment_matrix_root: roots for each dataset_name's
            Xenium cell_boundaries.csv.gz / nucleus_boundaries.parquet / alignment
            matrix (same layout as build_configs), wired in unconditionally so
            ResizedCellDataset can compute boundary_polygon(idx, 'cell'/'nucleus')
            for extract_embeddings(save_cell=..., save_nucleus=...). No-op unless
            those flags are also passed to extract_embeddings().
        mask / mask_grid_size / mask_token_size / mask_value: forwarded to
            ResizedCellDataset to mask the central mask_grid_size x mask_grid_size
            model tokens of the resized (size × size) output (see ResizedCellDataset
            and data.central_mask). mask_token_size should match the target model's
            ViT patch size (default 14).

    Returns:
        {model_name: {'inference_runs': [{'dataset_configs': ..., 'output_path': ...}]}}
    """
    runs = []
    for dataset_name, info in infos.items():
        wsi_root = _WSI_ROOT_CONVERTED if info['converted'] else _WSI_ROOT_RAW
        wsi_filename = info.get('wsi_filename', f'{dataset_name}_he_image.ome.tif')
        runs.append({
            'output_path': os.path.join(_OUTPUT_ROOT, f"{info['model_output_dir']}{output_suffix}", correction_name, dataset_name),
            'dataset_configs': {
                'wsi_path': os.path.join(wsi_root, wsi_filename),
                'cells_info_path': os.path.join(cells_info_root, dataset_name, 'patch_coordinates.h5'),
                'cells_csv_path': os.path.join(cell_boundaries_root, f'{dataset_name}_out', 'cell_boundaries.csv.gz'),
                'nucleus_boundaries_path': os.path.join(cell_boundaries_root, f'{dataset_name}_out', 'nucleus_boundaries.parquet'),
                'alignment_matrix_path': os.path.join(alignment_matrix_root, f'{dataset_name}_he_imagealignment.csv'),
                'size': size,
                'size_side': size_side,
                'transform': None,
                'cell_offset_x': offset,
                'cell_offset_y': offset,
                'mask': mask,
                'mask_grid_size': mask_grid_size,
                'mask_token_size': mask_token_size,
                'mask_value': mask_value,
            },
        })
    return {model_name: {'inference_runs': runs}}


def build_multicell_configs(
    model_name: str,
    infos: dict,
    x_size: int = 224,
    y_size: int = 224,
    output_suffix: str = '_multicell_h5',
    correction_name: str = 'raw',
    cell_offset_x: int = 112,
    cell_offset_y: int = 112,
    size_side_x: int | None = None,
    size_side_y: int | None = None,
    central_size_x: int | None = None,
    central_size_y: int | None = None,
    cells_info_root: str | os.PathLike = _CELLS_INFO_ROOT,
    with_boundaries: bool = False,
    cell_boundaries_root: str | os.PathLike = _CELL_BOUNDARIES_ROOT,
    alignment_matrix_root: str | os.PathLike = _ALIGNMENT_MATRIX_ROOT,
) -> dict:
    """Build a config dict wired for extract_embeddings_multicell (MultiCellPatchDataset).

    Args:
        model_name: key used by select_inference_provider (e.g. 'UNI2').
        infos: {dataset_name: {'converted': bool, 'model_output_dir': str, 'wsi_filename': str (optional)}}
            'wsi_filename' overrides the default '{dataset_name}_he_image.ome.tif' naming
            (e.g. for GSM samples named '{dataset_name}_registered_HE.ome.tif').
        x_size / y_size: model input tile size (the size fed to the model).
        correction_name: subfolder between the model folder and the dataset folder
            (see build_configs). Defaults to 'raw'.
        cell_offset_x / cell_offset_y: pixel offset from each cell's H5 x_start/y_start
            to its centroid (see MultiCellPatchDataset).
        size_side_x / size_side_y: if set, each tile is cropped at this native
            WSI-pixel size and resized down (or up) to x_size × y_size before it hits
            the model -- e.g. size_side_x=size_side_y=448 with x_size=y_size=224 crops
            a wider field of view per tile at native resolution then resizes it back
            down to the model's native 224×224 input, same "magnification-corrected"
            idea as build_resized_cell_configs' size_side, applied to the multicell
            tiling loop instead of a per-cell crop. Defaults (None) to x_size/y_size,
            i.e. no resize.
        central_size_x / central_size_y: restrict cell claiming to a centred
            sub-window of each (post-resize) tile, avoiding edge-of-tile field-of-view
            truncation for cells near a tile boundary (see MultiCellPatchDataset).
            Defaults (None) to x_size/y_size, i.e. no restriction.
        cells_info_root: root directory containing {dataset_name}/patch_coordinates.h5.
            Defaults to _CELLS_INFO_ROOT (Marc's ground-truth pipeline output); pass
            _POSITIONS_CONVERTED_ROOT to use the CellViT-centroid-corrected patch
            coordinates from src/python/OT/export_matched_patch_coordinates.py instead.
        with_boundaries: if True, also wire in each dataset_name's Xenium
            cell_boundaries.csv.gz / nucleus_boundaries.parquet and alignment matrix
            (same layout as build_configs), so MultiCellPatchDataset can compute
            boundary_polygon(...) for extract_embeddings_multicell(save_cell=...,
            save_nucleus=...). No-op unless those flags are also passed to
            extract_embeddings_multicell().
        cell_boundaries_root / alignment_matrix_root: roots for the boundary files
            (only used when with_boundaries=True).

    Returns:
        {model_name: {'inference_runs': [{'dataset_configs': ..., 'output_path': ...}]}}
    """
    runs = []
    for dataset_name, info in infos.items():
        wsi_root = _WSI_ROOT_CONVERTED if info['converted'] else _WSI_ROOT_RAW
        wsi_filename = info.get('wsi_filename', f'{dataset_name}_he_image.ome.tif')
        dataset_configs = {
            'wsi_path': os.path.join(wsi_root, wsi_filename),
            'cells_info_path': os.path.join(cells_info_root, dataset_name, 'patch_coordinates.h5'),
            'x_size': x_size,
            'y_size': y_size,
            'transform': None,
            'cell_offset_x': cell_offset_x,
            'cell_offset_y': cell_offset_y,
            'size_side_x': size_side_x,
            'size_side_y': size_side_y,
            'central_size_x': central_size_x,
            'central_size_y': central_size_y,
        }
        if with_boundaries:
            dataset_configs.update({
                'cells_csv_path': os.path.join(cell_boundaries_root, f'{dataset_name}_out', 'cell_boundaries.csv.gz'),
                'nucleus_boundaries_path': os.path.join(cell_boundaries_root, f'{dataset_name}_out', 'nucleus_boundaries.parquet'),
                'alignment_matrix_path': os.path.join(alignment_matrix_root, f'{dataset_name}_he_imagealignment.csv'),
            })
        runs.append({
            'output_path': os.path.join(_OUTPUT_ROOT, f"{info['model_output_dir']}{output_suffix}", correction_name, dataset_name),
            'dataset_configs': dataset_configs,
        })
    return {model_name: {'inference_runs': runs}}


def merge_configs(*configs: dict) -> dict:
    """Merge multiple config dicts (same or different models) into one."""
    merged = {}
    for cfg in configs:
        for model_name, model_cfg in cfg.items():
            if model_name not in merged:
                merged[model_name] = {'inference_runs': []}
            merged[model_name]['inference_runs'].extend(model_cfg['inference_runs'])
    return merged


# ── JSON run configs ──────────────────────────────────────────────────────────
#
# One JSON file = one extraction run: one model, one dataset/geometry
# configuration, one set of slides. Example (see configs/extract_embeddings/):
#
#   {
#     "model_name": "UNI2",
#     "mode": "patch",                      # "patch" | "resized" | "multicell"
#     "size": 224,                          # model input size (x_size = y_size = size)
#     "size_side": null,                    # native crop side before resizing to `size`
#                                           #   (resized / multicell only)
#     "output_suffix": "_h5",
#     "batch_size": 256,
#     "save_cls": true,
#     "samples_manifest": "../../notebooks/wsi_openslide_manifest.csv",
#     "cells_info_root_by_prefix": {"GSM79905": "positions_converted"}
#   }
#
# Output: {DATASETS_ROOT}/{model_output_dir}{output_suffix}/{correction_name}/{sample}/
# Relative paths inside the JSON (samples_manifest) are resolved against the JSON
# file's own directory. Keys starting with "_" are ignored (use them for comments).

_MODES = ('patch', 'resized', 'multicell')

_DEFAULT_OUTPUT_SUFFIX = {'patch': '_h5', 'resized': '_resized_h5', 'multicell': '_multicell_h5'}

_RUN_CONFIG_KEYS = {
    'model_name',                 # required: key for select_inference_provider ('UNI2', ...)
    'mode',                       # required: 'patch' (PatchDataset), 'resized' (ResizedCellDataset),
                                  #   'multicell' (MultiCellPatchDataset)
    'samples',                    # list of sample names, or {name: {'converted', 'wsi_filename', 'cells_info_root'}}
    'samples_manifest',           # CSV from notebooks/check_wsi_openslide.ipynb (dataset, converted,
                                  #   wsi_filename, status); rows with status == 'ok' are used
    'exclude_samples',            # list of sample names to drop
    'model_output_dir',           # default: model_name
    'output_suffix',              # default: '_h5' / '_resized_h5' / '_multicell_h5' by mode
    'correction_name',            # default: 'raw'
    'size',                       # default: 224
    'size_side',                  # default: null
    'batch_size',                 # default: 16 (4 for multicell)
    'save_cls',                   # default: true (not available for multicell)
    'save_cell',                  # default: false; true wires in the Xenium boundary files
    'save_nucleus',               # default: false; idem
    'cells_info_root',            # 'xenium_processed' (default), 'positions_converted', or a path
    'cells_info_root_by_prefix',  # {sample-name prefix: cells_info_root}, e.g. {'GSM79905': 'positions_converted'}
    'skip_existing',              # default: true (see is_extracted)
    'continue_on_error',          # default: true -- one broken slide doesn't stop the others
    'builder_kwargs',             # extra kwargs for the build_*configs function (offset, mask, central_size_x, ...)
}


def _resolve_cells_info_root(value: str) -> str | os.PathLike:
    match value:
        case 'xenium_processed':
            return _CELLS_INFO_ROOT
        case 'positions_converted':
            return _POSITIONS_CONVERTED_ROOT
        case _:
            return value


def _read_samples_manifest(path: Path) -> dict[str, dict]:
    infos = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('status', 'ok') != 'ok':
                continue
            infos[row['dataset']] = {
                'converted': row['converted'].strip().lower() in ('true', '1'),
                'wsi_filename': row['wsi_filename'],
            }
    return infos


def load_run_config(path: str | os.PathLike) -> dict:
    """Read + validate a JSON run config; resolve relative paths against its directory."""
    path = Path(path)
    with open(path) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith('_')}
    unknown = set(cfg) - _RUN_CONFIG_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; allowed: {sorted(_RUN_CONFIG_KEYS)}")
    for key in ('model_name', 'mode'):
        if key not in cfg:
            raise ValueError(f"{path}: missing required key {key!r}")
    if cfg['mode'] not in _MODES:
        raise ValueError(f"{path}: mode must be one of {_MODES}, got {cfg['mode']!r}")
    if 'samples' not in cfg and 'samples_manifest' not in cfg:
        raise ValueError(f"{path}: needs 'samples' and/or 'samples_manifest'")
    if 'samples_manifest' in cfg:
        cfg['samples_manifest'] = str((path.parent / cfg['samples_manifest']).resolve())
    cfg['_config_path'] = str(path)
    return cfg


def _resolve_samples(cfg: dict) -> dict[str, dict]:
    """{sample: {'converted', 'wsi_filename'?, 'cells_info_root'?}} from samples / samples_manifest."""
    manifest = _read_samples_manifest(Path(cfg['samples_manifest'])) if 'samples_manifest' in cfg else None
    samples = cfg.get('samples')
    if samples is None:
        infos = dict(manifest)
    elif isinstance(samples, list):
        if manifest is None:
            infos = {name: {'converted': False} for name in samples}
        else:
            not_ok = [name for name in samples if name not in manifest]
            if not_ok:
                raise ValueError(f"samples not marked 'ok' in {cfg['samples_manifest']}: {not_ok}")
            infos = {name: manifest[name] for name in samples}
    elif isinstance(samples, dict):
        infos = {name: {'converted': False, **(manifest or {}).get(name, {}), **(info or {})}
                 for name, info in samples.items()}
    else:
        raise ValueError("'samples' must be a list of names or a {name: info} dict")
    for name in cfg.get('exclude_samples', []):
        infos.pop(name, None)
    if not infos:
        raise ValueError(f"{cfg['_config_path']}: no samples left to extract")
    return infos


def build_run_configs(cfg: dict) -> dict:
    """Turn a loaded JSON run config into the {model_name: {'inference_runs': [...]}}
    dict the extract_embeddings* functions take."""
    mode = cfg['mode']
    model_name = cfg['model_name']
    size = cfg.get('size', 224)
    size_side = cfg.get('size_side')
    save_cell = cfg.get('save_cell', False)
    save_nucleus = cfg.get('save_nucleus', False)

    builder_kwargs = dict(cfg.get('builder_kwargs', {}))
    if 'mask_value' in builder_kwargs:
        builder_kwargs['mask_value'] = tuple(builder_kwargs['mask_value'])
    common = {
        'output_suffix': cfg.get('output_suffix', _DEFAULT_OUTPUT_SUFFIX[mode]),
        'correction_name': cfg.get('correction_name', 'raw'),
    }
    match mode:
        case 'patch':
            if size_side is not None:
                raise ValueError("mode 'patch' reads a native size x size patch; use mode 'resized' for size_side")
            builder = build_configs
            geometry = {'x_size': size, 'y_size': size, 'with_boundaries': save_cell or save_nucleus}
        case 'resized':
            builder = build_resized_cell_configs
            geometry = {'size': size, 'size_side': size_side}
        case 'multicell':
            builder = build_multicell_configs
            geometry = {'x_size': size, 'y_size': size, 'size_side_x': size_side, 'size_side_y': size_side,
                        'with_boundaries': save_cell or save_nucleus}
    clash = set(builder_kwargs) & (set(geometry) | set(common) | {'cells_info_root'})
    if clash:
        raise ValueError(f"builder_kwargs must not repeat top-level settings: {sorted(clash)}")

    # group samples by cells_info_root (per-sample > by-prefix > run-level default)
    default_root = cfg.get('cells_info_root', 'xenium_processed')
    by_prefix = cfg.get('cells_info_root_by_prefix', {})
    model_output_dir = cfg.get('model_output_dir', model_name)
    groups: dict[str, dict] = {}
    for name, info in _resolve_samples(cfg).items():
        info = dict(info)
        root = info.pop('cells_info_root', None) or next(
            (r for prefix, r in by_prefix.items() if name.startswith(prefix)), default_root)
        groups.setdefault(root, {})[name] = {**info, 'model_output_dir': model_output_dir}

    return merge_configs(*(
        builder(model_name, infos, cells_info_root=_resolve_cells_info_root(root),
                **geometry, **common, **builder_kwargs)
        for root, infos in groups.items()
    ))


def _input_paths(run: dict) -> list[str]:
    keys = ('wsi_path', 'cells_info_path', 'cells_csv_path', 'nucleus_boundaries_path', 'alignment_matrix_path')
    return [os.fspath(run['dataset_configs'][k]) for k in keys if run['dataset_configs'].get(k) is not None]


def run_from_config(cfg: dict, dry_run: bool = False) -> int:
    """Run one JSON run config. Returns an exit code: 0 all done, 2 no errors but at
    least one slide was skipped for missing input files, 3 at least one slide raised
    during extraction (1 stays Python's own code for an uncaught crash / bad config)."""
    mode = cfg['mode']
    save_cell = cfg.get('save_cell', False)
    save_nucleus = cfg.get('save_nucleus', False)
    if mode == 'multicell' and cfg.get('save_cls'):
        raise ValueError("save_cls is not available for mode 'multicell'")
    configs = build_run_configs(cfg)

    print(f"=== {cfg['_config_path']} ===")
    print(json.dumps({k: v for k, v in cfg.items() if not k.startswith('_')}, indent=2))

    # slides whose input files are missing (e.g. no access, CellViT/Xenium processing
    # not run yet) are reported and dropped instead of crashing the whole run
    missing = {}
    for model_name, model_cfg in configs.items():
        kept = []
        for run in model_cfg['inference_runs']:
            absent = [p for p in _input_paths(run) if not os.path.exists(p)]
            if absent:
                missing[run['output_path']] = absent
            else:
                kept.append(run)
        model_cfg['inference_runs'] = kept
    for output_path, absent in missing.items():
        print(f"[missing inputs] skipping {os.path.basename(output_path)}: " + ", ".join(absent))

    if dry_run:
        for model_name, model_cfg in configs.items():
            runs = model_cfg['inference_runs']
            done = [r for r in runs if is_extracted(r['output_path'])]
            print(f"[dry run] [{model_name}] {len(runs) + len(missing)} samples: {len(done)} already extracted, "
                  f"{len(runs) - len(done)} to run, {len(missing)} missing inputs")
            for r in runs:
                tag = 'done  ' if r in done else 'to run'
                print(f"  {tag} {r['dataset_configs']['wsi_path']} -> {r['output_path']}")
        return 2 if missing else 0

    skip_existing = cfg.get('skip_existing', True)
    continue_on_error = cfg.get('continue_on_error', True)
    batch_size = cfg.get('batch_size', 4 if mode == 'multicell' else 16)
    if mode == 'multicell':
        failed = extract_embeddings_multicell(
            configs, batch_size=batch_size, save_cell=save_cell, save_nucleus=save_nucleus,
            skip_existing=skip_existing, continue_on_error=continue_on_error,
        )
    else:
        failed = extract_embeddings(
            configs, batch_size=batch_size, save_cls=cfg.get('save_cls', True),
            save_cell=save_cell, save_nucleus=save_nucleus,
            dataset_cls=PatchDataset if mode == 'patch' else ResizedCellDataset,
            skip_existing=skip_existing, continue_on_error=continue_on_error,
        )

    print(f"=== finished {cfg['_config_path']}: {len(failed)} failed, {len(missing)} skipped for missing inputs ===")
    for output_path in failed:
        print(f"  FAILED  {output_path}")
    for output_path in missing:
        print(f"  MISSING {output_path}")
    return 3 if failed else 2 if missing else 0


# ── entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract embeddings for one JSON run config (one model configuration x one set of slides).")
    parser.add_argument('--config', required=True, help="path to the JSON run config")
    parser.add_argument('--dry-run', action='store_true',
                        help="only resolve samples and report done / to-run / missing inputs; no model is loaded")
    args = parser.parse_args(argv)
    return run_from_config(load_run_config(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
