from .inference_providers.inference_provider import InferenceProvider
from .data.patch_dataset import PatchDataset, MultiCellPatchDataset
from .data.resized_cell_dataset import ResizedCellDataset
from torch.utils.data import Dataset
import os

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
):
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
    """
    for model_name, model_cfg in configs.items():
        inference_provider = select_inference_provider(model_name)
        inference_provider.load_model()
        if getattr(inference_provider, 'use_pca', False):
            pca_datasets = [dataset_cls(**run['dataset_configs']) for run in model_cfg['inference_runs']]
            inference_provider.fit_pca(pca_datasets)
        for inference_run in model_cfg['inference_runs']:
            dataset = dataset_cls(**inference_run['dataset_configs'])
            print(f"Running inference [{model_name}] on {inference_run['dataset_configs']['wsi_path']}, cells: {len(dataset)}")
            print('-' * 20)
            run_attn_path = None
            if visualize_attention and attention_output_path is not None:
                wsi_name = os.path.basename(os.path.dirname(inference_run['output_path']))
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
            print('-' * 20)


def extract_embeddings_multicell(configs: dict, batch_size: int = 4):
    """Extract per-cell embeddings by tiling WSIs into patches and selecting each cell's token."""
    for model_name, model_cfg in configs.items():
        inference_provider = select_inference_provider(model_name)
        inference_provider.load_model()
        for inference_run in model_cfg['inference_runs']:
            dataset = MultiCellPatchDataset(**inference_run['dataset_configs'])
            print(f"Running multicell inference [{model_name}] on {inference_run['dataset_configs']['wsi_path']}, patches: {len(dataset)}, cells: {dataset.total_cells}")
            print('-' * 20)
            inference_provider.inference_multicell(dataset, inference_run['output_path'], batch_size=batch_size)
            print('-' * 20)


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
    offset_x: int = 0,
    offset_y: int = 0,
    cells_info_root: str | os.PathLike = _CELLS_INFO_ROOT,
    with_boundaries: bool = False,
    cell_boundaries_root: str | os.PathLike = _CELL_BOUNDARIES_ROOT,
    alignment_matrix_root: str | os.PathLike = _ALIGNMENT_MATRIX_ROOT,
) -> dict:
    """Build a config dict for extract_embeddings / run_attention_only.

    Args:
        model_name: key used by select_inference_provider (e.g. 'UNI2').
        infos: {dataset_name: {'converted': bool, 'model_output_dir': str, 'wsi_filename': str (optional)}}
            'wsi_filename' overrides the default '{dataset_name}_he_image.ome.tif' naming
            (e.g. for GSM samples named '{dataset_name}_registered_HE.ome.tif').
        output_suffix: appended to model_output_dir to form the output folder name.
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
        }
        if with_boundaries:
            dataset_configs.update({
                'cells_csv_path': os.path.join(cell_boundaries_root, f'{dataset_name}_out', 'cell_boundaries.csv.gz'),
                'nucleus_boundaries_path': os.path.join(cell_boundaries_root, f'{dataset_name}_out', 'nucleus_boundaries.parquet'),
                'alignment_matrix_path': os.path.join(alignment_matrix_root, f'{dataset_name}_he_imagealignment.csv'),
            })
        runs.append({
            'output_path': os.path.join(_OUTPUT_ROOT, f"{info['model_output_dir']}{output_suffix}", dataset_name),
            'dataset_configs': dataset_configs,
        })
    return {model_name: {'inference_runs': runs}}


def build_resized_cell_configs(
    model_name: str,
    infos: dict,
    size: int = 224,
    size_side: int | None = None,
    output_suffix: str = '_resized_h5',
    cells_info_root: str | os.PathLike = _CELLS_INFO_ROOT,
) -> dict:
    """Build a config dict wired for extract_embeddings(dataset_cls=ResizedCellDataset).

    Args:
        model_name: key used by select_inference_provider (e.g. 'UNI2').
        infos: {dataset_name: {'converted': bool, 'model_output_dir': str, 'wsi_filename': str (optional)}}
            'wsi_filename' overrides the default '{dataset_name}_he_image.ome.tif' naming
            (e.g. for GSM samples named '{dataset_name}_registered_HE.ome.tif').
        size: output patch size (patches are resized to size × size).
        size_side: if None, crop side is derived per-cell from its Xenium
            boundary polygon; otherwise every cell uses a fixed size_side ×
            size_side crop around its centroid (see ResizedCellDataset).
        cells_info_root: root directory containing {dataset_name}/patch_coordinates.h5.
            Defaults to _CELLS_INFO_ROOT (Marc's ground-truth pipeline output); pass
            _POSITIONS_CONVERTED_ROOT to use the CellViT-centroid-corrected patch
            coordinates from src/python/OT/export_matched_patch_coordinates.py instead.

    Returns:
        {model_name: {'inference_runs': [{'dataset_configs': ..., 'output_path': ...}]}}
    """
    runs = []
    for dataset_name, info in infos.items():
        wsi_root = _WSI_ROOT_CONVERTED if info['converted'] else _WSI_ROOT_RAW
        wsi_filename = info.get('wsi_filename', f'{dataset_name}_he_image.ome.tif')
        runs.append({
            'output_path': os.path.join(_OUTPUT_ROOT, f"{info['model_output_dir']}{output_suffix}", dataset_name),
            'dataset_configs': {
                'wsi_path': os.path.join(wsi_root, wsi_filename),
                'cells_info_path': os.path.join(cells_info_root, dataset_name, 'patch_coordinates.h5'),
                'cells_csv_path': os.path.join(_CELL_BOUNDARIES_ROOT, f'{dataset_name}_out', 'cell_boundaries.csv.gz'),
                'alignment_matrix_path': os.path.join(_ALIGNMENT_MATRIX_ROOT, f'{dataset_name}_he_imagealignment.csv'),
                'size': size,
                'size_side': size_side,
                'transform': None,
            },
        })
    return {model_name: {'inference_runs': runs}}


def build_multicell_configs(
    model_name: str,
    infos: dict,
    x_size: int = 224,
    y_size: int = 224,
    output_suffix: str = '_multicell_h5',
) -> dict:
    """Like build_configs but wired for extract_embeddings_multicell."""
    return build_configs(model_name, infos, x_size=x_size, y_size=y_size, output_suffix=output_suffix)


def merge_configs(*configs: dict) -> dict:
    """Merge multiple config dicts (same or different models) into one."""
    merged = {}
    for cfg in configs:
        for model_name, model_cfg in cfg.items():
            if model_name not in merged:
                merged[model_name] = {'inference_runs': []}
            merged[model_name]['inference_runs'].extend(model_cfg['inference_runs'])
    return merged


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    GSM_SAMPLE_IDS = [
        'GSM7990532', 'GSM7990533', 'GSM7990534', 'GSM7990537', 'GSM7990540',
        'GSM7990541', 'GSM7990543', 'GSM7990549', 'GSM7990550', 'GSM7990551',
        'GSM7990554', 'GSM7990556', 'GSM7990557', 'GSM7990558', 'GSM7990559',
    ]

    # Attention-map extraction (UNI2 only): 3 GSM samples + 3 colon samples, at
    # both the "normal" 224×224 centre patch and a tighter 100×100 size_side crop
    # (upsampled to 224×224) around each cell centroid -- same cell selection for
    # both, just a different field of view fed to the model.

    ATTN_COLON_SAMPLES = {
        'Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE': {'converted': False, 'model_output_dir': 'UNI2'},
        'Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE': {'converted': False, 'model_output_dir': 'UNI2'},
        'Xenium_V1_hColon_Cancer_Add_on_FFPE':             {'converted': False, 'model_output_dir': 'UNI2'},
    }

    # GSM samples use the CellViT-centroid-corrected patch coordinates
    # (_POSITIONS_CONVERTED_ROOT), same convention as elsewhere in this file for GSM.
    ATTN_GSM_SAMPLES = {
        gsm_id: {
            'converted': True,
            'model_output_dir': 'UNI2',
            'wsi_filename': f'{gsm_id}_registered_HE.ome.tif',
        }
        for gsm_id in GSM_SAMPLE_IDS[3:6]
    }

    # Normal size: standard 224×224 centre patch (PatchDataset).
    attn_normal_configs = merge_configs(
        build_configs('UNI2', ATTN_COLON_SAMPLES),
        build_configs('UNI2', ATTN_GSM_SAMPLES, cells_info_root=_POSITIONS_CONVERTED_ROOT),
    )
    run_attention_only(
        attn_normal_configs, os.path.join(_OUTPUT_ROOT, 'attention_maps_UNI2'), n_samples=250,
    )

    # 100 size_side: fixed 100×100 crop around each cell centroid, resized up to
    # 224×224 (ResizedCellDataset). Saved to its own attention_maps_UNI2_100 root
    # so it doesn't clash with the normal-size run above.
    attn_100_configs = merge_configs(
        build_resized_cell_configs('UNI2', ATTN_COLON_SAMPLES, size_side=100),
        build_resized_cell_configs('UNI2', ATTN_GSM_SAMPLES, size_side=100, cells_info_root=_POSITIONS_CONVERTED_ROOT),
    )
    run_attention_only(
        attn_100_configs, os.path.join(_OUTPUT_ROOT, 'attention_maps_UNI2_100'), n_samples=250,
        dataset_cls=ResizedCellDataset,
    )

    # ── UNI2 native-resolution embeddings, cross-cancer cohort ─────────────────
    # The 7 WSIs from configs/train.yaml's LWO splits (lung x2, colon, colorectal,
    # ovarian, liver, skin) -- raw (non-GSM) samples, same 'converted': False
    # convention as ATTN_COLON_SAMPLES above. Standard 224×224 native centre patch
    # (PatchDataset default), saving the four central tokens + CLS (as usual) plus
    # the boundary-overlap mean-pooled embeddings_cell / embeddings_nucleus
    # (with_boundaries=True wires in each sample's cell_boundaries.csv.gz /
    # nucleus_boundaries.parquet / alignment matrix). Written to its own
    # UNI2_specific_tokens_folder_h5 root so it doesn't clash with any existing
    # UNI2_h5 extraction that predates save_cell/save_nucleus.
    CROSS_CANCER_SAMPLES = {
        'Xenium_V1_humanLung_Cancer_FFPE':                 {'converted': False, 'model_output_dir': 'UNI2_specific_tokens_folder'},
        'Xenium_V1_Human_Lung_Cancer_Addon_FFPE':          {'converted': False, 'model_output_dir': 'UNI2_specific_tokens_folder'},
        'Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE': {'converted': False, 'model_output_dir': 'UNI2_specific_tokens_folder'},
        'Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE':    {'converted': False, 'model_output_dir': 'UNI2_specific_tokens_folder'},
        'Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE':       {'converted': False, 'model_output_dir': 'UNI2_specific_tokens_folder'},
        'Xenium_V1_hLiver_cancer_section_FFPE':            {'converted': False, 'model_output_dir': 'UNI2_specific_tokens_folder'},
        'Xenium_Prime_Human_Skin_FFPE':                    {'converted': False, 'model_output_dir': 'UNI2_specific_tokens_folder'},
    }

    specific_tokens_configs = build_configs('UNI2', CROSS_CANCER_SAMPLES, with_boundaries=True)
    extract_embeddings(specific_tokens_configs, save_cls=True, save_cell=True, save_nucleus=True)