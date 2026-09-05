# sc-central-tokens-code

Code for *"Cell-level representations in pathology foundation models are a question of
read-out"* (NeurIPS 2026 workshop submission, *Machine Learning for Spatially Resolved
High-dimensional Biology*).

Pathology foundation models (UNI2, Virchow2, H-Optimus-1, ...) are pretrained with
DINOv2 at the tile level, yet many downstream tasks are cell-level. The standard
practice — crop a window around a cell, resize it to the model's input size, take the
CLS token — forces a trade-off between cell specificity and spatial context. This
repo compares that baseline against **spatial indexing**: feeding an *uncropped*,
native-resolution tile centered on the cell and reading off the patch tokens that
overlap it, exploiting the fact that DINOv2's iBOT objective already makes those tokens
locally meaningful. Ground truth (cell type and spatial gene expression) comes from 10x
Xenium; H&E patches are located via CellViT nucleus detection.

> **Note on anonymity.** This README intentionally omits author names/affiliations —
> the paper is submitted under NeurIPS's double-blind option. Code comments still
> reference collaborators by first name in a few places (e.g. `ot/` docstrings); scrub
> those, and check `git log` authorship, before releasing this repo publicly if
> anonymity still matters at that point.

## Read-out notation

Every experiment reduces to picking (a) what image is fed to the frozen encoder and
(b) which output token(s) are read off. All conditions from the paper share the same
patch size (224×224, 14px ViT patches → 16×16 token grid for UNI2/Virchow2/H-Optimus-1)
and are produced by the config builders in `extract_embeddings.py` (Stage 1 below):

| Symbol | Name | Image fed in | Token read | Builder |
|---|---|---|---|---|
| $\rho_{\text{tok}}$ | **Spatial indexing (ours)** | 224×224 tile, native resolution, centered on the cell centroid | mean of the 4 patch tokens straddling the centroid | `build_configs` |
| $\rho_{\text{CLS}}^{\text{native}}$ | Uncropped-tile CLS | *same* 224×224 native tile | CLS token | `build_configs` (`save_cls=True`) |
| $\rho_{\text{CLS}}^{\text{resize}}$ | Crop-and-resize (baseline) | 100×100px crop around the cell, resized to 224×224 | CLS token | `build_resized_cell_configs(size_side=100)` |
| $\rho_{\text{tok}\|\text{CLS}}$ | Token+CLS pool (ablation) | native tile | mean of the 4 tokens *and* CLS together | either of the above, combined at load time |

$\rho_{\text{tok}}$ and $\rho_{\text{CLS}}^{\text{native}}$ come from **the same
extraction call** (`save_cls=True` writes both the four corner tokens and the CLS
token into one H5 file) — only which `data.embeddings_datasets` you train on differs.
$\rho_{\text{CLS}}^{\text{resize}}$ needs a separate extraction pass with
`ResizedCellDataset`. See "Reproducing the paper's read-out conditions" under Stage 1.

## Repository layout

```
configs/                          Hydra configs (see "Datasets" and Stage 2/DeepCell below)
  train.yaml                      classifier_training — cross-cancer cohort, 14-fold LOSO
  train_colon.yaml                classifier_training — colon-cancer cohort, 8-fold LOSO
  deepspot_train.yaml             deepspot_training — same WSI mix as train.yaml (general-purpose)
  deepspot_train_gsm.yaml         deepspot_training — lung-fibrosis (GSM) cohort, 15-fold LOSO

src/python/
  extract_embeddings/             Stage 1: run foundation models over per-cell H&E patches
    extract_embeddings.py           entry point + per-condition config builders
    data/                           PyTorch Datasets reading patch_coordinates.h5
    inference_providers/            one provider per foundation model (below)
  classifier_training/            Stage 2: cell-type classifier head, Leave-One-Slide-Out
    README.md                       full usage docs — start here for stage 2 detail
  deepspot_training/               Stage 2b (optional): DeepSpot DeepCell gene-expression regression
    README.md                       full usage docs
  ot/                              CellViT↔Xenium nucleus matching / re-centering (below)
  cellvit_outputs_processing/      geojson repair utility used by ot/
  code_configs/
    mappings.py                     cell-type label groupings (see "Cell-type mappings")
    paths.py                        environment-variable-backed data roots (see below)
  utils/                           shared HDF5 embeddings I/O helpers

.env.example                      template for the environment variables below
requirements.txt                  Python dependencies (a few installed from source, see below)
```

## Environment setup

```bash
conda create -n cellViT python=3.10
conda activate cellViT
pip install -r requirements.txt
```

`requirements.txt` is pinned to the exact versions verified in the `cellViT` conda env
this codebase was developed and run in — including `cellvit`, `conch`, and `deepspot`
(the three foundation-model/pathology-specific packages), each pip-installable
directly; see the fallback source-install commands in `requirements.txt`'s comments if
`pip install <name>` doesn't resolve to the intended package on a fresh machine.
`openslide-bin` bundles the native OpenSlide library, so no separate system package
should be needed; fall back to `apt install openslide-tools` / `brew install
openslide` / `conda install -c conda-forge openslide` if it doesn't work on your
platform.

UNI2, Virchow2, H-Optimus-1 and CONCH are gated on Hugging Face — request access on
each model's HF page, then either `huggingface-cli login` or set `HF_TOKEN` (below).

## Environment variables

All filesystem paths this repo touches are read from environment variables — nothing is
hardcoded. Copy `.env.example` to `.env`, fill in the paths for your machine, and

```bash
set -a; source .env; set +a
```

before running anything. You only need to set the variables the script you're running
actually touches (each is looked up lazily, at first use — see
`src/python/code_configs/paths.py`); the error message names the missing variable if you
forget one.

| Variable | What it points at |
|---|---|
| `DATASETS_ROOT` | Root for extracted embeddings and training I/O — `configs/*.yaml`'s `data.base_dir` |
| `XENIUM_PROCESSED_OUTPUT_ROOT` | Per-sample `patch_coordinates.h5` ground truth + `adata.h5ad` gene expression |
| `WSI_RAW_ROOT` | Raw H&E whole-slide images |
| `WSI_CONVERTED_ROOT` | Converted/registered H&E whole-slide images |
| `XENIUM_OUTPUT_ROOT` | Raw Xenium instrument output (nucleus/cell boundary parquet/csv, `experiment.xenium`) |
| `ALIGNMENT_MATRIX_ROOT` | Per-sample Xenium↔H&E affine alignment matrices |
| `POSITIONS_CONVERTED_ROOT` | CellViT-centroid-corrected `patch_coordinates.h5` (`src/python/ot/`) |
| `CELLVIT_GEOJSON_OUTPUT_ROOT` | CellViT nucleus-segmentation `cells.geojson` output (`src/python/ot/` only) |
| `CELLVIT_CACHE` | Directory holding CellViT checkpoint files |
| `CTRANSPATH_WEIGHTS` | Full path to the CTransPath checkpoint file |
| `HF_TOKEN` | Hugging Face token for gated model downloads |

## Datasets

Three Xenium cohorts, compiled from [HEST-1k](https://github.com/mahmoodlab/hest)
(cell types assigned via a Curated Cancer Cell Atlas reference-atlas consensus of
Tangram + TACCO + marker-gene signatures; ambiguous cells excluded):

| Cohort | Tissue | Sections | Cells | Mag. | Task | Config |
|---|---|---|---|---|---|---|
| Cross-cancer ("carcinomas") | lung, colon, colorectal, ovarian, liver, skin | 7 | 1,004,673 | 40× | classification | `configs/train.yaml` |
| Colon cancer | colon | 5 | 1,022,719 | 40× | classification | `configs/train_colon.yaml` |
| Lung fibrosis (GSM) | lung | 15 | 227,623 | 20× | gene expression | `configs/deepspot_train_gsm.yaml` |

(One slide is shared between the cross-cancer and colon-cancer cohorts.) The
lung-fibrosis cohort's Xenium-derived coordinates frequently miss the true H&E nucleus
at this resolution and are re-centered before extraction — see "CellViT↔Xenium
matching" below.

## Pipeline scope — what's included vs. assumed upstream

This repo picks up **after** whole-slide registration and nucleus segmentation:

- **Xenium↔H&E registration** (the affine matrices under `ALIGNMENT_MATRIX_ROOT`) is
  produced by an external registration pipeline, not included here.
- **CellViT nucleus segmentation itself** (producing `cells.geojson` per sample) is run
  separately, using [CellViT](https://github.com/TIO-IKIM/CellViT); `src/python/ot/`
  only *matches* those detections against Xenium's own nucleus calls and corrects
  patch-center coordinates — it does not run segmentation.
- **Batch correction** across WSIs (the `correction_name` path segment — `Base` for
  none, or e.g. `rpca`/`scanorama`) and the optional per-cell `matching` boolean flag
  (CellViT↔Xenium matched or not) are produced by a separate step, external to this
  repo, that reads `extract_embeddings.py`'s raw output and writes it back out under
  `$DATASETS_ROOT/{model_name}_h5/{correction_name}/{wsi}/embeddings_dataset.h5`. If
  you don't have a correction step, mirror/symlink the raw output described below under
  a folder named e.g. `Base`.
- **The mutual-nearest-neighbor / CellViT-own-mask-readout re-check** ("Is the CellViT
  comparison fair?" in the paper's appendix) and the **UMAP/PCA embedding
  visualizations** are reported in the paper but their analysis scripts are not part of
  this drop.

## Stage 1 — Extracting embeddings

### Input: `patch_coordinates.h5`

Every whole-slide image needs one `patch_coordinates.h5` (default location:
`$XENIUM_PROCESSED_OUTPUT_ROOT/{sample}/patch_coordinates.h5`) with four equal-length
top-level datasets:

| dataset | meaning |
|---|---|
| `x_start`, `y_start` | top-left corner (H&E pixel space) of that cell's 224×224-centered patch — cell center = `x_start + 112`, `y_start + 112` |
| `cell_id` | cell identifier string |
| `cell_type` | ground-truth cell-type label string |

`src/python/ot/export_matched_patch_coordinates.py` produces the CellViT-centroid-
corrected version of this file used for the lung-fibrosis cohort (same `cell_id`/
`cell_type` labels, corrected `x_start`/`y_start`) — pass
`cells_info_root=$POSITIONS_CONVERTED_ROOT` to the config builders below to use it.

### Supported foundation models

Selected in `extract_embeddings.select_inference_provider(model_name)`:

| `model_name` | embedding dim | CLS token | notes |
|---|---|---|---|
| `UNI2` | 1536 | yes (+8 registers) | gated HF (`MahmoodLab/UNI2-h`) — one of the paper's 3 main models |
| `VirchowV2` | 1280 | yes (+4 registers) | gated HF (`paige-ai/Virchow2`) — one of the paper's 3 main models |
| `HOptimus1` | 1536 | yes (+4 registers) | gated HF (`bioptimus/H-optimus-1`) — one of the paper's 3 main models |
| `CellViT-SAM` | 1280 | no | PanNuke-trained baseline; needs `CELLVIT_CACHE/CellViT-SAM-H-x40-AMP.pth` |
| `CellViT-HIPT` | 384 | yes | PanNuke-trained baseline; needs `CELLVIT_CACHE/CellViT-256-x40-AMP.pth` |
| `CONCH` | 768 | yes | vision-language baseline; gated HF (`MahmoodLab/conch`, via `HF_TOKEN`); requires 448×448 input, `offset_x=offset_y=-224` |
| `CTransPath` | 768 | no (Swin) | needs `CTRANSPATH_WEIGHTS` |
| `Dummy` | configurable (PCA) | no | flattened-pixel baseline — `use_pca=True, n_components=50` reproduces the paper's PCA(50) baseline |

Most ViT providers (UNI2, VirchowV2, HOptimus1, CellViT-SAM/HIPT, CONCH) pool the four
tokens straddling the patch center (`top_left`/`top_right`/`bottom_left`/
`bottom_right`, saved as `embeddings_top_left` etc.); CTransPath's coarser 7×7 Swin
grid uses a single `center` token; `Dummy` uses a single `pixels` token.

### Reproducing the paper's read-out conditions

```python
from src.python.extract_embeddings.extract_embeddings import (
    build_configs, build_resized_cell_configs, extract_embeddings,
)
from src.python.extract_embeddings.data.resized_cell_dataset import ResizedCellDataset

infos = {"Xenium_V1_humanLung_Cancer_FFPE": {"converted": False, "model_output_dir": "UNI2"}}

# rho_tok + rho_CLS^native in one pass (same native 224x224 tile, save_cls writes both):
native_configs = build_configs("UNI2", infos)
extract_embeddings(native_configs, save_cls=True)
# -> train with data.embeddings_datasets=[top_left,top_right,bottom_left,bottom_right]  (rho_tok)
# -> train with data.embeddings_datasets=[cls]                                          (rho_CLS^native)
# -> train with data.embeddings_datasets=[top_left,top_right,bottom_left,bottom_right,cls]  (rho_tok||CLS)

# rho_CLS^resize (crop-and-resize baseline): separate pass, 100px crop upsampled to 224px
resized_configs = build_resized_cell_configs("UNI2", infos, size=224, size_side=100)
extract_embeddings(resized_configs, save_cls=True, dataset_cls=ResizedCellDataset)
# -> train with data.embeddings_datasets=[cls]

# ablations: "dynamic resize" (crop to each cell's own Xenium boundary polygon) and a 32px crop
dynamic_configs = build_resized_cell_configs("UNI2", infos, size=224, size_side=None)
crop32_configs  = build_resized_cell_configs("UNI2", infos, size=224, size_side=32)

# baselines: PCA(50) over raw pixels, and CONCH (needs its own 448x448/offset input geometry)
pca_configs   = build_configs("Dummy", infos)
conch_configs = build_configs("CONCH", infos, x_size=448, y_size=448, offset_x=-224, offset_y=-224)
```

`converted: True` reads from `$WSI_CONVERTED_ROOT` instead of `$WSI_RAW_ROOT`;
`wsi_filename` in an entry overrides the default `{dataset_name}_he_image.ome.tif`
naming (e.g. for GSM samples, named `{dataset_name}_registered_HE.ome.tif`). See
`extract_embeddings.py`'s `__main__` block for a full worked example.

Three extraction modes, all consuming the same `configs` shape:

- `extract_embeddings(configs, ...)` — one patch per cell (used for every condition
  above).
- `extract_embeddings_multicell(configs, ...)` — tiles the WSI into large patches
  (`MultiCellPatchDataset`, via `build_multicell_configs`) and reads off each cell's
  token by position instead of a separate forward pass per cell. Much faster for dense
  slides; writes a single `embeddings_cell_token` dataset (no CLS, no four-corner split).
  `build_multicell_configs` supports the same optional features as the per-cell path:
  - `size_side_x` / `size_side_y`: crop each tile at this native pixel size and resize
    it down (or up) to `x_size × y_size` before it hits the model — the multicell
    analogue of `build_resized_cell_configs`'s `size_side`.
  - `central_size_x` / `central_size_y`: restrict cell extraction to a centred
    sub-window of each tile (e.g. the central 112×112 of a 224×224 patch) so no
    extracted cell sits near a tile's edge, where its field of view would otherwise be
    lopsided/truncated. Tiles still tile the WSI exactly once per cell — the central
    window just steps by its own (smaller) size while each tile keeps reading the full
    `x_size × y_size` (or `size_side_x × size_side_y`) neighbourhood around it.
  - `with_boundaries=True` (+ `extract_embeddings_multicell(..., save_cell=True,
    save_nucleus=True)`): also mean-pool the tokens overlapping each cell's/nucleus's
    Xenium boundary polygon, written as `embeddings_cell` / `embeddings_nucleus` (same
    convention as the per-cell path's `save_cell`/`save_nucleus`).
- `run_attention_only(configs, output_path, n_samples=...)` — no embeddings; dumps
  per-head attention-map visualizations for a stratified sample of cells (this is what
  produced the paper's 24-head UNI2 attention figures, at both the native-tile and
  `size_side=100` field of view). Unsupported for CTransPath and CONCH, whose
  architectures don't expose comparable attention.

### Output: `embeddings_dataset.h5`

Written to `{output_path}/{model_output_dir}{output_suffix}/{dataset_name}/
embeddings_dataset.h5` (default `output_suffix='_h5'`, so
`$DATASETS_ROOT/{model_output_dir}_h5/{dataset_name}/embeddings_dataset.h5` — see
"Pipeline scope" above regarding the `{correction_name}` folder classifier_training
expects on top of this). Datasets:

- `embeddings_{key}` for each `patches_to_save` key (`top_left`, `top_right`,
  `bottom_left`, `bottom_right`, or `cell_token` in multicell mode) — `(N, D)` float32
- `embeddings_cls` — `(N, D)` float32, only if `save_cls=True`
- `embeddings_cell` / `embeddings_nucleus` — `(N, D)` float32, only if
  `save_cell=True` / `save_nucleus=True` (both extraction modes)
- `cell_ids`, `cell_labels` — `(N,)` UTF-8 strings
- `dataset_stats.json` alongside it — patch size, class distribution, etc.

## Stage 2 — Training classifiers

Full documentation: [`src/python/classifier_training/README.md`](src/python/classifier_training/README.md).

The classifier head is a single linear layer (`hidden_dim=0`, softmax cross-entropy,
AdamW), trained for the full 30-epoch budget with final-epoch weights evaluated — no
early stopping. The grid searched per fold is 2 learning rates ($5\times10^{-4}$,
$1\times10^{-4}$) × 3 loss-reweighting schemes (none / inverse-WSI-frequency /
inverse-cell-type-frequency) = 6 trials; the config with the best *validation*-slide
macro-F1 is selected and reported on the held-out *test* slide. A single fixed seed
(`42`) drives every source of randomness.

Splits are strict Leave-One-Slide-Out (no cell appears on both sides of a fold):
`configs/train.yaml` (cross-cancer, 7 WSIs) holds out each slide as test once, with two
choices of validation slide among the rest — $7\times2=14$ folds; `configs/
train_colon.yaml` (colon cancer, 5 WSIs — 4 cancer + 1 non-diseased always kept in
train) does the same over the 4 cancer slides — $4\times2=8$ folds.

```bash
# Submit one SLURM job per Leave-One-Slide-Out split (+ same-WSI upper bound)
python -m src.python.classifier_training.launcher \
    data.model_name=UNI2 data.correction_name=Base

# Or run a single split directly, e.g. for local/interactive testing
python -m src.python.classifier_training.experiment \
    split_idx=0 data.model_name=UNI2 data.correction_name=Base
```

Every trial logs to MLflow (`mlflow ui --backend-store-uri mlruns`). Set
`data.embeddings_datasets` to the read-out condition you want (see "Read-out notation"
above) — this is the flag that switches between $\rho_{\text{tok}}$,
$\rho_{\text{CLS}}^{\text{native}}$, $\rho_{\text{CLS}}^{\text{resize}}$, and
$\rho_{\text{tok}\|\text{CLS}}$ at training time, once the corresponding embeddings
have been extracted.

### Cell-type mappings

`configs/train*.yaml`'s `mapping:` field selects a label grouping from
`src/python/code_configs/mappings.py`. The paper's five-class problem is
`simplified_broad`:

| `mapping` | classes |
|---|---|
| `simplified_broad` | `Epithelial`, `Malignant`, `Lymphoid`, `Myeloid`, `Stromal` (+ `Unknown`, always dropped) — **used in the paper** |
| `identity` | every raw CellViT/Xenium cell-type label, unchanged |
| `epithelial_malignant_vs_rest` | `epithelial_malignant` vs. `non_epithelial_malignant` |
| `malignant_vs_rest` | `epithelial_malignant` (malignant only) vs. `non_malignant` |

`Unknown` cells are always excluded from training/evaluation (see
`classifier_training/README.md`).

## Optional: gene-expression prediction (DeepCell)

`src/python/deepspot_training/` trains DeepSpot's `DeepCell` model on the same
embeddings to predict spatial gene expression instead of cell type. Full docs:
[`src/python/deepspot_training/README.md`](src/python/deepspot_training/README.md).
The paper's gene-expression results (Pearson correlation, and the "removing
neighbor-aggregation" ablation) use `configs/deepspot_train_gsm.yaml` specifically —
a 15-fold Leave-One-Sample-Out sweep over the lung-fibrosis/GSM cohort (14 train / 1
test per fold), predicting the top 300 highly-variable genes (scanpy, Seurat v3
flavor, batched by sample) from embeddings aggregated over each cell's `k=30` nearest
spatial neighbors (max-pooled). DeepSpot's `DeepCell` architecture is otherwise fixed:
a 10-network ensemble, hidden/embedding size 1024, MSE loss, AdamW (weight decay
$10^{-6}$, dropout 0.2), batch size 1024, up to 6 epochs with early stopping (patience
3, min delta 0.01); the grid searched per fold is learning rate ($10^{-4}$ vs.
$10^{-3}$) × per-gene standard-scaling (on/off) = 4 trials, seed `2024`.
`data.resolution` is deliberately set to `0` — a departure from DeepSpot's own default,
made for efficiency. Reported per-gene Pearson at $N\in\{50,100,200,300\}$ most
variable genes is a post-hoc aggregation over one model trained on all 300 genes, not
separate training runs (`data.gene_interval`, `data.num_genes=300`).

```bash
# 0. Precompute HVGs once, on a big-memory node (see deepspot_training/README.md)
sbatch scripts/compute_hvgs.sh

python -m src.python.deepspot_training.launcher \
    --config-name=deepspot_train_gsm \
    data.model_name=UNI2 data.correction_name=Base

# Ablation: neighbor aggregation on/off (k=30 vs k=0)
python -m src.python.deepspot_training.launcher \
    --config-name=deepspot_train_gsm \
    data.model_name=UNI2 data.correction_name=Base data.n_neighbors=0
```

`configs/deepspot_train.yaml` (same WSI splits as `configs/train.yaml`) is also
available as a general-purpose config, but is not what the paper's reported
gene-expression numbers come from.

## Optional: CellViT↔Xenium matching (`src/python/ot/`)

Two related but distinct pipelines live here:

- **`export_matched_patch_coordinates.py` + `cellvit_xenium_matching.py`** implement
  the re-centering procedure described in the paper's appendix ("Re-centring the lung
  fibrosis sections"), used to produce the lung-fibrosis cohort's corrected
  `patch_coordinates.h5`. For every CellViT-detected nucleus within `radius_px=80` of a
  Xenium nucleus centroid: `distance_score = 1/(1 + dist/15)`,
  `orientation_score = 1 - angle_diff/90` (long-axis agreement), `iou_score` = IoU
  after translating to a shared centroid; `weighted_score = 1.0·distance +
  1.2·orientation + 5.0·iou`. Matches with `weighted_score ≤ 4.5` are dropped; each
  Xenium cell keeps only its single best-scoring CellViT match (enforcing a 1-to-1
  mapping); the kept cells' `x_start`/`y_start` are replaced with the matched CellViT
  nucleus's own centroid, with `cell_id`/`cell_type` copied unchanged from the ground
  truth. This retains ~44% of cells, each uniquely matched to a nucleus close in shape
  and position:

  ```bash
  python -m src.python.ot.export_matched_patch_coordinates            # every GSM sample
  python -m src.python.ot.export_matched_patch_coordinates --sample GSM7990540
  ```

  Output: `$POSITIONS_CONVERTED_ROOT/{sample}/patch_coordinates.h5` — pass
  `cells_info_root=$POSITIONS_CONVERTED_ROOT` to Stage 1's config builders to extract
  on these corrected coordinates.

- **`centering_correction.py`** is a separate, offset-averaging approach (merge
  touching nuclei → best-match scoring → radius-averaged offset → broadcast back to
  per-cell IDs) explored in the same module — **not** the procedure reported in the
  paper. `export_matched_patch_coordinates.py` reuses only its two loading helpers
  (`load_cellvit_nuclei`, `load_xenium_nuclei`), not its matching pipeline.

`cellvit_outputs_processing/fix_geojson.py` repairs invalid/collection geometries in a
raw CellViT `cells.geojson` before either script reads it:

```bash
python -m src.python.cellvit_outputs_processing.fix_geojson cells.geojson cells.fixed.geojson
```
