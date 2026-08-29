# Classifier Evaluation Pipeline

Leave-One-WSI-Out (LWO) evaluation of WSI patch embeddings using a lightweight MLP.
Designed to quantify how well batch-correction methods remove batch effects.

## Architecture

```
launcher.py      — reads configs/train.yaml, submits one SLURM job per split
  └─ experiment.py — loads data once, runs full grid search, logs to MLflow
       └─ trainer.py   — pure training logic, returns metrics dict
```

## Quick start

### 1. Edit `configs/train.yaml`

Set the required fields:

```yaml
data:
  model_name: virchow_v2        # embedding model
  correction_name: scanorama    # batch-correction method
```

Optionally adjust `splits`, `hparams`, `training`, and `slurm` sections.

### 2. Submit all SLURM jobs (one per LWO split)

```bash
python -m src.python.classifier_training.launcher \
    data.model_name=virchow_v2 \
    data.correction_name=scanorama
```

#### Dry run (print scripts, don't submit)

```bash
python -m src.python.classifier_training.launcher \
    data.model_name=virchow_v2 \
    data.correction_name=scanorama \
    dry_run=true
```

### 3. Also run the same-WSI upper-bound experiment

```bash
python -m src.python.classifier_training.launcher \
    data.model_name=virchow_v2 \
    data.correction_name=scanorama \
    same_wsi_split=true
```

### 4. Run a single split directly (e.g. for local testing)

```bash
python -m src.python.classifier_training.experiment \
    split_idx=0 \
    data.model_name=virchow_v2 \
    data.correction_name=scanorama
```

### 5. Debug a failed SLURM job

```bash
# Print stdout + stderr captured by SLURM for split 0
python -m src.python.classifier_training.launcher \
    data.model_name=virchow_v2 \
    data.correction_name=scanorama \
    debug_split=0
```

## Output structure

```
outputs/
  {model_name}/{correction_name}/{mapping}/{matching}/{embeddings}/
    split_{i}/
      results_summary.yaml     # best hparams + per-class metrics
      grid_search_results.csv  # one row per trial
      split_informations.yaml  # WSI names, cell counts per class
    same_wsi_split/            # present only when same_wsi_split=true
      ...

slurm_logs/
  {model_name}/{correction_name}/
    split_{i}.out / split_{i}.err
    job_split_{i}.sh

mlruns/                        # MLflow tracking directory
```

- `{matching}` is `matching` when `data.consider_matching` is true (default — only
  cells matched between CellViT and Xenium are used) or `all_cells` when false
  (every detected cell is used, matched or not).
- `{embeddings}` is `default` when `data.embeddings_datasets` is empty (all
  embedding keys in the H5 file are averaged), otherwise the sorted, underscore-joined
  list of dataset keys used (e.g. `cell_token`, `cls_cell_token`).

## Hyperparameter tuning

Each hparam in `configs/train.yaml` has a `tune` flag and a `values` list.
Set `tune: false` with a single value to fix it:

```yaml
hparams:
  learning_rate:
    tune: false
    values: [1.0e-3]
  hidden_dim:
    tune: true
    values: [0, 64, 256]
```

A full grid search is run — all combinations of active hyperparameters.
The total trial count is the product of the lengths of all `tune: true` value lists.

## WSI split configuration

Each split entry uses fractional cell sampling per WSI:

```yaml
splits:
  - train:
      Xenium_V1_Human_Lung_Cancer_Addon_FFPE: 1.0  # keep 100 % of cells
      Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE: 0.5  # keep 50 %
    test:
      Xenium_V1_humanLung_Cancer_FFPE: 1.0
```

## MLflow

Results are logged to `mlruns/` (relative to the repo root).
View with:

```bash
mlflow ui --backend-store-uri mlruns
```

Each top-level MLflow run corresponds to one split.
Nested runs correspond to individual grid-search trials.
