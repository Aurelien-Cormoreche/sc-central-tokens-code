# DeepCell embedding evaluation pipeline

Answers "how good are these WSI embeddings for gene-expression prediction" the same way
`src/python/classifier_training/` answers it for cell-type classification: train
[DeepSpot](https://github.com/ratschlab/DeepSpot)'s `DeepCell` model on top of each embedding
model/correction combination, in a Leave-One-WSI-Out (LWO) fashion, and report per-gene Pearson
correlation on held-out WSIs.

The dataloader/model/training code is DeepSpot's own (`deepspot.cell.dataloader.DeepCellDataLoader`,
`deepspot.cell.model.DeepCell`, PyTorch Lightning training loop) — only *how data is loaded* is
adapted, via `data/custom_dataloader.py::CustomDeepCellDataLoader`, since our data lives in a
different layout than DeepSpot's `out_folder` convention. See that file's module docstring for
exactly what was changed vs. the original `DeepCellDataLoader.__init__`.

## Architecture (3 layers, same shape as classifier_training/, + one offline precompute step)

0. **HVG precompute** (`compute_hvgs.py`, run once via `scripts/compute_hvgs.sh` on a big-memory
   node, ~80G) — for each split's train WSIs, loads every WSI's full Xenium AnnData and ranks the
   top `data.hvg_max_genes` (default 500) genes by (batch-corrected) variability, no cell/gene
   subsampling shortcuts. Also flags, per gene, whether it's present in every one of that split's
   val/test WSI(s) (`present_in_test`) — a gene ranked highly variable from the train WSIs is
   useless if it's simply absent from the panel of the WSI we'll evaluate on. Writes one CSV per
   split to `data.hvg_dir` (default `data/HVGsSplit/`). Do this **before** running any experiments.
1. **Launcher** (`launcher.py`) — reads `configs/deepspot_train.yaml`, submits one SLURM job per
   split (+ one `same_wsi_split` job if enabled).
2. **Experiment** (`experiment.py`) — for one split: loads the precomputed HVG ranking for that
   split, loads the dataset once restricted to just those `num_genes` genes (`num_genes` and
   `n_neighbors` are fixed per run, so this is the only expensive I/O), then runs a full grid
   search over model/training hyperparameters (`learning_rate`, `dropout`, `loss_func`,
   `weight_decay`, `gene_normalization`, ...), logging every trial to MLflow.
3. **Trainer** (`trainer.py`) — stateless `train_and_evaluate()`: builds `deepspot.cell.model.DeepCell`,
   trains it with `lightning.Trainer` + `EarlyStopping` (mirrors
   `he2st/workflows/models/DeepCell.py`), and computes per-gene Pearson correlation on the val split.

## Environment setup

`deepspot` isn't on PyPI; install it from source into the `cellViT` conda env:

```bash
pip install git+https://github.com/ratschlab/DeepSpot.git
pip install lightning
```

## Running

```bash
# 0. Precompute HVGs once (all splits), on a big-memory node — required before running experiments
#    unless data.hvg_dir is set to null (on-the-fly fallback, only advisable on a large-memory node)
sbatch scripts/compute_hvgs.sh

# Submit all splits for one (model, correction) combination
python -m src.python.deepspot_training.launcher \
    data.model_name=virchow_v2 data.correction_name=scanorama

# Dry run (print SLURM scripts, don't submit)
python -m src.python.deepspot_training.launcher \
    data.model_name=virchow_v2 data.correction_name=scanorama dry_run=true

# Debug a specific split's SLURM logs
python -m src.python.deepspot_training.launcher \
    data.model_name=virchow_v2 data.correction_name=scanorama debug_split=0

# Run a single split directly (e.g. on an interactive GPU node)
python -m src.python.deepspot_training.experiment \
    split_idx=0 data.model_name=virchow_v2 data.correction_name=scanorama
```

To compare multiple `num_genes` (the trained gene-set size) or `n_neighbors` values, run separate
experiments overriding those fields — they are fixed per run by design (see
`configs/deepspot_train.yaml` comments), so each combination gets its own SLURM job(s) and its own
output subtree:

```bash
python -m src.python.deepspot_training.launcher \
    data.model_name=virchow_v2 data.correction_name=scanorama \
    data.num_genes=150 data.n_neighbors=15
```

`data.gene_interval` is different from `data.num_genes`: the model is trained **once**, on the
`num_genes` most highly variable genes (HVGs computed once, with rank info). `gene_interval` is a
purely post-hoc reporting slice — for each `n` in `gene_interval`, mean/median Pearson is
re-aggregated over just the `n` most variable genes *within that trained set* (by
`highly_variable_rank`), with no retraining and no further HVG computation. It answers "how does
correlation look if we only care about the most variable genes", not "train N separate models".

## Output structure

```
data/HVGsSplit/                        # written by compute_hvgs.py (data.hvg_dir)
  {split}.csv                          # gene_name, highly_variable_rank, present_in_test
  {split}_wsis.txt                     # train WSIs used for that split's ranking
  {split}_test_wsis.txt                # val/test WSI(s) checked for gene-panel presence

outputs_deepspot/{model_name}/{correction_name}/{all_cells|matching}/{token}/{num_genes}/{num_neighbours}/{split}/
  split_informations.yaml     # train/val WSI lists + sizes
  genes_used.csv               # the num_genes gene names used in this run
  grid_search_results.csv      # one row per trial (hparams + mean/median pearson)
  results_summary.yaml         # best config + best mean/median pearson + all trial rows
  best_train_loss_curve.npy
  best_val_loss_curve.npy
  best_training_curve.png
  pearson_distribution.png     # histogram of per-gene Pearson (best trial, all num_genes)
  top10_genes.csv
  worst10_genes.csv
  gene_interval_stats.csv      # mean/median pearson per gene_interval value (see above)
  gene_interval_stats.png
mlruns/                        # MLflow tracking directory (shared with classifier_training)
```

`{token}` = sorted, underscore-joined `data.embeddings_datasets` (e.g. `cls`), matching
`classifier_training`'s embeddings-folder naming.

## Notes / caveats

- **HVGs are precomputed, not computed inside the training pipeline.** `compute_hvgs.py` ranks the
  top `hvg_max_genes` genes (no cell subsampling — full train-WSI cell population) on a big-memory
  node — that's the only place the expensive computation happens. `experiment.py` then loads that
  ranking (`data.hvg_dir`) via `load_hvg_table`, which (1) drops genes not present in every val/test
  WSI of the split (`present_in_test` — a gene ranked highly variable from the train WSIs is useless
  if it's absent from the panel we'll evaluate on) and (2) selects the `num_genes` most variable
  genes among what's left. Since which genes to keep is now known before any WSI data is read,
  `load_data()` restricts each sample's AnnData to just those columns *before* densifying anything,
  so the regular (32G) training jobs never touch the full gene panel. Setting `data.hvg_dir: null`
  falls back to computing HVGs on the fly inside `load_data()` itself (dense, full `cells × all_genes`
  matrix, no presence filtering) — only advisable on a large-memory node, same as compute_hvgs.py.
- `same_wsi_split` (upper bound) pools all configured WSIs and splits at the **cell** level
  (80/20, WSI-of-origin-agnostic), same spirit as the classifier's same-WSI mode. LWO splits
  (`split_idx >= 0`) use each split's own precomputed train-WSI-only HVG file; `same_wsi_split`
  uses the `same_wsi_split.csv` file (ranked over every WSI pooled, train ∪ test of every split) —
  an optimistic upper bound, same caveat as the classifier's same_wsi_split.
- `gene_normalization` is swept as a grid hyperparameter without re-reading data from disk per
  trial: `CustomDeepCellDataLoader` caches the pre-scaler gene-expression matrix and exposes
  `set_normalization(...)` to cheaply refit/re-apply a scaler in memory.
- Real runs (SLURM, actual `adata.h5ad`/embeddings paths) must be validated on Euler — this code
  was developed and unit-tested locally against a synthetic fixture (no cluster access from the
  dev environment; DeepSpot's own `deepspot.cell.model.DeepCell`/`DeepCellDataLoader` and a real
  `lightning.Trainer.fit()` call were exercised, not mocked). The precomputed-HVG loading path
  (`load_hvg_table`, `_load_precomputed_hvgs`, per-sample column pre-filtering in `load_data`) was
  added after that verification run and has not yet been re-verified end-to-end.
- Inherited from DeepSpot's own neighbor encoding (unchanged): cell **barcodes** must not contain
  `_` (sample/WSI names may — see `data/custom_dataloader.py` module docstring). Standard 10x/Xenium
  barcodes (e.g. `aaaejtbb-1`) satisfy this.
