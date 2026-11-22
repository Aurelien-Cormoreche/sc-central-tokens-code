# `bench` — single-cell benchmark of pathology foundation models

One config (`configs/bench.yaml`), one CLI (`bench`), one storage layout, one static site.
Stage 1 compares UNI2-h, CONCH v1.5, Virchow2, H-optimus-1, LEMON and GenBio-PathFM (two readouts each) on
cell-type classification and gene-expression prediction. Stage 2 varies input size and tests a multicell
readout for the Gate-1 model. Stage 3 scales the training data and probe capacity.

Every command takes `--dry-run` (print jobs, estimates, done flags; run nothing), `--smoke` (smoke slides
only, outputs in `$BENCH_ROOT/smoke`), `--models`, `--datasets` (cohorts or slide patterns) and `--config`.
Job commands also take `--local` (run the bins in this process instead of submitting a SLURM array).

## Environment

Paths come from the repo's `.env` (the same file as the rest of the repo, see `.env.example`). bench needs:

```bash
BENCH_ROOT=/path/to/bench_outputs                     # everything bench writes
WSI_RAW_ROOT=/path/to/HE_images                        # {slide}_he_image.ome.tif etc. (quote paths with & or spaces)
WSI_CONVERTED_ROOT=/path/to/HE_images_converted        # re-encoded copies of unreadable files
XENIUM_PROCESSED_OUTPUT_ROOT=/path/to/xenium_processed # {slide}/patch_coordinates.h5, {slide}/adata.h5ad
HF_HOME=/path/to/hf_cache                              # HF cache with the weights in hub/; only read (read-only is fine)
# optional: BENCH_WSI_MANIFEST=/path/to/wsi_manifest.csv  (dataset, converted, wsi_filename[, status])
```

First run, with a cluster that has no internet but sees a client-filled folder read-only (e.g. customapps;
`scripts/cluster/`, run from the repo root):

```bash
# on the client WITH internet, writing into the folder the cluster reads:
hf auth login                                          # once; accept the gated models' terms first
bash scripts/cluster/download_weights.sh $HF_HOME      # `hf download` of the 6 checkpoints into $HF_HOME/hub
bash scripts/cluster/download_wheels.sh  DIR_W         # Linux wheels of bench's packages + TRIDENT (~80 MB)

# on the cluster:
WHEELS=DIR_W bash scripts/cluster/00_install.sh  # once: overlay venv .venv-bench on top of cellViT (cellViT is
                                                 #   not modified) + `bench`; runs the tests and the synthetic smoke
bash   scripts/cluster/01_weights.sh             # offline: checks every weight file, records the snapshot shas
sbatch scripts/cluster/02_prepare.sbatch         # CPU: inspect -> darkspots -> manifest for every slide
sbatch scripts/cluster/03_smoke.sbatch           # 1 GPU: real smoke + parity + calibration (runs alongside 02)
source scripts/cluster/bench_env.sh              # interactive shell: env + .env, then any `bench ...` command
```

bench never writes to `HF_HOME`: `bench_env.sh` sets `HF_HUB_OFFLINE=1` (set `HF_HUB_OFFLINE=0` in `.env` if
the cluster has internet) and moves transformers' module cache (GenBio's remote code, by default
`$HF_HOME/modules`) to `$TMPDIR` per job, `$BENCH_ROOT/.hf_modules` otherwise. With internet on the cluster,
`bash scripts/cluster/00_install.sh` (no `WHEELS`) and `bench fetch-weights` download directly.

Every array task sources `scripts/cluster/bench_env.sh` (`slurm.setup` in `configs/bench.yaml`). Check `slurm.partition` / `gres` / `account` / `cpu_partition` there: the GPU
type name (`gpu:rtx_4090:1`) must match your cluster's, in the config and in `03_smoke.sbatch`.

## Commands, in order

```bash
# ── 0. checks and data preparation (CPU) ──
bench smoke --synthetic            # anywhere, ~30 s: generated slides + tiny ViT through the whole CLI
bench fetch-weights --offline      # weights already in $HF_HOME/hub (download_weights.sh): checks every file, prints
                                   #   the `hf download` command for anything missing, records the snapshot shas in
                                   #   $BENCH_ROOT/weights_revisions.json (jobs pin them). Without --offline it downloads
bench inspect                      # slide table: WSI path, size, pixel size, scale -> $BENCH_ROOT/manifest/slides_inspect.csv
                                   #   stops on a missing file / unknown or mismatching pixel size (fix: mpp_overrides)
bench darkspots                    # CPU array: black / no-data / unreadable block maps
                                   #   review $BENCH_ROOT/darkspots/*.png (red = dark, blue = unreadable, grey = scanned)
bench manifest                     # global cell manifest + per-cohort drop counts (unknown label, border, dark spot)

# ── 1. real smoke (1 GPU, ~30–60 min) ──
srun --gres=gpu:rtx_4090:1 --cpus-per-task=16 --mem=64G --time=02:00:00 bench smoke
#   inspect -> darkspots -> manifest -> extract -> classify -> genexp -> 2a/2b -> scaling -> report on the 7 smoke
#   slides (<= 300 cells each) for every model, plus:
#   * parity: each model's `cls` (bf16, our loader) vs its model-card forward (fp32); PASS = mean cos >= 0.999
#   * centring overlays: $BENCH_ROOT/smoke/centring/{model}__{slide}.png (red cross = centroid, blue = central tokens)
#   * calibration: GPU throughput per model/input size -> $BENCH_ROOT/calibration.json (used by every --dry-run)

# ── 2. Stage 1, one model at a time (repeat for uni2 conch15 virchow2 hoptimus1 lemon genbio) ──
bench extract  --stage 1 --models uni2 --dry-run   # per-slide jobs, GPU-h, GB, bins
bench extract  --stage 1 --models uni2             # SLURM array; prints the plan id
bench classify --stage 1 --models uni2             # within / cross_loco / single_cohort x cls / central
bench genexp           --models uni2               # DeepSpot leave-one-patient-out, PRIME (9 folds) + VANNAN (14)
bench status --deletable                           # YES = every Stage-1 consumer of that model is done
#   rm -r $BENCH_ROOT/embeddings/stage1/uni2       # your call; keep the Gate-1 model's files (2a/3 reuse them)
bench report                                       # $BENCH_ROOT/site/index.html

#   >>> GATE 1 <<<  Overview page: ranking of every model x readout by the gate rule (mean test macro F1 over
#   present classes, over the within-cohort tasks). Write your choice into configs/bench.yaml:
#       gate1:
#         chosen: {model: <model>, readout: <cls|central>}
#   Stage 2 and 3 commands refuse to run until it is set.

# ── 3. Stage 2 (Gate-1 model, SPLIT + VANNAN, within-cohort) ──
bench extract  --stage 2a          # input sizes 56/112/224/448 (rounded up to the patch; the native size reuses Stage 1)
bench classify --stage 2a
bench report
#   >>> GATE 2 <<<  Overview page, "Gate 2": 224 vs 448. Set in configs/bench.yaml:
#       stage2:
#         multicell: {tile_px: <224|448>}
bench extract  --stage 2b          # multicell tiles, central readout
bench classify --stage 2b

# ── 4. Stage 3 (Gate-1 model) ──
bench scaling --probe linear       # fractions 0.5/1/5/100 % x 3 seeds, within + cross_loco
bench scaling --probe mlp          # 2-layer MLP, cross_loco
bench report
```

Housekeeping:

```bash
bench status [--stage 1 2 3] [-v]  # every job the config implies: done / running / failed / missing
bench retry --plan <plan_id>       # resubmit only the bins with failed or missing jobs ($BENCH_ROOT/jobs/<plan_id>)
bench calibrate --from-status      # refresh throughput from finished extraction jobs (better than the smoke benchmark)
```

Every job command is idempotent. Re-running it, or resubmitting a plan, skips jobs whose outputs are valid for
the current manifest and config. A worker re-checks each job before running it.

## What runs where

| Command | Resources | Unit of work | Done when |
|---|---|---|---|
| darkspots | CPU array (`slurm.cpu_*`) | slide | npz exists with the current filter parameters |
| extract | GPU array, LPT bins per (model, input size) | (stage, model, variant, slide) | h5 `complete`, row count and manifest hash match |
| classify / scaling | GPU array, one job per bin, memory = largest job | (stage, scheme, model, readout, variant[, probe]) | every `_DONE.json` has the current inputs hash |
| genexp | GPU array, one fold per task | (model, slide set, fold) | `_DONE.json` has the current inputs hash |

Bins pack jobs longest-first so each bin fits within `slurm.max_hours`. Embeddings are kept as fp16 in host RAM;
cross-cohort GenBio needs about 190 GB. Lower `--max-mem-gb` if the partition can't provide it.

## Outputs (`$BENCH_ROOT`)

```
manifest/slides_inspect.{parquet,csv}  slides.parquet  cells/{slide}.parquet  class_counts*.json  provenance_*/
darkspots/{slide}.npz|.png
embeddings/stage{1,2a,2b}/{model}/{native|fov056|...|mc448}/{slide}/embeddings_dataset.h5
results/{stage}/classify/{scheme}/{model}/{readout}/{variant}/[frac_{f}/seed_{s}/{probe}/]fold_{k}.json + _DONE.json
results/3/subsets/{scheme}/{fold}/frac_{f}_seed_{s}.npz          # Stage-3 subsets as (slide, manifest row) pairs
results/1/genexp/{set}/{model}/{readout}/native/fold_{k}/{result.json, _DONE.json, deepspot/}
genexp/hvgs/{set}/fold_{k}/split_0.csv                             # train-donor HVGs, shared by all models
jobs/{plan_id}/{plan.json, array.sbatch, logs/task_*.log}   status/{kind}/{job}.json   failed.jsonl
calibration.json   weights_revisions.json   site/index.html
```

h5 datasets are `embeddings_cls`, `embeddings_central`, `cell_ids`, `cell_labels` (raw), `labels` (mapped),
`manifest_row`, `x`, `y`. The key names are the ones `deepspot_training` and `utils/loading_functions` read.
The attributes record the model revision, input/patch/grid, scale and pixel sizes, readout definitions, the
manifest and config hashes, the git commit, package versions and throughput. Each result JSON and
`_DONE.json` carries provenance: the resolved config, the git commit (and diff size), package versions,
hardware and model revisions.

## Definitions

- **Geometry.**
  - Slides finer than 0.2 µm/px (SPLIT-Zeiss 0.086, three 10X slides at 0.137) are area-resampled to
    0.25 µm/px. All other slides are used as stored.
  - The crop is centred on the Xenium centroid, `patch_coordinates` `x_start + 112`. Its origin is
    `floor(e + 0.5) − N/2` in extraction pixels.
  - Crops come from tile-aligned region reads. They are pixel-identical to direct reads; after resampling they
    differ by at most 1 grey level (tested).
- **Readouts.** Both come from the same forward pass.
  - `cls` is the model-card standard embedding:
    - UNI2: CLS
    - CONCH v1.5: attentional pool + LN
    - Virchow2: CLS ‖ mean of patch tokens
    - H-optimus-1: CLS
    - LEMON: CLS
    - GenBio: 3 × CLS
  - `central` is the mean of the 2×2 patch tokens that meet at the centroid, or the single central token
    for odd grids.
- **Cell filter.** One manifest is shared by every model, readout and stage. It drops:
  - `Unknown` labels (classification only);
  - cells within 224 extraction px of the image edge;
  - cells within an L∞ box of 224 extraction px around a dark or unreadable 16-px block.
- **Classification.**
  - Linear probe on z-scored embeddings (statistics fit on train). The grid is lr {1e-3, 1e-4} × wd {1e-4, 1e-2}
    × loss weighting {none, by_cell_type}, with early stopping on val. All trials of all folds train in lockstep
    with `train_linear_jointly`.
  - **Within**: leave one donor out; val = the cohort's fixed val donors.
  - **cross_loco**: leave one cohort out; label space = classes present in both training and held-out cohorts.
  - **single_cohort**: train on one cohort, test on each other cohort.
  - Trials are selected on val. Selection is slide-mean macro F1, cohort-balanced for cross-cohort.
  - Reported metric: test macro F1 over present classes, plus balanced accuracy and others; per slide, per
    donor and per cohort; 95 % bootstrap CIs over test donors.
- **Gene expression.** DeepSpot DeepCell, with the template's fixed hyperparameters and early stopping on the
  val donor; the best epoch is restored.
  - Training uses the 300 HVGs of the training donors.
  - Reported: Pearson and Spearman per gene, as means over all genes and over the top 50 and 100 HVGs.
  - `readout: auto` uses that model's best readout under the Gate-1 rule (needs its Stage-1 within results).
- **Stage 3.** Subsets are class-stratified draws from each fold's training pool,
  `n_c = max(1, round(f · N_c))`. They are saved once as (slide, manifest row) pairs, so every probe and model
  sees identical cells. Val and test are fixed. The seed controls both the subset and the initialisation.

## Estimates (RTX 4090, bf16; priors until `bench smoke` writes calibration.json)

About 15.5 M cells to extract (14.1 M labelled plus unlabelled cells on the gene-expression slides).
`--dry-run` re-derives every number from the manifest and the calibration.

| Stage | GPU-hours | Disk |
|---|---|---|
| 0 inspect / darkspots / manifest | 0 (≈ 20–40 CPU-h, I/O bound) | < 5 GB |
| 1 extract | ≈ 127 + ≈ 10 slide overhead: UNI2 14 · Virchow2 13 · H-opt 23 · CONCH 22 · LEMON 2 · GenBio 54 | ≈ 675 GB if all kept: UNI2 95 · Virchow2 119 · H-opt 95 · CONCH 56 · LEMON 24 · GenBio 286 |
| 1 classify (6 models × 2 readouts × 3 schemes) | ≈ 10–30 (mostly loading) | < 5 GB |
| 1 gene expression (23 folds per model) | ≈ 55 (0.4 h per fold prior) | < 5 GB |
| 2a extract (3 new sizes on SPLIT + VANNAN) | ≈ 18 (UNI2) … 68 (GenBio) | 66–200 GB |
| 2b multicell | ≈ 2–15 | 10–35 GB |
| 2 classify | ≈ 3–6 | small |
| 3 scaling | ≈ 25–35 | small |
| **Total** | **≈ 255–390** | peak ≈ 300–400 GB when deleting per model |

## Tests

```bash
python -m pytest tests/bench          # 44 tests, CPU, ~40 s (incl. the synthetic end-to-end run)
```

Covered:
- crop centring and central tokens for every (input size, patch) used;
- region reads vs direct reads, at native and resampled resolution;
- the dark-spot and L∞ filter;
- LPT packing (≤ 4/3 of optimal);
- Stage-3 subsets: deterministic, order-independent, min 1 cell per class;
- atomic writes and `is_valid`;
- fold definitions on the real registry (43 within, 5 cross, 9 + 14 gene-expression folds);
- the gate rules;
- the full CLI chain, with site build and resumability.

## Changes to shared code (backward compatible)

- `classifier_training/trainer.py`: `FoldData.z_stats`. This is an optional precomputed z-score fit, which
  avoids gathering every training row at once for cross-cohort folds.
- `deepspot_training/trainer.py`:
  - opt-in `restore_best_epoch` (off by default);
  - per-gene Spearman alongside Pearson.
- `deepspot_training/experiment.py`:
  - optional `data.embeddings_dir`;
  - `training.restore_best_epoch`;
  - Spearman columns in `per_gene_pearson.csv` and `results_summary.yaml`.
