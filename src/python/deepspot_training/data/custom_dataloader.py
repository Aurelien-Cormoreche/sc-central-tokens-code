"""
Custom DeepSpot `DeepCellDataLoader` for this project's data layout.

DeepSpot (deepspot.cell.dataloader.DeepCellDataLoader) expects an `out_folder` with
`data/inputX/{sample}.pkl`, `data/h5ad/{sample}.h5ad` and `data/image_features/{model}[_{diameter}]/`.
Our data instead lives as:
  - per-sample Xenium AnnData: adata_path.format(sample=sample), raw counts in
    `layers['counts']`, spatial coordinates in `obs['x_centroid']`/`obs['y_centroid']`.
  - per-sample cell embeddings: `{embeddings_dir}/{sample}/embeddings_dataset.h5`
    (same HDF5 layout read by src/python/utils/loading_functions.py).

Constraint inherited from DeepSpot's own neighbor encoding (unchanged, in the base class's
__getitem__): it joins each neighbor as f"{barcode}_{sample}" and recovers the barcode via
str.split("_")[0], discarding the rest. Sample/WSI names may freely contain underscores (as ours
do, e.g. "Xenium_V1_Human_Lung_Cancer_Addon_FFPE") — only the cell **barcode** itself must not
contain "_", or it gets silently truncated. Standard 10x/Xenium cell barcodes (e.g. "aaaejtbb-1")
satisfy this.

`load_data()` below is the drop-in replacement for `deepspot.utils.utils_dataloader.load_data`
(the expensive I/O step). `CustomDeepCellDataLoader` subclasses `DeepCellDataLoader` and rewrites
only `__init__` (to call our `load_data` and to skip the second per-sample AnnData re-read the
original does purely to fetch coordinates — we already have them) and `_load_patch` (to drop the
disk-fallback branch, since `load_image_features_in_memory` is always True here). `__len__` and
`__getitem__` are inherited unchanged from `DeepCellDataLoader`.
"""
import os
from typing import Optional, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from deepspot.cell.dataloader import DeepCellDataLoader
from deepspot.utils.utils_dataloader import (
    get_balanced_index,
    log1p_normalization,
    spatial_upsample_and_smooth,
)


# ── data loading ─────────────────────────────────────────────────────────────────

def _load_embeddings_for_sample(
    embeddings_dir: str,
    sample: str,
    embeddings_datasets: Sequence[str],
    consider_matching: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Adapted from src/python/utils/loading_functions.py::load_unique_wsi — reads one sample's
    embeddings_dataset.h5, averaging over `embeddings_datasets` and applying the `matching` mask,
    same convention as the classifier pipeline."""
    path = os.path.join(embeddings_dir, sample, "embeddings_dataset.h5")
    with h5py.File(path, "r") as f:
        datasets = list(embeddings_datasets)
        if len(datasets) == 0:
            datasets = [key.split("embeddings_")[1] for key in f.keys() if key.startswith("embeddings_")]

        matching_mask = None
        if consider_matching and "matching" in f:
            matching_mask = f["matching"][()].astype(bool)

        embeddings = np.mean([f[f"embeddings_{key}"][()] for key in datasets], axis=0)
        cell_ids = f["cell_ids"][()]

        if matching_mask is not None:
            embeddings = embeddings[matching_mask]
            cell_ids = cell_ids[matching_mask]

    cell_ids = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cell_ids])
    cell_ids = np.array([c.split('_')[-1] for c in cell_ids])
    return embeddings, cell_ids


def _compute_highly_variable_genes(
    counts: pd.DataFrame,
    sample_ids: np.ndarray,
    num_genes: int,
    flavor: str = "seurat_v3_paper",
    batch_key: str = "sampleID",
) -> pd.DataFrame:
    """
    Select the `num_genes` most (batch-corrected) highly variable genes — this is the trained gene
    set. One `sc.pp.highly_variable_genes` call (flavor="seurat_v3_paper"/"seurat_v3" gives a
    `highly_variable_rank` per selected gene, 0 = most variable), so downstream reporting can slice
    "top-N most variable of the trained set" (`gene_interval` in experiment.py) without any further
    HVG (re)computation — training happens once, on `num_genes`; `gene_interval` only affects how
    validation-set correlations are aggregated afterwards, not what's trained on.
    """
    adata = ad.AnnData(
        counts.values.astype(np.float32),
        obs=pd.DataFrame({batch_key: sample_ids}, index=counts.index),
    )
    adata.var_names = counts.columns.astype(str)

    sc.pp.highly_variable_genes(adata, flavor=flavor, batch_key=batch_key, n_top_genes=num_genes)

    genes = pd.DataFrame({"gene_name": counts.columns.values}, index=counts.columns)
    genes["highly_variable"] = adata.var["highly_variable"].values
    genes["highly_variable_rank"] = adata.var["highly_variable_rank"].values
    return genes


def load_data(
    samples: Sequence[str],
    adata_path: str,
    embeddings_dir: str,
    embeddings_datasets: Sequence[str],
    factor: float = 10000,
    raw_counts: bool = False,
    consider_matching: bool = True,
    highly_variable_genes: Optional[pd.DataFrame] = None,
    num_genes: Optional[int] = None,
    hvg_flavor: str = "seurat_v3_paper",
    batch_key: str = "sampleID",
) -> dict:
    """
    Load gene expression, embeddings and spatial positions for `samples`, joined by cell barcode.

    Args:
        samples: WSI/sample names.
        adata_path: template with a `{sample}` placeholder, e.g.
            "$XENIUM_PROCESSED_OUTPUT_ROOT/{sample}/adata.h5ad" (see configs/deepspot_train.yaml's
            `data.adata_path_template`, resolved from the XENIUM_PROCESSED_OUTPUT_ROOT env var --
            see src/python/code_configs/paths.py and .env.example).
        embeddings_dir: directory containing `{sample}/embeddings_dataset.h5`
            (i.e. f"{base_dir}/{model_name}_h5/{correction_name}").
        factor: log1p normalization target sum (ignored if raw_counts=True).
        raw_counts: if True, skip log1p normalization and return raw counts.
        highly_variable_genes: gene table with `gene_name`/`highly_variable`/`highly_variable_rank`
            columns (see `load_hvg_table` for the precomputed-file case, or `_compute_highly_variable_genes`
            for the on-the-fly case). If given, each sample's gene panel is restricted to the
            selected genes *before* densifying — this is the normal path (HVGs precomputed once via
            compute_hvgs.py on a big-memory node, see configs/deepspot_train.yaml's `data.hvg_dir`).
            If None, HVGs are computed on the fly on this call's own (full-panel, un-shortcut) data —
            only advisable on a large-memory node; see the memory note below.
        num_genes: size of the highly-variable gene set to select (used only when
            highly_variable_genes is None).

    Returns:
        dict with keys:
          y: gene-expression DataFrame restricted to the selected genes, index
             f"{barcode}_{sample}", columns = gene names
          X: embeddings DataFrame, same index
          coords: DataFrame with x_centroid, y_centroid, barcode, sampleID, same index
          genes: highly-variable-genes table (see `highly_variable_genes` above)

    Memory note (on-the-fly path only, i.e. highly_variable_genes=None): this still builds one
    dense (all matched cells x full gene panel) matrix to rank genes by variability — for real
    WSI-scale data (100k+ cells, panels of hundreds-to-thousands of genes) that alone can OOM a
    regular training job. Precompute HVGs once via compute_hvgs.py (scripts/compute_hvgs.sh, big
    node) and pass the result in as `highly_variable_genes` instead — then this function never
    densifies more than `num_genes` columns per sample, independent of full panel size.
    """
    y_parts, X_parts, coords_parts = [], [], []

    kept_gene_names: Optional[np.ndarray] = None
    if highly_variable_genes is not None:
        kept_gene_names = highly_variable_genes.loc[
            highly_variable_genes["highly_variable"], "gene_name"
        ].to_numpy()

    for sample in tqdm(samples, desc="Loading samples"):
        adata = sc.read_h5ad(adata_path.format(sample=sample))

        # Gene panel already known (precomputed HVGs, or val split reusing train's table) — restrict
        # to just those columns before densifying anything, regardless of full panel size.
        if kept_gene_names is not None:
            adata = adata[:, kept_gene_names]

        barcodes = adata.obs['cell_id'].values

        embeddings, emb_cell_ids = _load_embeddings_for_sample(
            embeddings_dir, sample, embeddings_datasets, consider_matching
        )

        

        common = np.intersect1d(barcodes, emb_cell_ids)
        if len(common) == 0:
            continue

        barcode_to_row = {b: i for i, b in enumerate(barcodes)}
        emb_to_row = {b: i for i, b in enumerate(emb_cell_ids)}
        adata_idx = np.array([barcode_to_row[b] for b in common])
        emb_idx = np.array([emb_to_row[b] for b in common])

        # Row-subset (cheap on a sparse matrix) *before* densifying, and densify straight to
        # float32 — densifying the full (unfiltered, all-cells) counts matrix first was the main
        # driver of an OOM on real WSI-scale data (100k+ cells); see the memory note above.
        sample_counts = adata.layers["counts"][adata_idx]
        if hasattr(sample_counts, "todense"):
            sample_counts = np.asarray(sample_counts.todense(), dtype=np.float32)
        else:
            sample_counts = np.asarray(sample_counts, dtype=np.float32)

        sample_obs = adata.obs.iloc[adata_idx]
        index = [f"{b}_{sample}" for b in common]

        y_parts.append(pd.DataFrame(sample_counts, index=index, columns=adata.var_names))
        X_parts.append(pd.DataFrame(embeddings[emb_idx], index=index))
        coords_parts.append(pd.DataFrame({
            "x_centroid": sample_obs["x_centroid"].to_numpy(),
            "y_centroid": sample_obs["y_centroid"].to_numpy(),
            "barcode": common,
            "sampleID": sample,
        }, index=index))

    if not y_parts:
        raise ValueError(f"No cells matched between adata and embeddings for samples={list(samples)}")

    y = pd.concat(y_parts)
    X = pd.concat(X_parts)
    coords = pd.concat(coords_parts)

    genes = highly_variable_genes
    if genes is None:
        assert num_genes is not None, "num_genes is required when highly_variable_genes is None"
        genes = _compute_highly_variable_genes(
            y, coords["sampleID"].values, num_genes, flavor=hvg_flavor, batch_key=batch_key
        )
        # Trim down to the selected genes now, same as the pre-filtered path above, so `y`'s
        # contract is always "already restricted to the selected genes" regardless of which branch
        # produced `genes` (also frees the large full-panel matrix built just above).
        y = y.loc[:, genes.loc[genes["highly_variable"], "gene_name"].to_numpy()]

    gene_names = y.columns
    if not raw_counts:
        y = pd.DataFrame(log1p_normalization(y.values, factor=factor), index=y.index, columns=gene_names)

    return {"y": y, "X": X, "coords": coords, "genes": genes}


def load_hvg_table(path: str, num_genes: int) -> pd.DataFrame:
    """
    Load a precomputed HVG ranking (see compute_hvgs.py): drop genes not present in every val/test
    WSI of that split (the `present_in_test` column — ranked-variable-in-train is useless if the
    gene is simply absent from the panel we'll evaluate on), then select the `num_genes` most
    variable genes among what's left (by `highly_variable_rank`, ascending). The rank itself is
    independent of num_genes (computed once over the full train-WSI gene panel), so any num_genes
    cutoff can be derived from the same file without recomputing anything.
    """
    genes = pd.read_csv(path)
    if "present_in_test" in genes.columns:
        genes = genes[genes["present_in_test"]].copy()

    genes = genes.sort_values("highly_variable_rank", na_position="last").reset_index(drop=True)

    n_valid = int(genes["highly_variable_rank"].notna().sum())
    if n_valid < num_genes:
        raise ValueError(
            f"Only {n_valid} genes have a valid HVG rank after filtering to genes present in every "
            f"val/test WSI (at {path!r}), but num_genes={num_genes} were requested. Recompute HVGs "
            "with a higher data.hvg_max_genes, or lower data.num_genes."
        )

    genes["highly_variable"] = genes.index < num_genes
    return genes


# ── dataset ──────────────────────────────────────────────────────────────────────

class CustomDeepCellDataLoader(DeepCellDataLoader):
    """DeepCellDataLoader subclass reading from this project's data layout. Only __init__ and
    _load_patch are overridden; __len__ and __getitem__ are inherited unchanged."""

    def __init__(
        self,
        adata_path: str,
        embeddings_dir: str,
        embeddings_datasets: Sequence[str],
        samples: Sequence[str],
        num_genes: Optional[int] = None,
        target_sum: float = 10000,
        raw_counts: bool = False,
        resolution: int = 0,
        cell_context: str = "cell_neighbors",
        n_neighbors: int = 0,
        augmentation: str = "default",
        normalize: Optional[str] = None,
        scaler=None,
        resample_samples: bool = False,
        smooth_n: int = 0,
        consider_matching: bool = True,
        highly_variable_genes: Optional[pd.DataFrame] = None,
    ):
        Dataset.__init__(self)
        self.adata_path = adata_path
        self.embeddings_dir = embeddings_dir
        self.embeddings_datasets = embeddings_datasets
        self.samples = samples
        self.cell_context = cell_context
        self.target_sum = target_sum
        self.resolution = resolution
        self.n_neighbors = n_neighbors
        self.max_n_neighbors = 0
        self.augmentation = augmentation
        self.smooth_n = smooth_n
        self.normalize = normalize
        self.scaler = scaler
        self.resample_samples = resample_samples
        self.consider_matching = consider_matching
        self.load_image_features_in_memory = True
        print("loading data",flush=True)
        data = load_data(
            samples, adata_path, embeddings_dir, embeddings_datasets,
            factor=self.target_sum, raw_counts=raw_counts, consider_matching=consider_matching,
            highly_variable_genes=highly_variable_genes, num_genes=num_genes,
        )
        print("loaded data",flush=True)
        # Trained gene set = the num_genes most (batch-corrected) highly variable genes, selected
        # once (highly_variable_genes=None) or reused as-is (passed in, e.g. loaded from a
        # precomputed file, or a val split reusing train's table). `highly_variable_rank` lets
        # callers later slice "top-N most variable of this trained set" purely as a reporting
        # aggregation — see deepspot_training/experiment.py — with no further HVG (re)computation.
        # load_data() already restricts data["y"]'s columns to this set, so no further
        # column-subsetting is needed here.

        self.genes_table = data["genes"]
        self.genes_to_keep = self.genes_table["highly_variable"].values

        gene_names = data["y"].columns.values
        self.gene_names = gene_names
        transcriptomics = data["y"]

        patches = data["X"]
        self.patches = {b: e.values for b, e in patches.iterrows()}

        self.coordinates_df = data["coords"]


        assert (transcriptomics.index == self.coordinates_df.index).all()

        if "neighbors" in self.cell_context:
            neighbor_rows = []
            for sample in tqdm(samples, desc="Building neighbor graph"):
                sample_coords = self.coordinates_df[self.coordinates_df.sampleID == sample]
                if len(sample_coords) == 0:
                    continue

                neigh = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(sample_coords)))
                neigh.fit(sample_coords[["x_centroid", "y_centroid"]].values)
                neighbors = neigh.kneighbors(
                    sample_coords[["x_centroid", "y_centroid"]].values, return_distance=True
                )[1]
                neighbors = neighbors[:, 1:]  # remove the cell itself

                cell_ids = sample_coords.barcode.values
                sample_neighbors = pd.Series(
                    [
                        "___".join(f"{cell_ids[cell_id]}_{sample}" for cell_id in neigh_ids)
                        for neigh_ids in neighbors
                    ],
                    index=sample_coords.index,
                )
                neighbor_rows.append(sample_neighbors)

                max_n_neighbors = sample_neighbors.apply(lambda x: len(x.split("___"))).max()
                if self.max_n_neighbors < max_n_neighbors:
                    self.max_n_neighbors = max_n_neighbors

            self.coordinates_df["neighbors"] = pd.concat(neighbor_rows)

        if self.smooth_n > 0 or self.resolution > 0:
            res = self.resolution if self.resolution > 0 else 1
            resampled_idx, transcriptomics_smooth = spatial_upsample_and_smooth(
                transcriptomics.values, self.coordinates_df, transcriptomics.index,
                res, self.smooth_n, self.augmentation,
            )
            if self.smooth_n > 0:
                transcriptomics[:] = transcriptomics_smooth
            if self.resolution > 0:
                self.old_idx = transcriptomics.index
                transcriptomics = transcriptomics.loc[resampled_idx]
                self.coordinates_df = self.coordinates_df.loc[resampled_idx]
                assert (transcriptomics.index == self.coordinates_df.index).all()

        if self.resample_samples:
            assert (transcriptomics.index == self.coordinates_df.index).all()
            n_count = np.max(self.coordinates_df.sampleID.value_counts()).astype(int)
            temp_idx = np.array([f"{i}+++{idx}" for i, idx in enumerate(transcriptomics.index)])
            transcriptomics.index = temp_idx
            self.coordinates_df.index = temp_idx
            resampled_idx = get_balanced_index(temp_idx, self.coordinates_df.sampleID, n_count)
            transcriptomics = transcriptomics.loc[resampled_idx]
            self.coordinates_df = self.coordinates_df.loc[resampled_idx]
            new_idx = [i.split("+++")[1] for i in transcriptomics.index]
            transcriptomics.index = new_idx
            self.coordinates_df.index = new_idx
            assert (transcriptomics.index == self.coordinates_df.index).all()

        # Cached pre-scaler matrix so `set_normalization` can re-fit a scaler without re-reading
        # adata/embeddings from disk (needed to sweep `gene_normalization` as a grid hparam while
        # only loading data once per split — see deepspot_training/experiment.py).
        self._raw_transcriptomics = transcriptomics.copy()

        if self.scaler is not None:
            transcriptomics.values[:] = self.scaler.transform(transcriptomics.values)
        elif self.normalize is None:
            pass
        elif self.normalize == "standard":
            self.scaler = StandardScaler()
            transcriptomics.values[:] = self.scaler.fit_transform(transcriptomics.values)
        elif self.normalize == "robust":
            self.scaler = RobustScaler()
            transcriptomics.values[:] = self.scaler.fit_transform(transcriptomics.values)

        self.transcriptomics_df = transcriptomics
        self.transcriptomics = {b: t.values for b, t in transcriptomics.iterrows()}

    def set_normalization(self, normalize: Optional[str], scaler=None):
        """Re-derive transcriptomics_df/transcriptomics from the cached pre-scaler matrix, without
        touching disk or recomputing embeddings/neighbors. `scaler=...` reuses an already-fit
        scaler (e.g. the training split's scaler, for a validation split); otherwise a new scaler
        is fit here."""
        transcriptomics = self._raw_transcriptomics.copy()
        self.normalize = normalize
        self.scaler = scaler

        if self.scaler is not None:
            transcriptomics.values[:] = self.scaler.transform(transcriptomics.values)
        elif normalize is None:
            pass
        elif normalize == "standard":
            self.scaler = StandardScaler()
            transcriptomics.values[:] = self.scaler.fit_transform(transcriptomics.values)
        elif normalize == "robust":
            self.scaler = RobustScaler()
            transcriptomics.values[:] = self.scaler.fit_transform(transcriptomics.values)
        else:
            raise ValueError(f"Unknown normalize: {normalize!r}")

        self.transcriptomics_df = transcriptomics
        self.transcriptomics = {b: t.values for b, t in transcriptomics.iterrows()}
        return self.scaler

    def _load_patch(self, sampleID, cell_id):
        return self.patches[f"{cell_id}_{sampleID}"]
