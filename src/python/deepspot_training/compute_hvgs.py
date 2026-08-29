"""
Precompute highly-variable genes (HVGs) per split, offline, on a big-memory node.

For each split's TRAIN WSIs, loads the full Xenium AnnData (raw counts, `layers['counts']`) for
every WSI, concatenates them (no cell subsampling, no embeddings/matching filtering — HVG selection
is a property of the transcriptomics data alone), and ranks genes by (batch-corrected) variability
via scanpy (`sc.pp.highly_variable_genes`, flavor="seurat_v3_paper" by default, gives a
`highly_variable_rank` per gene) — up to `data.hvg_max_genes` (default 500; num_genes/gene_interval
never need more). This full-cell-population computation is intentionally memory-heavy — run it on a
large-memory node (see scripts/compute_hvgs.sh, ~80G), not as part of a regular training job.

Also checks, for each ranked gene, whether it's present in every one of the split's val/test WSI(s)
— a gene ranked highly variable from the train WSIs is useless if it's simply absent from the panel
of the WSI we'll evaluate on (e.g. differing Xenium panel versions across samples).

Writes, per split, into `data.hvg_dir` (default `data/HVGsSplit/`):
  {split_label}.csv         — one row per gene: gene_name, highly_variable_rank (ascending, 0 = most
                               variable), present_in_test (bool). Rank is independent of any later
                               `num_genes` cutoff — pick any top-N later (see
                               custom_dataloader.load_hvg_table, which also applies the
                               present_in_test filter) without recomputing.
  {split_label}_wsis.txt      — the train WSIs used to compute this split's ranking (provenance).
  {split_label}_test_wsis.txt — the val/test WSI(s) checked for gene-panel presence (provenance).

deepspot_training/experiment.py loads these (via data.hvg_dir in configs/deepspot_train.yaml)
instead of computing HVGs on the fly.

Run via SLURM: sbatch scripts/compute_hvgs.sh
Or directly:   python -m src.python.deepspot_training.compute_hvgs
"""
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import scanpy as sc
from omegaconf import DictConfig, OmegaConf


def _resolve_train_wsis(cfg: DictConfig, split_idx: int) -> list[str]:
    """Train WSIs only for this split — mirrors deepspot_training/experiment.py::_resolve_split
    (train side); split_idx == -1 is the pooled same_wsi_split set (train == test == every WSI)."""
    if split_idx == -1:
        all_wsis: set[str] = set()
        for sp in cfg.splits:
            all_wsis.update(k for k in sp.train.keys() if k != "cell_type_proportions")
            all_wsis.update(k for k in sp.test.keys() if k != "cell_type_proportions")
        return sorted(all_wsis)

    sp = cfg.splits[split_idx]
    return [k for k in sp.train.keys() if k != "cell_type_proportions"]


def _resolve_val_wsis(cfg: DictConfig, split_idx: int) -> list[str]:
    """Val/test WSIs for this split, used only to check gene-panel presence. split_idx == -1
    (same_wsi_split): train and test cells come from the same pooled WSI set, so presence is
    checked against that same pool (already covered by train_wsis — nothing extra to check)."""
    if split_idx == -1:
        return _resolve_train_wsis(cfg, split_idx)
    sp = cfg.splits[split_idx]
    return [k for k in sp.test.keys() if k != "cell_type_proportions"]


def compute_split_hvgs(
    train_wsis: list[str],
    val_wsis: list[str],
    adata_path_template: str,
    flavor: str = "seurat_v3_paper",
    batch_key: str = "sampleID",
    max_genes: int = 500,
) -> pd.DataFrame:
    """Concatenate every train WSI's raw counts (full cell population, no cell subsampling — uses
    everything) and rank genes by (batch-corrected) variability, keeping rank info for only the top
    `max_genes` (num_genes/gene_interval never need more than that; genes outside the top `max_genes`
    get a NaN rank and are simply never selected downstream). Also flags, per gene, whether it's
    present in every WSI in `val_wsis`."""
    adatas = []
    for wsi in train_wsis:
        print(f"[compute_hvgs]   reading {wsi}")
        a = sc.read_h5ad(adata_path_template.format(sample=wsi))
        a.X = a.layers["counts"]
        a.obs[batch_key] = wsi
        a.obs_names = [f"{bc}_{wsi}" for bc in a.obs_names]
        adatas.append(a)

    combined = ad.concat(adatas, join="inner")
    gene_names = combined.var_names.to_numpy()
    n_top_genes = min(max_genes, len(gene_names))
    print(f"[compute_hvgs]   ranking top {n_top_genes} of {len(gene_names)} genes over {combined.n_obs} cells")

    sc.pp.highly_variable_genes(combined, flavor=flavor, batch_key=batch_key, n_top_genes=n_top_genes)

    print(f"[compute_hvgs]   checking gene-panel presence in {len(val_wsis)} val/test WSI(s)")
    present_in_test = np.ones(len(gene_names), dtype=bool)
    for wsi in val_wsis:
        # backed="r" avoids loading counts into memory — we only need var_names here.
        val_adata = sc.read_h5ad(adata_path_template.format(sample=wsi), backed="r")
        present_in_test &= np.isin(gene_names, val_adata.var_names.to_numpy())

    genes = pd.DataFrame({
        "gene_name": gene_names,
        "highly_variable_rank": combined.var["highly_variable_rank"].to_numpy(),
        "present_in_test": present_in_test,
    })
    genes = genes.sort_values("highly_variable_rank", na_position="last").reset_index(drop=True)
    return genes


@hydra.main(config_path="../../../configs", config_name="deepspot_train_gsm", version_base="1.3")
def main(cfg: DictConfig) -> None:
    repo_root = Path(hydra.utils.get_original_cwd())
    out_dir = repo_root / str(OmegaConf.select(cfg, "data.hvg_dir", default="data/HVGsSplit"))
    out_dir.mkdir(parents=True, exist_ok=True)

    flavor = OmegaConf.select(cfg, "data.hvg_flavor", default="seurat_v3_paper")
    batch_key = OmegaConf.select(cfg, "data.hvg_batch_key", default="sampleID")
    max_genes = int(OmegaConf.select(cfg, "data.hvg_max_genes", default=500))

    split_indices = list(range(len(cfg.splits)))
    if cfg.same_wsi_split:
        split_indices.append(-1)

    for split_idx in split_indices:
        split_label = "same_wsi_split" if split_idx == -1 else f"split_{split_idx}"
        train_wsis = _resolve_train_wsis(cfg, split_idx)
        val_wsis = _resolve_val_wsis(cfg, split_idx)
        print(f"[compute_hvgs] {split_label}: {len(train_wsis)} train WSIs, {len(val_wsis)} val/test WSIs")

        genes = compute_split_hvgs(
            train_wsis, val_wsis, cfg.data.adata_path_template,
            flavor=flavor, batch_key=batch_key, max_genes=max_genes,
        )

        genes.to_csv(out_dir / f"{split_label}.csv", index=False)
        (out_dir / f"{split_label}_wsis.txt").write_text("\n".join(train_wsis) + "\n")
        (out_dir / f"{split_label}_test_wsis.txt").write_text("\n".join(val_wsis) + "\n")

        n_dropped = int((~genes["present_in_test"]).sum())
        print(f"[compute_hvgs] {split_label}: wrote {len(genes)} genes -> {out_dir / f'{split_label}.csv'} "
              f"({n_dropped} not present in all val/test WSI(s))")

    print("[compute_hvgs] done.")


if __name__ == "__main__":
    main()
