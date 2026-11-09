"""Per-slide, per-gene expression statistics for every slide of one or more deepspot_training
configs: do the HVGs a fold trains on (ranked on its train slides by compute_hvgs.py) actually
vary on its test slides? One row per (slide, gene) of that slide's panel:

  n_cells         cells in the slide's adata.h5ad
  detect_frac     fraction of cells with >= 1 count
  mean_counts     mean raw count per cell
  mean_log1p,     mean / variance of log1p(counts / cell total * target_sum), cell totals over
  var_log1p       the slide's full panel
  variances_norm  scanpy seurat_v3 normalised variance within the slide -- the per-slide score
                  compute_hvgs.py ranks genes by (it is computed per batch = per slide there too)

Reads layers['counts'] one slide at a time (the 10x Prime lymph node, ~0.65M cells x 5K genes,
needs ~40G).

Usage (repo root, .env sourced):
    python scripts/gene_slide_stats.py deepspot_lopo_vannan deepspot_lopo_prime --out gene_slide_stats.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[1]


def config_slides(config_names: list[str]) -> dict[str, list[tuple[str, str, float]]]:
    """config name -> [(wsi, adata path, target_sum)] over every slide of its splits."""
    out = {}
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        for name in config_names:
            cfg = compose(name, overrides=["data.model_name=UNI2", "data.correction_name=raw"])
            wsis = sorted({w for split in cfg.splits for side in split.values() for w in side})
            out[name] = [(w, cfg.data.adata_path_template.format(sample=w), float(cfg.data.target_sum)) for w in wsis]
    return out


def slide_stats(adata_path: str, target_sum: float) -> pd.DataFrame:
    adata = sc.read_h5ad(adata_path)
    counts = adata.layers["counts"]
    counts = (counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)).astype(np.float32)
    n_cells = counts.shape[0]

    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    log1p = sp.diags((target_sum / totals).astype(np.float32)) @ counts
    log1p.data = np.log1p(log1p.data)
    mean_log1p = np.asarray(log1p.mean(axis=0)).ravel()
    var_log1p = np.asarray(log1p.multiply(log1p).mean(axis=0)).ravel() - mean_log1p ** 2
    del log1p

    stats = pd.DataFrame({
        "gene_name": adata.var_names.astype(str),
        "n_cells": n_cells,
        "detect_frac": np.asarray((counts > 0).sum(axis=0)).ravel() / n_cells,
        "mean_counts": np.asarray(counts.mean(axis=0)).ravel(),
        "mean_log1p": mean_log1p,
        "var_log1p": var_log1p,
    })
    hvg = sc.AnnData(X=counts)
    hvg.var_names = stats.gene_name.to_numpy()
    sc.pp.highly_variable_genes(hvg, flavor="seurat_v3", n_top_genes=hvg.n_vars)
    stats["variances_norm"] = hvg.var["variances_norm"].to_numpy()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="config names under configs/, e.g. deepspot_lopo_prime")
    parser.add_argument("--out", type=Path, default=Path("gene_slide_stats.csv"))
    args = parser.parse_args()

    parts = []
    for name, slides in config_slides(args.configs).items():
        for i, (wsi, path, target_sum) in enumerate(slides, 1):
            print(f"[{name} {i}/{len(slides)}] {wsi}", flush=True)
            parts.append(slide_stats(path, target_sum).assign(config=name, wsi=wsi))
    out = pd.concat(parts, ignore_index=True)
    out = out[["config", "wsi", *[c for c in out.columns if c not in ("config", "wsi")]]]
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} rows, {out.wsi.nunique()} slides")


if __name__ == "__main__":
    main()
