"""
Layer 3 — Trainer for the DeepCell embedding-evaluation pipeline.

Pure training logic: given a preloaded train/val CustomDeepCellDataLoader pair and a config,
train DeepSpot's own `DeepCell` LightningModule (deepspot.cell.model.DeepCell, unmodified) with
`lightning.Trainer`, exactly as DeepSpot/he2st do (see he2st/workflows/models/DeepCell.py), then
report per-gene Pearson correlation on the val split. Stateless and MLflow-free — call
train_and_evaluate() from experiment.py.
"""
from dataclasses import dataclass
from typing import Optional

import lightning as L
import numpy as np
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from deepspot.cell.model import DeepCell
from deepspot.utils.utils import run_inference_from_dataloader


@dataclass
class TrainConfig:
    input_size: int
    output_size: int
    loss_func: str
    lr: float
    p: float
    weight_decay: float
    n_ensemble: int
    phi2rho_size: int
    emb_size: int
    agg_neighbors: str
    cell_context: str
    num_epochs: int
    batch_size: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    random_seed: int
    num_workers: int = 8


def _per_gene_pearson(y_pred: np.ndarray, y_true: np.ndarray, gene_names: np.ndarray) -> dict:
    """Per-gene Pearson r between predictions and target. Correlating against the (possibly
    scaler-transformed) stored target rather than an inverse-transformed one is mathematically
    equivalent here: Pearson r is invariant to each side's independent affine rescaling, and
    StandardScaler/RobustScaler are affine per-gene."""
    per_gene = {}
    for i, gene in enumerate(gene_names):
        pred_col = y_pred[:, i]
        true_col = y_true[:, i]
        if np.std(pred_col) == 0 or np.std(true_col) == 0:
            per_gene[gene] = float("nan")
            continue
        r, _ = pearsonr(pred_col, true_col)
        per_gene[gene] = float(r)
    return per_gene


def train_and_evaluate(
    train_dataset,
    val_dataset,
    gene_names: np.ndarray,
    config: TrainConfig,
    device: str,
) -> dict:
    """
    Train a DeepCell model on `train_dataset` and evaluate per-gene Pearson correlation on
    `val_dataset`.

    Args:
        train_dataset / val_dataset: CustomDeepCellDataLoader instances (already loaded/normalized).
        gene_names: gene names in the same column order as the datasets' transcriptomics.
        config: training hyperparameters.
        device: "cuda" or "cpu".

    Returns:
        dict with keys: mean_pearson, median_pearson, per_gene_pearson, train_loss_curve,
                        val_loss_curve
    """
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, drop_last=False,
    )

    model = DeepCell(
        input_size=config.input_size,
        output_size=config.output_size,
        loss_func=config.loss_func,
        lr=config.lr,
        p=config.p,
        n_ensemble=config.n_ensemble,
        phi2rho_size=config.phi2rho_size,
        emb_size=config.emb_size,
        weight_decay=config.weight_decay,
        random_seed=config.random_seed,
        agg_neighbors=config.agg_neighbors,
        scaler=train_dataset.scaler,
        cell_context=config.cell_context,
    )

    accelerator = "gpu" if device == "cuda" else "cpu"
    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=[
            EarlyStopping(
                monitor="val",
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                mode="min",
            )
        ],
    )
    trainer.fit(model, train_loader, val_loader)

    predictions = run_inference_from_dataloader(model, val_loader, device)
    y_true = val_dataset.transcriptomics_df.values

    per_gene_pearson = _per_gene_pearson(predictions, y_true, gene_names)
    valid_r = np.array([r for r in per_gene_pearson.values() if not np.isnan(r)])

    return {
        "mean_pearson": float(np.mean(valid_r)) if len(valid_r) else float("nan"),
        "median_pearson": float(np.median(valid_r)) if len(valid_r) else float("nan"),
        "per_gene_pearson": per_gene_pearson,
        "train_loss_curve": list(model.training_loss),
        "val_loss_curve": list(model.validation_loss),
    }
