"""
Pure training logic for the LWO classifier evaluation pipeline.
Stateless and MLflow-free — call train_and_evaluate() from experiment.py.
"""
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


@dataclass
class TrainConfig:
    input_dim: int
    num_classes: int
    class_names: list[str]
    learning_rate: float
    dropout: float
    hidden_dim: int          # 0 = logistic regression (no hidden layer)
    pca_components: Optional[int]
    normalization: str       # "none" | "z_score"
    # "none" | "by_wsi" | "by_cell_type" | "by_wsi_and_cell_type" reweight the CE loss.
    # "oversampling_cell_type" | "oversampling_wsi" instead draw train batches with a
    # WeightedRandomSampler using the same balanced weights, leaving the loss unweighted.
    loss_weights: str
    num_epochs: int
    batch_size: int
    seed: int = 42


def _is_oversampling(loss_weights: str) -> bool:
    return loss_weights.startswith("oversampling")


def _seed_everything(seed: int) -> None:
    """Seed every RNG this module touches, so a given config.seed is fully reproducible.

    Called once per train_and_evaluate() call (i.e. once per grid trial) with the
    same seed -- trials differ only by hyperparameters, not by init/shuffling noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        if hidden_dim == 0:
            self.net: nn.Module = nn.Linear(input_dim, num_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _apply_preprocessing(
    train_emb: np.ndarray,
    eval_embs: list[np.ndarray],
    pca_components: Optional[int],
    normalization: str,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Fit PCA/z-score stats once on train_emb and transform train plus every eval array."""
    if pca_components is not None:
        n_components = min(pca_components, train_emb.shape[1], train_emb.shape[0] - 1)
        # random_state pins PCA's randomized SVD solver (auto-selected for
        # high-dim embeddings with n_components << n_features) for reproducibility.
        pca = PCA(n_components=n_components, random_state=seed)
        train_emb = pca.fit_transform(train_emb)
        eval_embs = [pca.transform(e) for e in eval_embs]
    if normalization == "z_score":
        mean = train_emb.mean(axis=0)
        std = train_emb.std(axis=0) + 1e-8
        train_emb = (train_emb - mean) / std
        eval_embs = [(e - mean) / std for e in eval_embs]
    return train_emb, eval_embs


def _compute_sample_weights(
    labels: torch.Tensor,
    wsi_origins: torch.Tensor,
    num_classes: int,
    loss_weights: str,
) -> torch.Tensor:
    """Balanced (inverse-frequency) per-sample weights, shared by both use sites:
    "by_*" values multiply these into the CE loss; "oversampling_*" values instead
    feed them to a WeightedRandomSampler (see _is_oversampling / train_and_evaluate).
    Matched by substring so "by_cell_type"/"oversampling_cell_type" both trigger the
    class term and "by_wsi"/"oversampling_wsi" both trigger the WSI term.
    """
    n = len(labels)
    weights = torch.ones(n, dtype=torch.float32)

    if "cell_type" in loss_weights:
        class_counts = torch.bincount(labels, minlength=num_classes).float()
        class_w = n / (num_classes * class_counts.clamp(min=1))
        weights = weights * class_w[labels]

    if "wsi" in loss_weights:
        unique_wsis = wsi_origins.unique()
        wsi_w = torch.ones(n, dtype=torch.float32)
        for wsi_id in unique_wsis:
            mask = wsi_origins == wsi_id
            count = mask.sum().float()
            wsi_w[mask] = n / (len(unique_wsis) * count)
        weights = weights * wsi_w

    weights = weights / weights.mean()
    return weights


def _make_loader(
    emb_t: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    batch_size: int,
    shuffle: bool = False,
    sampler: Optional[WeightedRandomSampler] = None,
) -> DataLoader:
    dataset = TensorDataset(emb_t, labels, weights)
    # DataLoader forbids passing both shuffle=True and an explicit sampler.
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, drop_last=False, num_workers=16,
    )


def _run_forward_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, np.ndarray]:
    """One full pass over `loader`. Returns (weighted mean loss, predictions)."""
    total_loss = 0.0
    n = 0
    all_preds: list[np.ndarray] = []
    for emb_b, label_b, weight_b in loader:
        emb_b = emb_b.to(device)
        label_b = label_b.to(device)
        weight_b = weight_b.to(device)
        logits = model(emb_b)
        losses = criterion(logits, label_b)              # shape: [batch_size]
        total_loss += (losses * weight_b).sum().item()   # weighted sum
        n += len(label_b)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
    return total_loss / n, np.concatenate(all_preds)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    labels: torch.Tensor,
    criterion: nn.Module,
    device: str,
    num_classes: int,
    class_names: list[str],
) -> dict:
    """Run one forward pass over `loader` and compute the full metric set."""
    model.eval()
    with torch.no_grad():
        loss, preds = _run_forward_pass(model, loader, criterion, device)

    labels_np = labels.numpy()
    class_range = list(range(num_classes))

    acc = float(accuracy_score(labels_np, preds))
    macro_f1 = float(f1_score(labels_np, preds, average="macro", zero_division=0, labels=class_range))
    weighted_f1 = float(f1_score(labels_np, preds, average="weighted", zero_division=0, labels=class_range))
    balanced_acc = float(balanced_accuracy_score(labels_np, preds))
    per_class_f1_arr = f1_score(labels_np, preds, average=None, zero_division=0, labels=class_range)
    cm = confusion_matrix(labels_np, preds, labels=class_range).tolist()

    return {
        "loss": loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_acc,
        "per_class_f1": {
            class_names[i]: float(per_class_f1_arr[i]) for i in range(num_classes)
        },
        "confusion_matrix": cm,
    }


def _prefix_metrics(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def train_and_evaluate(
    train_embeddings: np.ndarray,
    train_labels: torch.Tensor,
    train_wsi_origins: torch.Tensor,
    test_embeddings: np.ndarray,
    test_labels: torch.Tensor,
    test_wsi_origins: torch.Tensor,
    config: TrainConfig,
    device: str,
    val_embeddings: Optional[np.ndarray] = None,
    val_labels: Optional[torch.Tensor] = None,
    val_wsi_origins: Optional[torch.Tensor] = None,
) -> dict:
    """
    Train an MLP on frozen embeddings and return evaluation metrics.

    Args:
        train_embeddings: (N_train, D) float array
        train_labels: (N_train,) long tensor
        train_wsi_origins: (N_train,) long tensor — WSI index per cell
        test_embeddings: (N_test, D) float array
        test_labels: (N_test,) long tensor
        test_wsi_origins: (N_test,) long tensor
        config: full training configuration
        device: "cuda" or "cpu"
        val_embeddings, val_labels, val_wsi_origins: optional validation set.

    If a validation set is given, the per-epoch curve and hyperparameter-selection
    metrics are computed on it (never on test), and test is only evaluated once
    at the end for reporting — returned keys are prefixed `val_*`/`test_*`, plus
    `train_loss_curve`/`val_loss_curve`.

    If no validation set is given (e.g. grid_experiment.py's spatial-checkerboard
    mode, which has no val concept), behavior is unchanged from before: the
    per-epoch curve and final metrics are computed against test, and returned
    keys are unprefixed (`accuracy`, `macro_f1`, ..., `test_loss_curve`).
    """
    _seed_everything(config.seed)

    has_val = val_embeddings is not None
    eval_embs = [val_embeddings, test_embeddings] if has_val else [test_embeddings]
    train_emb, eval_embs = _apply_preprocessing(
        train_embeddings, eval_embs, config.pca_components, config.normalization, config.seed
    )
    if has_val:
        val_emb, test_emb = eval_embs
    else:
        (test_emb,) = eval_embs

    train_t = torch.tensor(train_emb, dtype=torch.float32)
    test_t = torch.tensor(test_emb, dtype=torch.float32)
    input_dim = train_t.shape[1]

    model = MLP(input_dim, config.hidden_dim, config.num_classes, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction="none")

    oversample = _is_oversampling(config.loss_weights)

    def _weights_for(labels: torch.Tensor, wsi_origins: torch.Tensor) -> torch.Tensor:
        # Under oversampling, balance is enforced by the sampler (below), so the
        # CE loss itself stays unweighted -- same as "none" -- for train and eval alike.
        if config.loss_weights == "none" or oversample:
            return torch.ones(len(labels), dtype=torch.float32)
        return _compute_sample_weights(labels, wsi_origins, config.num_classes, config.loss_weights)

    sample_weights = _weights_for(train_labels, train_wsi_origins)
    test_weights = _weights_for(test_labels, test_wsi_origins)

    if oversample:
        sampling_weights = _compute_sample_weights(
            train_labels, train_wsi_origins, config.num_classes, config.loss_weights
        )
        train_sampler = WeightedRandomSampler(
            sampling_weights, num_samples=len(train_labels), replacement=True
        )
        loader = _make_loader(train_t, train_labels, sample_weights, config.batch_size, sampler=train_sampler)
    else:
        loader = _make_loader(train_t, train_labels, sample_weights, config.batch_size, shuffle=True)
    test_loader = _make_loader(test_t, test_labels, test_weights, config.batch_size, shuffle=False)

    if has_val:
        val_t = torch.tensor(val_emb, dtype=torch.float32)
        val_weights = _weights_for(val_labels, val_wsi_origins)
        val_loader = _make_loader(val_t, val_labels, val_weights, config.batch_size, shuffle=False)
        curve_loader, curve_labels = val_loader, val_labels
    else:
        curve_loader, curve_labels = test_loader, test_labels

    train_loss_curve: list[float] = []
    curve_loss_curve: list[float] = []

    model.train()
    for _ in range(config.num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for emb_b, label_b, w_b in loader:
            emb_b = emb_b.to(device)
            label_b = label_b.to(device)
            w_b = w_b.to(device)
            logits = model(emb_b)
            loss = (criterion(logits, label_b) * w_b).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss_curve.append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            epoch_curve_loss, _ = _run_forward_pass(model, curve_loader, criterion, device)
            curve_loss_curve.append(epoch_curve_loss)
        model.train()

    if has_val:
        val_metrics = _evaluate(
            model, val_loader, val_labels, criterion, device, config.num_classes, config.class_names
        )
        test_metrics = _evaluate(
            model, test_loader, test_labels, criterion, device, config.num_classes, config.class_names
        )
        return {
            **_prefix_metrics(val_metrics, "val"),
            **_prefix_metrics(test_metrics, "test"),
            "train_loss_curve": train_loss_curve,
            "val_loss_curve": curve_loss_curve,
        }

    test_metrics = _evaluate(
        model, test_loader, test_labels, criterion, device, config.num_classes, config.class_names
    )
    return {
        **test_metrics,
        "train_loss_curve": train_loss_curve,
        "test_loss_curve": curve_loss_curve,
    }
