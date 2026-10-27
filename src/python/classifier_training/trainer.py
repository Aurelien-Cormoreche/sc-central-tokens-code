"""
Pure training logic for the LWO classifier evaluation pipeline.
Stateless and MLflow-free — call train_and_evaluate() from experiment.py.
"""
import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
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
    num_epochs: int          # maximum number of epochs when early stopping is on
    batch_size: int
    seed: int = 42
    # Early stopping on the val selection score (train_and_evaluate's val_score_fn):
    # stop once it hasn't improved by more than min_delta for `patience` epochs, then
    # restore the best epoch's weights. None = train all num_epochs, keep the last epoch.
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0


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


@dataclass
class Preprocessor:
    """Train-fitted PCA / z-score state, kept as plain arrays so a saved classifier
    can be re-applied to embeddings it never saw (see saved_models.py). Each stage is
    skipped when its arrays are None.
    """
    pca_mean: Optional[np.ndarray] = None        # (D,)
    pca_components: Optional[np.ndarray] = None  # (n_components, D)
    z_mean: Optional[np.ndarray] = None
    z_std: Optional[np.ndarray] = None           # already includes the +1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.pca_components is not None:
            # Same two-matmul form as sklearn's PCA.transform (whiten=False), so
            # applying a reloaded Preprocessor matches what training did.
            x = x @ self.pca_components.T - self.pca_mean @ self.pca_components.T
        if self.z_mean is not None:
            x = (x - self.z_mean) / self.z_std
        return x


def _fit_preprocessing(
    train_emb: np.ndarray,
    pca_components: Optional[int],
    normalization: str,
    seed: int,
) -> tuple[np.ndarray, Preprocessor]:
    """Fit PCA/z-score stats once on train_emb; returns (transformed train, fitted
    Preprocessor) -- eval arrays go through Preprocessor.transform."""
    pre = Preprocessor()
    if pca_components is not None:
        n_components = min(pca_components, train_emb.shape[1], train_emb.shape[0] - 1)
        # random_state pins PCA's randomized SVD solver (auto-selected for
        # high-dim embeddings with n_components << n_features) for reproducibility.
        pca = PCA(n_components=n_components, random_state=seed)
        train_emb = pca.fit_transform(train_emb)
        pre.pca_mean, pre.pca_components = pca.mean_, pca.components_
    if normalization == "z_score":
        pre.z_mean = train_emb.mean(axis=0)
        pre.z_std = train_emb.std(axis=0) + 1e-8
        train_emb = (train_emb - pre.z_mean) / pre.z_std
    return train_emb, pre


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
) -> tuple[float, np.ndarray, np.ndarray]:
    """One full pass over `loader`. Returns (weighted mean loss, predictions,
    per-class softmax probabilities) -- predictions/probabilities are in the
    same order as `loader`'s dataset (shuffle=False for every eval loader), so
    callers can zip them back against the original labels/ids/wsi arrays."""
    total_loss = 0.0
    n = 0
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    for emb_b, label_b, weight_b in loader:
        emb_b = emb_b.to(device)
        label_b = label_b.to(device)
        weight_b = weight_b.to(device)
        logits = model(emb_b)
        losses = criterion(logits, label_b)              # shape: [batch_size]
        total_loss += (losses * weight_b).sum().item()   # weighted sum
        n += len(label_b)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return total_loss / n, np.concatenate(all_preds), np.concatenate(all_probs)


def classification_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    num_classes: int,
    class_names: list[str],
) -> dict:
    """Label-only metric set, shared by the pooled evaluation (_evaluate) and the
    per-slide breakdown in experiment.py so both are computed identically.

    `macro_f1` averages over every model class (a class absent from `labels` and never
    predicted scores 0 under zero_division=0), so it is deflated on any set missing some
    classes -- typical for a single slide or a held-out cohort. `macro_f1_present`
    averages only over classes with support > 0 in `labels`.
    `balanced_accuracy` is mean recall over classes present in `labels` -- identical
    to sklearn's balanced_accuracy_score, taken from the confusion matrix so slides
    whose predictions include classes absent from their labels don't warn.
    """
    class_range = list(range(num_classes))
    cm = confusion_matrix(labels, preds, labels=class_range)
    support = cm.sum(axis=1)
    present = support > 0
    per_class_f1_arr = f1_score(labels, preds, average=None, zero_division=0, labels=class_range)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0, labels=class_range)),
        "macro_f1_present": float(per_class_f1_arr[present].mean()),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0, labels=class_range)),
        "balanced_accuracy": float((np.diag(cm)[present] / support[present]).mean()),
        "per_class_f1": {
            class_names[i]: float(per_class_f1_arr[i]) for i in range(num_classes)
        },
        "confusion_matrix": cm.tolist(),
    }


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
        loss, preds, probs = _run_forward_pass(model, loader, criterion, device)

    return {
        "loss": loss,
        **classification_metrics(labels.numpy(), preds, num_classes, class_names),
        # Raw per-cell outputs (same order as `loader`'s dataset), for callers that
        # want to save individual predictions rather than just aggregate metrics
        # (see classifier_training/experiment.py's save_test_predictions).
        "predictions": preds,
        "probabilities": probs,
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
    return_model: bool = False,
    val_score_fn: Optional[Callable[[np.ndarray], float]] = None,
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

    With a validation set, every epoch's val predictions (in val order) are also
    scored by val_score_fn (higher = better; default: pooled val macro F1) -- the
    score early stopping monitors (config.early_stopping_patience). Also returned:
    `val_score_curve`, `selected_epoch` (1-based epoch whose weights are evaluated),
    `epochs_trained` and `val_selection_score` (val_score_fn of the evaluated model,
    what experiment.py compares across trials).

    If no validation set is given (e.g. grid_experiment.py's spatial-checkerboard
    mode, which has no val concept), behavior is unchanged from before: the
    per-epoch curve and final metrics are computed against test, and returned
    keys are unprefixed (`accuracy`, `macro_f1`, ..., `test_loss_curve`).

    If return_model, the result also carries `model_state_dict` (CPU copies of the
    evaluated, i.e. selected-epoch, weights) and `preprocessor` (the train-fitted Preprocessor) -- everything
    saved_models.save_classifier needs to re-apply this classifier later.
    """
    _seed_everything(config.seed)

    has_val = val_embeddings is not None
    train_emb, preprocessor = _fit_preprocessing(
        train_embeddings, config.pca_components, config.normalization, config.seed
    )
    test_emb = preprocessor.transform(test_embeddings)
    if has_val:
        val_emb = preprocessor.transform(val_embeddings)

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
        if val_score_fn is None:
            # A copy: the DataLoader may move val_labels' storage to shared memory.
            val_labels_np = val_labels.numpy().copy()
            val_score_fn = lambda preds: classification_metrics(  # noqa: E731
                val_labels_np, preds, config.num_classes, config.class_names
            )["macro_f1"]
    else:
        curve_loader, curve_labels = test_loader, test_labels

    # Early stopping only ever monitors val -- without one (grid_experiment.py) the
    # per-epoch curve is on test, which must not drive any choice.
    early_stopping = has_val and config.early_stopping_patience is not None
    train_loss_curve: list[float] = []
    curve_loss_curve: list[float] = []
    val_score_curve: list[float] = []
    best_score = -np.inf
    best_epoch = 0
    best_state: Optional[dict[str, torch.Tensor]] = None

    model.train()
    for epoch in range(1, config.num_epochs + 1):
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
            epoch_curve_loss, epoch_preds, _ = _run_forward_pass(model, curve_loader, criterion, device)
            curve_loss_curve.append(epoch_curve_loss)
        model.train()

        if not has_val:
            continue
        score = float(val_score_fn(epoch_preds))
        val_score_curve.append(score)
        if score > best_score + config.early_stopping_min_delta:
            best_score, best_epoch = score, epoch
            if early_stopping:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif early_stopping and epoch - best_epoch >= config.early_stopping_patience:
            break

    if early_stopping:
        model.load_state_dict(best_state)
        selected_epoch = best_epoch
    else:
        selected_epoch = len(train_loss_curve)
    model.eval()

    model_outputs = (
        {
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "preprocessor": preprocessor,
        }
        if return_model else {}
    )

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
            "val_selection_score": float(val_score_fn(val_metrics["predictions"])),
            "selected_epoch": selected_epoch,
            "epochs_trained": len(train_loss_curve),
            "train_loss_curve": train_loss_curve,
            "val_loss_curve": curve_loss_curve,
            "val_score_curve": val_score_curve,
            **model_outputs,
        }

    test_metrics = _evaluate(
        model, test_loader, test_labels, criterion, device, config.num_classes, config.class_names
    )
    return {
        **test_metrics,
        "train_loss_curve": train_loss_curve,
        "test_loss_curve": curve_loss_curve,
        **model_outputs,
    }
