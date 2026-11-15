"""
Pure training logic for the LWO classifier evaluation pipeline.
Stateless and MLflow-free — call train_and_evaluate() from experiment.py.
"""
import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


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
    # AdamW's decoupled weight decay: each step multiplies the weights by (1 - lr * weight_decay).
    # 1e-2 is torch's default, which every trial used before this was a hyperparameter.
    weight_decay: float = 1e-2
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
        pre.z_mean, pre.z_std = _z_score_stats(train_emb)
        train_emb = (train_emb - pre.z_mean) / pre.z_std
    return train_emb, pre


def _z_score_stats(train_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z-score fit: per-feature mean and std (+1e-8) of the train rows. Shared by
    train_and_evaluate and train_linear_jointly, so both standardise with the same numbers."""
    return train_emb.mean(axis=0), train_emb.std(axis=0) + 1e-8


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


def _loss_weights(config: TrainConfig, labels: torch.Tensor, wsi_origins: torch.Tensor) -> torch.Tensor:
    """Per-row CE loss weights of a trial. Under oversampling, balance is enforced by the
    sampled order (see _epoch_order), so the loss itself stays unweighted -- same as "none" --
    for train and eval alike."""
    if config.loss_weights == "none" or _is_oversampling(config.loss_weights):
        return torch.ones(len(labels), dtype=torch.float32)
    return _compute_sample_weights(labels, wsi_origins, config.num_classes, config.loss_weights)


# Batches gathered and copied to the device together: one index_select and one transfer
# per 64 batches instead of a DataLoader collate per batch.
_CHUNK_BATCHES = 64


def _draw_loader_seed(generator: Optional[torch.Generator] = None) -> None:
    """Take the int64 base seed that creating a DataLoader iterator draws from torch's
    global RNG (or from `generator`, which train_linear_jointly gives each split).

    The trainer used to iterate DataLoaders, which consumed one such draw per pass (train
    and eval alike) plus RandomSampler's permutation seed. Making the same draws keeps the
    global RNG stream unchanged, so for a given seed the shuffles, oversampling draws and
    CPU dropout masks -- hence every result -- match runs of the DataLoader trainer.
    """
    torch.empty((), dtype=torch.int64).random_(generator=generator)


def _epoch_order(
    n: int, sampling_weights: Optional[torch.Tensor], generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Train row order for one epoch, drawn exactly as DataLoader(shuffle=True) or
    DataLoader(sampler=WeightedRandomSampler(sampling_weights, n, replacement=True)) did."""
    _draw_loader_seed(generator)
    if sampling_weights is not None:
        return torch.multinomial(sampling_weights, n, replacement=True, generator=generator)
    seed = int(torch.empty((), dtype=torch.int64).random_(generator=generator).item())
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))


def _device_chunks(
    tensors: tuple[torch.Tensor, ...],
    device: str,
    chunk_rows: int,
    order: Optional[torch.Tensor] = None,
):
    """Yield (offset, tensors on `device`) for consecutive chunks of chunk_rows rows of
    tensors[order] (order=None: rows in order). On CUDA each chunk is gathered on the CPU
    into pinned memory and copied without blocking, so the copy overlaps the previous
    chunk's compute; torch's pinned-memory cache doesn't reuse a buffer before its copy ends."""
    cuda = torch.device(device).type == "cuda"
    n = len(tensors[0]) if order is None else len(order)
    for lo in range(0, n, chunk_rows):
        hi = min(lo + chunk_rows, n)
        chunk = []
        for t in tensors:
            if order is None and not cuda:
                chunk.append(t[lo:hi])
                continue
            buf = torch.empty((hi - lo,) + t.shape[1:], dtype=t.dtype, pin_memory=cuda)
            if order is None:
                buf.copy_(t[lo:hi])
            else:
                torch.index_select(t, 0, order[lo:hi], out=buf)
            chunk.append(buf.to(device, non_blocking=True))
        yield lo, chunk


def _run_forward_pass(
    model: nn.Module,
    emb_t: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    criterion: nn.Module,
    device: str,
    batch_size: int,
    return_probs: bool = True,
) -> tuple[float, np.ndarray, Optional[np.ndarray]]:
    """One pass over every row, in order. Returns (weighted mean loss, predictions,
    per-class softmax probabilities or None if not return_probs), in row order, so callers
    can zip them back against the original labels/ids/wsi arrays.

    The model still sees batch_size-row batches, so logits, predictions and the loss are
    bit-identical to the former DataLoader pass; only the transfer is done in large chunks,
    and per-batch loss sums are added on the device in float64 (as the former Python float
    sum did) instead of syncing on every batch."""
    _draw_loader_seed()
    n = len(labels)
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    preds = torch.empty(n, dtype=torch.long, device=device)
    probs: Optional[torch.Tensor] = None
    for lo, (emb_c, label_c, weight_c) in _device_chunks(
        (emb_t, labels, weights), device, batch_size * _CHUNK_BATCHES
    ):
        for i in range(0, len(label_c), batch_size):
            logits = model(emb_c[i:i + batch_size])
            losses = criterion(logits, label_c[i:i + batch_size])           # shape: [batch_size]
            total_loss += (losses * weight_c[i:i + batch_size]).sum()        # weighted sum
            rows = slice(lo + i, lo + i + len(logits))
            preds[rows] = logits.argmax(dim=1)
            if return_probs:
                if probs is None:
                    probs = torch.empty((n, logits.shape[1]), dtype=logits.dtype, device=device)
                probs[rows] = torch.softmax(logits, dim=1)
    return (
        total_loss.item() / n,
        preds.cpu().numpy(),
        probs.cpu().numpy() if probs is not None else None,
    )


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
    emb_t: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    criterion: nn.Module,
    device: str,
    batch_size: int,
    num_classes: int,
    class_names: list[str],
) -> dict:
    """Run one forward pass over every row and compute the full metric set."""
    model.eval()
    with torch.no_grad():
        loss, preds, probs = _run_forward_pass(model, emb_t, labels, weights, criterion, device, batch_size)

    return {
        "loss": loss,
        **classification_metrics(labels.numpy(), preds, num_classes, class_names),
        # Raw per-cell outputs (same order as `emb_t`), for callers that
        # want to save individual predictions rather than just aggregate metrics
        # (see classifier_training/experiment.py's save_test_predictions).
        "predictions": preds,
        "probabilities": probs,
    }


def _as_float_tensor(x: np.ndarray) -> torch.Tensor:
    """float32 CPU tensor sharing x's memory when x already is contiguous float32 --
    the embeddings are the job's largest allocation, so they are not copied."""
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


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

    train_t = _as_float_tensor(train_emb)
    test_t = _as_float_tensor(test_emb)
    input_dim = train_t.shape[1]
    batch_size = config.batch_size

    model = MLP(input_dim, config.hidden_dim, config.num_classes, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(reduction="none")

    oversample = _is_oversampling(config.loss_weights)

    def _weights_for(labels: torch.Tensor, wsi_origins: torch.Tensor) -> torch.Tensor:
        return _loss_weights(config, labels, wsi_origins)

    sample_weights = _weights_for(train_labels, train_wsi_origins)
    test_weights = _weights_for(test_labels, test_wsi_origins)

    # Oversampling draws each epoch's rows with replacement, proportionally to these
    # weights (in float64, as WeightedRandomSampler stored them); otherwise a permutation.
    sampling_weights = (
        _compute_sample_weights(
            train_labels, train_wsi_origins, config.num_classes, config.loss_weights
        ).double()
        if oversample else None
    )

    if has_val:
        val_t = _as_float_tensor(val_emb)
        val_weights = _weights_for(val_labels, val_wsi_origins)
        curve_set = (val_t, val_labels, val_weights)
        if val_score_fn is None:
            val_labels_np = val_labels.numpy()
            val_score_fn = lambda preds: classification_metrics(  # noqa: E731
                val_labels_np, preds, config.num_classes, config.class_names
            )["macro_f1"]
    else:
        curve_set = (test_t, test_labels, test_weights)

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
        order = _epoch_order(len(train_labels), sampling_weights)
        # Summed on the device in float64 -- the same additions as the former per-step
        # `epoch_loss += loss.item()`, without a GPU sync on every step.
        epoch_loss = torch.zeros((), dtype=torch.float64, device=device)
        n_batches = 0
        for _, (emb_c, label_c, w_c) in _device_chunks(
            (train_t, train_labels, sample_weights), device, batch_size * _CHUNK_BATCHES, order
        ):
            for i in range(0, len(label_c), batch_size):
                logits = model(emb_c[i:i + batch_size])
                loss = (criterion(logits, label_c[i:i + batch_size]) * w_c[i:i + batch_size]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach()
                n_batches += 1
        train_loss_curve.append(epoch_loss.item() / n_batches)

        model.eval()
        with torch.no_grad():
            epoch_curve_loss, epoch_preds, _ = _run_forward_pass(
                model, *curve_set, criterion, device, batch_size, return_probs=False
            )
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

    def _final(emb_t: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> dict:
        return _evaluate(
            model, emb_t, labels, weights, criterion, device, batch_size,
            config.num_classes, config.class_names,
        )

    if has_val:
        val_metrics = _final(val_t, val_labels, val_weights)
        test_metrics = _final(test_t, test_labels, test_weights)
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

    test_metrics = _final(test_t, test_labels, test_weights)
    return {
        **test_metrics,
        "train_loss_curve": train_loss_curve,
        "test_loss_curve": curve_loss_curve,
        **model_outputs,
    }


# ── joint training: every linear trial of several splits at once ─────────────────
#
# A cross-cohort job spends most of its time reading embeddings and gathering batches,
# not in the tiny linear model. train_linear_jointly shares both: the job holds each cell's
# embedding once for all splits, and all trials of a split that see the same batches
# (a stream, see _Stream) are trained from the same gathered rows, every split's streams
# stepping in lockstep as one batched matmul on the device.

# Rows gathered per chunk by train_linear_jointly, over all its streams together (~400 MB
# at 1536-d float32): one pinned buffer and one host-to-device copy per chunk, built on a
# worker thread while the device works on the previous chunk.
_JOINT_CHUNK_ROWS = 65536


def supports_joint_training(config: TrainConfig) -> bool:
    """Trials train_linear_jointly can train: linear (no hidden layer, hence no dropout),
    without PCA (whose fit would give each split its own input space); z-score is applied on
    the device to the shared batches."""
    return config.hidden_dim == 0 and config.pca_components is None and config.normalization in ("none", "z_score")


@dataclass
class FoldData:
    """One split for train_linear_jointly. Its train/val/test sets are rows of the shared
    embeddings (embeddings[train_rows] is what train_and_evaluate would get as
    train_embeddings, and so on), with their labels and WSI origins; `configs` are its
    trials and val_score_fn the val score they are selected on (see train_and_evaluate)."""
    train_rows: torch.Tensor     # (N_train,) long
    train_labels: torch.Tensor
    train_wsi_origins: torch.Tensor
    val_rows: torch.Tensor
    val_labels: torch.Tensor
    val_wsi_origins: torch.Tensor
    test_rows: torch.Tensor
    test_labels: torch.Tensor
    test_wsi_origins: torch.Tensor
    configs: list[TrainConfig]
    val_score_fn: Callable[[np.ndarray], float]


@dataclass
class _Stream:
    """Trials of one split that see the same batches. train_and_evaluate seeds torch's RNG,
    initialises the model and only then draws each epoch's batch order (plus one draw per val
    pass), so its trials with the same seed and sampling -- a shuffle, or oversampling with the
    same weights -- draw identical orders (z-score draws nothing and leaves the model's shape
    alone). `generator` carries that RNG stream on for them; each batch is gathered once."""
    fold: int
    generator: torch.Generator
    sampling_weights: Optional[torch.Tensor]
    n_batches: int


@dataclass
class _Unit:
    """Trials of one stream with the same input transform: a block of the stacked parameters,
    fed the stream's batches either raw or z-scored with the split's train statistics."""
    stream: int
    trials: list[int]                     # indices into the fold's configs
    z_stats: Optional[tuple[np.ndarray, np.ndarray]]   # (mean, std) of _z_score_stats; None = raw
    loss_weights: torch.Tensor            # (len(trials), N_train) per-row CE weights


@dataclass
class _MemberState:
    """Training record of one trial in train_linear_jointly (train_and_evaluate's locals)."""
    train_loss_curve: list[float]
    val_loss_curve: list[float]
    val_score_curve: list[float]
    best_score: float = -np.inf
    best_epoch: int = 0
    best_params: Optional[torch.Tensor] = None
    stopped: bool = False


def _prefetched(make_chunk: Callable, specs) -> Iterator:
    """Yield make_chunk(spec) for each spec, the next one being built on a worker thread while
    the caller uses the current one (torch's CPU gathers release the GIL)."""
    specs = list(specs)
    if not specs:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(make_chunk, specs[0])
        for nxt in specs[1:] + [None]:
            chunk = future.result()
            if nxt is not None:
                future = pool.submit(make_chunk, nxt)
            yield chunk


def _pinned(t: torch.Tensor, cuda: bool) -> torch.Tensor:
    """t in pinned memory (on CUDA), so its copy to the device doesn't block the host."""
    if not cuda:
        return t
    buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    buf.copy_(t)
    return buf


def _masked_adamw_(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    mask: torch.Tensor,
    lr: torch.Tensor,
    step: torch.Tensor,
    weight_decay: torch.Tensor,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> None:
    """torch.optim.AdamW's update (betas and eps at its defaults, as train_and_evaluate uses
    it) of parameters stacked over slots: p, grad and the moments are (S, T, ...), mask / lr /
    weight_decay / step (S, T), step counting each slot's updates including this one. Slots
    outside mask are left untouched."""
    shape = mask.shape + (1,) * (p.dim() - mask.dim())
    beta1, beta2 = betas
    step = step.clamp(min=1)  # slots that haven't stepped yet are masked out anyway
    # Per-slot factors in float64, rounded once to float32 -- as AdamW's Python-float scalars are.
    decay = (1 - lr * weight_decay).float().view(shape)
    step_size = (lr / (1 - beta1 ** step)).float().view(shape)
    bias_correction2_sqrt = (1 - beta2 ** step).sqrt().float().view(shape)
    new_avg = exp_avg.lerp(grad, 1 - beta1)
    new_avg_sq = exp_avg_sq.mul(beta2).addcmul_(grad, grad, value=1 - beta2)
    denom = new_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps)
    new_p = p.mul(decay).sub_(new_avg.div(denom).mul_(step_size))
    m = mask.view(shape)
    p.copy_(torch.where(m, new_p, p))
    exp_avg.copy_(torch.where(m, new_avg, exp_avg))
    exp_avg_sq.copy_(torch.where(m, new_avg_sq, exp_avg_sq))


def _joint_eval(
    embeddings: torch.Tensor,
    rows: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    params: torch.Tensor,
    device: str,
    want_probs: tuple[int, ...] = (),
    z: Optional[tuple[list[int], torch.Tensor, torch.Tensor]] = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """One in-order pass of M linear models -- params (M, C, D+1), bias last -- over
    embeddings[rows]: each model's weighted mean CE loss (weights (M, N)), its predictions
    (M, N), and the softmax probabilities (N, C) of the models listed in want_probs.
    z = (models, mean, std): those models see the rows z-scored, (x - mean) / std on the
    device in the embeddings' dtype -- the arithmetic of Preprocessor.transform."""
    cuda = torch.device(device).type == "cuda"
    n_models, n_classes, d1 = params.shape
    dim = d1 - 1
    z_models = list(z[0]) if z is not None else []
    raw_models = [m for m in range(n_models) if m not in set(z_models)]
    groups = []   # (model indices, their weight (M_g*C, D) and bias, z stats or None)
    for models, stats in ((raw_models, None), (z_models, z[1:] if z is not None else None)):
        if models:
            p = params[torch.tensor(models, device=params.device)]
            groups.append((
                torch.tensor(models, device=device),
                p[..., :dim].reshape(len(models) * n_classes, dim),
                p[..., dim].reshape(len(models) * n_classes),
                stats,
            ))
    n = len(rows)
    total = torch.zeros(n_models, dtype=torch.float64, device=device)
    preds = torch.empty((n_models, n), dtype=torch.long, device=device)
    probs = {m: torch.empty((n, n_classes), device=device) for m in want_probs}

    def make_chunk(lo: int):
        hi = min(lo + _JOINT_CHUNK_ROWS, n)
        x = torch.empty((hi - lo, dim), dtype=embeddings.dtype, pin_memory=cuda)
        torch.index_select(embeddings, 0, rows[lo:hi], out=x)
        return lo, x, _pinned(labels[lo:hi], cuda), _pinned(weights[:, lo:hi].T.contiguous(), cuda)

    with torch.no_grad():
        for lo, x, y, w in _prefetched(make_chunk, range(0, n, _JOINT_CHUNK_ROWS)):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            logits = torch.empty((len(x), n_models, n_classes), device=device)
            for models, weight, bias, stats in groups:
                xg = x if stats is None else (x - stats[0]) / stats[1]
                logits[:, models] = torch.addmm(bias, xg.float(), weight.T).view(len(x), len(models), n_classes)
            nll = -torch.log_softmax(logits, -1).gather(-1, y[:, None, None].expand(-1, n_models, 1)).squeeze(-1)
            total += (nll * w).double().sum(0)
            hi = lo + len(x)
            preds[:, lo:hi] = logits.argmax(-1).T
            for m, out in probs.items():
                out[lo:hi] = torch.softmax(logits[:, m], -1)
    return (total / n).cpu().numpy(), preds.cpu().numpy(), {m: t.cpu().numpy() for m, t in probs.items()}


def train_linear_jointly(
    embeddings: torch.Tensor,
    folds: list[FoldData],
    device: str,
    return_model: bool = False,
) -> tuple[list[list[dict]], int]:
    """Train every trial of every fold, all at once, as train_and_evaluate would one by one.

    embeddings: (N, D) CPU tensor holding the cells of every fold once (float32, or float16
    converted on the device). All trials must be linear without PCA (supports_joint_training)
    and share batch_size and num_epochs.

    Each trial gets the initial weights, batch order, early stopping and selected epoch of
    its own train_and_evaluate call: trials of a fold drawing the same batches are grouped in
    a _Stream whose generator continues their RNG stream, and the streams of all folds step
    together, one round = one batch of each. A stream's trials are split by input transform
    into _Units (raw, or z-scored on the device with the fold's train statistics -- the same
    numbers and arithmetic as train_and_evaluate's); the units' models run as one batched
    matmul with AdamW's update applied per trial (_masked_adamw_). Only the float rounding of
    the batched matmuls and sums differs, so metrics agree closely rather than bit for bit.

    Returns (results, n_cell_epochs): results[f][i] is train_and_evaluate's dict (with val)
    for folds[f].configs[i]. To bound memory, per-cell outputs (val_/test_predictions and
    _probabilities) are kept only for each fold's trial with the highest val_selection_score
    (the first on ties -- the one experiment.py selects) and are None for the others.
    n_cell_epochs counts the train rows (per stream) and val rows (per fold) processed over all
    epochs, the unit of slurm.time_estimate.joint_seconds_per_cell_epoch.
    """
    configs = [c for fold in folds for c in fold.configs]
    unsupported = [c for c in configs if not supports_joint_training(c)]
    if unsupported:
        raise ValueError(f"train_linear_jointly needs linear trials without PCA, got {unsupported[0]}")
    batch_size, num_epochs = configs[0].batch_size, configs[0].num_epochs
    if any(c.batch_size != batch_size or c.num_epochs != num_epochs for c in configs):
        raise ValueError("train_linear_jointly needs every trial to share batch_size and num_epochs")
    cuda = torch.device(device).type == "cuda"
    dim = embeddings.shape[1]

    # ── streams (per fold, its trials grouped by seed and sampling) and their units ──
    streams: list[_Stream] = []
    units: list[_Unit] = []
    inits: list[torch.Tensor] = []
    z_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}    # per fold with z-score trials
    for f, fold in enumerate(folds):
        num_classes = fold.configs[0].num_classes
        if any(c.num_classes != num_classes for c in fold.configs):
            raise ValueError("train_linear_jointly needs the trials of a fold to share num_classes")
        if any(c.normalization == "z_score" for c in fold.configs):
            # train_and_evaluate's fit, on the same rows in the same order (embeddings[train_rows]
            # is its train_embeddings); one fit serves every z-score trial of the fold.
            z_stats[f] = _z_score_stats(embeddings[fold.train_rows].numpy())
        groups: dict[tuple[int, Optional[str]], list[int]] = {}
        for i, c in enumerate(fold.configs):
            sampling = c.loss_weights if _is_oversampling(c.loss_weights) else None
            groups.setdefault((c.seed, sampling), []).append(i)
        for (seed, sampling), trials in groups.items():
            first = fold.configs[trials[0]]
            # The draws train_and_evaluate makes before its first epoch: seeding, then the init.
            _seed_everything(seed)
            linear = MLP(dim, 0, num_classes, first.dropout).net
            generator = torch.Generator()
            generator.set_state(torch.get_rng_state())
            init = torch.cat([linear.weight.detach(), linear.bias.detach()[:, None]], dim=1)
            sampling_weights = (
                _compute_sample_weights(fold.train_labels, fold.train_wsi_origins, num_classes, sampling).double()
                if sampling else None
            )
            streams.append(_Stream(f, generator, sampling_weights, math.ceil(len(fold.train_labels) / batch_size)))
            by_transform: dict[str, list[int]] = {}
            for i in trials:
                by_transform.setdefault(fold.configs[i].normalization, []).append(i)
            for normalization, unit_trials in by_transform.items():
                units.append(_Unit(
                    len(streams) - 1, unit_trials, z_stats[f] if normalization == "z_score" else None,
                    torch.stack([
                        _loss_weights(fold.configs[i], fold.train_labels, fold.train_wsi_origins) for i in unit_trials
                    ]),
                ))
                inits.append(init)

    # ── parameters stacked as (unit, slot, class, D + bias), padded to the largest ──
    n_units = len(units)
    n_slots = max(len(un.trials) for un in units)
    n_classes = max(init.shape[0] for init in inits)
    params = torch.zeros((n_units, n_slots, n_classes, dim + 1))
    class_pad = torch.ones((n_units, n_classes), dtype=torch.bool)
    lr = torch.zeros((n_units, n_slots), dtype=torch.float64)
    weight_decay = torch.zeros((n_units, n_slots), dtype=torch.float64)
    # Per unit, the z-score applied to its rows on the device (0 / 1 for raw units: exact no-ops).
    z_mean = torch.zeros((n_units, dim), dtype=embeddings.dtype)
    z_std = torch.ones((n_units, dim), dtype=embeddings.dtype)
    for u, un in enumerate(units):
        k, c = len(un.trials), inits[u].shape[0]
        params[u, :k, :c] = inits[u]
        class_pad[u, :c] = False
        unit_configs = [folds[streams[un.stream].fold].configs[i] for i in un.trials]
        lr[u, :k] = torch.tensor([c.learning_rate for c in unit_configs], dtype=torch.float64)
        weight_decay[u, :k] = torch.tensor([c.weight_decay for c in unit_configs], dtype=torch.float64)
        if un.z_stats is not None:
            z_mean[u], z_std[u] = (torch.from_numpy(np.asarray(a)) for a in un.z_stats)
    any_z = any(un.z_stats is not None for un in units)
    params = params.to(device).requires_grad_()
    exp_avg, exp_avg_sq = torch.zeros_like(params), torch.zeros_like(params)
    lr, weight_decay = lr.to(device), weight_decay.to(device)
    class_pad = class_pad.to(device)
    z_mean, z_std = z_mean.to(device), z_std.to(device)
    steps = torch.zeros((n_units, n_slots), dtype=torch.float64, device=device)

    states = {(u, t): _MemberState([], [], []) for u, un in enumerate(units) for t in range(len(un.trials))}
    # Per fold, its members (unit, slot) in trial order, their val CE weights, and which of
    # them (positions in that order) are z-score trials.
    fold_members: list[list[tuple[int, int]]] = []
    val_weights: list[torch.Tensor] = []
    fold_z: list[Optional[tuple[list[int], torch.Tensor, torch.Tensor]]] = []
    for f, fold in enumerate(folds):
        members = sorted(
            (un.trials[t], u, t) for u, un in enumerate(units) if streams[un.stream].fold == f
            for t in range(len(un.trials))
        )
        fold_members.append([(u, t) for _, u, t in members])
        val_weights.append(torch.stack([
            _loss_weights(fold.configs[i], fold.val_labels, fold.val_wsi_origins) for i, _, _ in members
        ]))
        z_positions = [m for m, (_, u, _) in enumerate(members) if units[u].z_stats is not None]
        fold_z.append(
            (z_positions, *(torch.from_numpy(np.asarray(a)).to(device) for a in z_stats[f])) if z_positions else None
        )

    def config_of(u: int, t: int) -> TrainConfig:
        return folds[streams[units[u].stream].fold].configs[units[u].trials[t]]

    def fold_params(f: int, source: Callable[[int, int], torch.Tensor]) -> torch.Tensor:
        c = folds[f].configs[0].num_classes
        return torch.stack([source(u, t)[:c] for u, t in fold_members[f]])

    n_cell_epochs = 0
    for epoch in range(1, num_epochs + 1):
        live = [u for u, un in enumerate(units) if any(not states[(u, t)].stopped for t in range(len(un.trials)))]
        if not live:
            break
        # Streams still feeding a live unit: their order is drawn and their rows gathered.
        live_streams = sorted({units[u].stream for u in live})
        stream_pos = {s: j for j, s in enumerate(live_streams)}
        active = torch.zeros((n_units, n_slots), dtype=torch.bool)
        for (u, t), state in states.items():
            active[u, t] = not state.stopped
        active = active.to(device)
        orders = {
            s: _epoch_order(len(folds[streams[s].fold].train_labels), streams[s].sampling_weights, streams[s].generator)
            for s in live_streams
        }
        n_live, n_live_streams = len(live), len(live_streams)
        live_idx = torch.tensor(live, device=device)
        unit_src = torch.tensor([stream_pos[units[u].stream] for u in live], device=device)
        pad_live = class_pad[live_idx].view(n_live, 1, 1, n_classes)
        z_live = (z_mean[live_idx].view(n_live, 1, dim), z_std[live_idx].view(n_live, 1, dim)) if any_z else None
        n_rounds = max(streams[s].n_batches for s in live_streams)
        rounds_per_chunk = max(1, _JOINT_CHUNK_ROWS // (n_live_streams * batch_size))

        def make_chunk(r0: int):
            r1 = min(r0 + rounds_per_chunk, n_rounds)
            n_rows = (r1 - r0) * batch_size
            # rows and labels once per stream; loss weights per unit (its own trials)
            x = torch.empty((n_live_streams, n_rows, dim), dtype=embeddings.dtype, pin_memory=cuda)
            y = torch.zeros((n_live_streams, n_rows), dtype=torch.long, pin_memory=cuda)
            w = torch.zeros((n_live, n_slots, n_rows), pin_memory=cuda)
            counts = torch.zeros((n_live_streams, r1 - r0), pin_memory=cuda)   # real rows per round
            has_rows = torch.zeros((r1 - r0, n_units), dtype=torch.bool, pin_memory=cuda)
            round_starts = torch.arange(r1 - r0) * batch_size
            for j, s in enumerate(live_streams):
                fold = folds[streams[s].fold]
                order = orders[s][r0 * batch_size:r1 * batch_size]
                n = len(order)
                if n:
                    torch.index_select(embeddings, 0, fold.train_rows[order], out=x[j, :n])
                    torch.index_select(fold.train_labels, 0, order, out=y[j, :n])
                if n < n_rows:
                    x[j, n:].zero_()  # padding: zero rows with zero weight
                counts[j] = (n - round_starts).clamp(0, batch_size)
            for k, u in enumerate(live):
                un = units[u]
                order = orders[un.stream][r0 * batch_size:r1 * batch_size]
                if len(order):
                    w[k, :len(un.trials), :len(order)] = un.loss_weights[:, order]
                has_rows[:, u] = counts[stream_pos[un.stream]] > 0
            return x, y, w, counts, has_rows

        epoch_loss = torch.zeros((n_units, n_slots), dtype=torch.float64, device=device)
        for chunk in _prefetched(make_chunk, range(0, n_rounds, rounds_per_chunk)):
            x, y, w, counts, has_rows = (t.to(device, non_blocking=True) for t in chunk)
            # Each live unit's rows: its stream's batch, z-scored for z-score units (the
            # arithmetic of train_and_evaluate's (x - mean) / std, in the embeddings' dtype).
            x = x.index_select(0, unit_src)
            if z_live is not None:
                x = (x - z_live[0]) / z_live[1]
            x = x.float()
            y, counts = y.index_select(0, unit_src), counts.index_select(0, unit_src)
            for i in range(counts.shape[1]):
                rows = slice(i * batch_size, (i + 1) * batch_size)
                p = params[live_idx]                                         # (L, T, C, D+1)
                logits = torch.baddbmm(
                    p[..., dim].reshape(n_live, 1, n_slots * n_classes),
                    x[:, rows],
                    p[..., :dim].reshape(n_live, n_slots * n_classes, dim).transpose(1, 2),
                ).view(n_live, -1, n_slots, n_classes).masked_fill(pad_live, float("-inf"))
                target = y[:, rows, None, None].expand(-1, -1, n_slots, 1)
                nll = -torch.log_softmax(logits, -1).gather(-1, target).squeeze(-1)       # (L, B, T)
                # Mean over the batch's real rows, as train_and_evaluate's (ce * w).mean().
                loss = (nll * w[:, :, rows].transpose(1, 2)).sum(1) / counts[:, i:i + 1].clamp(min=1)
                grad, = torch.autograd.grad(loss.sum(), params)
                with torch.no_grad():
                    step_mask = active & has_rows[i].view(n_units, 1)
                    steps += step_mask
                    _masked_adamw_(params, grad, exp_avg, exp_avg_sq, step_mask, lr, steps, weight_decay)
                    epoch_loss.index_add_(0, live_idx, torch.where(step_mask[live_idx], loss.detach().double(), 0.0))
        n_cell_epochs += sum(len(folds[streams[s].fold].train_labels) for s in live_streams)

        # ── per-epoch val pass and early stopping, per fold ──
        for s in live_streams:
            _draw_loader_seed(streams[s].generator)   # the draw of train_and_evaluate's val pass
        epoch_loss = epoch_loss.cpu()
        for f, fold in enumerate(folds):
            members = fold_members[f]
            if all(states[m].stopped for m in members):
                continue
            val_loss, val_preds, _ = _joint_eval(
                embeddings, fold.val_rows, fold.val_labels, val_weights[f],
                fold_params(f, lambda u, t: params[u, t].detach()), device, z=fold_z[f],
            )
            n_cell_epochs += len(fold.val_labels)
            for m, (u, t) in enumerate(members):
                state = states[(u, t)]
                if state.stopped:
                    continue
                config = config_of(u, t)
                state.train_loss_curve.append(epoch_loss[u, t].item() / streams[units[u].stream].n_batches)
                state.val_loss_curve.append(float(val_loss[m]))
                score = float(fold.val_score_fn(val_preds[m]))
                state.val_score_curve.append(score)
                early_stopping = config.early_stopping_patience is not None
                if score > state.best_score + config.early_stopping_min_delta:
                    state.best_score, state.best_epoch = score, epoch
                    if early_stopping:
                        state.best_params = params[u, t].detach().clone()
                elif early_stopping and epoch - state.best_epoch >= config.early_stopping_patience:
                    state.stopped = True

    # ── final evaluation of each trial's selected weights ──
    def final_params(u: int, t: int) -> torch.Tensor:
        state = states[(u, t)]
        return state.best_params if config_of(u, t).early_stopping_patience is not None else params[u, t].detach()

    def subset_z(z, models: list[int]):
        """fold_z restricted to `models` (positions in the fold's order), renumbered."""
        if z is None:
            return None
        picked = [j for j, m in enumerate(models) if m in set(z[0])]
        return (picked, z[1], z[2]) if picked else None

    results: list[list[dict]] = []
    for f, fold in enumerate(folds):
        members = fold_members[f]
        num_classes, class_names = fold.configs[0].num_classes, fold.configs[0].class_names
        final = fold_params(f, final_params)
        val_loss, val_preds, _ = _joint_eval(
            embeddings, fold.val_rows, fold.val_labels, val_weights[f], final, device, z=fold_z[f],
        )
        scores = [float(fold.val_score_fn(val_preds[m])) for m in range(len(members))]
        best = 0
        for m in range(1, len(members)):
            if scores[m] > scores[best]:
                best = m
        _, _, best_val_probs = _joint_eval(
            embeddings, fold.val_rows, fold.val_labels, val_weights[f][best:best + 1],
            final[best:best + 1], device, want_probs=(0,), z=subset_z(fold_z[f], [best]),
        )
        test_weights = torch.stack([
            _loss_weights(config_of(u, t), fold.test_labels, fold.test_wsi_origins) for u, t in members
        ])
        test_loss, test_preds, test_probs = _joint_eval(
            embeddings, fold.test_rows, fold.test_labels, test_weights, final, device,
            want_probs=(best,), z=fold_z[f],
        )
        val_labels_np, test_labels_np = fold.val_labels.numpy(), fold.test_labels.numpy()
        fold_results = []
        for m, (u, t) in enumerate(members):
            state, config = states[(u, t)], config_of(u, t)
            is_best = m == best
            val_metrics = {
                "loss": float(val_loss[m]),
                **classification_metrics(val_labels_np, val_preds[m], num_classes, class_names),
                "predictions": val_preds[m] if is_best else None,
                "probabilities": best_val_probs[0] if is_best else None,
            }
            test_metrics = {
                "loss": float(test_loss[m]),
                **classification_metrics(test_labels_np, test_preds[m], num_classes, class_names),
                "predictions": test_preds[m] if is_best else None,
                "probabilities": test_probs[best] if is_best else None,
            }
            result = {
                **_prefix_metrics(val_metrics, "val"),
                **_prefix_metrics(test_metrics, "test"),
                "val_selection_score": scores[m],
                "selected_epoch": (
                    state.best_epoch if config.early_stopping_patience is not None else len(state.train_loss_curve)
                ),
                "epochs_trained": len(state.train_loss_curve),
                "train_loss_curve": state.train_loss_curve,
                "val_loss_curve": state.val_loss_curve,
                "val_score_curve": state.val_score_curve,
            }
            if return_model:
                result["model_state_dict"] = {
                    "net.weight": final[m, :, :dim].cpu().clone(),
                    "net.bias": final[m, :, dim].cpu().clone(),
                }
                z_fit = units[u].z_stats
                result["preprocessor"] = (
                    Preprocessor(z_mean=z_fit[0], z_std=z_fit[1]) if z_fit is not None else Preprocessor()
                )
            fold_results.append(result)
        results.append(fold_results)
    return results, n_cell_epochs
