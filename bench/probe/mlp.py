"""2-layer MLP probe on the shared float16 host tensor.

Same recipe as classifier_training.trainer.train_and_evaluate (AdamW, weighted CE, z-score fit on
train, early stopping on the val selection score with the best epoch restored) but batches are
gathered from the shared embeddings by row index, so a cross-cohort fold never materialises a
float32 copy of its training set (train_and_evaluate would need ~4x the embeddings in RAM).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.python.classifier_training.trainer import MLP, FoldData, TrainConfig, _loss_weights, _seed_everything, \
    classification_metrics

_CHUNK = 65536


def _batches(emb: torch.Tensor, rows: torch.Tensor, device: torch.device, bs: int):
    cuda = device.type == "cuda"
    for a in range(0, len(rows), _CHUNK):
        r = rows[a:a + _CHUNK]
        buf = torch.empty((len(r), emb.shape[1]), dtype=emb.dtype, pin_memory=cuda)
        torch.index_select(emb, 0, r, out=buf)
        x = buf.to(device, non_blocking=True).float()
        for b in range(0, len(r), bs):
            yield a + b, x[b:b + bs]


@torch.no_grad()
def predict(model: nn.Module, emb: torch.Tensor, rows: torch.Tensor, mean, std, device, bs: int) -> np.ndarray:
    model.eval()
    out = torch.empty(len(rows), dtype=torch.long, device=device)
    for lo, x in _batches(emb, rows, device, bs * 8):
        out[lo:lo + len(x)] = model((x - mean) / std).argmax(1)
    return out.cpu().numpy()


def train_mlp(emb: torch.Tensor, fd: FoldData, config: TrainConfig, device: str,
              z: Optional[tuple] = None) -> dict:
    dev = torch.device(device)
    _seed_everything(config.seed)
    k = config.num_classes
    model = MLP(emb.shape[1], config.hidden_dim, k, config.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ce = nn.CrossEntropyLoss(reduction="none")
    if z is not None and config.normalization == "z_score":
        mean = torch.as_tensor(np.asarray(z[0], dtype=np.float32), device=dev)
        std = torch.as_tensor(np.asarray(z[1], dtype=np.float32), device=dev)
    else:
        mean, std = torch.zeros(emb.shape[1], device=dev), torch.ones(emb.shape[1], device=dev)
    w = _loss_weights(config, fd.train_labels, fd.train_wsi_origins).to(dev)
    y = fd.train_labels.to(dev)
    gen = torch.Generator().manual_seed(config.seed)
    best_score, best_epoch, best_state = -np.inf, 0, None
    train_curve, val_curve = [], []
    epoch = 0
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        order = torch.randperm(len(fd.train_rows), generator=gen)
        rows = fd.train_rows[order]
        tot, nb = 0.0, 0
        for lo, x in _batches(emb, rows, dev, config.batch_size):
            idx = order[lo:lo + len(x)].to(dev)
            loss = (ce(model((x - mean) / std), y[idx]) * w[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        train_curve.append(tot / max(nb, 1))
        score = float(fd.val_score_fn(predict(model, emb, fd.val_rows, mean, std, dev, config.batch_size)))
        val_curve.append(score)
        if score > best_score + config.early_stopping_min_delta:
            best_score, best_epoch = score, epoch
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        elif config.early_stopping_patience is not None and epoch - best_epoch >= config.early_stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    names = config.class_names
    vp = predict(model, emb, fd.val_rows, mean, std, dev, config.batch_size)
    tp = predict(model, emb, fd.test_rows, mean, std, dev, config.batch_size)
    vm = classification_metrics(fd.val_labels.numpy(), vp, k, names)
    tm = classification_metrics(fd.test_labels.numpy(), tp, k, names)
    return {**{f"val_{a}": b for a, b in vm.items()}, **{f"test_{a}": b for a, b in tm.items()},
            "val_predictions": vp, "test_predictions": tp, "val_selection_score": float(fd.val_score_fn(vp)),
            "selected_epoch": best_epoch, "epochs_trained": epoch, "train_loss_curve": train_curve,
            "val_loss_curve": [], "val_score_curve": val_curve}
