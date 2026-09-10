"""Closed-form / torch-native solvers for the probing package's linear probes --
deliberately NOT gradient-descent training loops (unlike classifier_training's
MLP, see classifier_training/trainer.py): ridge regression has an exact
closed-form normal-equations solution, and logistic regression is solved to
convergence in a single call via torch's L-BFGS (a proper quasi-Newton batch
solver, not a manual epoch/minibatch SGD loop).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @staticmethod
    def fit(X: torch.Tensor) -> "Standardizer":
        mean = X.mean(dim=0)
        std = X.std(dim=0) + 1e-8
        return Standardizer(mean, std)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std


@dataclass
class RidgeModel:
    weight: torch.Tensor   # (D, K)
    bias: torch.Tensor     # (K,)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.weight + self.bias


def ridge_fit(X: torch.Tensor, y: torch.Tensor, lam: float) -> RidgeModel:
    """Closed-form ridge regression via the centred normal equations:
    w = (Xc^T Xc + lam*I)^-1 Xc^T yc, b = mean(y) - mean(X) @ w. Centring first
    means the bias term is never itself penalized (the standard, correct
    convention -- penalizing it would arbitrarily shrink predictions towards 0
    instead of towards the training mean). `y` may be 2D (multi-output, e.g. the
    composition probe's per-class fractions, or the orientation probe's
    [cos(2*theta), sin(2*theta)] pair) -- every output shares the same lambda and
    feature matrix, solved in one batched linear solve.
    """
    if y.dim() == 1:
        y = y.unsqueeze(1)
    x_mean = X.mean(dim=0)
    y_mean = y.mean(dim=0)
    Xc = X - x_mean
    yc = y - y_mean
    D = Xc.shape[1]
    A = Xc.T @ Xc + lam * torch.eye(D, dtype=X.dtype, device=X.device)
    B = Xc.T @ yc
    W = torch.linalg.solve(A, B)
    b = y_mean - x_mean @ W
    return RidgeModel(W, b)


@dataclass
class LogisticModel:
    weight: torch.Tensor   # (D, C)
    bias: torch.Tensor     # (C,)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        return torch.softmax(X @ self.weight + self.bias, dim=1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(X).argmax(dim=1)


def logistic_fit(
    X: torch.Tensor,
    y: torch.Tensor,
    lam: float,
    num_classes: int,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> LogisticModel:
    """L2-regularized multinomial logistic regression, solved to convergence by
    torch.optim.LBFGS in a single `optimizer.step(closure)` call (LBFGS's closure
    re-evaluates the *full* batch each internal iteration -- this is a batch
    quasi-Newton solve to convergence, not an SGD-style epoch loop). Included for
    framework completeness/extensibility (see probes/base.py's Probe.task_type);
    none of the probes this module ships with (area/eccentricity/orientation/
    composition/count/density) are classification tasks, so this path isn't
    exercised by the default config -- it's here for the next probe someone adds
    that is (e.g. a discrete cell-type-identity probe).
    """
    D = X.shape[1]
    weight = torch.zeros(D, num_classes, dtype=X.dtype, device=X.device, requires_grad=True)
    bias = torch.zeros(num_classes, dtype=X.dtype, device=X.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=max_iter, tolerance_grad=tol, line_search_fn='strong_wolfe'
    )

    def closure():
        optimizer.zero_grad()
        logits = X @ weight + bias
        loss = torch.nn.functional.cross_entropy(logits, y) + lam * (weight ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return LogisticModel(weight.detach(), bias.detach())


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Per-output R^2 (coefficient of determination), shape (K,). A degenerate
    output (every value in y_true identical, e.g. a probe target that happens to
    be constant on this particular split) would otherwise divide by zero -- it's
    defined as 0 when predictions also match exactly, else -inf, so it never
    silently wins a "best lambda" comparison by producing a NaN."""
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(1)
        y_pred = y_pred.unsqueeze(1)
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    y_mean = y_true.mean(dim=0)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=0)
    degenerate = ss_tot < 1e-12
    r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-12)
    r2 = torch.where(
        degenerate,
        torch.where(ss_res < 1e-12, torch.zeros_like(r2), torch.full_like(r2, float('-inf'))),
        r2,
    )
    return r2


def mean_finite(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float('-inf')
