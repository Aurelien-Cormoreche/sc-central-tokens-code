"""Turn registry folds into trainer inputs, and draw Stage-3 training subsets.

Label space of a fold: its `label_space` (cross-cohort / single-cohort: classes present in both
train and test cohorts), or the classes present in train (within), restricted to classes with
train cells. Train/val cells outside it are dropped; test cells outside it are dropped and
counted per slide (they could never be predicted correctly).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from bench.config import BenchConfig, Grid
from bench.probe.data import Loaded
from bench.probe.metrics import make_scorer
from bench.registry import Fold

from src.python.classifier_training.trainer import FoldData, TrainConfig


@dataclass
class Prepared:
    fold: Fold
    classes: List[str]
    train: np.ndarray            # rows of Loaded
    val: np.ndarray
    test: np.ndarray
    local: np.ndarray            # (N,) local class index per Loaded row (-1 outside the label space)
    dropped_val: int = 0
    dropped_test: Dict[int, int] = field(default_factory=dict)


def prepare(fold: Fold, data: Loaded) -> Prepared:
    names = data.slide_names
    pos = {s: i for i, s in enumerate(names)}
    sl = data.slide_idx

    def rows_of(side: List[str]) -> np.ndarray:
        idx = [pos[s] for s in side if s in pos]
        return np.flatnonzero(np.isin(sl, idx))

    tr, va, te = rows_of(fold.train), rows_of(fold.val), rows_of(fold.test)
    present = set(np.unique(data.label[tr]).tolist())
    space = [c for c in (fold.label_space if fold.label_space is not None else data.classes)
             if c in data.classes and data.classes.index(c) in present]
    lut = np.full(len(data.classes), -1, dtype=np.int64)
    for i, c in enumerate(space):
        lut[data.classes.index(c)] = i
    local = lut[data.label]
    keep_tr = tr[local[tr] >= 0]
    keep_va = va[local[va] >= 0]
    keep_te = te[local[te] >= 0]
    drop_te = te[local[te] < 0]
    dropped = {int(s): int(n) for s, n in zip(*np.unique(sl[drop_te], return_counts=True))}
    return Prepared(fold, space, keep_tr, keep_va, keep_te, local, int(len(va) - len(keep_va)), dropped)


def trial_configs(cfg: BenchConfig, dim: int, classes: List[str], grid: Grid, seed: int, hidden: int = 0,
                  dropout: float = 0.0) -> List[TrainConfig]:
    c = cfg.classify
    out = []
    for lw in grid.loss_weights:
        for wd in grid.weight_decay:
            for lr in grid.lr:
                out.append(TrainConfig(
                    input_dim=dim, num_classes=len(classes), class_names=list(classes), learning_rate=lr,
                    dropout=dropout, hidden_dim=hidden, pca_components=None,
                    normalization="z_score" if c.zscore else "none", loss_weights=lw, num_epochs=c.epochs,
                    batch_size=c.batch_size, seed=seed, weight_decay=wd, early_stopping_patience=c.patience,
                    early_stopping_min_delta=c.min_delta))
    return out


def fold_data(cfg: BenchConfig, prep: Prepared, data: Loaded, configs: List[TrainConfig],
              selection_metric: str, train_rows: Optional[np.ndarray] = None,
              z: Optional[tuple] = None) -> FoldData:
    tr = prep.train if train_rows is None else train_rows
    cohorts = [s.cohort for s in data.slides]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64))  # noqa: E731
    scorer = make_scorer(selection_metric, prep.local[prep.val], data.slide_idx[prep.val], cohorts, len(prep.classes))
    return FoldData(
        train_rows=t(tr), train_labels=t(prep.local[tr]), train_wsi_origins=t(data.slide_idx[tr]),
        val_rows=t(prep.val), val_labels=t(prep.local[prep.val]), val_wsi_origins=t(data.slide_idx[prep.val]),
        test_rows=t(prep.test), test_labels=t(prep.local[prep.test]), test_wsi_origins=t(data.slide_idx[prep.test]),
        configs=configs, val_score_fn=scorer, z_stats=z)


# ── Stage 3 subsets ──

def subset_path(cfg: BenchConfig, fold: Fold, frac: float, seed: int) -> Path:
    return cfg.root / "results" / "3" / "subsets" / fold.scheme / fold.key / f"frac_{frac:g}_seed_{seed}.npz"


def stratified_subset(slides: np.ndarray, mrows: np.ndarray, labels: np.ndarray, frac: float, seed: int,
                      min_per_class: int, key: str) -> np.ndarray:
    """Indices into the given (slide, mrow, label) pool: per class n_c = max(min_per_class,
    round(frac * N_c)) (capped at N_c), drawn without replacement. The pool is put in canonical
    (slide, mrow) order first, so the draw depends only on the manifest, never on a model."""
    order = np.lexsort((mrows, slides))
    rng = np.random.default_rng([seed, zlib.crc32(key.encode()), int(round(frac * 1e6))])
    picked = []
    for c in np.unique(labels):
        members = order[labels[order] == c]
        n_c = min(len(members), max(min_per_class, int(round(frac * len(members)))))
        picked.append(rng.choice(members, size=n_c, replace=False))
    return np.sort(np.concatenate(picked)) if picked else np.zeros(0, dtype=np.int64)


def stage3_train_rows(cfg: BenchConfig, prep: Prepared, data: Loaded, frac: float, seed: int) -> np.ndarray:
    """Rows of `data` in the (frac, seed) subset of prep's train pool; the subset is saved once as
    (slide name, manifest row) pairs and reused by every model, readout and probe."""
    p = subset_path(cfg, prep.fold, frac, seed)
    names = np.array(data.slide_names)
    pool_slide = names[data.slide_idx[prep.train]]
    pool_mrow = data.mrow[prep.train]
    if frac >= 1.0:
        return prep.train
    if p.exists():
        z = np.load(p, allow_pickle=False)
        want = set(zip(z["slide"].tolist(), z["mrow"].tolist()))
    else:
        idx = stratified_subset(pool_slide, pool_mrow, prep.local[prep.train], frac, seed,
                                cfg.stage3.min_per_class, prep.fold.key)
        want = set(zip(pool_slide[idx].tolist(), pool_mrow[idx].tolist()))
        p.parent.mkdir(parents=True, exist_ok=True)
        pairs = sorted(want)
        from bench.utils import atomic_path
        with atomic_path(p) as tmp:
            with open(tmp, "wb") as fh:
                np.savez(fh, slide=np.array([a for a, _ in pairs]), mrow=np.array([b for _, b in pairs], dtype=np.int64))
    hit = np.array([(a, b) in want for a, b in zip(pool_slide.tolist(), pool_mrow.tolist())], dtype=bool)
    if hit.sum() != len(want):
        raise RuntimeError(f"{p}: {len(want) - int(hit.sum())} subset cells are not in this model's train pool "
                           f"(manifest changed?) -- delete the subsets directory and rerun")
    return prep.train[hit]
