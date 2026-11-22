"""Classification metrics and val selection scorers (same definitions as
src/python/classifier_training/experiment.py; metric set from trainer.classification_metrics)."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from src.python.classifier_training.trainer import classification_metrics

SCALARS = ("macro_f1_present", "macro_f1", "balanced_accuracy", "accuracy", "weighted_f1")


def slide_macro_f1(labels: np.ndarray, preds: np.ndarray, slides: np.ndarray, n_slides: int, k: int) -> np.ndarray:
    """Per slide: macro F1 over the classes present in that slide's labels (NaN without cells)."""
    cm = np.bincount((slides * k + labels) * k + preds, minlength=n_slides * k * k).reshape(n_slides, k, k)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    support = cm.sum(axis=2)
    denom = support + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(tp.shape, dtype=float), where=denom > 0)
    present = support > 0
    n_present = present.sum(axis=1)
    return np.where(n_present > 0, (f1 * present).sum(axis=1) / np.maximum(n_present, 1), np.nan)


def make_scorer(metric: str, labels: np.ndarray, slides: np.ndarray, slide_cohorts: List[str],
                k: int) -> Callable[[np.ndarray], float]:
    labels, slides = np.asarray(labels).copy(), np.asarray(slides).copy()
    n = len(slide_cohorts)
    if metric == "pooled_macro_f1":
        names = [str(i) for i in range(k)]
        return lambda p: classification_metrics(labels, p, k, names)["macro_f1"]
    if metric == "slide_macro_f1":
        return lambda p: float(np.nanmean(slide_macro_f1(labels, p, slides, n, k)))
    if metric == "cohort_balanced_slide_macro_f1":
        groups: Dict[str, List[int]] = {}
        for s in np.unique(slides).tolist():
            groups.setdefault(slide_cohorts[s], []).append(s)

        def score(p):
            per = slide_macro_f1(labels, p, slides, n, k)
            return float(np.mean([np.nanmean(per[idx]) for idx in groups.values()]))
        return score
    raise ValueError(f"unknown selection metric {metric!r}")


def summarize(labels: np.ndarray, preds: np.ndarray, class_names: List[str]) -> dict:
    m = classification_metrics(labels, preds, len(class_names), class_names)
    m["n"] = int(len(labels))
    m["support"] = {c: int(v) for c, v in zip(class_names, np.bincount(labels, minlength=len(class_names)))}
    return m


def per_slide(labels: np.ndarray, preds: np.ndarray, slides: np.ndarray, slide_names: List[str],
              slide_cohort: List[str], slide_donor: List[str], class_names: List[str],
              dropped: Optional[Dict[int, int]] = None) -> List[dict]:
    rows = []
    dropped = dropped or {}
    for s in sorted(set(np.unique(slides).tolist()) | set(dropped)):
        mask = slides == s
        row = {"slide": slide_names[s], "cohort": slide_cohort[s], "donor": slide_donor[s],
               "n_cells": int(mask.sum()), "n_dropped_absent_from_train": int(dropped.get(s, 0))}
        if mask.any():
            m = classification_metrics(labels[mask], preds[mask], len(class_names), class_names)
            row.update({k: m[k] for k in SCALARS})
            row["per_class_f1"] = m["per_class_f1"]
            row["confusion_matrix"] = m["confusion_matrix"]
        rows.append(row)
    return rows


def group_mean(rows: List[dict], by: str) -> Dict[str, Optional[float]]:
    """Mean over groups (slides, donors or cohorts) of each scalar, every group weighted equally;
    a donor/cohort value is the mean over its slides."""
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        if "macro_f1_present" in r:
            groups.setdefault(r[by], []).append(r)
    out: Dict[str, Optional[float]] = {}
    for k in SCALARS:
        vals = [np.mean([r[k] for r in rs]) for rs in groups.values()]
        out[k] = float(np.mean(vals)) if vals else None
    out["n_groups"] = len(groups)
    return out
