"""Collect every result JSON under {bench_root}/results into tidy records for the site and gates.

Classification: one record per fold (stage, scheme, task, model, readout, variant, probe, frac,
seed), plus `summary` rows per task: the task score is the mean over folds of the fold's test
metric (a within-cohort fold = one held-out donor; a cross-cohort fold = one held-out cohort),
with a bootstrap 95% CI over test donors.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from bench.config import BenchConfig
from bench.probe.metrics import SCALARS

KEYS = ("stage", "scheme", "task", "model", "readout", "variant", "probe", "frac", "seed")


def _load(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def classify_folds(cfg: BenchConfig) -> List[dict]:
    root = cfg.root / "results"
    out = []
    if not root.exists():
        return out
    for done in root.glob("*/classify/**/_DONE.json"):
        d = done.parent
        for p in sorted(d.glob("fold_*.json")):
            r = _load(p)
            if r:
                r["_path"] = str(p.relative_to(cfg.root))
                out.append(r)
    return out


def genexp_folds(cfg: BenchConfig) -> List[dict]:
    root = cfg.root / "results" / "1" / "genexp"
    out = []
    if not root.exists():
        return out
    for p in root.glob("**/result.json"):
        if (p.parent / "_DONE.json").exists():
            r = _load(p)
            if r:
                out.append(r)
    return out


def bootstrap_ci(values: List[float], n_boot: int = 1000, seed: int = 0) -> tuple:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return None, None
    if len(v) == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _donor_values(folds: List[dict], metric: str) -> List[float]:
    """Per test donor: within -> the fold's pooled value (one donor per fold); otherwise the mean
    over that donor's slides of the per-slide value."""
    if folds and folds[0]["scheme"] == "within":
        return [f["test"][metric] for f in folds]
    by: Dict[str, List[float]] = defaultdict(list)
    for f in folds:
        for s in f.get("test_slides", []):
            if s.get(metric) is not None:
                by[s["donor"]].append(s[metric])
    return [float(np.mean(v)) for v in by.values()]


def summarize_classify(folds: List[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for f in folds:
        groups[tuple(f.get(k) for k in KEYS)].append(f)
    rows = []
    for key, fs in groups.items():
        row = dict(zip(KEYS, key))
        row["n_folds"] = len(fs)
        for m in SCALARS:
            vals = [f["test"][m] for f in fs]
            row[m] = float(np.mean(vals))
            row[f"val_{m}"] = float(np.mean([f["val"][m] for f in fs if f.get("val")])) if any(f.get("val") for f in fs) else None
            row[f"slide_mean_{m}"] = float(np.nanmean([f["test_slide_mean"][m] for f in fs if f["test_slide_mean"].get(m) is not None])) \
                if any(f["test_slide_mean"].get(m) is not None for f in fs) else None
        lo, hi = bootstrap_ci(_donor_values(fs, "macro_f1_present"))
        row["ci_lo"], row["ci_hi"] = lo, hi
        row["n_donors"] = len(_donor_values(fs, "macro_f1_present"))
        rows.append(row)
    return rows


def compact_fold(f: dict) -> dict:
    """What the site needs from a fold record (per-slide confusion matrices dropped)."""
    return {
        **{k: f.get(k) for k in KEYS}, "fold_key": f["fold"]["key"], "note": f["fold"].get("note", ""),
        "test_cohorts": f["fold"].get("test_cohorts"), "classes": f["classes"], "n": f["n"], "dropped": f["dropped"],
        "best_trial": f["best_trial"], "selected_epoch": f.get("selected_epoch"),
        "test": {k: f["test"][k] for k in (*SCALARS, "per_class_f1", "confusion_matrix", "support")},
        "val": {k: f["val"][k] for k in SCALARS} if f.get("val") else None,
        "slides": [{k: s.get(k) for k in ("slide", "cohort", "donor", "n_cells", "n_dropped_absent_from_train", *SCALARS)}
                   for s in f.get("test_slides", [])],
        "trials": f.get("trials", []),
    }


def summarize_genexp(recs: List[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in recs:
        groups[(r["set"], r["model"], r["readout"])].append(r)
    rows = []
    for (s, m, rd), rs in groups.items():
        row = {"set": s, "model": m, "readout": rd, "n_folds": len(rs)}
        for k in sorted({k for r in rs for k in r["test"]}):
            vals = [r["test"][k] for r in rs if r["test"].get(k) is not None]
            row[k] = float(np.mean(vals)) if vals else None
        lo, hi = bootstrap_ci([r["test"].get("mean_pearson_all") for r in rs])
        row["ci_lo"], row["ci_hi"] = lo, hi
        # per-gene mean over folds (genes differ per fold: grouped by name)
        by_gene: Dict[str, List[float]] = defaultdict(list)
        for r in rs:
            for g in r["per_gene"]:
                if g.get("pearson") is not None and np.isfinite(g["pearson"]):
                    by_gene[g["gene_name"]].append(g["pearson"])
        q = np.array([np.mean(v) for v in by_gene.values()]) if by_gene else np.zeros(0)
        row["gene_quantiles"] = [float(x) for x in np.percentile(q, [5, 25, 50, 75, 95])] if len(q) else None
        rows.append(row)
    return rows


def collect(cfg: BenchConfig) -> dict:
    folds = classify_folds(cfg)
    gx = genexp_folds(cfg)
    return {"classify_summary": summarize_classify(folds), "classify_folds": [compact_fold(f) for f in folds],
            "genexp_summary": summarize_genexp(gx),
            "genexp_folds": [{k: r[k] for k in ("set", "model", "readout", "fold", "test", "n_genes")} for r in gx]}
