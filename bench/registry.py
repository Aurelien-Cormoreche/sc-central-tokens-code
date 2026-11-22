"""Slide registry and fold definitions.

The registry (configs/bench_slides.csv, written by scripts/make_bench_slides.py from
_metadata_work/splits.csv) holds slide -> cohort, donor, fixed-val flag. Folds are pure functions
of the registry plus per-cohort class counts (from the cell manifest), so every model, readout,
stage and probe sees exactly the same folds.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from bench.config import BenchConfig
from bench.utils import matches_any

UNKNOWN = "Unknown"


@dataclass(frozen=True)
class Slide:
    slide: str
    cohort: str
    donor: str
    val: bool


def load_slides(cfg: BenchConfig) -> List[Slide]:
    with open(cfg.repo_path(cfg.slides_csv), newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = [Slide(r["slide"], r["cohort"], r["donor"], r["val"].strip() in ("1", "true", "True"))
           for r in rows]
    out = [s for s in out if s.cohort in cfg.cohorts and s.slide not in set(cfg.exclude)]
    if cfg.is_smoke:
        out = [s for s in out if s.slide in set(cfg.smoke.slides)]
    return out


def select_slides(slides: List[Slide], datasets: Optional[List[str]]) -> List[Slide]:
    """--datasets filter: cohort names or slide-name patterns."""
    if not datasets:
        return slides
    return [s for s in slides if s.cohort in datasets or matches_any(s.slide, datasets)]


def class_names(cfg: BenchConfig) -> List[str]:
    from src.python.code_configs.mappings import mapping_factory
    return sorted({v for v in mapping_factory(cfg.mapping).values() if v != UNKNOWN})


def genexp_slides(cfg: BenchConfig, slides: List[Slide]) -> Dict[str, List[Slide]]:
    out = {}
    for name, spec in cfg.genexp.sets.items():
        sel = [s for s in slides if (spec.cohort and s.cohort == spec.cohort)
               or (spec.slides and matches_any(s.slide, spec.slides))]
        out[name] = sel
    return out


# ── folds ──────────────────────────────────────────────────────────────────────

@dataclass
class Fold:
    scheme: str                 # within | cross_loco | single_cohort | genexp
    task: str                   # site grouping, e.g. within:SPLIT, cross:VANNAN, single:SPLIT
    index: int
    key: str                    # stable id, used in paths and seeds
    train: List[str]
    val: List[str]
    test: List[str]
    # None = the classes present in train (within). Otherwise the fixed label space; train/val
    # cells outside it are dropped, test cells outside it are dropped and counted.
    label_space: Optional[List[str]] = None
    test_cohorts: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _donors(slides: List[Slide]) -> Dict[str, List[str]]:
    d: Dict[str, List[str]] = defaultdict(list)
    for s in slides:
        d[s.donor].append(s.slide)
    return {k: sorted(v) for k, v in d.items()}


def _safe(s: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
    return keep[:80]


def within_folds(slides: List[Slide], cohort: str) -> List[Fold]:
    """Leave-one-donor-out within a cohort: val = the cohort's fixed val donors (same in every
    fold), test = one other donor, train = the remaining donors (same rule as
    scripts/make_within_cohort_configs.py)."""
    cs = [s for s in slides if s.cohort == cohort]
    donors = _donors(cs)
    val_donors = {s.donor for s in cs if s.val}
    val = sorted(w for d in val_donors for w in donors[d])
    test_donors = sorted(set(donors) - val_donors, key=lambda d: (donors[d][0], d))
    folds = []
    for i, td in enumerate(test_donors):
        train = sorted(w for d in test_donors if d != td for w in donors[d])
        folds.append(Fold("within", f"within:{cohort}", i, f"{cohort}__{_safe(td)}", train, val,
                          donors[td], None, [cohort], note=td))
    return folds


def present_classes(class_counts: Dict[str, Dict[str, int]], cohorts: List[str], min_cells: int) -> set:
    out: set = set()
    for c in cohorts:
        out |= {k for k, n in class_counts.get(c, {}).items() if n >= min_cells}
    return out


def cross_loco_folds(slides: List[Slide], cohorts: List[str], class_counts: Dict[str, Dict[str, int]],
                     min_cells: int = 1) -> List[Fold]:
    """Leave-one-cohort-out: test = every slide of the held-out cohort; train/val = the other
    cohorts' train/val donors. Label space = classes present in both train and test cohorts."""
    folds = []
    for i, held in enumerate(cohorts):
        others = [c for c in cohorts if c != held]
        train = sorted(s.slide for s in slides if s.cohort in others and not s.val)
        val = sorted(s.slide for s in slides if s.cohort in others and s.val)
        test = sorted(s.slide for s in slides if s.cohort == held)
        space = sorted(present_classes(class_counts, others, min_cells) & present_classes(class_counts, [held], min_cells))
        folds.append(Fold("cross_loco", f"cross:{held}", i, f"test_{held}", train, val, test, space, [held]))
    return folds


def single_cohort_folds(slides: List[Slide], cohorts: List[str], class_counts: Dict[str, Dict[str, int]],
                        min_cells: int = 1) -> List[Fold]:
    """Train on one cohort (its train donors; val = its val donors), test on each other cohort.
    Test cohorts sharing a label space (classes present in both train and test cohort) form one
    fold, as in scripts/make_single_cohort_config.py."""
    folds = []
    i = 0
    for t in cohorts:
        train = sorted(s.slide for s in slides if s.cohort == t and not s.val)
        val = sorted(s.slide for s in slides if s.cohort == t and s.val)
        t_cls = present_classes(class_counts, [t], min_cells)
        groups: Dict[tuple, List[str]] = defaultdict(list)
        for u in cohorts:
            if u != t:
                groups[tuple(sorted(t_cls & present_classes(class_counts, [u], min_cells)))].append(u)
        for space, us in sorted(groups.items(), key=lambda kv: kv[1]):
            test = sorted(s.slide for s in slides if s.cohort in us)
            folds.append(Fold("single_cohort", f"single:{t}", i, f"train_{t}__test_{'-'.join(us)}",
                              train, val, test, list(space), list(us)))
            i += 1
    return folds


def classification_folds(cfg: BenchConfig, slides: List[Slide], scheme: str,
                         class_counts: Dict[str, Dict[str, int]]) -> List[Fold]:
    present = sorted({s.cohort for s in slides})
    cohorts = [c for c in cfg.cohorts if c in present]
    m = cfg.classify.min_class_cells
    if scheme == "within":
        folds = [f for c in cfg.classify.within_cohorts if c in present for f in within_folds(slides, c)]
    elif scheme == "cross_loco":
        folds = cross_loco_folds(slides, cohorts, class_counts, m)
    elif scheme == "single_cohort":
        folds = single_cohort_folds(slides, cohorts, class_counts, m)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    folds = [f for f in folds if f.train and f.test]
    if cfg.is_smoke:
        # one fold per task keeps the smoke run short
        seen, keep = set(), []
        for f in folds:
            if f.task not in seen:
                seen.add(f.task)
                keep.append(f)
        folds = keep
    for i, f in enumerate(folds):
        f.index = i
    return folds


def stage2_slides(cfg: BenchConfig, slides: List[Slide]) -> List[Slide]:
    return [s for s in slides if s.cohort in cfg.stage2.cohorts]


def genexp_folds(slides: List[Slide], set_name: str) -> List[Fold]:
    """Leave-one-patient-out over a gene-expression slide set: donors sorted by donor key; fold i
    tests donor i, validates on donor i+1 (cyclic), trains on the rest (same rule as
    scripts/make_deepspot_lopo_configs.py)."""
    donors = _donors(slides)
    order = sorted(donors)
    folds = []
    if len(order) < 3:
        return folds
    for i, td in enumerate(order):
        vd = order[(i + 1) % len(order)]
        train = sorted(w for d in order if d not in (td, vd) for w in donors[d])
        folds.append(Fold("genexp", f"genexp:{set_name}", i, f"{set_name}__{_safe(td)}", train,
                          donors[vd], donors[td], None, [set_name], note=td))
    return folds
