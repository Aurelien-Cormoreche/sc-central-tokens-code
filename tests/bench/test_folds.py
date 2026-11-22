"""Fold definitions on the real slide registry (configs/bench_slides.csv)."""
import pytest

from bench.config import load_config
from bench.registry import (classification_folds, cross_loco_folds, genexp_folds, genexp_slides,
                            load_slides, within_folds)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def slides(cfg):
    return load_slides(cfg)


def test_registry(slides):
    by = {}
    for s in slides:
        by[s.cohort] = by.get(s.cohort, 0) + 1
    assert by == {"10X": 35, "JOHNSON": 5, "MCCORD": 11, "SPLIT": 29, "VANNAN": 18}
    assert not any(s.slide.startswith("Yu_DRG") for s in slides)


def test_within_folds(cfg, slides):
    folds = [f for c in cfg.classify.within_cohorts for f in within_folds(slides, c)]
    assert len(folds) == 43
    donor = {s.slide: s.donor for s in slides}
    for f in folds:
        tr, va, te = ({donor[x] for x in side} for side in (f.train, f.val, f.test))
        assert len(te) == 1 and not (tr & te) and not (tr & va) and not (va & te)


def test_cross_and_single(cfg, slides):
    cc = {c: {"A": 10, "B": 10} for c in cfg.cohorts}
    cc["JOHNSON"] = {"A": 10}
    folds = cross_loco_folds(slides, cfg.cohorts, cc)
    assert len(folds) == 5
    j = next(f for f in folds if f.task == "cross:JOHNSON")
    assert j.label_space == ["A"]
    assert all(not (set(f.train) & set(f.test)) for f in folds)
    single = classification_folds(cfg, slides, "single_cohort", cc)
    assert {f.task for f in single} == {f"single:{c}" for c in cfg.cohorts}


def test_genexp_folds(cfg, slides):
    sets = genexp_slides(cfg, slides)
    n = {k: len(genexp_folds(v, k)) for k, v in sets.items()}
    assert n == {"PRIME": 9, "VANNAN": 14}
    for k, v in sets.items():
        donor = {s.slide: s.donor for s in v}
        for f in genexp_folds(v, k):
            tr, va, te = ({donor[x] for x in side} for side in (f.train, f.val, f.test))
            assert len(te) == 1 and len(va) == 1 and not (tr & (va | te))
