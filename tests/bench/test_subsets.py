"""Stage-3 class-stratified training subsets."""
import numpy as np

from bench.probe.folds import stratified_subset


def _pool(seed=0, n=5000):
    rng = np.random.default_rng(seed)
    slides = np.array([f"s{i}" for i in rng.integers(0, 8, n)])
    mrows = np.arange(n)
    labels = rng.choice(5, size=n, p=[0.6, 0.3, 0.07, 0.025, 0.005])
    return slides, mrows, labels


def test_min_per_class_and_fraction():
    s, m, y = _pool()
    idx = stratified_subset(s, m, y, 0.005, 0, 1, "fold")
    got = np.bincount(y[idx], minlength=5)
    full = np.bincount(y, minlength=5)
    assert (got >= 1).all()
    for c in range(5):
        assert got[c] == min(full[c], max(1, round(0.005 * full[c])))


def test_deterministic_and_order_independent():
    s, m, y = _pool()
    a = stratified_subset(s, m, y, 0.05, 1, 1, "fold")
    b = stratified_subset(s, m, y, 0.05, 1, 1, "fold")
    assert np.array_equal(a, b)
    perm = np.random.default_rng(9).permutation(len(s))
    c = stratified_subset(s[perm], m[perm], y[perm], 0.05, 1, 1, "fold")
    # same (slide, manifest row) pairs whatever the row order (= whatever model produced it)
    assert set(zip(s[a], m[a])) == set(zip(s[perm][c], m[perm][c]))


def test_seeds_and_folds_differ():
    s, m, y = _pool()
    a = stratified_subset(s, m, y, 0.05, 0, 1, "fold")
    assert not np.array_equal(a, stratified_subset(s, m, y, 0.05, 1, 1, "fold"))
    assert not np.array_equal(a, stratified_subset(s, m, y, 0.05, 0, 1, "other"))


def test_full_fraction_keeps_everything():
    s, m, y = _pool(n=300)
    assert len(stratified_subset(s, m, y, 1.0, 0, 1, "f")) == 300
