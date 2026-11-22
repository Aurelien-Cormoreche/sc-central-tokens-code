"""LPT packing."""
import itertools

import numpy as np

from bench.jobs import lpt, pack


def _opt(costs, k):
    best = float("inf")
    for assign in itertools.product(range(k), repeat=len(costs)):
        loads = [0.0] * k
        for c, b in zip(costs, assign):
            loads[b] += c
        best = min(best, max(loads))
    return best


def test_lpt_within_4_3_of_optimal():
    rng = np.random.default_rng(0)
    for _ in range(30):
        costs = rng.uniform(1, 10, size=7).tolist()
        bins = lpt(costs, 3)
        span = max(sum(costs[i] for i in b) for b in bins)
        assert span <= 4 / 3 * _opt(costs, 3) + 1e-9
        assert sorted(i for b in bins for i in b) == list(range(7))


def test_pack_respects_budget_and_groups():
    jobs = [{"key": f"a{i}", "group": "a", "est_s": 100.0} for i in range(10)] + \
           [{"key": f"b{i}", "group": "b", "est_s": 250.0} for i in range(3)] + \
           [{"key": "huge", "group": "c", "est_s": 5000.0}]
    bins = pack(jobs, max_s=300.0)
    for b in bins:
        assert len({j["group"] for j in b}) == 1
        if len(b) > 1:
            assert sum(j["est_s"] for j in b) <= 300.0
    assert sorted(j["key"] for b in bins for j in b) == sorted(j["key"] for j in jobs)
    assert [len(b) for b in bins if b[0]["key"] == "huge"] == [1]
