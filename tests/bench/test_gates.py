"""Gate rules on summary rows."""
import csv

import pytest

from bench.config import BenchConfig
from bench.report.gates import gate1, gate2


@pytest.fixture()
def cfg(tmp_path):
    p = tmp_path / "slides.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slide", "cohort", "donor", "val"])
        for c in ("A", "B"):
            for k in range(1, 5):
                w.writerow([f"{c}{k}", c, f"{c}d{k}", int(k == 4)])
    return BenchConfig.model_validate({
        "paths": {"bench_root": str(tmp_path / "root")}, "slides_csv": str(p), "cohorts": ["A", "B"],
        "models": {"m1": {"input_px": 224, "patch": 14}, "m2": {"input_px": 224, "patch": 14}},
        "classify": {"within_cohorts": ["A", "B"]}})


def _row(model, readout, task, score, n=3, stage="1", variant="native"):
    return {"stage": stage, "scheme": "within", "task": task, "model": model, "readout": readout,
            "variant": variant, "probe": "linear", "frac": None, "seed": None, "n_folds": n,
            "macro_f1_present": score, "val_macro_f1_present": 1 - score}


def test_gate1_ranks_complete_rows_by_test_mean(cfg):
    rows = [_row("m1", "cls", "within:A", 0.7), _row("m1", "cls", "within:B", 0.7),
            _row("m1", "central", "within:A", 0.8), _row("m1", "central", "within:B", 0.75),
            _row("m2", "cls", "within:A", 0.95), _row("m2", "cls", "within:B", 0.95, n=2),   # incomplete
            _row("m2", "central", "within:A", 0.6), _row("m2", "central", "within:B", 0.6)]
    g = gate1(cfg, rows)
    assert g["suggested"] == {"model": "m1", "readout": "central"}
    assert g["ranking"][0]["score"] == pytest.approx(0.775)
    assert g["ranking"][-1]["model"] == "m2" and not g["ranking"][-1]["complete"]


def test_gate1_val_rule(cfg):
    d = cfg.dump()
    d["gate1"]["rule"]["split"] = "val"
    cfg2 = BenchConfig.model_validate(d)
    rows = [_row("m1", "cls", t, 0.9) for t in ("within:A", "within:B")] + \
           [_row("m2", "cls", t, 0.5) for t in ("within:A", "within:B")]
    assert gate1(cfg2, rows)["suggested"]["model"] == "m2"


def test_gate2(cfg):
    d = cfg.dump()
    d["gate1"]["chosen"] = {"model": "m1", "readout": "central"}
    cfg2 = BenchConfig.model_validate(d)
    rows = [_row("m1", "central", t, s, stage="2a", variant=v) for t in ("within:A", "within:B")
            for v, s in (("fov056", 0.5), ("fov224", 0.7), ("fov448", 0.8))]
    g = gate2(cfg2, rows)
    assert g["suggested_tile_px"] == 448 and g["by_fov"][224] == pytest.approx(0.7)


def test_config_rejects_half_set_gate(cfg):
    d = cfg.dump()
    d["gate1"]["chosen"] = {"model": "m1", "readout": None}
    with pytest.raises(ValueError):
        BenchConfig.model_validate(d)
    d["gate1"]["chosen"] = {"model": None, "readout": None}
    d["stage2"]["multicell"]["tile_px"] = 300
    with pytest.raises(ValueError):
        BenchConfig.model_validate(d)
