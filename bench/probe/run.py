"""Classification jobs (`bench classify`, `bench scaling`).

Job = (stage, scheme, model, readout, variant[, probe]); all folds of the scheme are trained
together: linear probes with trainer.train_linear_jointly (every fold x trial in lockstep, one
shared float16 embedding tensor), MLP probes with bench/probe/mlp.py.

Results: {bench_root}/results/{stage}/classify/{scheme}/{model}/{readout}/{variant}/
            [frac_{f}/seed_{s}/{probe}/]fold_{k}.json  + _DONE.json (inputs hash, provenance)
A job is done when every _DONE.json exists with the current inputs hash (config, folds, manifest).
"""
from __future__ import annotations

import math
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from bench.config import BenchConfig, stable_hash, without_planning
from bench.extract import planner
from bench.registry import Fold, Slide, class_names, classification_folds, load_slides, stage2_slides
from bench.utils import LOG, fmt_hours, read_json, write_json

STAGES = ("1", "2a", "2b", "3")


def results_dir(cfg: BenchConfig, stage: str, scheme: str, model: str, readout: str, variant: str) -> Path:
    return cfg.root / "results" / stage / "classify" / scheme / model / readout / variant


def sub_dir(base: Path, frac: Optional[float], seed: Optional[int], probe: Optional[str]) -> Path:
    if frac is None:
        return base
    return base / f"frac_{frac:g}" / f"seed_{seed}" / probe


def require_gate1(cfg: BenchConfig) -> tuple:
    ch = cfg.gate1.chosen
    if ch.model is None:
        raise SystemExit("gate1.chosen is not set. Run `bench report`, read the Gate 1 suggestion on the Overview "
                         "page, then set gate1.chosen.model and gate1.chosen.readout in the config.")
    return ch.model, ch.readout


# ── planning ──

def _scheme_slides(cfg: BenchConfig, stage: str, slides: List[Slide]) -> List[Slide]:
    return stage2_slides(cfg, slides) if stage in ("2a", "2b") else slides


def _folds(cfg: BenchConfig, stage: str, scheme: str, slides: List[Slide]) -> List[Fold]:
    from bench.data.manifest import load_class_counts
    cc = load_class_counts(cfg)
    if stage in ("2a", "2b"):
        d = cfg.dump()
        d["classify"]["within_cohorts"] = [c for c in cfg.classify.within_cohorts if c in cfg.stage2.cohorts]
        d.pop("source", None)
        cfg = BenchConfig.model_validate(d)
    return classification_folds(cfg, slides, scheme, cc)


def _cells(cfg: BenchConfig, names: List[str]) -> Dict[str, int]:
    sm = planner.slides_manifest(cfg)
    return {s: int(sm.loc[s, "n_classify"]) for s in names}


def _make_job(cfg: BenchConfig, stage: str, scheme: str, model: str, readout: str, variant: str,
              slides: List[Slide], probe: Optional[str] = None) -> dict:
    folds = _folds(cfg, stage, scheme, slides)
    used = sorted({s for f in folds for s in f.train + f.val + f.test})
    counts = _cells(cfg, used)
    n = sum(counts.values())
    d_std, d_tok = planner.DIMS.get(model, (1024, 1024))
    dim = d_std if readout == "cls" else d_tok
    train_cells = sum(sum(counts[s] for s in f.train) for f in folds)
    c = cfg.classify
    if stage == "3":
        fr = sum(min(1.0, f) for f in cfg.stage3.fractions) * len(cfg.stage3.seeds)
        if probe == "mlp":
            g = cfg.stage3.mlp.grid
            trials = len(g.lr) * len(g.weight_decay) * len(g.loss_weights)
            train_s = trials * fr * train_cells * c.epochs * 0.5 / cfg.stage3.mlp.gpu_cells_per_s
        else:
            train_s = fr * train_cells * c.epochs * 0.6 / c.gpu_cells_per_s
    else:
        train_s = train_cells * c.epochs * 0.6 / c.gpu_cells_per_s
    load_s = n * dim * 2 / 0.8e9
    var = "native" if stage == "3" else variant
    key = f"classify/{stage}/{scheme}/{model}/{readout}/{var}" + (f"/{probe}" if probe else "")
    return {
        "kind": "classify", "key": key, "group": key, "stage": stage, "scheme": scheme, "model": model,
        "readout": readout, "variant": var, "probe": probe, "n_cells": n, "n_folds": len(folds),
        "slides": [[s.slide, s.cohort, s.donor, s.val] for s in slides if s.slide in set(used)],
        "est_s": load_s + train_s + 120, "mem_gb": int(math.ceil(n * dim * 2 * 1.3 / 1e9 + 16)),
        "out": str(results_dir(cfg, stage, scheme, model, readout, var)),
    }


def plan_jobs(cfg: BenchConfig, stage: str, models: List[str], slides: List[Slide],
              probe: Optional[str] = None) -> List[dict]:
    jobs = []
    if stage == "1":
        for m in models:
            for scheme in cfg.classify.schemes:
                for r in cfg.readouts:
                    jobs.append(_make_job(cfg, "1", scheme, m, r, "native", slides))
    elif stage in ("2", "2a", "2b"):
        m, r = require_gate1(cfg)
        s2 = stage2_slides(cfg, slides)
        if stage in ("2", "2a"):
            for v, _, _ in planner.stage_variants(cfg, "2a", m):
                jobs.append(_make_job(cfg, "2a", "within", m, r, v, s2))
        if stage in ("2", "2b") and cfg.stage2.multicell.tile_px is not None:
            v = planner.stage_variants(cfg, "2b", m)[0][0]
            jobs.append(_make_job(cfg, "2b", "within", m, "central", v, s2))
    elif stage == "3":
        m, r = require_gate1(cfg)
        for p, schemes in cfg.stage3.probes.items():
            if probe and p != probe:
                continue
            for scheme in schemes:
                jobs.append(_make_job(cfg, "3", scheme, m, r, "native", slides, probe=p))
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    return jobs


def inputs_hash(cfg: BenchConfig, job: dict) -> str:
    sm = planner.slides_manifest(cfg)
    parts = {"job": {k: job[k] for k in ("stage", "scheme", "model", "readout", "variant", "probe", "slides")},
             "classify": without_planning(cfg.dump()["classify"]), "mapping": cfg.mapping,
             "manifest": {s[0]: str(sm.loc[s[0], "hash_classify"]) for s in job["slides"] if s[0] in sm.index}}
    if job["stage"] == "3":
        parts["stage3"] = without_planning(cfg.dump()["stage3"])
    return stable_hash(parts)


def _units(cfg: BenchConfig, job: dict) -> List[tuple]:
    if job["stage"] != "3":
        return [(None, None, None)]
    return [(f, s, job["probe"]) for s in cfg.stage3.seeds for f in cfg.stage3.fractions]


def _unit_done(cfg: BenchConfig, job: dict, unit: tuple, h: str) -> bool:
    m = read_json(sub_dir(Path(job["out"]), *unit) / "_DONE.json")
    return bool(m) and m.get("inputs_hash") == h


def job_done(cfg: BenchConfig, job: dict) -> bool:
    try:
        h = inputs_hash(cfg, job)
    except (SystemExit, KeyError):
        return False
    return all(_unit_done(cfg, job, u, h) for u in _units(cfg, job))


# ── running ──

def _fold_record(cfg, job, prep, data, best: dict, trials: List[dict], unit: tuple, n_train: int) -> dict:
    from bench.probe.metrics import group_mean, per_slide, summarize
    names, cohorts, donors = data.slide_names, [s.cohort for s in data.slides], [s.donor for s in data.slides]
    te = prep.test
    y = prep.local[te]
    pred = np.asarray(best["test_predictions"])
    rows = per_slide(y, pred, data.slide_idx[te], names, cohorts, donors, prep.classes, prep.dropped_test)
    vy = prep.local[prep.val]
    return {
        "stage": job["stage"], "scheme": job["scheme"], "model": job["model"], "readout": job["readout"],
        "variant": job["variant"], "probe": unit[2] or "linear", "frac": unit[0], "seed": unit[1],
        "task": prep.fold.task, "fold": prep.fold.to_dict(), "classes": prep.classes,
        "n": {"train": int(n_train), "train_pool": int(len(prep.train)), "val": int(len(prep.val)), "test": int(len(te))},
        "dropped": {"val": prep.dropped_val, "test": {names[k]: v for k, v in prep.dropped_test.items()}},
        "best_trial": best["trial"], "selected_epoch": best.get("selected_epoch"), "epochs_trained": best.get("epochs_trained"),
        "val": summarize(vy, np.asarray(best["val_predictions"]), prep.classes) if best.get("val_predictions") is not None else None,
        "test": summarize(y, pred, prep.classes),
        "test_slides": rows, "test_slide_mean": group_mean(rows, "slide"), "test_donor_mean": group_mean(rows, "donor"),
        "test_cohort_mean": group_mean(rows, "cohort"),
        "trials": trials,
        "curves": {k: best.get(k) for k in ("train_loss_curve", "val_loss_curve", "val_score_curve")},
    }


def _trial_info(c) -> dict:
    return {"lr": c.learning_rate, "weight_decay": c.weight_decay, "loss_weights": c.loss_weights,
            "hidden": c.hidden_dim, "seed": c.seed}


def _pick(fold_results: List[dict], configs) -> tuple:
    scores = [r["val_selection_score"] for r in fold_results]
    b = int(np.argmax(scores))
    trials = [{**_trial_info(c), "val_score": r["val_selection_score"],
               "val_macro_f1_present": r["val_macro_f1_present"], "test_macro_f1_present": r["test_macro_f1_present"],
               "selected_epoch": r["selected_epoch"]} for r, c in zip(fold_results, configs)]
    best = dict(fold_results[b])
    best["trial"] = _trial_info(configs[b])
    return best, trials


def _write_unit(cfg, job, unit, records: List[dict], h: str, seconds: float) -> None:
    from bench.provenance import collect
    d = sub_dir(Path(job["out"]), *unit)
    for k, rec in enumerate(records):
        write_json(d / f"fold_{k}.json", rec)
    write_json(d / "_DONE.json", {"inputs_hash": h, "n_folds": len(records), "seconds": round(seconds, 1),
                                  "provenance": collect(cfg)})


def run_probe_job(cfg: BenchConfig, job: dict, device: str) -> None:
    from bench.probe.data import load, z_stats
    from bench.probe.folds import fold_data, prepare, stage3_train_rows, trial_configs
    from src.python.classifier_training.trainer import train_linear_jointly

    t0 = time.time()
    h = inputs_hash(cfg, job)
    todo = [u for u in _units(cfg, job) if not _unit_done(cfg, job, u, h)]
    if not todo:
        return
    all_slides = load_slides(cfg)
    slides = _scheme_slides(cfg, job["stage"], all_slides)
    folds = _folds(cfg, job["stage"], job["scheme"], slides)
    used = {s for f in folds for s in f.train + f.val + f.test}
    data = load(cfg, job["stage"] if job["stage"] != "3" else "1", job["model"], job["variant"], job["readout"],
                [s for s in slides if s.slide in used], class_names(cfg))
    preps = [prepare(f, data) for f in folds]
    preps = [p for p in preps if len(p.train) and len(p.val) and len(p.test) and len(p.classes) > 1]
    if len(preps) < len(folds):
        LOG.warning("%s: %d fold(s) skipped (empty train/val/test or a single class)", job["key"], len(folds) - len(preps))
    metric = cfg.classify.selection_metric[job["scheme"]]
    dim = data.emb.shape[1]
    zs = {}

    def z_of(i, rows):
        if not cfg.classify.zscore:
            return None
        import zlib
        key = (i, len(rows), zlib.crc32(np.ascontiguousarray(rows, dtype=np.int64).tobytes()))
        if key not in zs:
            zs[key] = z_stats(data.emb, rows)
        return zs[key]

    by_seed: Dict[Optional[int], List[tuple]] = {}
    for u in todo:
        by_seed.setdefault(u[1], []).append(u)
    for seed, units in by_seed.items():
        t1 = time.time()
        entries = []   # (unit, prep index, train rows, configs)
        for u in units:
            for i, p in enumerate(preps):
                rows = p.train if u[0] is None else stage3_train_rows(cfg, p, data, u[0], u[1])
                s = cfg.classify.seed if seed is None else seed
                if u[2] == "mlp":
                    configs = trial_configs(cfg, dim, p.classes, cfg.stage3.mlp.grid, s, cfg.stage3.mlp.hidden,
                                            cfg.stage3.mlp.dropout)
                else:
                    configs = trial_configs(cfg, dim, p.classes, cfg.classify.grid, s)
                entries.append((u, i, rows, configs))
        results: Dict[tuple, tuple] = {}
        lin = [e for e in entries if e[0][2] != "mlp"]
        if lin:
            fds = [fold_data(cfg, preps[i], data, cf, metric, rows, z_of(i, rows)) for (_, i, rows, cf) in lin]
            out, _ = train_linear_jointly(data.emb, fds, device)
            for e, fr in zip(lin, out):
                results[(e[0], e[1])] = (*_pick(fr, e[3]), len(e[2]))
        for (u, i, rows, cf) in [e for e in entries if e[0][2] == "mlp"]:
            from bench.probe.mlp import train_mlp
            z = z_of(i, rows)
            fd = fold_data(cfg, preps[i], data, cf, metric, rows, z)
            fr = [train_mlp(data.emb, fd, c, device, z) for c in cf]
            results[(u, i)] = (*_pick(fr, cf), len(rows))
        for u in units:
            recs = [_fold_record(cfg, job, preps[i], data, *results[(u, i)][:2], u, results[(u, i)][2])
                    for i in range(len(preps))]
            _write_unit(cfg, job, u, recs, h, time.time() - t1)
    LOG.info("%s done in %s", job["key"], fmt_hours((time.time() - t0) / 3600))


def run_probe_bin(cfg: BenchConfig, jobs: List[dict], device: str) -> int:
    from bench.jobs import set_status
    failed = 0
    for j in jobs:
        if job_done(cfg, j):
            continue
        set_status(cfg, j, "running")
        t0 = time.time()
        try:
            run_probe_job(cfg, j, device)
            set_status(cfg, j, "done", seconds=round(time.time() - t0, 1))
        except Exception as e:
            failed += 1
            LOG.error("%s FAILED: %s", j["key"], e)
            set_status(cfg, j, "failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
    return failed
