"""Decision gates. The pipeline only *suggests*; the user writes the choice into the config.

Gate 1 (after Stage 1): rank every (model, readout) by gate1.rule -- by default the mean, over
the within-cohort tasks (classify.within_cohorts), of the task score (mean over held-out donors of
the test macro F1 over present classes). Ranking on test scores is the user's choice; the site
flags every Stage 2-3 number as selected on test.

Gate 2 (after Stage 2a): for the chosen (model, readout), the multicell tile size suggestion is
the better of the 224 and 448 px single-cell inputs (mean over the Stage-2 within-cohort tasks).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from bench.config import BenchConfig
from bench.report.collect import classify_folds, summarize_classify


def _expected_folds(cfg: BenchConfig) -> Dict[str, int]:
    from bench.registry import load_slides, within_folds
    slides = load_slides(cfg)
    out = {}
    for c in cfg.classify.within_cohorts:
        n = len(within_folds(slides, c))
        if n:
            out[f"within:{c}"] = 1 if cfg.is_smoke else n
    return out


def model_readout_scores(cfg: BenchConfig, summary: Optional[List[dict]] = None) -> Dict[Tuple[str, str], dict]:
    rule = cfg.gate1.rule
    summary = summary if summary is not None else summarize_classify(classify_folds(cfg))
    expected = _expected_folds(cfg)
    metric = rule.metric if rule.split == "test" else f"val_{rule.metric}"
    out: Dict[Tuple[str, str], dict] = {}
    for r in summary:
        if r["stage"] != "1" or r["scheme"] != "within" or r["variant"] != "native":
            continue
        d = out.setdefault((r["model"], r["readout"]), {"per_task": {}, "folds": {}, "val_per_task": {}})
        d["per_task"][r["task"]] = r.get(metric)
        d["val_per_task"][r["task"]] = r.get(f"val_{rule.metric}")
        d["folds"][r["task"]] = r["n_folds"]
    for k, d in out.items():
        d["complete"] = all(d["folds"].get(t, 0) >= n for t, n in expected.items())
        vals = [v for t, v in d["per_task"].items() if t in expected and v is not None]
        d["score"] = float(np.mean(vals)) if vals else None
        vv = [v for t, v in d["val_per_task"].items() if t in expected and v is not None]
        d["val_score"] = float(np.mean(vv)) if vv else None
        d["missing"] = sorted(t for t, n in expected.items() if d["folds"].get(t, 0) < n)
    return out


def gate1(cfg: BenchConfig, summary: Optional[List[dict]] = None) -> dict:
    scores = model_readout_scores(cfg, summary)
    ranking = sorted(
        [{"model": m, "readout": r, **d} for (m, r), d in scores.items()],
        key=lambda x: (not x["complete"], -(x["score"] if x["score"] is not None else -1)))
    complete = [x for x in ranking if x["complete"]]
    sug = complete[0] if complete else None
    return {"rule": cfg.gate1.rule.model_dump(), "ranking": ranking,
            "suggested": {"model": sug["model"], "readout": sug["readout"]} if sug else None,
            "chosen": cfg.gate1.chosen.model_dump(),
            "all_models_complete": {m for m, _ in scores} >= set(cfg.enabled_models()) and all(
                x["complete"] for x in ranking)}


def gate2(cfg: BenchConfig, summary: Optional[List[dict]] = None) -> dict:
    ch = cfg.gate1.chosen
    out = {"chosen_tile_px": cfg.stage2.multicell.tile_px, "suggested_tile_px": None, "by_fov": {}}
    if ch.model is None:
        return out
    summary = summary if summary is not None else summarize_classify(classify_folds(cfg))
    by: Dict[int, List[float]] = {}
    for r in summary:
        if r["stage"] == "2a" and r["model"] == ch.model and r["readout"] == ch.readout:
            fov = int(r["variant"].removeprefix("fov"))
            by.setdefault(fov, []).append(r[cfg.gate1.rule.metric])
    out["by_fov"] = {f: float(np.mean(v)) for f, v in sorted(by.items())}
    cands = {f: v for f, v in out["by_fov"].items() if f in (224, 448)}
    if len(cands) == 2:
        out["suggested_tile_px"] = max(cands, key=cands.get)
    return out
