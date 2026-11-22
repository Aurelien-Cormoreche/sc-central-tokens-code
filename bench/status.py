"""`bench status`: every job the current config implies, and whether it is done.

States: done (output valid for the current manifest/config), running / failed (last status file),
missing (never run, or output stale/invalid). `--deletable` lists, per model, whether its Stage-1
embeddings are no longer needed: all its Stage-1 classification and gene-expression jobs are done
and it is not the Gate-1 model (whose Stage-1 files Stage 2a and Stage 3 reuse).
"""
from __future__ import annotations

import subprocess
from collections import Counter
from typing import List, Optional

from bench.config import BenchConfig
from bench.registry import Slide, load_slides
from bench.utils import LOG, print_table


def planned_jobs(cfg: BenchConfig, models: Optional[List[str]] = None, stages: Optional[List[str]] = None,
                 slides: Optional[List[Slide]] = None) -> List[dict]:
    from bench.extract import planner
    from bench.genexp import run as genexp
    from bench.probe import run as probe
    slides = slides if slides is not None else load_slides(cfg)
    models = models or cfg.enabled_models()
    stages = stages or ["1", "2", "3"]
    jobs: List[dict] = []

    def safe(fn, *a, **k):
        try:
            return fn(*a, **k)
        except SystemExit as e:   # not plannable yet (gate unset, manifest missing, ...)
            LOG.debug("not planned: %s", e)
            return []

    if "1" in stages:
        jobs += safe(planner.plan_jobs, cfg, "1", models, slides)
        jobs += safe(probe.plan_jobs, cfg, "1", models, slides)
        for m in models:
            jobs += safe(genexp.plan_jobs, cfg, [m], slides)
    if "2" in stages and cfg.gate1.chosen.model:
        m = [cfg.gate1.chosen.model]
        jobs += safe(planner.plan_jobs, cfg, "2a", m, slides)
        if cfg.stage2.multicell.tile_px:
            jobs += safe(planner.plan_jobs, cfg, "2b", m, slides)
        jobs += safe(probe.plan_jobs, cfg, "2", m, slides)
    if "3" in stages and cfg.gate1.chosen.model:
        jobs += safe(probe.plan_jobs, cfg, "3", [cfg.gate1.chosen.model], slides)
    return jobs


def job_states(cfg: BenchConfig, jobs: List[dict]) -> List[dict]:
    from bench.jobs import get_status, is_done
    out = []
    for j in jobs:
        st = get_status(cfg, j) or {}
        if is_done(cfg, j):
            state = "done"
        elif st.get("state") in ("failed", "running"):
            state = st["state"]
        else:
            state = "missing"
        out.append({"kind": j["kind"], "key": j["key"], "state": state, "model": j.get("model"),
                    "stage": j.get("stage", "1"), "time": st.get("time"), "host": st.get("host"),
                    "error": (st.get("error") or "")[:300] if state != "done" else "",
                    "seconds": st.get("seconds"), "cells": st.get("cells", j.get("n_cells")),
                    "tiles": st.get("tiles"), "cells_per_s": st.get("cells_per_s"),
                    "est_h": round(j.get("est_s", 0) / 3600, 2)})
    return out


def deletable(cfg: BenchConfig, states: List[dict]) -> List[dict]:
    rows = []
    for m in cfg.enabled_models():
        mine = [s for s in states if s["model"] == m and s["stage"] in ("1", None)]
        need = [s for s in mine if s["kind"] in ("classify", "genexp")]
        ext = [s for s in mine if s["kind"] == "extract"]
        d = cfg.root / "embeddings" / "stage1" / m
        size = ""
        if d.exists():
            r = subprocess.run(["du", "-sh", str(d)], capture_output=True, text=True)
            size = r.stdout.split()[0] if r.returncode == 0 and r.stdout else "?"
        pending = [s["key"] for s in need if s["state"] != "done"]
        gate = cfg.gate1.chosen.model == m
        ok = bool(need) and not pending and not gate
        why = ("Gate-1 model: Stage 2a/3 reuse these files" if gate else
               f"{len(pending)} classify/genexp job(s) not done" if pending else
               "no classify/genexp jobs planned" if not need else "all Stage-1 consumers done")
        if ok and cfg.gate1.chosen.model is None:
            why += " (Gate 1 not chosen yet: if this model wins, Stage 2a/3 need its Stage-1 files re-extracted)"
        rows.append({"model": m, "deletable": "YES" if ok else "no", "reason": why,
                     "extracted": f"{sum(s['state'] == 'done' for s in ext)}/{len(ext)}", "size": size,
                     "path": str(d)})
    return rows


def run_status(cfg: BenchConfig, models=None, stages=None, show_deletable=False, verbose=False) -> List[dict]:
    states = job_states(cfg, planned_jobs(cfg, models, stages))
    counts = Counter((s["kind"], s["stage"], s["state"]) for s in states)
    keys = sorted({(k, st) for k, st, _ in counts})
    rows = [{"kind": k, "stage": st, **{x: counts.get((k, st, x), 0) for x in ("done", "running", "failed", "missing")}}
            for k, st in keys]
    print_table(rows)
    if verbose:
        print()
        print_table([s for s in states if s["state"] != "done"], ["kind", "key", "state", "time", "error"])
    failed = [s for s in states if s["state"] == "failed"]
    if failed:
        print(f"\n{len(failed)} failed job(s); details in {cfg.root / 'failed.jsonl'} -- resubmit with `bench retry --plan <id>` "
              "or rerun the stage command (it skips what is done)")
    if show_deletable:
        print()
        print_table(deletable(cfg, states))
    return states
