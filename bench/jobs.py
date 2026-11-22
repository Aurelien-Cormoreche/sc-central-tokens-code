"""Plans, bins, SLURM arrays, status and retry -- shared by every command that runs jobs.

A job is a dict {kind, key, est_s, out, ...}; `key` is unique and stable across plans, `out` is
what makes the job "done" (checked by the kind's `is_done`). A plan packs jobs into bins
(longest-processing-time-first on est_s; bins of one `group` share a loaded model) and is saved
as {bench_root}/jobs/{plan_id}/plan.json together with the resolved config. Each bin is one
SLURM array task (`bench run-bin`), or runs in-process with --local. Workers re-check every job
before running it, so re-submitting a plan only redoes what is missing.

Per-job status: {bench_root}/status/{kind}/{key}.json ; failures also go to {bench_root}/failed.jsonl.
"""
from __future__ import annotations

import heapq
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from bench import REPO
from bench.config import BenchConfig
from bench.utils import LOG, append_jsonl, fmt_hours, print_table, read_json, write_json

# ── LPT packing ──────────────────────────────────────────────────────────────


def lpt(costs: List[float], k: int) -> List[List[int]]:
    """Longest-processing-time-first: indices into `costs` per bin (<= 4/3 OPT makespan)."""
    k = max(1, min(k, len(costs))) if costs else 1
    heap = [(0.0, b) for b in range(k)]
    bins: List[List[int]] = [[] for _ in range(k)]
    for i in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        load, b = heapq.heappop(heap)
        bins[b].append(i)
        heapq.heappush(heap, (load + costs[i], b))
    return [b for b in bins if b]


def pack(jobs: List[dict], max_s: float, overhead_s: float = 0.0, max_bins: Optional[int] = None) -> List[List[dict]]:
    """Bins per job `group` (default: one group), each bin's estimated time <= max_s when
    possible (a job longer than max_s gets a bin of its own)."""
    groups: Dict[str, List[dict]] = {}
    for j in jobs:
        groups.setdefault(j.get("group", ""), []).append(j)
    out: List[List[dict]] = []
    budget = max(1.0, max_s - overhead_s)
    for _, gj in sorted(groups.items()):
        costs = [float(j["est_s"]) for j in gj]
        k = max(1, math.ceil(sum(costs) / budget))
        while True:
            bins = lpt(costs, k)
            span = max(sum(costs[i] for i in b) for b in bins)
            if span <= budget or k >= len(gj) or (max_bins and k >= max_bins):
                break
            k += 1
        out.extend([[gj[i] for i in sorted(b, key=lambda i: -costs[i])] for b in bins])
    return out


# ── status ───────────────────────────────────────────────────────────────────

def _safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.=" else "__" if c == "/" else "_" for c in key)


def status_path(cfg: BenchConfig, job: dict) -> Path:
    return cfg.root / "status" / job["kind"] / f"{_safe(job['key'])}.json"


def set_status(cfg: BenchConfig, job: dict, state: str, **extra) -> None:
    rec = {"key": job["key"], "kind": job["kind"], "state": state, "time": datetime.now().isoformat(timespec="seconds"),
           "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
           "task": os.environ.get("SLURM_ARRAY_TASK_ID"), **extra}
    write_json(status_path(cfg, job), rec)
    if state == "failed":
        append_jsonl(cfg.root / "failed.jsonl", {**rec, "job": job})


def get_status(cfg: BenchConfig, job: dict) -> Optional[dict]:
    return read_json(status_path(cfg, job))


# ── kinds ────────────────────────────────────────────────────────────────────

def _done_check(kind: str) -> Callable:
    """Is-done check of a job kind: file checks only, no torch (planning runs on login nodes)."""
    if kind == "extract":
        from bench.extract.planner import job_done
    elif kind == "classify":
        from bench.probe.run import job_done
    elif kind == "genexp":
        from bench.genexp.run import job_done
    elif kind == "darkspots":
        from bench.data.darkspots import job_done
    else:
        raise KeyError(f"unknown job kind {kind!r}")
    return job_done


def _runner(kind: str) -> Callable:
    """Bin runner of a job kind (imports torch & co.; only in array tasks / --local)."""
    if kind == "extract":
        from bench.extract.worker import run_extract_bin as f
    elif kind == "classify":
        from bench.probe.run import run_probe_bin as f
    elif kind == "genexp":
        from bench.genexp.run import run_genexp_bin as f
    elif kind == "darkspots":
        from bench.data.darkspots import run_darkspots_bin as f
    else:
        raise KeyError(f"unknown job kind {kind!r}")
    return f


def is_done(cfg: BenchConfig, job: dict) -> bool:
    return bool(_done_check(job["kind"])(cfg, job))


# ── plans ────────────────────────────────────────────────────────────────────

def new_plan(cfg: BenchConfig, name: str, jobs: List[dict], resources: dict, max_s: Optional[float] = None,
             overhead_s: float = 0.0) -> dict:
    max_s = max_s if max_s is not None else cfg.slurm.max_hours * 3600 * 0.85
    bins = pack(jobs, max_s, overhead_s)
    pid = f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return {"plan_id": pid, "name": name, "created": datetime.now().isoformat(timespec="seconds"),
            "config": cfg.dump(), "resources": resources, "overhead_s": overhead_s,
            "bins": bins}


def plan_dir(cfg: BenchConfig, plan_id: str) -> Path:
    return cfg.root / "jobs" / plan_id


def save_plan(cfg: BenchConfig, plan: dict) -> Path:
    d = plan_dir(cfg, plan["plan_id"])
    write_json(d / "plan.json", plan)
    return d


def load_plan(path_or_id: str, cfg: Optional[BenchConfig] = None) -> dict:
    p = Path(path_or_id)
    if p.is_dir():
        p = p / "plan.json"
    if not p.exists() and cfg is not None:
        p = plan_dir(cfg, path_or_id) / "plan.json"
    plan = read_json(p)
    if plan is None:
        raise SystemExit(f"plan not found: {path_or_id}")
    return plan


def bin_hours(plan: dict, b: List[dict]) -> float:
    return (sum(j["est_s"] for j in b) + plan.get("overhead_s", 0.0)) / 3600


def print_plan(cfg: BenchConfig, plan: dict, show_jobs: bool = True) -> None:
    rows, tot_h, tot_gb, n_done = [], 0.0, 0.0, 0
    for i, b in enumerate(plan["bins"]):
        for j in b:
            done = is_done(cfg, j)
            n_done += done
            if show_jobs:
                rows.append({"bin": i, "job": j["key"], "cells": f"{j.get('n_cells', 0):,}",
                             "est": fmt_hours(j["est_s"] / 3600), "out_GB": f"{j.get('out_gb', 0):.2f}",
                             "done": "yes" if done else ""})
            if not done:
                tot_h += j["est_s"] / 3600
                tot_gb += j.get("out_gb", 0.0)
    if show_jobs:
        print_table(rows)
    n = sum(len(b) for b in plan["bins"])
    print(f"\nplan {plan['plan_id']}: {n} jobs in {len(plan['bins'])} bins, {n_done} already done; "
          f"remaining ~{tot_h:.1f} {plan['resources'].get('unit', 'GPU')}-h"
          f" (+{plan.get('overhead_s', 0) * len(plan['bins']) / 3600:.1f}h task overhead), ~{tot_gb:.1f} GB output; "
          f"longest bin {max((bin_hours(plan, b) for b in plan['bins']), default=0):.1f}h")


def run_bin(cfg: BenchConfig, plan: dict, index: int, device: Optional[str] = None) -> int:
    """Run one bin; returns the number of failed jobs. Jobs already done are skipped."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    jobs = plan["bins"][index]
    todo = []
    for j in jobs:
        if is_done(cfg, j):
            LOG.info("skip (done) %s", j["key"])
            continue
        todo.append(j)
    if not todo:
        return 0
    runner: Callable = _runner(todo[0]["kind"])
    failed = 0
    try:
        res = runner(cfg, todo, device)
        if isinstance(res, int):
            failed = res
    except Exception as e:  # runner-level failure (e.g. model load): every remaining job fails
        tb = traceback.format_exc()
        LOG.error("bin %d failed: %s", index, e)
        for j in todo:
            if not is_done(cfg, j):
                set_status(cfg, j, "failed", error=f"{type(e).__name__}: {e}", traceback=tb[-4000:],
                           plan=plan["plan_id"])
                failed += 1
    return failed


def run_local(cfg: BenchConfig, plan: dict, device: Optional[str] = None) -> int:
    failed = 0
    for i in range(len(plan["bins"])):
        LOG.info("── bin %d/%d (~%s)", i + 1, len(plan["bins"]), fmt_hours(bin_hours(plan, plan["bins"][i])))
        failed += run_bin(cfg, plan, i, device)
    return failed


# ── SLURM ────────────────────────────────────────────────────────────────────

def _fmt_time(hours: float) -> str:
    m = int(math.ceil(hours * 60))
    return f"{m // 1440}-{(m % 1440) // 60:02d}:{m % 60:02d}:00" if m >= 1440 else f"{m // 60:02d}:{m % 60:02d}:00"


def write_sbatch(cfg: BenchConfig, plan: dict, bins: Optional[List[int]] = None) -> Path:
    d = plan_dir(cfg, plan["plan_id"])
    res = plan["resources"]
    indices = bins if bins is not None else list(range(len(plan["bins"])))
    longest = max(bin_hours(plan, plan["bins"][i]) for i in indices)
    hours = min(cfg.slurm.max_hours, max(0.25, 1.5 * longest + 0.25))
    tmpl = (REPO / "scripts" / "slurm" / "array.sbatch.tmpl").read_text()
    s = cfg.slurm
    opt = []
    part = res.get("partition", s.partition)
    if part:
        opt.append(f"#SBATCH --partition={part}")
    if res.get("gres", s.gres):
        opt.append(f"#SBATCH --gres={res.get('gres', s.gres)}")
    if s.account:
        opt.append(f"#SBATCH --account={s.account}")
    opt += [f"#SBATCH {x}" for x in s.extra]
    text = tmpl.format(
        name=f"bench-{plan['name']}"[:60], array=",".join(map(str, indices)) + f"%{s.max_array}",
        time=_fmt_time(hours), cpus=res.get("cpus", s.cpus), mem=res.get("mem_gb", s.mem_gb),
        log_dir=d / "logs", options="\n".join(opt), setup=s.setup, repo=REPO, python=s.python, plan_dir=d)
    (d / "logs").mkdir(parents=True, exist_ok=True)
    out = d / ("array.sbatch" if bins is None else f"retry-{datetime.now().strftime('%H%M%S')}.sbatch")
    out.write_text(text)
    return out


def submit(cfg: BenchConfig, plan: dict, bins: Optional[List[int]] = None) -> str:
    script = write_sbatch(cfg, plan, bins)
    r = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"sbatch failed: {r.stderr.strip()}\nscript: {script}")
    job_id = r.stdout.strip().split(";")[0]
    plan.setdefault("submissions", []).append({"job_id": job_id, "bins": bins, "script": str(script),
                                               "time": datetime.now().isoformat(timespec="seconds")})
    save_plan(cfg, plan)
    print(f"submitted {script.name}: SLURM job {job_id} ({len(bins) if bins else len(plan['bins'])} tasks); "
          f"logs in {plan_dir(cfg, plan['plan_id']) / 'logs'}")
    return job_id


def execute(cfg: BenchConfig, plan: dict, dry_run: bool, local: bool, device: Optional[str] = None) -> int:
    """The tail of every job-running command: print, then run locally or submit."""
    print_plan(cfg, plan, show_jobs=dry_run or local)
    todo_bins = [i for i, b in enumerate(plan["bins"]) if not all(is_done(cfg, j) for j in b)]
    if dry_run:
        return 0
    if not todo_bins:
        print("nothing to do: every job is done")
        return 0
    save_plan(cfg, plan)
    if local:
        failed = run_local(cfg, plan, device)
        print(f"local run finished: {failed} failed job(s)" + (" -- see bench status / failed.jsonl" if failed else ""))
        return 1 if failed else 0
    submit(cfg, plan, todo_bins if len(todo_bins) < len(plan["bins"]) else None)
    return 0


def retry(cfg: BenchConfig, plan_id: str, dry_run: bool, local: bool) -> int:
    plan = load_plan(plan_id, cfg)
    pcfg = BenchConfig.model_validate(plan["config"])
    todo = [i for i, b in enumerate(plan["bins"]) if not all(is_done(pcfg, j) for j in b)]
    print(f"plan {plan['plan_id']}: {len(todo)}/{len(plan['bins'])} bins have missing or failed jobs")
    if not todo or dry_run:
        return 0
    if local:
        return 1 if sum(run_bin(pcfg, plan, i) for i in todo) else 0
    submit(pcfg, plan, todo)
    return 0


def main_run_bin(plan_path: str, index: int) -> int:
    plan = load_plan(plan_path)
    cfg = BenchConfig.model_validate(plan["config"])
    from bench.utils import setup_logging
    setup_logging()
    t0 = time.time()
    LOG.info("plan %s bin %d: %d jobs, est %s", plan["plan_id"], index, len(plan["bins"][index]),
             fmt_hours(bin_hours(plan, plan["bins"][index])))
    failed = run_bin(cfg, plan, index)
    LOG.info("bin %d finished in %s: %d failed", index, fmt_hours((time.time() - t0) / 3600), failed)
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main_run_bin(sys.argv[1], int(sys.argv[2])))
