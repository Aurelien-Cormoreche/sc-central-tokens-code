"""`bench` command line. Every command takes --config, --smoke, --dry-run, --models, --datasets.

Data preparation (CPU):   fetch-weights, inspect, darkspots, manifest
Jobs (SLURM arrays):      extract, classify, genexp, scaling   (--local: run the bins here)
Results:                  report, status, retry, calibrate
Checks:                   smoke [--synthetic]
Internal:                 run-bin (one array task)
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from bench.config import BenchConfig, load_config, smoke_config
from bench.utils import LOG, print_table, setup_logging


def _common(p: argparse.ArgumentParser, jobs: bool = False) -> None:
    p.add_argument("--config", help="bench YAML (default: $BENCH_CONFIG or configs/bench.yaml)")
    p.add_argument("--smoke", action="store_true", help="smoke variant: smoke slides, few cells, outputs in {bench_root}/smoke")
    p.add_argument("--dry-run", action="store_true", help="print what would run (jobs, estimates, done flags) and exit")
    p.add_argument("--models", nargs="+", help="restrict to these models (config names)")
    p.add_argument("--datasets", nargs="+", help="restrict to these cohorts or slide-name patterns")
    p.add_argument("-v", "--verbose", action="store_true")
    if jobs:
        p.add_argument("--local", action="store_true", help="run the bins in this process instead of submitting to SLURM")
        p.add_argument("--device", help="torch device for --local (default: cuda if available)")


def _cfg(args) -> BenchConfig:
    setup_logging(getattr(args, "verbose", False))
    cfg = load_config(args.config)
    return smoke_config(cfg) if args.smoke else cfg


def _slides(cfg: BenchConfig, args):
    from bench.registry import load_slides, select_slides
    slides = select_slides(load_slides(cfg), args.datasets)
    if not slides:
        raise SystemExit("no slides selected (check --datasets / cohorts / exclude)")
    return slides


def _gate_models(cfg: BenchConfig, args) -> List[str]:
    from bench.probe.run import require_gate1
    m, _ = require_gate1(cfg)
    if args.models and args.models != [m]:
        raise SystemExit(f"stage 2/3 run on the Gate-1 model only ({m}); drop --models or change gate1.chosen")
    return [m]


def _run_jobs(cfg: BenchConfig, args, name: str, jobs: List[dict], resources: dict, overhead_s: float,
              max_s: Optional[float] = None) -> int:
    from bench.jobs import execute, new_plan
    if not jobs:
        print("no jobs")
        return 0
    plan = new_plan(cfg, name, jobs, resources, max_s=max_s, overhead_s=overhead_s)
    return execute(cfg, plan, args.dry_run, getattr(args, "local", False), getattr(args, "device", None))


def _gpu(cfg: BenchConfig, mem_gb: Optional[int] = None) -> dict:
    return {"unit": "GPU", "partition": cfg.slurm.partition, "gres": cfg.slurm.gres, "cpus": cfg.slurm.cpus,
            "mem_gb": mem_gb or cfg.slurm.mem_gb}


# ── commands ──

def cmd_fetch_weights(cfg: BenchConfig, args) -> int:
    """Make sure every enabled model's weights are in the local HF cache ($HF_HOME/hub) and record their
    snapshot shas in {bench_root}/weights_revisions.json; extraction then loads exactly those snapshots.

    Default: download what is missing (needs internet). --offline (or HF_HUB_OFFLINE=1): no network -- check
    the cache, filled with `hf download` on a machine with internet, and print the commands for what is missing.
    --dry-run: print the `hf download` commands only."""
    import os
    from datetime import datetime
    from bench.models.catalog import (REPOS, cached_snapshot, hf_cache_dir, hf_download_command,
                                      missing_weight_files, weight_spec)
    from bench.provenance import model_revisions
    from bench.utils import write_json
    offline = args.offline or os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on")
    root = cfg.root.parent if cfg.is_smoke else cfg.root
    revs = dict(model_revisions(root))
    rows, todo = [], []
    for m in cfg.enabled_models(args.models):
        if m not in REPOS:
            continue
        mcfg = cfg.models[m]
        repo, rev = REPOS[m], mcfg.revision
        files, _ = weight_spec(m, mcfg)
        cmd = hf_download_command(repo, rev, files)
        if args.dry_run:
            rows.append({"model": m, "command": cmd})
            continue
        if not offline:
            from huggingface_hub import snapshot_download
            LOG.info("downloading %s (%s)", m, repo)
            snapshot_download(repo, revision=rev, allow_patterns=files)
        snap = cached_snapshot(repo, rev)
        miss = missing_weight_files(m, mcfg, snap)
        if miss:
            rows.append({"model": m, "repo": repo, "state": "MISSING: " + "; ".join(miss)})
            todo.append(cmd)
            continue
        revs[m] = {"repo": repo, "requested": rev or "main", "sha": snap.name, "snapshot": str(snap),
                   "time": datetime.now().isoformat(timespec="seconds")}
        rows.append({"model": m, "repo": repo, "state": "ok", "sha": snap.name})
    print(f"HF cache: {hf_cache_dir()}")
    print_table(rows)
    if args.dry_run:
        return 0
    write_json(root / "weights_revisions.json", revs)
    print(f"\nrecorded in {root / 'weights_revisions.json'} (jobs load these snapshots unless models.<name>.revision is set)")
    if todo:
        print("\nmissing -- on a machine with internet, with HF_HOME pointing at this cache (or copy its hub/ here):")
        print("\n".join("  " + c for c in todo))
        print("then rerun this command. Gated repos: `hf auth login` there, and accept the terms on the model pages.")
        return 1
    return 0


def cmd_inspect(cfg: BenchConfig, args) -> int:
    from bench.data.inspect import run_inspect
    from bench.provenance import write_run_provenance
    slides = _slides(cfg, args)
    if args.dry_run:
        print(f"would inspect {len(slides)} slides -> {cfg.root / 'manifest' / 'slides_inspect.parquet'}")
        return 0
    run_inspect(cfg, slides, workers=args.workers, strict=not args.no_strict)
    write_run_provenance(cfg.root / "manifest" / "provenance_inspect", cfg)
    return 0


def cmd_darkspots(cfg: BenchConfig, args) -> int:
    from bench.data import darkspots
    from bench.data.inspect import load_inspect
    slides = _slides(cfg, args)
    jobs = darkspots.plan_jobs(cfg, slides, load_inspect(cfg))
    res = {"unit": "CPU", "partition": cfg.slurm.cpu_partition, "gres": "", "cpus": cfg.slurm.cpus,
           "mem_gb": cfg.slurm.cpu_mem_gb}
    return _run_jobs(cfg, args, "darkspots", jobs, res, overhead_s=30, max_s=4 * 3600)


def cmd_manifest(cfg: BenchConfig, args) -> int:
    from bench.data.darkspots import darkspot_path
    from bench.data.inspect import load_inspect
    from bench.data.manifest import run_manifest
    from bench.provenance import write_run_provenance
    slides = _slides(cfg, args)
    missing = [s.slide for s in slides if not darkspot_path(cfg, s.slide).exists()]
    if args.dry_run or missing:
        print(f"{len(slides)} slides; {len(missing)} without a dark-spot map" + (f": {missing[:10]}" if missing else ""))
        return 1 if missing else 0
    run_manifest(cfg, slides, load_inspect(cfg), workers=args.workers)
    write_run_provenance(cfg.root / "manifest" / "provenance_manifest", cfg)
    return 0


def cmd_extract(cfg: BenchConfig, args) -> int:
    from bench.extract import planner
    stage = args.stage
    models = cfg.enabled_models(args.models) if stage == "1" else _gate_models(cfg, args)
    jobs = planner.plan_jobs(cfg, stage, models, _slides(cfg, args))
    return _run_jobs(cfg, args, f"extract{stage}", jobs, _gpu(cfg), cfg.extract.bin_overhead_s)


def cmd_classify(cfg: BenchConfig, args) -> int:
    from bench.probe import run as probe
    stage = args.stage
    models = cfg.enabled_models(args.models) if stage == "1" else _gate_models(cfg, args)
    jobs = probe.plan_jobs(cfg, stage, models, _slides(cfg, args))
    if args.schemes:
        jobs = [j for j in jobs if j["scheme"] in args.schemes]
    mem = min(max([j["mem_gb"] for j in jobs] + [cfg.slurm.mem_gb]), args.max_mem_gb)
    return _run_jobs(cfg, args, f"classify{stage}", jobs, _gpu(cfg, mem), 120)


def cmd_scaling(cfg: BenchConfig, args) -> int:
    from bench.probe import run as probe
    _gate_models(cfg, args)
    jobs = probe.plan_jobs(cfg, "3", [], _slides(cfg, args), probe=args.probe)
    mem = min(max([j["mem_gb"] for j in jobs] + [cfg.slurm.mem_gb]), args.max_mem_gb)
    return _run_jobs(cfg, args, "scaling", jobs, _gpu(cfg, mem), 120)


def cmd_genexp(cfg: BenchConfig, args) -> int:
    from bench.genexp import run as genexp
    jobs = genexp.plan_jobs(cfg, cfg.enabled_models(args.models), _slides(cfg, args), readout=args.readout)
    for j in jobs:
        j["group"] = j["key"]   # one fold per array task
    return _run_jobs(cfg, args, "genexp", jobs, _gpu(cfg, cfg.genexp.mem_gb), 120)


def cmd_report(cfg: BenchConfig, args) -> int:
    from pathlib import Path
    from bench.report.site import build_site
    if args.dry_run:
        from bench.report.collect import collect
        d = collect(cfg)
        print(f"{len(d['classify_folds'])} classification folds, {len(d['genexp_folds'])} gene-expression folds")
        return 0
    out = build_site(cfg, Path(args.out) if args.out else None, with_status=not args.no_status)
    print(out)
    return 0


def cmd_status(cfg: BenchConfig, args) -> int:
    from bench.status import run_status
    run_status(cfg, cfg.enabled_models(args.models) if args.models else None, args.stage,
               show_deletable=args.deletable, verbose=args.verbose)
    return 0


def cmd_retry(cfg: BenchConfig, args) -> int:
    from bench.jobs import retry
    return retry(cfg, args.plan, args.dry_run, args.local)


def cmd_calibrate(cfg: BenchConfig, args) -> int:
    from bench.smoke import calibrate
    calibrate(cfg, cfg.enabled_models(args.models), from_status=args.from_status, device=args.device,
              dry_run=args.dry_run)
    return 0


def cmd_smoke(cfg: BenchConfig, args) -> int:
    from bench import smoke
    if args.synthetic:
        return smoke.run_synthetic(args.out, keep=args.keep)
    return smoke.run_real(cfg, args)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch-weights", help="model weights in the HF cache: download or check, record snapshot shas")
    _common(p)
    p.add_argument("--offline", action="store_true", help="no network: check the cache, print `hf download` commands")
    p.set_defaults(fn=cmd_fetch_weights)

    p = sub.add_parser("inspect", help="slide table: WSI path, size, pixel size, scale")
    _common(p)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-strict", action="store_true", help="write the table even if some slides have problems")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("darkspots", help="black/no-data/unreadable block maps (CPU array)")
    _common(p, jobs=True)
    p.set_defaults(fn=cmd_darkspots)

    p = sub.add_parser("manifest", help="global cell manifest with drop reasons and hashes")
    _common(p)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_manifest)

    p = sub.add_parser("extract", help="embeddings (SLURM array)")
    _common(p, jobs=True)
    p.add_argument("--stage", choices=["1", "2a", "2b"], default="1")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("classify", help="linear probes, stage 1 or 2")
    _common(p, jobs=True)
    p.add_argument("--stage", choices=["1", "2", "2a", "2b"], default="1")
    p.add_argument("--schemes", nargs="+", choices=["within", "cross_loco", "single_cohort"])
    p.add_argument("--max-mem-gb", type=int, default=480)
    p.set_defaults(fn=cmd_classify)

    p = sub.add_parser("genexp", help="gene-expression prediction (DeepSpot, leave one patient out)")
    _common(p, jobs=True)
    p.add_argument("--readout", choices=["cls", "central"], help="override genexp.readout (default: config, auto)")
    p.set_defaults(fn=cmd_genexp)

    p = sub.add_parser("scaling", help="stage 3: training-fraction scaling, linear and MLP probes")
    _common(p, jobs=True)
    p.add_argument("--probe", choices=["linear", "mlp"])
    p.add_argument("--max-mem-gb", type=int, default=480)
    p.set_defaults(fn=cmd_scaling)

    p = sub.add_parser("report", help="build the static site")
    _common(p)
    p.add_argument("--out", help="output HTML (default {bench_root}/site/index.html)")
    p.add_argument("--no-status", action="store_true", help="skip the run-status page (faster)")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("status", help="done / running / failed / missing per planned job")
    _common(p)
    p.add_argument("--stage", nargs="+", choices=["1", "2", "3"])
    p.add_argument("--deletable", action="store_true", help="which models' stage-1 embeddings can be deleted")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("retry", help="resubmit the failed/missing bins of a plan")
    _common(p, jobs=True)
    p.add_argument("--plan", required=True, help="plan id or directory")
    p.set_defaults(fn=cmd_retry)

    p = sub.add_parser("calibrate", help="measure throughput -> {bench_root}/calibration.json")
    _common(p)
    p.add_argument("--from-status", action="store_true", help="use finished extraction jobs instead of a GPU benchmark")
    p.add_argument("--device")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("smoke", help="end-to-end check: real (cluster GPU) or --synthetic (CPU, generated data)")
    _common(p)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", help="--synthetic: work directory (default: a temporary directory)")
    p.add_argument("--keep", action="store_true", help="--synthetic: keep the work directory")
    p.add_argument("--device")
    p.add_argument("--skip", nargs="+", default=[], choices=["parity", "genexp", "stage2", "stage3"])
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("run-bin", help=argparse.SUPPRESS)
    p.add_argument("--plan", required=True)
    p.add_argument("--bin", type=int, required=True)

    args = ap.parse_args(argv)
    if args.cmd == "run-bin":
        from bench.jobs import main_run_bin
        return main_run_bin(args.plan, args.bin)
    if args.cmd == "smoke" and args.synthetic:
        setup_logging(args.verbose)
        return cmd_smoke(None, args)
    try:
        return int(args.fn(_cfg(args), args) or 0)
    except EnvironmentError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
