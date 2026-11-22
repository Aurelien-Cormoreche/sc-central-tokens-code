"""Gene-expression prediction (`bench genexp`) with DeepSpot's DeepCell, leave-one-patient-out.

Job = (model, slide set, fold). Each job builds a DeepSpot config from genexp.template (training,
hparams, data defaults) pointing at {bench_root}/embeddings/stage1/{model}/native, computes the
fold's train-HVG table once (model-independent, shared by every model), runs
deepspot_training.experiment.run_split_experiment with best-epoch restore, and condenses the
outputs into one fold JSON: per-gene Pearson and Spearman, means over all trained genes and over
the top-k HVGs (genexp.top), per slide.

readout: auto = this model's best Stage-1 readout by the Gate-1 rule (needs its Stage-1
within-cohort classification results), recorded in every result.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np

from bench.config import BenchConfig, stable_hash, without_planning
from bench.extract import planner
from bench.extract.store import embeddings_dir
from bench.registry import Slide, genexp_folds, genexp_slides
from bench.utils import LOG, atomic_path, fmt_hours, read_json, write_json


def resolve_readout(cfg: BenchConfig, model: str) -> str:
    if cfg.genexp.readout != "auto":
        return cfg.genexp.readout
    from bench.report.gates import model_readout_scores
    scores = {r: v for (m, r), v in model_readout_scores(cfg).items() if m == model and v.get("complete")}
    if not scores:
        raise SystemExit(f"genexp.readout=auto needs {model}'s Stage-1 within-cohort classification results "
                         f"for every readout: run `bench classify --stage 1 --models {model}` first, "
                         f"or set genexp.readout to cls/central")
    return max(scores, key=lambda r: scores[r]["score"])


def fold_dir(cfg: BenchConfig, set_name: str, model: str, readout: str, i: int) -> Path:
    return cfg.root / "results" / "1" / "genexp" / set_name / model / readout / "native" / f"fold_{i}"


def hvg_dir(cfg: BenchConfig, set_name: str, i: int) -> Path:
    return cfg.root / "genexp" / "hvgs" / set_name / f"fold_{i}"


def plan_jobs(cfg: BenchConfig, models: List[str], slides: List[Slide], readout: Optional[str] = None) -> List[dict]:
    jobs = []
    sets = genexp_slides(cfg, slides)
    for m in models:
        r = readout or resolve_readout(cfg, m)
        for set_name, ss in sets.items():
            folds = genexp_folds(ss, set_name)
            if cfg.is_smoke:
                folds = folds[:1]
            for f in folds:
                n = sum(planner.expected(cfg, "1", s)[0] for s in f.train + f.val + f.test)
                jobs.append({"kind": "genexp", "key": f"genexp/1/{set_name}/{m}/{r}/fold_{f.index}",
                             "group": f"genexp/{m}/{set_name}/{f.index}", "model": m, "readout": r, "set": set_name,
                             "fold_index": f.index, "fold": f.to_dict(), "n_cells": n,
                             "est_s": cfg.genexp.gpu_hours_per_fold * 3600, "mem_gb": cfg.genexp.mem_gb,
                             "out": str(fold_dir(cfg, set_name, m, r, f.index))})
    return jobs


def inputs_hash(cfg: BenchConfig, job: dict) -> str:
    f = job["fold"]
    hashes = {s: planner.expected(cfg, "1", s)[1] for s in f["train"] + f["val"] + f["test"]}
    return stable_hash({"fold": f, "model": job["model"], "readout": job["readout"],
                        "genexp": without_planning(cfg.dump()["genexp"]), "manifest": hashes, "smoke": cfg.is_smoke})


def job_done(cfg: BenchConfig, job: dict) -> bool:
    m = read_json(Path(job["out"]) / "_DONE.json")
    try:
        return bool(m) and m.get("inputs_hash") == inputs_hash(cfg, job)
    except (SystemExit, KeyError):
        return False


def _ensure_hvgs(cfg: BenchConfig, dc, job: dict) -> Path:
    from src.python.deepspot_training.compute_hvgs import compute_split_hvgs
    d = hvg_dir(cfg, job["set"], job["fold_index"])
    out = d / "split_0.csv"
    if out.exists():
        return d
    f = job["fold"]
    genes = compute_split_hvgs(f["train"], f["val"] + f["test"], dc.data.adata_path_template,
                               flavor=dc.data.hvg_flavor, batch_key=dc.data.hvg_batch_key,
                               max_genes=int(dc.data.hvg_max_genes))
    with atomic_path(out) as tmp:
        genes.to_csv(tmp, index=False)
    return d


def deepspot_config(cfg: BenchConfig, job: dict):
    from omegaconf import OmegaConf
    dc = OmegaConf.load(cfg.repo_path(cfg.genexp.template))
    f = job["fold"]
    dc.splits = [{"train": {s: 1.0 for s in f["train"]}, "val": {s: 1.0 for s in f["val"]},
                  "test": {s: 1.0 for s in f["test"]}}]
    dc.same_wsi_split = False
    dc.output_dir = str(cfg.root / "results" / "1" / "genexp")
    d = dc.data
    d.base_dir = str(cfg.root)
    d.embeddings_dir = str(embeddings_dir(cfg.root, "1", job["model"], "native"))
    d.model_name, d.correction_name = job["model"], job["readout"]
    d.embeddings_datasets = [job["readout"]]
    d.consider_matching = False
    d.adata_path_template = str(cfg.paths.get("xenium_processed")) + "/{sample}/adata.h5ad"
    d.num_genes = cfg.genexp.num_genes
    d.gene_interval = sorted(set(cfg.genexp.top))
    d.hvg_max_genes = cfg.genexp.hvg_max_genes
    dc.training.restore_best_epoch = True
    dc.mlflow.tracking_uri = f"file:{cfg.root / 'mlruns'}"
    dc.mlflow.experiment_name = f"bench_genexp_{job['set']}"
    if cfg.is_smoke:
        d.num_genes = min(20, cfg.genexp.num_genes)
        d.gene_interval = [10, d.num_genes]
        dc.training.num_epochs = 1
    return dc


def _condense(cfg: BenchConfig, job: dict, ds_out: Path, seconds: float) -> dict:
    import pandas as pd
    import yaml
    summary = yaml.safe_load(open(ds_out / "results_summary.yaml"))
    genes = pd.read_csv(ds_out / "per_gene_pearson.csv").sort_values("highly_variable_rank").reset_index(drop=True)
    tops = sorted(set(cfg.genexp.top) | {len(genes)})
    test = {}
    for k in tops:
        sub = genes.iloc[:min(k, len(genes))]
        tag = "all" if k >= len(genes) else f"top{k}"
        test[f"mean_pearson_{tag}"] = float(np.nanmean(sub["pearson"])) if len(sub) else None
        if "spearman" in sub:
            test[f"mean_spearman_{tag}"] = float(np.nanmean(sub["spearman"])) if len(sub) else None
    test["median_pearson_all"] = float(np.nanmedian(genes["pearson"]))
    per_slide = pd.read_csv(ds_out / "per_slide_metrics.csv").to_dict("records") \
        if (ds_out / "per_slide_metrics.csv").exists() else []
    return {
        "stage": "1", "task": f"genexp:{job['set']}", "set": job["set"], "model": job["model"],
        "readout": job["readout"], "readout_rule": cfg.genexp.readout, "fold": job["fold"],
        "n_genes": int(len(genes)), "test": test, "best_metrics": summary.get("best_metrics", {}),
        "best_config": summary.get("best_config", {}),
        "per_gene": genes[[c for c in ("gene_name", "highly_variable_rank", "pearson", "spearman") if c in genes]]
        .to_dict("records"),
        "per_slide": per_slide, "seconds": round(seconds, 1),
    }


def run_genexp_job(cfg: BenchConfig, job: dict, device: str) -> None:
    from src.python.deepspot_training.experiment import run_split_experiment
    from bench.provenance import collect
    t0 = time.time()
    dc = deepspot_config(cfg, job)
    dc.data.hvg_dir = str(_ensure_hvgs(cfg, dc, job))
    out = Path(job["out"])
    ds_out = out / "deepspot"
    run_split_experiment(0, dc, device, ds_out)
    rec = _condense(cfg, job, ds_out, time.time() - t0)
    write_json(out / "result.json", rec)
    write_json(out / "_DONE.json", {"inputs_hash": inputs_hash(cfg, job), "provenance": collect(cfg)})


def run_genexp_bin(cfg: BenchConfig, jobs: List[dict], device: str) -> int:
    from bench.jobs import set_status
    failed = 0
    for j in jobs:
        if job_done(cfg, j):
            continue
        set_status(cfg, j, "running")
        t0 = time.time()
        try:
            run_genexp_job(cfg, j, device)
            set_status(cfg, j, "done", seconds=round(time.time() - t0, 1))
            LOG.info("%s done in %s", j["key"], fmt_hours((time.time() - t0) / 3600))
        except Exception as e:
            failed += 1
            LOG.error("%s FAILED: %s", j["key"], e)
            set_status(cfg, j, "failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
    return failed
