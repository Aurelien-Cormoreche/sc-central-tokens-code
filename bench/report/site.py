"""`bench report`: one self-contained static page, {bench_root}/site/index.html.

All data is embedded as JSON; the page works from file:// with no network access (no external
scripts, styles or fonts).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from bench.config import BenchConfig
from bench.provenance import git_state
from bench.report import collect as C
from bench.report import gates
from bench.utils import LOG, atomic_path, dumps

TEMPLATE = Path(__file__).parent / "templates" / "site.html"


def build_data(cfg: BenchConfig, with_status: bool = True) -> dict:
    data = C.collect(cfg)
    g1 = gates.gate1(cfg, data["classify_summary"])
    g2 = gates.gate2(cfg, data["classify_summary"])
    status = []
    if with_status:
        from bench.status import job_states, planned_jobs
        try:
            status = job_states(cfg, planned_jobs(cfg))
        except SystemExit as e:
            LOG.warning("status not available: %s", e)
    g = git_state()
    data["meta"] = {
        "generated": datetime.now().isoformat(timespec="seconds"), "git_commit": g["commit"], "git_dirty": g["dirty"],
        "config_source": cfg.source, "smoke": cfg.is_smoke, "models": cfg.enabled_models(), "readouts": cfg.readouts,
        "cohorts": cfg.cohorts, "within_cohorts": cfg.classify.within_cohorts,
        "gate1": {**g1, "all_models_complete": bool(g1["all_models_complete"])}, "gate2": g2,
        "stage2": cfg.stage2.model_dump(), "stage3": cfg.stage3.model_dump(), "genexp_top": cfg.genexp.top, "genexp_num_genes": cfg.genexp.num_genes,
        "selection_metric": cfg.classify.selection_metric,
    }
    data["status"] = status
    return data


def build_site(cfg: BenchConfig, out: Path | None = None, with_status: bool = True) -> Path:
    data = build_data(cfg, with_status)
    blob = dumps(data).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__BENCH_DATA__*/null", blob)
    out = out or cfg.root / "site" / "index.html"
    with atomic_path(out) as tmp:
        tmp.write_text(html, encoding="utf-8")
    n = len(data["classify_folds"])
    LOG.info("wrote %s (%d classification folds, %d gene-expression folds, %d jobs, %.1f MB)", out, n,
             len(data["genexp_folds"]), len(data["status"]), len(html) / 1e6)
    return out
