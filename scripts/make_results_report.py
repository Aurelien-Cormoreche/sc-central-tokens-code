"""Write a self-contained HTML report of classifier results, slide by slide.

Scans an experiment's output folder (e.g. outputs_classifier/cross_cohort) for runs: every
{model}/{correction}/{mapping}/{matching}/{readout}/ folder whose split_k/ subfolders hold
experiment.py's results_summary.yaml + per_slide_metrics.csv. Several runs (e.g. the corner-token
and CLS read-outs of one model) are shown side by side. Slide metadata (tissue, disease, donor,
H&E) comes from xenium_metadata.csv. The page needs no network: open the file in a browser.

Usage (repo root):
    python scripts/make_results_report.py [--results-dir outputs_classifier/cross_cohort] \
        [--out outputs_classifier/cross_cohort/results_report.html]
"""
import argparse
import csv
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "results_report_template.html"

try:
    _YAML_LOADER = yaml.CSafeLoader
except AttributeError:
    _YAML_LOADER = yaml.SafeLoader

CORNER_READOUT = "bottom_left_bottom_right_top_left_top_right"
SLIDE_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_f1_present", "weighted_f1", "ce_loss")
SUMMARY_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_f1_present", "weighted_f1", "loss", "ce_loss")
META_COLUMNS = {
    "tissue": "tissue_of_origin", "site": "anatomical_site", "disease_state": "disease_state",
    "disease_type": "disease_type", "subtype": "subtype", "age": "developmental_stage_or_age",
    "donor": "donor_patient_id", "study": "study_name", "publication": "publication_name",
    "doi": "paper_doi", "preservation": "preservation", "tissue_format": "tissue_format",
    "chemistry": "chemistry", "panel": "panel_name", "panel_size": "panel_size",
    "he_magnification": "he_magnification", "he_scanner": "he_scanner", "he_pixel_size_um": "he_pixel_size_um",
}


def _load_yaml(path: Path):
    with open(path, encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_YAML_LOADER)


def _num(x):
    """JSON-safe number: NaN/inf (and blanks) become null."""
    if x is None or x == "":
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _readout_label(readout: str) -> str:
    if readout == CORNER_READOUT:
        return "4 corner tokens"
    if readout == "cls":
        return "CLS"
    return readout.replace("_", " ")


def _metrics(d: dict | None, keys=SUMMARY_METRICS) -> dict:
    return {k: _num(d[k]) for k in keys if d and k in d}


def _curve(path: Path) -> list | None:
    if not path.exists():
        return None
    return [_num(v) for v in np.load(path, allow_pickle=False).ravel().tolist()]


def _read_fold(fold_dir: Path, class_names_hint: list[str] | None) -> tuple[dict, list[dict]]:
    s = _load_yaml(fold_dir / "results_summary.yaml")
    class_names = s.get("class_names") or class_names_hint
    info = _load_yaml(fold_dir / "split_informations.yaml") if (fold_dir / "split_informations.yaml").exists() else {}
    cms = _load_yaml(fold_dir / "per_slide_confusion_matrices.yaml") if (fold_dir / "per_slide_confusion_matrices.yaml").exists() else {}
    fold_idx = int(fold_dir.name.split("_")[-1])

    slides = []
    with open(fold_dir / "per_slide_metrics.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            slides.append({
                "set": r["set"], "fold": fold_idx, "wsi": r["wsi"], "cohort": r.get("cohort") or None,
                "n_cells": int(float(r["n_cells"])),
                "n_dropped": int(float(r.get("n_dropped_absent_from_train") or 0)),
                **{k: _num(r.get(k)) for k in SLIDE_METRICS},
                "f1": [_num(r.get(f"f1_{c}")) for c in class_names],
                "n": [int(float(r.get(f"n_{c}") or 0)) for c in class_names],
                "cm": (cms.get(r["set"]) or {}).get(r["wsi"]),
            })

    test_cohorts = sorted({x["cohort"] for x in slides if x["set"] == "test" and x["cohort"]})
    sel = s.get("selection") or {}
    trials = [{
        "loss_weights": t.get("loss_weights"), "learning_rate": _num(t.get("learning_rate")),
        "dropout": _num(t.get("dropout")), "hidden_dim": t.get("hidden_dim"),
        "normalization": t.get("normalization"), "pca_components": t.get("pca_components"),
        "val_selection_score": _num(t.get("val_selection_score")), "test_macro_f1": _num(t.get("test_macro_f1")),
        "test_accuracy": _num(t.get("test_accuracy")), "selected_epoch": t.get("selected_epoch"),
        "epochs_trained": t.get("epochs_trained"), "trial_seconds": _num(t.get("trial_seconds")),
    } for t in s.get("all_trial_results") or []]
    fold = {
        "fold": fold_idx,
        "test_cohorts": test_cohorts,
        "class_names": class_names,
        "n_slides": {side: len(info.get(side) or {}) for side in ("train", "val", "test")},
        "test_size": info.get("test_size"),
        "best_config": s.get("best_config"),
        "selection": {k: (_num(v) if isinstance(v, float) else v) for k, v in sel.items()},
        "test": {
            "pooled": _metrics(s.get("best_test_metrics")),
            "slide_mean": _metrics(s.get("best_test_slide_mean_metrics")),
            "cohort_mean": _metrics(s.get("best_test_cohort_mean_metrics")),
            "cm": s.get("test_confusion_matrix"),
        },
        "val": {
            "pooled": _metrics(s.get("best_val_metrics")),
            "slide_mean": _metrics(s.get("best_val_slide_mean_metrics")),
            "cohort_mean": _metrics(s.get("best_val_cohort_mean_metrics")),
            "cm": s.get("val_confusion_matrix"),
        },
        "trials": trials,
        "curves": {
            "train_loss": _curve(fold_dir / "best_train_loss_curve.npy"),
            "val_loss": _curve(fold_dir / "best_val_loss_curve.npy"),
            "val_score": _curve(fold_dir / "best_val_score_curve.npy"),
        },
        "timing": {k: _num(v) for k, v in (s.get("timing") or {}).items()},
        "provenance": {k: v for k, v in (s.get("provenance") or {}).items() if k != "saved_model_dir"},
    }
    return fold, slides


def _settings(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        return {}
    cfg = _load_yaml(cfg_path)
    data, training = cfg.get("data") or {}, cfg.get("training") or {}
    hparams = {
        name: (spec.get("values") if spec.get("tune") else (spec.get("values") or [None])[0])
        for name, spec in (cfg.get("hparams") or {}).items()
    }
    return {
        "mapping": cfg.get("mapping"), "seed": cfg.get("seed"),
        "embeddings_datasets": data.get("embeddings_datasets"),
        "consider_matching": data.get("consider_matching"), "max_cells_per_wsi": data.get("max_cells_per_wsi"),
        "num_epochs": training.get("num_epochs"), "batch_size": training.get("batch_size"),
        "early_stopping": training.get("early_stopping"), "selection_metric": training.get("selection_metric"),
        "hparams": hparams,
    }


def _find_runs(results_dir: Path) -> list[Path]:
    return sorted({p.parent.parent for p in results_dir.rglob("split_*/results_summary.yaml")})


def _read_metadata(path: Path, wsis: set[str]) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        by_folder = {r["xenium_folder"]: r for r in csv.DictReader(fh)}
    out = {}
    for w in wsis:
        row = next((by_folder[k] for k in (w, f"{w}_out", f"{w}_outs") if k in by_folder), None)
        if row is not None:
            out[w] = {k: (None if row.get(col) in (None, "", "NA", "unknown") else row[col]) for k, col in META_COLUMNS.items()}
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=REPO / "outputs_classifier" / "cross_cohort")
    parser.add_argument("--out", type=Path, default=None, help="default: {results-dir}/results_report.html")
    parser.add_argument("--metadata", type=Path, default=REPO / "xenium_metadata.csv")
    parser.add_argument("--title", default=None, help="default: '<experiment> results'")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    out = args.out or results_dir / "results_report.html"

    run_dirs = _find_runs(results_dir)
    if not run_dirs:
        raise SystemExit(f"no split_*/results_summary.yaml under {results_dir}")

    runs, all_wsis = [], set()
    for run_dir in run_dirs:
        parts = run_dir.relative_to(results_dir).parts
        folds, slides, class_names = [], [], None
        fold_dirs = sorted((d for d in run_dir.glob("split_*") if (d / "results_summary.yaml").exists()),
                           key=lambda d: int(d.name.split("_")[-1]))
        for fold_dir in fold_dirs:
            fold, fold_slides = _read_fold(fold_dir, class_names)
            class_names = class_names or fold["class_names"]
            assert fold["class_names"] == class_names, f"{fold_dir}: class names differ from the run's other folds"
            folds.append(fold)
            slides.extend(fold_slides)
        all_wsis.update(x["wsi"] for x in slides)
        incomplete = sorted(d.name for d in run_dir.glob("split_*") if not (d / "results_summary.yaml").exists())
        named = dict(zip(("model", "correction", "mapping", "matching", "readout"), parts)) if len(parts) == 5 else {}
        runs.append({
            "id": "/".join(parts),
            "parts": list(parts),
            **named,
            "readout_label": _readout_label(named["readout"]) if named else None,
            "class_names": class_names,
            "folds": folds,
            "incomplete_folds": incomplete,
            "settings": _settings(fold_dirs[0] / "config_resolved.yaml"),
            "slides": slides,
        })

    # Label each run by the path parts that differ between runs (all of them when there is one).
    n_parts = {len(r["parts"]) for r in runs}
    varying = (
        [i for i in range(max(n_parts)) if len({r["parts"][i] for r in runs}) > 1]
        if len(n_parts) == 1 and len(runs) > 1 else None
    )
    for r in runs:
        pretty = [_readout_label(p) if i == len(r["parts"]) - 1 else p for i, p in enumerate(r["parts"])]
        r["full_label"] = " · ".join(pretty)
        r["label"] = " · ".join(pretty[i] for i in varying) if varying else r["full_label"]
    class_names = runs[0]["class_names"]
    assert all(r["class_names"] == class_names for r in runs), "runs use different class sets"

    experiment = results_dir.name
    data = {
        "title": args.title or f"{experiment.replace('_', '-').capitalize()} results",
        "experiment": experiment,
        "results_dir": str(results_dir.relative_to(REPO)) if results_dir.is_relative_to(REPO) else str(results_dir),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "report_commit": _git_commit(),
        "class_names": class_names,
        "runs": runs,
        "meta": _read_metadata(args.metadata, all_wsis),
    }
    # Inside <script type="application/json">: no '<' may survive, or slide names could end the tag.
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    html = TEMPLATE.read_text(encoding="utf-8")
    assert html.count("__REPORT_DATA__") == 1
    html = html.replace("__REPORT_TITLE__", data["title"].replace("&", "&amp;").replace("<", "&lt;"))
    html = html.replace("__REPORT_DATA__", payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    n_slides = len({(x["set"], x["fold"], x["wsi"]) for r in runs for x in r["slides"] if x["set"] == "test"})
    print(f"wrote {out}  ({len(runs)} run(s): {', '.join(r['label'] for r in runs)}; "
          f"{n_slides} test slides; {len(data['meta'])}/{len(all_wsis)} slides with metadata)")
    for r in runs:
        if r["incomplete_folds"]:
            print(f"  {r['id']}: no results_summary.yaml yet in {', '.join(r['incomplete_folds'])}")


if __name__ == "__main__":
    main()
