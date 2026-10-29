"""Write a self-contained HTML report of classifier results, slide by slide.

Scans a results folder (default: all of outputs_classifier) for experiment.py runs: every
folder whose split_k/ subfolders hold results_summary.yaml + per_slide_metrics.csv. A run is
identified from its config_resolved.yaml (experiment, model, correction, mapping, matching,
read-out), not from the folder layout. The within_cohort_* experiments (one per cohort,
leave-one-donor-out) are merged into one "within-cohort" run per read-out, so each slide can be
compared between holding out its cohort (cross_cohort) and holding out only its donor. Several
runs (schemes, read-outs) are shown side by side. Slide metadata (tissue, disease, donor, H&E)
comes from xenium_metadata.csv. The page needs no network: open the file in a browser.

Usage (repo root):
    python scripts/make_results_report.py [--results-dir outputs_classifier] \
        [--experiments cross_cohort 'within_cohort_*'] [--out outputs_classifier/results_report.html]
"""
import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime
from fnmatch import fnmatch
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
DEFAULT_EXPERIMENTS = ("cross_cohort", "within_cohort_*")
SCHEME_LABELS = {"cross_cohort": "cross-cohort", "within_cohort": "within-cohort"}
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


def _scheme(experiment: str) -> str:
    """within_cohort_SPLIT, within_cohort_VANNAN, ... are one scheme, merged into one run."""
    return "within_cohort" if experiment.startswith("within_cohort") else experiment


def _wsi_cohort(wsi: str, cohorts: dict | None) -> str | None:
    return next((c for c, pats in (cohorts or {}).items() if any(fnmatch(wsi, p) for p in pats)), None)


def _metrics(d: dict | None, keys=SUMMARY_METRICS) -> dict:
    return {k: _num(d[k]) for k in keys if d and k in d}


def _curve(path: Path) -> list | None:
    if not path.exists():
        return None
    return [_num(v) for v in np.load(path, allow_pickle=False).ravel().tolist()]


def _held_out(fold_idx: int, experiment: str, test_wsis: list[str], test_cohorts: list[str],
              train_cohorts: set, by_folder: dict) -> dict:
    """Fold name and what it holds out: a whole cohort (none of its slides in train) or, when
    the test cohort is also in train (within-cohort), a donor."""
    if test_cohorts and not set(test_cohorts) & train_cohorts:
        return {"unit": "cohort", "name": f"Fold {fold_idx}", "held_out": "+".join(test_cohorts)}
    donors = {(_slide_meta(by_folder, w) or {}).get("donor") for w in test_wsis}
    if len(donors) == 1 and None not in donors:
        held = f"donor {donors.pop()}"
    else:
        held = test_wsis[0] if len(test_wsis) == 1 else f"{len(test_wsis)} slides"
    prefix = "+".join(test_cohorts) + " fold" if test_cohorts else experiment + " fold"
    return {"unit": "donor", "name": f"{prefix} {fold_idx}", "held_out": held}


def _read_fold(fold_dir: Path, experiment: str, cohorts: dict | None, by_folder: dict) -> tuple[dict, list[dict]]:
    s = _load_yaml(fold_dir / "results_summary.yaml")
    class_names = s["class_names"]  # this fold's model classes (a within-cohort model may lack some)
    info = _load_yaml(fold_dir / "split_informations.yaml") if (fold_dir / "split_informations.yaml").exists() else {}
    cms = _load_yaml(fold_dir / "per_slide_confusion_matrices.yaml") if (fold_dir / "per_slide_confusion_matrices.yaml").exists() else {}
    fold_idx = int(fold_dir.name.split("_")[-1])
    fold_id = f"{experiment}/{fold_idx}"

    slides = []
    with open(fold_dir / "per_slide_metrics.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            slides.append({
                "set": r["set"], "fold": fold_id, "wsi": r["wsi"], "cohort": r.get("cohort") or None,
                "n_cells": int(float(r["n_cells"])),
                "n_dropped": int(float(r.get("n_dropped_absent_from_train") or 0)),
                **{k: _num(r.get(k)) for k in SLIDE_METRICS},
                "f1": [_num(r.get(f"f1_{c}")) for c in class_names],
                "n": [int(float(r.get(f"n_{c}") or 0)) for c in class_names],
                "cm": (cms.get(r["set"]) or {}).get(r["wsi"]),
            })

    test_wsis = sorted(info.get("test") or {x["wsi"] for x in slides if x["set"] == "test"})
    test_cohorts = sorted({x["cohort"] for x in slides if x["set"] == "test" and x["cohort"]})
    train_cohorts = {_wsi_cohort(w, cohorts) for w in info.get("train") or {}} - {None}
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
        "id": fold_id,
        "experiment": experiment,
        "fold": fold_idx,
        **_held_out(fold_idx, experiment, test_wsis, test_cohorts, train_cohorts, by_folder),
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


def _expand(cm: list | None, idx: list) -> list | None:
    if cm is None:
        return None
    return [[cm[a][b] if a is not None and b is not None else 0 for b in idx] for a in idx]


def _to_classes(fold: dict, slides: list[dict], classes: list[str]) -> None:
    """Re-index a fold's per-class arrays and confusion matrices onto the report's class list.
    A within-cohort model only knows its cohort's classes; `in_model` marks them."""
    own = fold["class_names"]
    idx = [own.index(c) if c in own else None for c in classes]
    fold["in_model"] = [i is not None for i in idx]
    for side in ("test", "val"):
        fold[side]["cm"] = _expand(fold[side]["cm"], idx)
    for x in slides:
        x["f1"] = [x["f1"][i] if i is not None else None for i in idx]
        x["n"] = [x["n"][i] if i is not None else 0 for i in idx]
        x["cm"] = _expand(x["cm"], idx)
        x["in_model"] = fold["in_model"]


def _settings(cfg: dict) -> dict:
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


def _run_identity(run_dir: Path, results_dir: Path, cfg: dict) -> tuple[str, tuple[str, ...]]:
    """(experiment, (scheme, model, correction, mapping, matching, readout)) of a run folder.
    The folder layout is only a fallback: copies of runs need not keep it."""
    parts = run_dir.relative_to(results_dir).parts
    data = cfg.get("data") or {}
    if cfg.get("experiment") and data.get("model_name") and data.get("embeddings_datasets"):
        experiment = cfg["experiment"]
        return experiment, (
            _scheme(experiment), data["model_name"], data.get("correction_name") or "raw", cfg.get("mapping") or "?",
            "matching" if data.get("consider_matching") else "all_cells", "_".join(sorted(data["embeddings_datasets"])),
        )
    experiment = parts[0] if len(parts) == 6 else results_dir.name
    return experiment, (_scheme(experiment), *(parts[-5:] if len(parts) >= 5 else parts))


def _find_runs(results_dir: Path) -> list[Path]:
    return sorted({p.parent.parent for p in results_dir.rglob("split_*/results_summary.yaml")})


def _read_metadata(path: Path) -> dict:
    """xenium_folder -> report metadata fields (blanks and 'unknown' become None)."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["xenium_folder"]: {k: (None if r.get(col) in (None, "", "NA", "unknown") else r[col])
                                     for k, col in META_COLUMNS.items()} for r in csv.DictReader(fh)}


def _slide_meta(by_folder: dict, wsi: str) -> dict | None:
    return next((by_folder[k] for k in (wsi, f"{wsi}_out", f"{wsi}_outs") if k in by_folder), None)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=REPO / "outputs_classifier")
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS),
                        help="config `experiment` names to include (shell patterns); default: %(default)s")
    parser.add_argument("--out", type=Path, default=None, help="default: {results-dir}/results_report.html")
    parser.add_argument("--metadata", type=Path, default=REPO / "xenium_metadata.csv")
    parser.add_argument("--title", default=None, help="default: from the schemes found")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    out = args.out or results_dir / "results_report.html"
    by_folder = _read_metadata(args.metadata)

    # Group run folders by identity: the within_cohort_* experiments of one read-out become one run.
    groups: dict[tuple, list] = defaultdict(list)
    skipped = defaultdict(int)
    for run_dir in _find_runs(results_dir):
        fold_dirs = sorted((d for d in run_dir.glob("split_*") if (d / "results_summary.yaml").exists()),
                           key=lambda d: int(d.name.split("_")[-1]))
        cfg_path = fold_dirs[0] / "config_resolved.yaml"
        cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
        experiment, key = _run_identity(run_dir, results_dir, cfg)
        if not any(fnmatch(experiment, p) for p in args.experiments):
            skipped[experiment] += 1
            continue
        if cfg.get("same_wsi_split"):  # splits slides into train/test: not comparable to held-out slides
            skipped[f"{experiment} (same_wsi_split)"] += 1
            continue
        groups[key].append((experiment, run_dir, fold_dirs, cfg))
    if not groups:
        raise SystemExit(f"no split_*/results_summary.yaml of experiments {args.experiments} under {results_dir}"
                         + (f" (skipped: {', '.join(skipped)})" if skipped else ""))

    # Cohort order: the cohorts map with the most cohorts (cross_cohort's) sets it.
    cohort_order = list(max((cfg.get("cohorts") or {} for g in groups.values() for *_, cfg in g), key=len))
    def exp_order(name: str, cfg: dict) -> tuple:
        cohorts = list(cfg.get("cohorts") or {})
        first = min((cohort_order.index(c) for c in cohorts if c in cohort_order), default=99)
        return (_scheme(name) != "cross_cohort", _scheme(name), first, name)

    runs, experiments = [], {}
    for key, members in groups.items():
        members.sort(key=lambda m: exp_order(m[0], m[3]))
        folds, incomplete, run_exps = [], [], []
        for experiment, run_dir, fold_dirs, cfg in members:
            for fold_dir in fold_dirs:
                fold, fold_slides = _read_fold(fold_dir, experiment, cfg.get("cohorts"), by_folder)
                folds.append((fold, fold_slides))
            found = {int(d.name.split("_")[-1]) for d in fold_dirs}
            expected = len(cfg.get("splits") or []) or max(found) + 1
            missing = [f"split_{i}" for i in range(expected) if i not in found]
            incomplete += [f"{experiment} {m}" for m in missing]
            run_exps.append({"name": experiment, "n_folds": len(found), "n_expected": expected,
                             "path": str(run_dir.relative_to(results_dir))})
            if experiment not in experiments:
                scheme = _scheme(experiment)
                if scheme == "within_cohort":
                    short = "+".join(cfg.get("cohorts") or {}) or experiment
                    label = f"within-cohort {short}"
                else:
                    short = label = SCHEME_LABELS.get(scheme, experiment)
                experiments[experiment] = {"name": experiment, "scheme": scheme, "short": short, "label": label,
                                           "order": exp_order(experiment, cfg)}
        runs.append({
            "id": "/".join(key),
            "parts": list(key),
            **dict(zip(("scheme", "model", "correction", "mapping", "matching", "readout"), key)),
            "scheme_label": SCHEME_LABELS.get(key[0], key[0]),
            "readout_label": _readout_label(key[-1]),
            "experiments": run_exps,
            "folds": folds,
            "incomplete_folds": incomplete,
            "settings": _settings(members[0][3]),
        })

    # One class list for the report: the largest model's, in its order. Within-cohort models lack
    # the classes their cohort has no cells of (e.g. VANNAN: no Malignant).
    own_lists = [f["class_names"] for r in runs for f, _ in r["folds"]]
    longest = max(own_lists, key=len)
    class_names = list(longest) + sorted({c for cs in own_lists for c in cs} - set(longest))
    for r in runs:
        for fold, fold_slides in r["folds"]:
            _to_classes(fold, fold_slides, class_names)
        r["slides"] = [x for _, fold_slides in r["folds"] for x in fold_slides]
        r["folds"] = [fold for fold, _ in r["folds"]]
        r["model_classes"] = sorted({tuple(f["class_names"]) for f in r["folds"]}, key=len, reverse=True)

    # Scheme first (cross-cohort before within-cohort), then 4-corner tokens before other read-outs.
    runs.sort(key=lambda r: (r["scheme"] != "cross_cohort", r["scheme"], r["parts"][-1] != CORNER_READOUT, r["parts"]))
    # Label each run by the identity parts that differ between runs (all of them when there is one).
    n_parts = {len(r["parts"]) for r in runs}
    varying = (
        [i for i in range(max(n_parts)) if len({r["parts"][i] for r in runs}) > 1]
        if len(n_parts) == 1 and len(runs) > 1 else None
    )
    for r in runs:
        pretty = [r["scheme_label"], *r["parts"][1:-1], r["readout_label"]]
        r["full_label"] = " · ".join(pretty)
        r["label"] = " · ".join(pretty[i] for i in varying) if varying else r["full_label"]

    schemes = list(dict.fromkeys(r["scheme"] for r in runs))
    scheme_names = [SCHEME_LABELS.get(s, s.replace("_", "-")) for s in schemes]
    all_wsis = {x["wsi"] for r in runs for x in r["slides"]}
    data = {
        "title": args.title or (" vs ".join(scheme_names) + " results").capitalize(),
        "experiment": "_vs_".join(schemes),
        "results_dir": str(results_dir.relative_to(REPO)) if results_dir.is_relative_to(REPO) else str(results_dir),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "report_commit": _git_commit(),
        "class_names": class_names,
        "cohort_order": cohort_order,
        "experiments": [{k: v for k, v in e.items() if k != "order"}
                        for e in sorted(experiments.values(), key=lambda e: e["order"])],
        "runs": runs,
        "meta": {w: m for w in all_wsis if (m := _slide_meta(by_folder, w))},
    }
    # Inside <script type="application/json">: no '<' may survive, or slide names could end the tag.
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    html = TEMPLATE.read_text(encoding="utf-8")
    assert html.count("__REPORT_DATA__") == 1
    html = html.replace("__REPORT_TITLE__", data["title"].replace("&", "&amp;").replace("<", "&lt;"))
    html = html.replace("__REPORT_DATA__", payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    print(f"wrote {out}  ({len(runs)} run(s); {len(data['meta'])}/{len(all_wsis)} slides with metadata)")
    for r in runs:
        n_test = len({x["wsi"] for x in r["slides"] if x["set"] == "test"})
        exps = ", ".join(f"{e['name']} {e['n_folds']}/{e['n_expected']}" for e in r["experiments"])
        print(f"  {r['label']}: {n_test} test slides; folds {exps}")
        if r["incomplete_folds"]:
            print(f"    no results_summary.yaml yet: {', '.join(r['incomplete_folds'])}")
    if skipped:
        print(f"  skipped (not in --experiments {' '.join(args.experiments)}, or same_wsi_split): "
              + ", ".join(f"{k} ({v} run folder{'s' if v > 1 else ''})" for k, v in sorted(skipped.items())))


if __name__ == "__main__":
    main()
