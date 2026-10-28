"""Write configs/train_within_cohort_{cohort}.yaml: within-cohort leave-one-donor-out folds.

One config per cohort (SPLIT, VANNAN, MCCORD, JOHNSON; 10X is not included). Fold i
of a cohort's config: test = every slide of one donor; val = the cohort's val slides from
configs/train_cross_cohort.yaml (same donors in every fold); train = the cohort's other
slides. Donors (splits.csv donor_key) are never split across train/val/test, e.g. SPLIT's
lung donor L1 (4 slides on 3 panels) is always tested as a whole. Slides, val donors and
everything else come from scripts/make_cross_cohort_config.py, so rerun that first if it
changes.

Usage (repo root):
    python scripts/make_within_cohort_configs.py [--splits-csv _metadata_work/splits.csv]
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_cross_cohort_config as cc  # noqa: E402

REPO = cc.REPO

# make_splits.py source -> (cohort name, cohorts-map patterns + comment), in config order.
COHORTS = {
    "NM": ("SPLIT", '["GSM9509*"]              # breast + lung cancer (SPLIT, Nature Methods)'),
    "VANNAN": ("VANNAN", '["GSM7990*"]             # pulmonary fibrosis TMAs, 20x'),
    "MCCORD": ("MCCORD", '["GSM9054*"]             # HNSCC, fresh frozen'),
    "JOHNSON": ("JOHNSON", '["fetal_*", "tHIO_*"]   # fetal small intestine (+ organoid transplant), 20x'),
}


def _cross_cohort_rows(splits_csv: Path) -> list[dict]:
    """E4_loso rows after make_cross_cohort_config.py's exclusions and val overrides
    (same filtering as its main)."""
    with open(splits_csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["experiment"] == "E4_loso"]
    rows = [r for r in rows if cc._wsi_name(r["xenium_folder"]) not in cc.EXCLUDED_WSIS]
    for r in rows:
        if r["source"] in cc.VAL_DONOR_OVERRIDES and r["split"] in ("train", "val"):
            r["split"] = "val" if r["donor_key"] in cc.VAL_DONOR_OVERRIDES[r["source"]] else "train"
    return rows


def _cohort_donors(rows: list[dict], source: str) -> tuple[dict[str, list[str]], set[str]]:
    """(donor_key -> sorted WSI names, val donor keys) of one cohort. Val donors are the
    cohort's val donors in the cross-cohort folds (identical in every fold it isn't test)."""
    donors: dict[str, set[str]] = defaultdict(set)
    val_by_fold: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["source"] != source:
            continue
        donors[r["donor_key"]].add(cc._wsi_name(r["xenium_folder"]))
        if r["split"] == "val":
            val_by_fold[r["fold"]].add(r["donor_key"])
    vals = {frozenset(v) for v in val_by_fold.values()}
    assert len(vals) == 1, f"{source}: val donors differ across cross-cohort folds: {vals}"
    return {d: sorted(w) for d, w in donors.items()}, set(next(iter(vals)))


def _donor_label(donor_key: str) -> str:
    return cc._wsi_name(donor_key.split("::", 1)[-1])  # JOHNSON's donor_key is its folder


def _splits_block(cohort: str, donors: dict[str, list[str]], val_donors: set[str]) -> str:
    val_wsis = sorted(w for d in val_donors for w in donors[d])
    test_donors = sorted(set(donors) - val_donors, key=lambda d: (donors[d][0], d))
    lines = [
        f"# Scheme: within-cohort leave-one-donor-out over {cohort} -- regenerate with",
        "# scripts/make_within_cohort_configs.py. Fold i: test = every slide of one donor;",
        f"# val = {', '.join(sorted(_donor_label(d) for d in val_donors))} (the cohort's val slides in",
        "# configs/train_cross_cohort.yaml, fixed across folds); train = the cohort's other",
        "# slides. A donor's slides are never split across train/val/test.",
        "splits:",
    ]
    for i, test_donor in enumerate(test_donors):
        train = sorted(w for d in test_donors if d != test_donor for w in donors[d])
        test = donors[test_donor]
        lines.append(
            f"  # split {i}: test = {_donor_label(test_donor)} ({len(test)} slide{'s' if len(test) > 1 else ''}); "
            f"val = {len(val_wsis)} slide{'s' if len(val_wsis) > 1 else ''}; train = {len(train)} slides"
        )
        for j, (side, wsis) in enumerate((("train", train), ("val", val_wsis), ("test", test))):
            lines.append(f"  {'- ' if j == 0 else '  '}{side}:")
            lines.extend(f"      {w}: 1.0" for w in wsis)
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits-csv", type=Path, default=REPO / "_metadata_work" / "splits.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO / "configs")
    args = parser.parse_args()

    rows = _cross_cohort_rows(args.splits_csv)
    cross = (REPO / "configs" / "train_cross_cohort.yaml").read_text()

    for source, (cohort, patterns) in COHORTS.items():
        donors, val_donors = _cohort_donors(rows, source)
        text = cross
        head, rest = text.split("# Scheme:", 1)
        _, tail = rest.split("# When true, the launcher also submits", 1)
        text = head + _splits_block(cohort, donors, val_donors) + "\n# When true, the launcher also submits" + tail

        text = cc._replace_once(
            text,
            "# Leave-one-cohort-out (cross-cohort) classifier evaluation -- generated by\n"
            "# scripts/make_cross_cohort_config.py from configs/train.yaml; edit there or here.\n",
            f"# Within-cohort ({cohort}) leave-one-donor-out classifier evaluation -- generated by\n"
            "# scripts/make_within_cohort_configs.py from configs/train_cross_cohort.yaml.\n",
        )
        # One experiment per cohort: split_<i> of different cohorts must not share a folder.
        text = cc._replace_once(text, "experiment: cross_cohort\n", f"experiment: within_cohort_{cohort}\n")
        text = cc._replace_once(text, cc.COHORTS_BLOCK, f"cohorts:\n  {cohort}: {patterns}\n")
        text = cc._replace_once(
            text,
            "  experiment_name: embeddings_eval_cross_cohort\n",
            f"  experiment_name: embeddings_eval_within_cohort_{cohort}\n",
        )
        # A single cohort: cohort balancing would be a no-op, plain slide mean says what it does.
        text = cc._replace_once(
            text, "  selection_metric: cohort_balanced_slide_macro_f1\n", "  selection_metric: slide_macro_f1\n"
        )
        # Folds are one cohort, far smaller than cross-cohort's ~100 slides: estimate wall
        # time from the cell counts again (configs/train.yaml's default).
        text = cc._replace_once(text, "  auto_time: false\n", "  auto_time: true\n")
        text = re.sub(r"  mem: 256G[^\n]*\n", "  mem: 256G\n", text, count=1)

        out = args.out_dir / f"train_within_cohort_{cohort.lower()}.yaml"
        out.write_text(text)
        n_val = sum(len(donors[d]) for d in val_donors)
        print(f"wrote {out} ({len(donors) - len(val_donors)} folds, "
              f"{sum(map(len, donors.values()))} slides, {n_val} val)")


if __name__ == "__main__":
    main()
