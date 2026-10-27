"""Write configs/train_cross_cohort.yaml: leave-one-cohort-out (E4_loso) classifier folds.

Folds come from _metadata_work/splits.csv (written by _metadata_work/make_splits.py,
donor-grouped, leakage-checked): fold i holds out one cohort as test; val is a fixed set
of donors drawn from each of the other four cohorts; train is the rest. Everything
else is taken from configs/train.yaml, with the cross-cohort settings below swapped in.

Usage (repo root):
    python scripts/make_cross_cohort_config.py [--splits-csv _metadata_work/splits.csv]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# make_splits.py's source -> cohort name used in this config (NM is the SPLIT study).
COHORT_NAMES = {"NM": "SPLIT", "VANNAN": "VANNAN", "MCCORD": "MCCORD", "JOHNSON": "JOHNSON", "10X": "10X"}

# splits.csv slides (WSI names) left out of every fold because they have no embeddings:
EXCLUDED_WSIS = {
    # Base-panel runs of two 10x breast blocks; only their _With_Addon runs (serial sections,
    # same donors, also in splits.csv) were extracted.
    "Xenium_V1_FFPE_Human_Breast_IDC",
    "Xenium_V1_FFPE_Human_Breast_ILC",
    # VANNAN VUILD105 (IPF): slide not available.
    "GSM7990546",
}

# Val donors (splits.csv donor_key) replacing E4_loso's val for these cohorts; the cohort's
# other donors go to train in the folds where it trains. One single-slide donor per group:
VAL_DONOR_OVERRIDES = {
    "NM": {"NM::B16", "NM::L7"},                          # breast IDC, lung LUAD
    "VANNAN": {"VANNAN::THD0008", "VANNAN::VUILD106"},    # healthy, IPF (one donor per site)
}

COHORTS_BLOCK = """cohorts:
  SPLIT: ["GSM9509*"]              # breast + lung cancer (SPLIT, Nature Methods)
  VANNAN: ["GSM7990*"]             # pulmonary fibrosis TMAs, 20x
  MCCORD: ["GSM9054*"]             # HNSCC, fresh frozen
  JOHNSON: ["fetal_*", "tHIO_*"]   # fetal small intestine (+ organoid transplant), 20x
  10X: ["Xenium*"]                 # 10x Genomics public datasets, various tissues
"""


def _wsi_name(xenium_folder: str) -> str:
    """splits.csv uses Xenium output folder names; embeddings live under the WSI name
    (same thing without the trailing _out)."""
    return xenium_folder[: -len("_out")] if xenium_folder.endswith("_out") else xenium_folder


def _splits_block(rows: list[dict]) -> str:
    folds: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        folds[int(r["fold"])][r["split"]].append(r)

    lines = [
        "# Scheme: leave-one-cohort-out over the 5 cohorts (see `cohorts`), from",
        "# _metadata_work/splits.csv experiment E4_loso -- regenerate with",
        "# scripts/make_cross_cohort_config.py. Fold i: test = every slide of one cohort;",
        "# val = fixed donors from each of the 4 others (SPLIT: B16 breast + L7 lung; VANNAN:",
        "# THD0008 healthy + VUILD106 IPF; others: splits.csv's every-5th donor); train = the",
        "# remaining slides. Splits are donor-grouped. Left out (no embeddings):",
        f"# {', '.join(sorted(EXCLUDED_WSIS))}.",
        "splits:",
    ]
    for fold in sorted(folds):
        sides = folds[fold]
        held = sorted({COHORT_NAMES[r["source"]] for r in sides["test"]})
        val_cohorts = defaultdict(int)
        for r in sides["val"]:
            val_cohorts[COHORT_NAMES[r["source"]]] += 1
        lines.append(
            f"  # split {fold}: test = {'+'.join(held)} ({len(sides['test'])} slides); "
            f"val = {len(sides['val'])} slides ({', '.join(f'{c} {n}' for c, n in sorted(val_cohorts.items()))}); "
            f"train = {len(sides['train'])} slides"
        )
        for i, side in enumerate(("train", "val", "test")):
            lines.append(f"  {'- ' if i == 0 else '  '}{side}:")
            for wsi in sorted(_wsi_name(r["xenium_folder"]) for r in sides[side]):
                lines.append(f"      {wsi}: 1.0")
        lines.append("")
    return "\n".join(lines) + "\n"


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"configs/train.yaml no longer contains exactly one {old!r}"
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits-csv", type=Path, default=REPO / "_metadata_work" / "splits.csv")
    parser.add_argument("--out", type=Path, default=REPO / "configs" / "train_cross_cohort.yaml")
    args = parser.parse_args()

    with open(args.splits_csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["experiment"] == "E4_loso"]
    assert rows, f"no E4_loso rows in {args.splits_csv}"
    unknown = EXCLUDED_WSIS - {_wsi_name(r["xenium_folder"]) for r in rows}
    assert not unknown, f"excluded WSIs not in splits.csv: {unknown}"
    rows = [r for r in rows if _wsi_name(r["xenium_folder"]) not in EXCLUDED_WSIS]
    for source, donors in VAL_DONOR_OVERRIDES.items():
        missing = donors - {r["donor_key"] for r in rows if r["source"] == source}
        assert not missing, f"val donors not in splits.csv: {missing}"
    for r in rows:
        if r["source"] in VAL_DONOR_OVERRIDES and r["split"] in ("train", "val"):
            r["split"] = "val" if r["donor_key"] in VAL_DONOR_OVERRIDES[r["source"]] else "train"

    text = (REPO / "configs" / "train.yaml").read_text()
    head, rest = text.split("# Scheme:", 1)
    _, tail = rest.split("# When true, the launcher also submits", 1)
    text = head + _splits_block(rows) + "\n# When true, the launcher also submits" + tail

    text = _replace_once(
        text,
        "# Top-level config for the LWO classifier evaluation pipeline.\n",
        "# Leave-one-cohort-out (cross-cohort) classifier evaluation -- generated by\n"
        "# scripts/make_cross_cohort_config.py from configs/train.yaml; edit there or here.\n",
    )
    text = _replace_once(text, "experiment: null\n", "experiment: cross_cohort\n")
    text = _replace_once(text, "cohorts: {}\n", COHORTS_BLOCK)
    # Cell-level random split pooled over all ~100 slides: a different (and very large) job.
    text = _replace_once(text, "same_wsi_split: true\n", "same_wsi_split: false\n")
    text = _replace_once(text, "  experiment_name: embeddings_eval\n", "  experiment_name: embeddings_eval_cross_cohort\n")
    text = _replace_once(text, "  selection_metric: slide_macro_f1\n", "  selection_metric: cohort_balanced_slide_macro_f1\n")
    # One learning rate: the grid is then only loss_weights (3 trials per fold).
    text = _replace_once(
        text,
        "  learning_rate:\n    tune: true\n    values: [5.0e-4, 1.0e-4]\n",
        "  learning_rate:\n    tune: false\n    values: [5.0e-4]\n",
    )
    # Every cell of every slide is used: ~12-13M cells loaded per fold, ~78 GB per copy of the
    # 1536-d float32 embeddings, and training holds ~2 copies (numpy + torch).
    text = _replace_once(text, "  mem: 32G\n", "  mem: 256G   # ~2 x 78 GB of embeddings per fold, all cells\n")
    # Fixed wall time for every fold instead of the per-split estimate.
    text = _replace_once(text, "  auto_time: true\n", "  auto_time: false\n")
    text = _replace_once(text, '  time: "05:00:00"\n', '  time: "24:00:00"\n')

    args.out.write_text(text)
    print(f"wrote {args.out} ({len({r['fold'] for r in rows})} folds, "
          f"{len({r['xenium_folder'] for r in rows})} slides)")


if __name__ == "__main__":
    main()
