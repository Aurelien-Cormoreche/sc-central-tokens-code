"""Write configs/deepspot_lopo_{vannan,prime}.yaml: leave-one-patient-out gene-expression
prediction (deepspot_training) on

  vannan -- the VANNAN pulmonary-fibrosis cohort (343-gene lung panel, healthy + fibrotic),
  prime  -- every human Xenium Prime 5K slide (SPLIT's 6 NSCLC slides + 10x's lymph node,
            prostate and skin; the 10x ones use development versions of the 5K panel).

Patients are splits.csv's donor_key (donor-grouped, as for the classifier folds); slides
without embeddings (make_cross_cohort_config.EXCLUDED_WSIS) are left out. Patients in donor_key
order, fold i: test = every slide of patient i, val = every slide of patient i+1 (cyclic, so each
patient is val once), train = every other slide. Val drives early stopping (hyperparameters are
fixed, as in configs/deepspot_train_colon.yaml, which everything else is copied from: single-cell
context, 300 HVGs); test is only evaluated. HVGs are ranked per fold over the inner join of the
train slides' panels, then restricted to genes present in the val and test slides (compute_hvgs.py).

Usage (repo root):
    python scripts/make_deepspot_lopo_configs.py [--splits-csv _metadata_work/splits.csv]
Then on the cluster, per config:
    sbatch scripts/compute_hvgs.sh --config-name=deepspot_lopo_vannan
    python -m src.python.deepspot_training.launcher --config-name=deepspot_lopo_vannan \\
        data.model_name=UNI2 data.correction_name=raw
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_cross_cohort_config as cc  # noqa: E402

REPO = cc.REPO
TEMPLATE = "deepspot_train_colon"
CORNERS = "[bottom_left, bottom_right, top_left, top_right]"
DATASETS = {
    # name: (description, slide filter on (splits.csv row, xenium_metadata row), slurm.mem)
    "vannan": ("the VANNAN pulmonary-fibrosis cohort (GSE250346; 343-gene lung panel)",
               lambda r, meta: r["source"] == "VANNAN", "64G"),
    # Largest slide: 10x lymph node, ~0.65M typed cells; deepspot reads every cell's 4 tokens.
    "prime": ("every human Xenium Prime 5K slide (SPLIT NSCLC + 10x lymph node, prostate, skin)",
              lambda r, meta: meta["chemistry"] == "Xenium Prime 5K" and meta["species"] == "Homo sapiens", "128G"),
}


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"configs/{TEMPLATE}.yaml no longer contains exactly one {old!r}"
    return text.replace(old, new)


def _splits_block(slides: dict[str, str], excluded: set[str]) -> str:
    """slides: WSI -> donor_key. One fold per patient, in donor_key order, val = the next patient
    (cyclic); `excluded` only goes in the comment."""
    by_donor = defaultdict(list)
    for wsi, donor in slides.items():
        by_donor[donor].append(wsi)
    donors = sorted(by_donor)
    assert len(donors) >= 3, "need a train, a val and a test patient"
    n_slides = lambda d: f"{len(by_donor[d])} slide{'s' * (len(by_donor[d]) > 1)}"
    lines = [
        "# ── WSI splits ─────────────────────────────────────────────────────────────",
        f"# Leave-one-patient-out over {len(slides)} slides of {len(donors)} patients (splits.csv donor_key);",
        "# val = the next patient in donor_key order (cyclic); regenerate with",
        "# scripts/make_deepspot_lopo_configs.py. Left out (no embeddings):",
        f"# {', '.join(sorted(excluded)) or 'none'}.",
        "splits:",
    ]
    for i, donor in enumerate(donors):
        val_donor = donors[(i + 1) % len(donors)]
        test, val = sorted(by_donor[donor]), sorted(by_donor[val_donor])
        train = sorted(w for w in slides if w not in test and w not in val)
        lines.append(f"  # split {i}: test = {donor} ({n_slides(donor)}); val = {val_donor} ({n_slides(val_donor)}); "
                     f"train = {len(train)} slides of {len(donors) - 2} patients")
        for j, (side, wsis) in enumerate((("train", train), ("val", val), ("test", test))):
            lines.append(f"  {'- ' if j == 0 else '  '}{side}:")
            lines += [f"      {w}: 1.0" for w in wsis]
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits-csv", type=Path, default=REPO / "_metadata_work" / "splits.csv")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    args = parser.parse_args()

    with open(args.splits_csv, newline="", encoding="utf-8") as fh:
        rows = {r["xenium_folder"]: r for r in csv.DictReader(fh) if r["experiment"] == "E4_loso"}
    with open(REPO / "xenium_metadata.csv", newline="", encoding="utf-8") as fh:
        meta = {r["xenium_folder"]: r for r in csv.DictReader(fh)}
    template = (REPO / "configs" / f"{TEMPLATE}.yaml").read_text()
    _, body = template.split("defaults:\n", 1)
    head, rest = body.split("# ── WSI splits", 1)
    _, tail = rest.split("# No same-WSI", 1)

    for name in args.datasets:
        description, keep, mem = DATASETS[name]
        chosen = {cc._wsi_name(f): r["donor_key"] for f, r in rows.items() if keep(r, meta[f])}
        excluded = set(chosen) & cc.EXCLUDED_WSIS
        slides = {w: d for w, d in chosen.items() if w not in excluded}
        assert slides, name

        text = (
            "# Leave-one-patient-out gene-expression prediction (deepspot_training) on\n"
            f"# {description} -- generated by\n"
            f"# scripts/make_deepspot_lopo_configs.py from configs/{TEMPLATE}.yaml; edit there.\n"
            "# Fold i: test = every slide of one patient; val = the next patient (early stopping; never in\n"
            "# train); train = the rest. Hyperparameters are fixed, as in the template.\n"
            "# Before the experiments, precompute the per-fold HVGs (data.hvg_dir):\n"
            f"#   sbatch scripts/compute_hvgs.sh --config-name=deepspot_lopo_{name}\n"
            "\ndefaults:\n" + head + _splits_block(slides, excluded) + "\n# No same-WSI" + tail
        )
        text = _replace_once(text, "output_dir: outputs_deepspot_colon\n", f"output_dir: outputs_deepspot_lopo_{name}\n")
        text = _replace_once(text, "  embeddings_datasets:  [cls]\n", f"  embeddings_datasets: {CORNERS}\n")
        text = _replace_once(text, "  gene_interval: [10,20,30,40,50]\n", "  gene_interval: [50,100,200,300]\n")
        text = _replace_once(text, "  hvg_dir: data/HVGsSplitColon\n", f"  hvg_dir: data/HVGsSplitLOPO_{name}\n")
        # load_hvg_table drops ranked genes absent from the test slides and needs num_genes left:
        # rank more than num_genes so they are replaced (the Prime lymph node's development panel
        # lacks 377 of the 5,001 release genes).
        text = _replace_once(
            text, "  hvg_max_genes: 300\n",
            "  hvg_max_genes: 500   # > num_genes: ranked genes absent from a test slide get replaced\n",
        )
        text = _replace_once(text, "  experiment_name: deepcell_embeddings_eval_colon\n",
                             f"  experiment_name: deepcell_embeddings_eval_lopo_{name}\n")
        text = _replace_once(text, "  mem: 64G\n", f"  mem: {mem}\n")
        text = _replace_once(text, "  output_dir: slurm_logs_deepspot_colon\n", f"  output_dir: slurm_logs_deepspot_lopo_{name}\n")

        out = REPO / "configs" / f"deepspot_lopo_{name}.yaml"
        out.write_text(text)
        n_patients = len(set(slides.values()))
        print(f"wrote {out.relative_to(REPO)}: {len(slides)} slides, {n_patients} patients -> {n_patients} folds"
              + (f" (left out, no embeddings: {sorted(excluded)})" if excluded else ""))


if __name__ == "__main__":
    main()
