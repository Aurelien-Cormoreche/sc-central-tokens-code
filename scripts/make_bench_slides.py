"""Write configs/bench_slides.csv: the slide registry of the `bench` pipeline.

One row per slide of the E4_loso experiment of _metadata_work/splits.csv (all 5 human cohorts),
after the same exclusions and val-donor overrides as scripts/make_cross_cohort_config.py, so the
bench folds match the branch's within/cross/single-cohort configs:

    slide, cohort, donor, val

`val` = 1 for the cohort's fixed validation donors (the slides that are `val` in every
leave-one-cohort-out fold where that cohort is not held out). Yu_DRG and the mouse samples are
not in E4_loso, so they are not in the registry.

Usage (repo root):
    python scripts/make_bench_slides.py --splits-csv <path to splits.csv> [--out configs/bench_slides.csv]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_cross_cohort_config as cc  # noqa: E402


def build_rows(splits_csv: Path) -> list[dict]:
    with open(splits_csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["experiment"] == "E4_loso"]
    unknown = cc.EXCLUDED_WSIS - {cc._wsi_name(r["xenium_folder"]) for r in rows}
    assert not unknown, f"excluded WSIs not in splits.csv: {unknown}"
    rows = [r for r in rows if cc._wsi_name(r["xenium_folder"]) not in cc.EXCLUDED_WSIS]
    for r in rows:
        if r["source"] in cc.VAL_DONOR_OVERRIDES and r["split"] in ("train", "val"):
            r["split"] = "val" if r["donor_key"] in cc.VAL_DONOR_OVERRIDES[r["source"]] else "train"

    roles: dict[str, set[str]] = defaultdict(set)
    info: dict[str, tuple[str, str]] = {}
    for r in rows:
        slide = cc._wsi_name(r["xenium_folder"])
        cohort = cc.COHORT_NAMES[r["source"]]
        if slide in info:
            assert info[slide] == (cohort, r["donor_key"]), f"{slide}: inconsistent cohort/donor"
        info[slide] = (cohort, r["donor_key"])
        if r["split"] != "test":
            roles[slide].add(r["split"])
    out = []
    for slide, (cohort, donor) in sorted(info.items(), key=lambda kv: (kv[1][0], kv[0])):
        assert len(roles[slide]) == 1, f"{slide}: train/val role differs across folds: {roles[slide]}"
        out.append({"slide": slide, "cohort": cohort, "donor": donor, "val": int("val" in roles[slide])})
    # a donor is never split between train and val
    by_donor: dict[str, set[int]] = defaultdict(set)
    for r in out:
        by_donor[r["donor"]].add(r["val"])
    assert all(len(v) == 1 for v in by_donor.values()), "a donor has both train and val slides"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=cc.REPO / "configs" / "bench_slides.csv")
    args = ap.parse_args()
    rows = build_rows(args.splits_csv)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["slide", "cohort", "donor", "val"])
        w.writeheader()
        w.writerows(rows)
    counts = defaultdict(lambda: [0, 0, set()])
    for r in rows:
        c = counts[r["cohort"]]
        c[0] += 1
        c[1] += r["val"]
        c[2].add(r["donor"])
    for cohort, (n, n_val, donors) in counts.items():
        print(f"{cohort:8s} {n:3d} slides ({n_val} val), {len(donors)} donors")
    print(f"wrote {len(rows)} slides -> {args.out}")


if __name__ == "__main__":
    main()
