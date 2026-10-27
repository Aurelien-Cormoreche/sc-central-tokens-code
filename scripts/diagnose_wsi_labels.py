"""Check every WSI of a classifier config for cells the classifier can actually use.

For each WSI in the config's splits, reads {data.base_dir}/{model}_h5/{correction}/{wsi}/
embeddings_dataset.h5 (labels and ids only, no embeddings) and reports: number of cells,
unfinished rows (empty cell_ids: extraction killed mid-run), raw cell_labels, labels the
mapping doesn't know (silently become 'Unknown'), and the usable cells per mapped class --
what experiment.py's _load_wsi_data keeps. Ends with a detailed section per problem WSI.

Usage (repo root, after `set -a; source .env; set +a`):
    python scripts/diagnose_wsi_labels.py --config configs/train_cross_cohort.yaml \
        --model-name UNI2 --correction-name raw [--wsis WSI ...]
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
from omegaconf import OmegaConf

# Run as a plain script (`python scripts/...`), Python only puts scripts/ on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.python.code_configs.mappings import mapping_factory  # noqa: E402

LOW_USABLE_FRACTION = 0.10


def _side_wsis(side_cfg) -> list[str]:
    return [k for k in side_cfg.keys() if k != "cell_type_proportions"] if side_cfg is not None else []


def _decode(label) -> str:
    return label.decode("utf-8", "replace") if isinstance(label, bytes) else str(label)


def _inspect(h5_path: Path, mapping: dict, consider_matching: bool) -> dict:
    with h5py.File(h5_path, "r", locking=False) as f:
        labels = f["cell_labels"][()]
        n_empty_ids = int(sum(1 for c in f["cell_ids"][()] if len(c) == 0))
        matching = f["matching"][()].astype(bool) if "matching" in f else None
        emb_keys = sorted(k[len("embeddings_"):] for k in f.keys() if k.startswith("embeddings_"))
    if consider_matching and matching is not None:
        labels = labels[matching]
    raw_counts = Counter(labels.tolist())
    unmapped = {lab: n for lab, n in raw_counts.items() if lab not in mapping}
    mapped = Counter()
    for lab, n in raw_counts.items():
        mapped[mapping.get(lab, "Unknown")] += n
    usable = sum(n for cls, n in mapped.items() if cls != "Unknown")
    return {
        "n": len(labels), "n_empty_ids": n_empty_ids, "raw": raw_counts, "unmapped": unmapped,
        "mapped": mapped, "usable": usable, "emb_keys": emb_keys,
        "n_matching": None if matching is None else int(matching.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/train_cross_cohort.yaml"))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--correction-name", required=True)
    parser.add_argument("--wsis", nargs="*", help="only these WSIs (default: every WSI in the splits)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    mapping = mapping_factory(cfg.mapping)
    consider_matching = bool(OmegaConf.select(cfg, "data.consider_matching", default=True))
    wanted_keys = list(cfg.data.embeddings_datasets or [])
    try:
        base = Path(str(cfg.data.base_dir)) / f"{args.model_name}_h5" / args.correction_name
    except Exception as e:
        raise SystemExit(f"can't resolve data.base_dir ({e}) -- run `set -a; source .env; set +a` first")

    where: dict[str, list[str]] = defaultdict(list)
    for i, sp in enumerate(cfg.splits):
        for side in ("train", "val", "test"):
            for w in _side_wsis(OmegaConf.select(sp, side)):
                where[w].append(f"{i}:{side}")
    wsis = args.wsis or sorted(where)
    print(f"{len(wsis)} WSIs under {base} | mapping={cfg.mapping} consider_matching={consider_matching}\n")
    print(f"{'status':12} {'wsi':58} {'cells':>9} {'usable':>9} {'unknown%':>8}  top raw labels")

    problems: dict[str, tuple[str, dict]] = {}
    for w in wsis:
        path = base / w / "embeddings_dataset.h5"
        if not path.exists():
            print(f"{'MISSING':12} {w:58}")
            problems[w] = ("MISSING", {})
            continue
        try:
            r = _inspect(path, mapping, consider_matching)
        except Exception as e:
            print(f"{'UNREADABLE':12} {w:58} {type(e).__name__}: {e}")
            problems[w] = ("UNREADABLE", {"error": f"{type(e).__name__}: {e}"})
            continue
        missing_keys = [k for k in wanted_keys if k not in r["emb_keys"]]
        if r["usable"] == 0:
            status = "ZERO_USABLE"
        elif r["n_empty_ids"]:
            status = "INCOMPLETE"
        elif missing_keys:
            status = "NO_READOUT"
        elif r["unmapped"]:
            status = "UNMAPPED"
        elif r["usable"] < LOW_USABLE_FRACTION * max(1, r["n"]):
            status = "LOW_USABLE"
        else:
            status = "ok"
        unknown_pct = 100 * (r["n"] - r["usable"]) / max(1, r["n"])
        top = ", ".join(f"{_decode(lab)}={n}" for lab, n in r["raw"].most_common(4))
        print(f"{status:12} {w:58} {r['n']:9,} {r['usable']:9,} {unknown_pct:7.1f}%  {top}")
        if status != "ok":
            problems[w] = (status, {**r, "missing_keys": missing_keys})

    print(f"\n{len(wsis) - len(problems)} ok, {len(problems)} with problems")
    for w, (status, r) in problems.items():
        print(f"\n## {w}: {status}  (folds:sides {', '.join(where.get(w, ['-']))})")
        if not r or "error" in r:
            print(f"   {r.get('error', 'no embeddings_dataset.h5')}")
            continue
        print(f"   cells={r['n']:,}  empty cell_ids={r['n_empty_ids']:,}  matching={r['n_matching']}  "
              f"embeddings datasets={r['emb_keys']}" + (f"  MISSING read-out {r['missing_keys']}" if r["missing_keys"] else ""))
        print("   raw cell_labels: " + ", ".join(f"{_decode(lab)!r}={n:,}" for lab, n in r["raw"].most_common()))
        if r["unmapped"]:
            print(f"   not in mapping '{cfg.mapping}' (-> Unknown): "
                  + ", ".join(f"{_decode(lab)!r}={n:,}" for lab, n in sorted(r["unmapped"].items(), key=lambda x: -x[1])))
        print("   usable per class: " + (", ".join(f"{c}={n:,}" for c, n in r["mapped"].most_common() if c != "Unknown") or "none"))
        if status == "ZERO_USABLE":
            only_unknown = set(r["raw"]) <= {b"Unknown", "Unknown"}
            only_empty = set(r["raw"]) <= {b"", ""}
            print("   -> " + (
                "every label is empty: the extraction didn't finish (or wrote no labels) -- re-extract" if only_empty else
                "every cell is typed 'Unknown' upstream (cell typing in patch_coordinates.h5 failed or found no "
                "known type) -- fix the annotation or drop the WSI" if only_unknown else
                "labels exist but none is in the mapping -- extend the mapping or drop the WSI"))


if __name__ == "__main__":
    main()
