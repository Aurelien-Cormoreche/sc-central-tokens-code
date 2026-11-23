"""End-to-end checks and throughput calibration.

`bench smoke --synthetic` (CPU, minutes): generates tiled TIFF slides with painted "cells" (one
colour per class, a black no-data patch, one slide at 0.125 um/px so the resampling path runs),
patch_coordinates.h5 and adata.h5ad files, a config using the tiny random ViT, then runs the real
CLI: inspect -> darkspots -> manifest -> extract 1 -> classify 1 -> genexp (if DeepSpot's deps
are installed) -> Gate 1 -> extract/classify 2a, 2b -> scaling -> report -> status, and asserts
that every planned job is done, re-running is a no-op, and the site holds every result.

`bench smoke` (cluster, one GPU, ~30-60 min): the same chain on the smoke slides with every
enabled model (outputs in {bench_root}/smoke), plus
  * calibration: GPU throughput per model and input size -> {bench_root}/calibration.json
  * parity: each model's `cls` (bf16, our loader) vs its model-card forward (fp32), cosine
  * centring overlays: crops with the centroid and the central-token box drawn,
    {bench_root}/smoke/centring/{model}__{slide}.png
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from bench.config import BenchConfig, load_config, smoke_config
from bench.utils import LOG, print_table, read_json, write_json


# ── calibration ──

def _main_root(cfg: BenchConfig) -> Path:
    return cfg.root.parent if cfg.is_smoke else cfg.root


def _bench_sizes(cfg: BenchConfig, model: str) -> List[int]:
    from bench.models.catalog import round_to_patch
    m = cfg.models[model]
    return sorted({m.input_px} | {round_to_patch(f, m.patch, cfg.stage2.round_to_patch) for f in cfg.stage2.fov_px})


def gpu_throughput(cfg: BenchConfig, model: str, n_px: int, device: str, seconds: float = 8.0) -> float:
    """Cells/s of the forward pass + readouts on random crops (no I/O)."""
    import torch
    from bench.models.encoders import build_encoder
    from bench.provenance import model_revisions
    mcfg = cfg.models[model]
    rev = model_revisions(cfg.paths.optional("bench_root")).get(model, {}).get("sha")
    if mcfg.revision is None and rev:
        mcfg = mcfg.model_copy(update={"revision": rev})
    enc = build_encoder(model, mcfg, n_px, device, cfg.extract.precision)
    bs = max(1, int(mcfg.batch_size * min(64.0, (mcfg.input_px / n_px) ** 2)))
    x = torch.randint(0, 256, (bs, n_px, n_px, 3), dtype=torch.uint8, device=enc.device)
    sync = torch.cuda.synchronize if enc.device.type == "cuda" else (lambda: None)
    for _ in range(2):
        enc.encode(x)
    sync()
    n, t0 = 0, time.time()
    while time.time() - t0 < seconds or n < 3 * bs:
        enc.encode(x)
        n += bs
    sync()
    cps = n / (time.time() - t0)
    del enc
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return cps


def calibrate(cfg: BenchConfig, models: List[str], from_status: bool = False, device: Optional[str] = None,
              dry_run: bool = False, io_factor: float = 0.85) -> dict:
    """Write {main bench_root}/calibration.json = {model: {input_px: cells/s}}. GPU benchmark
    values are scaled by io_factor (reading, gathering, writing); --from-status uses finished
    extraction jobs with >= 20k cells instead (end to end, wins over the benchmark)."""
    out = _main_root(cfg) / "calibration.json"
    cal = read_json(out, {}) or {}
    rows = []
    if from_status:
        by: Dict[tuple, List[float]] = {}
        for p in (cfg.root / "status" / "extract").glob("*.json"):
            s = read_json(p) or {}
            if s.get("state") == "done" and s.get("cells", 0) >= 20000 and s.get("input_px") and s.get("model") in models:
                by.setdefault((s["model"], int(s["input_px"])), []).append(float(s["cells_per_s"]))
        for (m, n), v in sorted(by.items()):
            cal.setdefault(m, {})[str(n)] = round(float(np.median(v)), 1)
            rows.append({"model": m, "input_px": n, "cells_per_s": round(float(np.median(v)), 1), "slides": len(v),
                         "source": "extraction jobs"})
    else:
        import torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        for m in models:
            for n in _bench_sizes(cfg, m):
                if dry_run:
                    rows.append({"model": m, "input_px": n})
                    continue
                cps = gpu_throughput(cfg, m, n, device) * io_factor
                cal.setdefault(m, {})[str(n)] = round(cps, 1)
                rows.append({"model": m, "input_px": n, "cells_per_s": round(cps, 1), "source": f"benchmark x{io_factor}"})
                LOG.info("calibrate %s @ %d px: %.0f cells/s", m, n, cps)
    cal["_meta"] = {"device": device, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "io_factor": io_factor}
    print_table(rows)
    if not dry_run and rows:
        write_json(out, cal)
        print(f"wrote {out}")
    return cal


# ── parity and centring (real smoke) ──

def _sample_crops(cfg: BenchConfig, slide: str, n_px: int, k: int = 16):
    from bench.data.inspect import load_inspect
    from bench.data.manifest import extraction_cells
    from bench.data.reader import WSIReader
    from bench.data.regions import crop_origins
    row = load_inspect(cfg).loc[slide]
    df = extraction_cells(cfg, slide).head(k)
    ox, oy = crop_origins(df["ex"].to_numpy(), df["ey"].to_numpy(), n_px)
    rd = WSIReader(row["wsi_path"], scale=float(row["scale"]))
    try:
        crops = np.stack([rd.read_ext(int(x), int(y), n_px, n_px) for x, y in zip(ox, oy)])
    finally:
        rd.close()
    return crops


def parity(cfg: BenchConfig, models: List[str], slide: str, device: str) -> List[dict]:
    import torch
    from bench.models.encoders import build_encoder
    from bench.provenance import model_revisions
    revs = model_revisions(cfg.paths.optional("bench_root"))
    rows = []
    for m in models:
        mcfg = cfg.models[m]
        if mcfg.revision is None and revs.get(m, {}).get("sha"):
            mcfg = mcfg.model_copy(update={"revision": revs[m]["sha"]})
        try:
            enc = build_encoder(m, mcfg, None, device, cfg.extract.precision)
            crops = torch.from_numpy(_sample_crops(cfg, slide, enc.input_px)).to(enc.device)
            ours = enc.encode(crops)["cls"]
            ref = enc.build_reference()
            with torch.inference_mode():
                theirs = ref(enc.normalize(crops)).float()
            cos = torch.nn.functional.cosine_similarity(ours, theirs, dim=1).cpu().numpy()
            ok = cos.mean() >= 0.999 and cos.min() >= 0.995
            rows.append({"model": m, "dim": int(ours.shape[1]), "ref_dim": int(theirs.shape[1]),
                         "cos_mean": f"{cos.mean():.5f}", "cos_min": f"{cos.min():.5f}", "result": "PASS" if ok else "FAIL"})
            del enc, ref
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:
            rows.append({"model": m, "result": f"ERROR {type(e).__name__}: {str(e)[:120]}"})
    print_table(rows)
    return rows


def centring_overlays(cfg: BenchConfig, models: List[str], slides: List[str]) -> List[Path]:
    from PIL import Image, ImageDraw
    from bench.models.readouts import central_pixel_box
    out = []
    for m in models:
        mc = cfg.models[m]
        g = mc.input_px // mc.patch
        lo, hi = central_pixel_box(g, mc.patch)
        for s in slides:
            try:
                crops = _sample_crops(cfg, s, mc.input_px, 16)
            except Exception as e:
                LOG.warning("centring overlay %s/%s: %s", m, s, e)
                continue
            scale = max(1, 160 // mc.input_px)
            tiles = []
            for c in crops:
                im = Image.fromarray(c).resize((mc.input_px * scale,) * 2, Image.NEAREST)
                d = ImageDraw.Draw(im)
                d.rectangle([lo * scale, lo * scale, hi * scale - 1, hi * scale - 1], outline=(0, 160, 255))
                h = mc.input_px * scale / 2
                d.line([h - 6, h, h + 6, h], fill=(255, 0, 0))
                d.line([h, h - 6, h, h + 6], fill=(255, 0, 0))
                tiles.append(im)
            w = tiles[0].width
            sheet = Image.new("RGB", (4 * w, ((len(tiles) + 3) // 4) * w), "white")
            for i, t in enumerate(tiles):
                sheet.paste(t, ((i % 4) * w, (i // 4) * w))
            p = cfg.root / "centring" / f"{m}__{s}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(p)
            out.append(p)
    LOG.info("wrote %d centring overlays to %s", len(out), cfg.root / "centring")
    return out


# ── shared driver ──

def _ns(**kw) -> argparse.Namespace:
    base = dict(config=None, smoke=False, dry_run=False, models=None, datasets=None, verbose=False, local=True,
                device=None, workers=4, no_strict=False, stage="1", schemes=None, max_mem_gb=10_000, readout=None,
                probe=None, out=None, no_status=False, deletable=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _step(results: list, name: str, fn, *a, **k):
    t0 = time.time()
    try:
        rc = fn(*a, **k)
        ok = rc in (0, None)
        results.append({"step": name, "result": "PASS" if ok else f"FAIL (rc={rc})", "s": round(time.time() - t0, 1)})
        return ok
    except SystemExit as e:
        results.append({"step": name, "result": f"FAIL: {e}"[:200], "s": round(time.time() - t0, 1)})
    except Exception as e:
        LOG.exception("step %s", name)
        results.append({"step": name, "result": f"ERROR {type(e).__name__}: {e}"[:200], "s": round(time.time() - t0, 1)})
    return False


def _with_gates(cfg: BenchConfig, model: str, readout: str, tile: int) -> BenchConfig:
    d = cfg.dump()
    d["gate1"]["chosen"] = {"model": model, "readout": readout}
    d["stage2"]["multicell"]["tile_px"] = tile
    src = d.pop("source", None)
    out = BenchConfig.model_validate(d)
    out.source = src
    return out


def _pipeline(cfg: BenchConfig, results: list, device: Optional[str], skip: List[str], gate_model: str) -> None:
    from bench import cli
    from bench.jobs import execute, new_plan
    from bench.data import darkspots
    from bench.data.inspect import load_inspect
    from bench.registry import load_slides
    a = _ns(device=device)
    _step(results, "inspect", cli.cmd_inspect, cfg, a)
    _step(results, "darkspots", lambda: execute(cfg, new_plan(cfg, "darkspots", darkspots.plan_jobs(
        cfg, load_slides(cfg), load_inspect(cfg)), {"unit": "CPU"}), False, True, device))
    _step(results, "manifest", cli.cmd_manifest, cfg, a)
    _step(results, "extract 1", cli.cmd_extract, cfg, _ns(device=device, stage="1"))
    _step(results, "classify 1", cli.cmd_classify, cfg, _ns(device=device, stage="1"))
    if "genexp" not in skip:
        _step(results, "genexp", cli.cmd_genexp, cfg, _ns(device=device))
    g = cfg
    if not ({"stage2", "stage3"} <= set(skip)):
        from bench.report.gates import gate1
        sug = gate1(cfg)["suggested"] or {"model": gate_model, "readout": "central"}
        results.append({"step": f"gate 1 -> {sug['model']}/{sug['readout']} (smoke: auto-set), gate 2 -> 224",
                        "result": "PASS", "s": 0})
        g = _with_gates(cfg, sug["model"], sug["readout"], 224)
    if "stage2" not in skip:
        _step(results, "extract 2a", cli.cmd_extract, g, _ns(device=device, stage="2a"))
        _step(results, "extract 2b", cli.cmd_extract, g, _ns(device=device, stage="2b"))
        _step(results, "classify 2", cli.cmd_classify, g, _ns(device=device, stage="2"))
    if "stage3" not in skip:
        _step(results, "scaling", cli.cmd_scaling, g, _ns(device=device))
    _step(results, "report", cli.cmd_report, g, _ns())

    def check_status():
        from bench.status import job_states, planned_jobs
        st = job_states(g, planned_jobs(g))
        if "genexp" in skip:
            st = [s for s in st if s["kind"] != "genexp"]
        bad = [s for s in st if s["state"] != "done"]
        for s in bad[:20]:
            print(f"  not done: {s['key']} ({s['state']}) {s['error']}")
        return 0 if not bad and st else 1
    _step(results, "status: every planned job done", check_status)

    def check_rerun():
        from bench.extract import planner
        from bench.jobs import is_done
        jobs = planner.plan_jobs(g, "1", g.enabled_models(), load_slides(g))
        return 0 if all(is_done(g, j) for j in jobs) else 1
    _step(results, "resume: re-planning extract 1 finds nothing to do", check_rerun)
    return g


# ── real smoke ──

def run_real(cfg: BenchConfig, args) -> int:
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = cfg
    cfg = smoke_config(cfg)
    models = cfg.enabled_models(args.models)
    if args.models:
        d = cfg.dump()
        for m in d["models"]:
            d["models"][m]["enabled"] = m in models
        src = d.pop("source", None)
        cfg = BenchConfig.model_validate(d)
        cfg.source = src
    results: list = []
    print(f"smoke run: {len(cfg.smoke.slides)} slides, <= {cfg.smoke.max_cells_per_slide} cells each, models "
          f"{', '.join(models)}, device {device}; outputs in {cfg.root}")
    g = _pipeline(cfg, results, device, args.skip, models[0])
    if "parity" not in args.skip:
        _step(results, "parity (cls vs model card)", lambda: 0 if all(
            r.get("result") == "PASS" for r in parity(cfg, models, cfg.smoke.slides[-1], device)) else 1)
    _step(results, "centring overlays", lambda: 0 if centring_overlays(cfg, models, cfg.smoke.slides[:2]) else 1)
    _step(results, "calibration (GPU throughput)", lambda: calibrate(base, models, device=device) and 0)
    print()
    print_table(results)
    cal = read_json(_main_root(cfg) / "calibration.json", {}) or {}
    rows = []
    for m in models:
        n = str(base.models[m].input_px)
        cps = cal.get(m, {}).get(n)
        if cps:
            rows.append({"model": m, "cells_per_s": cps, "stage1_GPU_h_per_15.5M_cells": round(15.5e6 / cps / 3600, 1)})
    if rows:
        print("\nstage-1 extraction projection (calibrated):")
        print_table(rows)
    print(f"\nsite: {g.root / 'site' / 'index.html'}   centring overlays: {cfg.root / 'centring'}")
    return 0 if all(r["result"] == "PASS" for r in results) else 1


# ── synthetic smoke ──

RAW_TYPES = [b"B_cell", b"Epithelial", b"Fibroblast", b"Myeloid", b"Malignant", b"Unknown"]
COLORS = {b"B_cell": (60, 40, 150), b"Epithelial": (200, 90, 60), b"Fibroblast": (230, 150, 200),
          b"Myeloid": (90, 160, 80), b"Malignant": (120, 20, 40), b"Unknown": (150, 150, 150)}


def _paint(rng, w: int, h: int, cx, cy, types, r: int, dark: Optional[tuple]) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[...] = (236, 212, 228)
    img = np.clip(img.astype(np.int16) + rng.integers(-12, 13, size=img.shape), 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = xx ** 2 + yy ** 2 <= r * r
    for x, y, t in zip(cx, cy, types):
        x, y = int(round(x)), int(round(y))
        if r <= x < w - r and r <= y < h - r:
            img[y - r:y + r + 1, x - r:x + r + 1][disk] = COLORS[t]
    if dark:
        x0, y0, x1, y1 = dark
        img[y0:y1, x0:x1] = 0
    return img


def make_synthetic(work: Path, seed: int = 0) -> Path:
    """Slides, coordinates, adata and a config under `work`; returns the config path."""
    import h5py
    import tifffile
    rng = np.random.default_rng(seed)
    wsi, xen, root = work / "wsi", work / "xenium", work / "bench"
    for d in (wsi, xen, root, work / "wsi_converted"):
        d.mkdir(parents=True, exist_ok=True)
    slides = []
    for cohort in ("A", "B"):
        for k in range(1, 5):
            slides.append({"slide": f"{cohort}_s{k}", "cohort": cohort, "donor": f"{cohort.lower()}{k}", "val": int(k == 4)})
    genes = [f"G{i:02d}" for i in range(30)]
    for i, s in enumerate(slides):
        mpp = 0.125 if s["slide"] == "B_s2" else 0.25
        f = 2 if mpp == 0.125 else 1
        w = h = 1100 * f
        n = 220
        cx = rng.uniform(30 * f, w - 30 * f, n)
        cy = rng.uniform(30 * f, h - 30 * f, n)
        p = np.array([0.18, 0.2, 0.2, 0.16, 0.16, 0.1]) if s["cohort"] == "A" else np.array([0.25, 0.2, 0.25, 0.2, 0.0, 0.1])
        types = np.array(RAW_TYPES, dtype=object)[rng.choice(len(RAW_TYPES), size=n, p=p / p.sum())]
        dark = (300 * f, 300 * f, 420 * f, 420 * f) if s["slide"] == "A_s1" else None
        img = _paint(rng, w, h, cx, cy, types, 5 * f, dark)
        tifffile.imwrite(wsi / f"{s['slide']}_he_image.tif", img, tile=(256, 256), photometric="rgb",
                         resolution=(1e4 / mpp, 1e4 / mpp), resolutionunit="CENTIMETER")
        d = xen / s["slide"]
        d.mkdir(parents=True, exist_ok=True)
        ids = [f"{s['slide']}_c{j:05d}".encode() for j in range(n)]
        with h5py.File(d / "patch_coordinates.h5", "w") as fh:
            fh["x_start"] = (cx - 112).astype(np.float64)
            fh["y_start"] = (cy - 112).astype(np.float64)
            fh["cell_id"] = np.array(ids)
            fh["cell_type"] = np.array(list(types))
        # expression: class-dependent Poisson rates
        import anndata as ad
        import pandas as pd
        cls = np.array([RAW_TYPES.index(t) for t in types])
        rates = rng.gamma(2.0, 1.0, size=(len(RAW_TYPES), len(genes)))
        counts = rng.poisson(rates[cls] * 3).astype(np.float32)
        obs = pd.DataFrame({"cell_id": [f"c{j:05d}" for j in range(n)], "x_centroid": cx * mpp, "y_centroid": cy * mpp},
                           index=[f"c{j:05d}" for j in range(n)])
        a = ad.AnnData(X=counts.copy(), obs=obs, var=pd.DataFrame(index=genes))
        a.layers["counts"] = counts
        a.write_h5ad(d / "adata.h5ad")
    import csv
    with open(work / "slides.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["slide", "cohort", "donor", "val"])
        wr.writeheader()
        wr.writerows(slides)
    cfg = {
        "paths": {"bench_root": str(root), "wsi_raw": str(wsi), "wsi_converted": str(work / "wsi_converted"),
                  "xenium_processed": str(xen)},
        "slides_csv": str(work / "slides.csv"), "cohorts": ["A", "B"], "mapping": "simplified_broad",
        # resampling on here (bench.yaml has it off) so the resampled path (B_s2 at 0.125 um/px) stays tested
        "resolution": {"resample_below_mpp": 0.2, "target_mpp": 0.25},
        "filters": {"linf_px": 24, "border_px": 40, "block_px": 16, "scan_tile_px": 512},
        "models": {"tiny": {"input_px": 32, "patch": 8, "batch_size": 64, "cells_per_s": 400}},
        "extract": {"region_px": 512, "num_workers": 0, "slide_overhead_s": 1, "bin_overhead_s": 1},
        "slurm": {"partition": None, "max_hours": 1, "cpus": 2},
        "classify": {"within_cohorts": ["A", "B"], "epochs": 6, "patience": 3, "batch_size": 64,
                     "grid": {"lr": [1e-2, 1e-3], "weight_decay": [1e-4], "loss_weights": ["none", "by_cell_type"]}},
        "genexp": {"sets": {"GX": {"cohort": "A"}}, "num_genes": 20, "top": [5, 10, 20], "hvg_max_genes": 30,
                   "readout": "auto"},
        "stage2": {"cohorts": ["A", "B"], "fov_px": [16, 32, 64]},
        "stage3": {"fractions": [0.1, 1.0], "seeds": [0, 1], "mlp": {"hidden": 16, "grid": {
            "lr": [1e-2], "weight_decay": [1e-4], "loss_weights": ["none"]}}},
    }
    p = work / "bench_synthetic.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def _deepspot_available() -> Optional[str]:
    for mod in ("lightning", "mlflow", "hydra"):
        try:
            __import__(mod)
        except Exception as e:
            return f"{mod}: {type(e).__name__}"
    return None


def run_synthetic(out: Optional[str] = None, keep: bool = False) -> int:
    work = Path(out) if out else Path(tempfile.mkdtemp(prefix="bench_synth_"))
    work.mkdir(parents=True, exist_ok=True)
    results: list = []
    try:
        cfg_path = make_synthetic(work)
        cfg = load_config(cfg_path)
        missing = _deepspot_available()
        skip = ["genexp"] if missing else []
        if missing:
            results.append({"step": "genexp", "result": f"SKIPPED (DeepSpot deps missing: {missing})", "s": 0})
        g = _pipeline(cfg, results, "cpu", skip, "tiny")

        def check_site():
            html = (g.root / "site" / "index.html").read_text()
            from bench.report.collect import collect
            d = collect(g)
            stages = sorted({f["stage"] for f in d["classify_folds"]})
            want = ["1", "2a", "2b", "3"]
            print(f"  site: {len(html) / 1e3:.0f} kB, classification folds per stage: "
                  + ", ".join(f"{s}={sum(f['stage'] == s for f in d['classify_folds'])}" for s in stages))
            return 0 if stages == want and "classify_summary" in html else 1
        _step(results, "site holds stage 1, 2a, 2b, 3 results", check_site)

        def check_darkspot_drop():
            from bench.data.manifest import load_cells
            df = load_cells(g, "A_s1")
            n = int((df["drop"] == "darkspot").sum())
            print(f"  A_s1: {n} cells dropped next to the black patch, {int((df['drop'] == 'border').sum())} at the border")
            return 0 if n > 0 else 1
        _step(results, "dark-spot filter drops cells near the black patch", check_darkspot_drop)
    finally:
        print()
        print_table(results)
        if keep or out:
            print(f"work directory: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    bad = [r for r in results if not (r["result"] == "PASS" or r["result"].startswith("SKIPPED"))]
    print("\nSYNTHETIC SMOKE " + ("PASSED" if not bad else f"FAILED ({len(bad)} step(s))"))
    return 0 if not bad else 1
