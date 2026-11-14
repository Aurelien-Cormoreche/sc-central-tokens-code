#!/usr/bin/env python3
"""Profile embedding extraction and classifier training on the cluster, to decide what to optimise.

Sections (all by default; pick with --sections):
  env        GPU / CPU / RAM / filesystems / library versions of the node it runs on
  cluster    sinfo, partition and account limits, fairshare, queue, and your sacct history
             (queue wait, CPU efficiency, MaxRSS per job family)
  history    throughput of past extraction jobs (parsed from their logs) and of classifier
             runs (results_summary.yaml `timing`)
  inventory  per slide: cells, WSI tiling/compression, embeddings h5 layout, and how many WSI
             tiles a batch decodes in the current cell order vs a tile-sorted order
  extract    one slide: WSI read rates (cold/warm, orders, cache), a fork-safety check of the
             OpenSlide handle DataLoader workers inherit, DataLoader worker sweep, UNI2
             throughput per precision x batch size, numerical drift vs fp32 and vs the stored
             embeddings, h5 write cost, end to end (current code vs a prototype of the
             suggested changes), local-scratch staging, torch.compile
  classify   split sizes, embeddings read throughput (threads vs processes, 4 datasets vs 1),
             label pass, training-loop variants on synthetic data, per-split extrapolation

Usage (from the repo root, .env next to it) -- one job per GPU type you may use:
    sbatch --gres=gpu:rtx4090:1 --cpus-per-task=32 scripts/profile_pipeline.sh
    sbatch --gres=gpu:rtx5090:1 scripts/profile_pipeline.sh --sections env extract
    sbatch --gres=gpu:h200:1    scripts/profile_pipeline.sh --sections env extract
The report is printed and saved to profile_reports/<gpu>_<job>.{txt,json} (+ inventory csv).
Datasets are only read. Temporary files go to <DATASETS_ROOT>/_profile_tmp_<job> and the
node-local scratch dir, and are removed at the end (--keep-tmp keeps them).
"""
import argparse
import contextlib
import copy
import csv
import glob
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SECTIONS = ['env', 'cluster', 'history', 'inventory', 'extract', 'classify']
CENTRAL = ['bottom_left', 'bottom_right', 'top_left', 'top_right']
_now = time.perf_counter


# ── report ───────────────────────────────────────────────────────────────────

class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.data: dict = {}

    def p(self, *parts):
        line = " ".join(str(x) for x in parts)
        print(line, flush=True)
        self.lines.append(line)

    def put(self, key, value):
        self.data[key] = value


R = Report()


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if not math.isfinite(v):
            return str(v)
        if v == 0:
            return "0"
        if abs(v) >= 100:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.3g}"
    return str(v)


def table(rows, cols, indent="  "):
    if not rows:
        R.p(indent + "(none)")
        return
    cells = [[_fmt(r.get(c)) for c in cols] for r in rows]
    w = [max(len(c), *(len(x[i]) for x in cells)) for i, c in enumerate(cols)]
    R.p(indent + "  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
    for x in cells:
        R.p(indent + "  ".join(v.ljust(w[i]) for i, v in enumerate(x)))


@contextlib.contextmanager
def block(name, top=False):
    """A titled (sub)section; an exception is reported and the run goes on."""
    t0 = _now()
    R.p(f"\n=== {name} {'=' * max(3, 72 - len(name))}" if top else f"\n--- {name}")
    try:
        yield
    except Exception as e:  # noqa: BLE001 -- a failed measurement must not stop the others
        R.p(f"!! {name} failed: {type(e).__name__}: {e}")
        tb = traceback.format_exc().strip().splitlines()
        for line in tb[-8:]:
            R.p("   " + line)
    finally:
        if top:
            R.p(f"[{name}: {_now() - t0:.0f}s]")


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = r.stdout.strip()
        if r.returncode:
            out = (out + "\n" + r.stderr.strip()).strip() or f"<exit {r.returncode}>"
        return out
    except Exception as e:  # noqa: BLE001
        return f"<{cmd.split()[0]} failed: {e}>"


def _dur(s):
    """'1-02:03:04' / '02:03:04' / '03:04' / '03:04.123' -> seconds."""
    s = (s or "").strip()
    if not s or not s[0].isdigit():
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    try:
        parts = [float(p) for p in s.split(":")]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + sec


def _bytes(s):
    m = re.match(r"([\d.]+)([KMGTP]?)", s or "")
    if not m:
        return None
    return float(m[1]) * 1024 ** " KMGTP".index(m[2] or " ")


def _pct(values, q):
    values = [v for v in values if v is not None]
    return float(np.percentile(values, q)) if values else None


def _ncpu():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def _tmp_dir(ctx, root):
    d = Path(root) / f"_profile_tmp_{ctx['job']}"
    d.mkdir(parents=True, exist_ok=True)
    ctx.setdefault("tmp_dirs", []).append(d)
    return d


def _local_scratch():
    for cand in [os.environ.get("TMPDIR"), "/scratch", "/tmp"]:
        if cand and os.path.isdir(cand) and os.access(cand, os.W_OK):
            return cand
    return None


# ── env ──────────────────────────────────────────────────────────────────────

def sec_env(args, ctx):
    R.p(f"host {socket.gethostname()} | python {platform.python_version()} | job {ctx['job']} | "
        f"{datetime.now().isoformat(timespec='seconds')}")
    slurm = {k: os.environ[k] for k in ["SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
                                         "SLURM_MEM_PER_CPU", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE",
                                         "CUDA_VISIBLE_DEVICES", "SLURM_JOB_NODELIST", "TMPDIR"]
             if os.environ.get(k)}
    R.p("slurm:", slurm)
    R.p(f"usable cpus {_ncpu()} (node has {os.cpu_count()})")
    R.p(sh("lscpu | grep -E '^(Model name|Socket\\(s\\)|Core\\(s\\) per socket|Thread\\(s\\) per core|NUMA node[0-9]* CPU)'"))
    try:
        mem = {l.split(":")[0]: l.split(":")[1].strip() for l in open("/proc/meminfo") if ":" in l}
        R.p(f"RAM total {mem.get('MemTotal')}, available {mem.get('MemAvailable')}")
    except OSError:
        pass
    for f in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        if os.path.exists(f):
            v = open(f).read().strip()
            R.p(f"cgroup memory limit: {v if not v.isdigit() else f'{int(v) / 2**30:.1f} GiB'}")
            break
    R.p(sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.max,"
           "pcie.link.width.current,power.limit,clocks.max.sm --format=csv"))
    R.p(sh("nvidia-smi topo -m"))

    versions = {}
    for mod in ["numpy", "h5py", "torch", "torchvision", "timm", "openslide", "PIL", "sklearn", "tifffile",
                "pyvips", "mlflow"]:
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            versions[mod] = None
    try:
        versions["hdf5"] = h5py.version.hdf5_version
        import openslide
        versions["libopenslide"] = getattr(openslide, "__library_version__", "?")
        versions["OpenSlideCache"] = hasattr(openslide, "OpenSlideCache")
        from PIL import features
        versions["libjpeg_turbo"] = features.check_feature("libjpeg_turbo")
    except Exception:  # noqa: BLE001
        pass
    R.p("versions:", versions)
    R.put("versions", versions)

    try:
        import torch
        R.p(f"torch {torch.__version__} cuda {torch.version.cuda} cudnn {torch.backends.cudnn.version()} "
            f"| default float32 matmul precision {torch.get_float32_matmul_precision()}, "
            f"matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}, cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            R.p(f"gpu0 {p.name} cc {p.major}.{p.minor}, {p.total_memory / 2**30:.1f} GiB, "
                f"{p.multi_processor_count} SMs, bf16 {torch.cuda.is_bf16_supported()}, visible gpus {torch.cuda.device_count()}")
            R.put("gpu", {"name": p.name, "cc": f"{p.major}.{p.minor}", "mem_gib": p.total_memory / 2**30})
            _sdpa_check(torch)
    except Exception as e:  # noqa: BLE001
        R.p(f"torch unavailable: {e}")

    R.p("filesystems (df -PTh):")
    for label in ["DATASETS_ROOT", "XENIUM_PROCESSED_OUTPUT_ROOT", "WSI_RAW_ROOT", "WSI_CONVERTED_ROOT", "HF_HOME", "TMPDIR"]:
        path = os.environ.get(label)
        if path and os.path.exists(path):
            R.p(f"  {label:<29} {sh(f'df -PTh {path} | tail -1')}")
    for path in ["/scratch", "/tmp", "/dev/shm", str(REPO_ROOT)]:
        if os.path.exists(path):
            R.p(f"  {path:<29} {sh(f'df -PTh {path} | tail -1')}")


def _sdpa_check(torch):
    import torch.nn.functional as F
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        R.p("SDPA backend check needs torch>=2.3")
        return
    res = {}
    dtypes = [torch.float32, torch.float16] + ([torch.bfloat16] if torch.cuda.is_bf16_supported() else [])
    for be in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH] + (
            [SDPBackend.CUDNN_ATTENTION] if hasattr(SDPBackend, "CUDNN_ATTENTION") else []):
        ok = []
        for dt in dtypes:
            q = torch.randn(2, 24, 265, 64, device="cuda", dtype=dt)
            try:
                with sdpa_kernel(be):
                    F.scaled_dot_product_attention(q, q, q)
                ok.append(str(dt).replace("torch.", ""))
            except Exception:  # noqa: BLE001
                pass
        res[be.name] = ok
    R.p("SDPA kernels usable for ViT-H shapes:", res)


# ── cluster ──────────────────────────────────────────────────────────────────

def sec_cluster(args, ctx):
    user = os.environ.get("USER", "")
    with block("gpu nodes"):
        nodes = {}
        for line in sh("sinfo -p gpu -N -h -o '%N|%c|%m|%G|%f'").splitlines():
            f = line.split("|")
            if len(f) < 5 or f[0] in nodes:
                continue
            gpus = re.findall(r"gpu:([\w.\-]+):(\d+)", f[3])
            ngpu = sum(int(n) for _, n in gpus) or None
            cpus, mem_gb = int(f[1]), int(re.sub(r"\D", "", f[2]) or 0) / 1024
            nodes[f[0]] = {"node": f[0], "cpus": cpus, "mem_GB": mem_gb, "gpus": ",".join(f"{t}:{n}" for t, n in gpus),
                           "cpus/gpu": cpus / ngpu if ngpu else None, "GB/gpu": mem_gb / ngpu if ngpu else None,
                           "features": f[4]}
        table(list(nodes.values()), ["node", "gpus", "cpus", "mem_GB", "cpus/gpu", "GB/gpu", "features"])
        R.put("nodes", list(nodes.values()))
    with block("current gpu usage (sinfo GresUsed / CPUs A/I/O/T)"):
        R.p(sh("sinfo -p gpu -h -N -O 'NodeList:22,Gres:34,GresUsed:46,CPUsState:22,AllocMem:10,Memory:10'"))
        counts = {}
        for line in sh("squeue -p gpu -h -o '%T|%b|%u'").splitlines():
            f = line.split("|")
            if len(f) == 3:
                key = (f[0], re.sub(r"^gres/|^gres:", "", f[1]) or "-", "me" if f[2] == user else "others")
                counts[key] = counts.get(key, 0) + 1
        table([{"state": k[0], "gres": k[1], "who": k[2], "jobs": v} for k, v in sorted(counts.items())],
              ["state", "gres", "who", "jobs"])
    with block("limits"):
        part = sh("scontrol show partition gpu")
        keep = re.findall(r"\b(MaxTime|DefaultTime|DefMemPerCPU|DefMemPerNode|MaxMemPerNode|MaxMemPerCPU|MaxCPUsPerNode|"
                          r"QoS|AllowQos|PriorityTier|TRESBillingWeights|OverSubscribe|PreemptMode)=(\S+)", part)
        R.p("partition gpu:", " ".join(f"{k}={v}" for k, v in keep) or part[:400])
        assoc = sh(f"sacctmgr -nP show assoc user={user} format=Cluster,Account,Partition,QOS,GrpTRES,GrpJobs,"
                   f"MaxTRES,MaxTRESPerNode,MaxJobs,MaxSubmit,MaxWall")
        R.p("assoc (Cluster|Account|Partition|QOS|GrpTRES|GrpJobs|MaxTRES|MaxTRESPerNode|MaxJobs|MaxSubmit|MaxWall):")
        R.p("  " + assoc.replace("\n", "\n  "))
        qos_names = sorted({q for line in assoc.splitlines() for q in (line.split("|")[3:4] or [""])[0].split(",") if q})
        if qos_names:
            R.p("qos (Name|Priority|MaxWall|MaxTRESPU|MaxJobsPU|MaxSubmitPU|GrpTRES|MaxTRES):")
            R.p("  " + sh(f"sacctmgr -nP show qos {','.join(qos_names)} format=Name,Priority,MaxWall,MaxTRESPU,"
                          f"MaxJobsPU,MaxSubmitPU,GrpTRES,MaxTRES").replace("\n", "\n  "))
        R.p("fairshare:", sh(f"sshare -U -u {user} -o Account,User,RawShares,NormShares,RawUsage,EffectvUsage,FairShare"))
        R.p("my pending jobs (id|name|state|reason|gres):")
        R.p("  " + (sh(f"squeue -u {user} -h -t PD -o '%i|%j|%T|%r|%b'") or "(none)").replace("\n", "\n  "))
    with block(f"my sacct history (last {args.history_days} days)"):
        _sacct_history(user, args.history_days)


def _sacct_history(user, days):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    out = sh(f"sacct -u {user} -S {start} -nP --format=JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,AllocTRES,"
             f"NodeList,Submit,Start,Timelimit", timeout=180)
    jobs = {}
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 11:
            continue
        jid, name, state, elapsed, totcpu, maxrss, tres, nodes, submit, start_t, tlimit = f[:11]
        base, _, step = jid.partition(".")
        if not step:
            tr = dict(kv.split("=", 1) for kv in tres.split(",") if "=" in kv)
            gtype = next((k.split(":", 1)[1] for k in tr if k.startswith("gres/gpu:")), None)
            wait = None
            try:
                wait = (datetime.fromisoformat(start_t) - datetime.fromisoformat(submit)).total_seconds()
            except ValueError:
                pass
            el, cpu_s, ncpu = _dur(elapsed), _dur(totcpu), int(tr.get("cpu", 0) or 0)
            jobs[base] = {"job": base, "name": name, "family": re.sub(r"\d+", "#", name)[:44], "state": state.split()[0],
                          "elapsed_h": el / 3600 if el else None, "wait_h": wait / 3600 if wait is not None else None,
                          "cpus": ncpu, "mem_GB": (_bytes(tr.get("mem", "")) or 0) / 2**30 or None,
                          "gpu": gtype or (tr.get("gres/gpu") and "gpu"), "node": nodes,
                          "cpu_eff": cpu_s / (el * ncpu) if el and ncpu and cpu_s is not None else None,
                          "submit": submit, "limit_h": (_dur(tlimit) or 0) / 3600 or None}
        elif step == "batch" and base in jobs:
            rss = _bytes(maxrss)
            jobs[base]["maxrss_GB"] = rss / 2**30 if rss else None
    if not jobs:
        R.p("  (no jobs or sacct unavailable)", out[:300])
        return
    fams = {}
    for j in jobs.values():
        fams.setdefault(j["family"], []).append(j)
    rows = []
    for fam, js in sorted(fams.items(), key=lambda kv: -len(kv[1])):
        states = {}
        for j in js:
            states[j["state"]] = states.get(j["state"], 0) + 1
        rows.append({"family": fam, "jobs": len(js), "states": " ".join(f"{k}:{v}" for k, v in states.items()),
                     "gpus": ",".join(sorted({str(j['gpu']) for j in js if j['gpu']})),
                     "med_elapsed_h": _pct([j["elapsed_h"] for j in js], 50),
                     "max_elapsed_h": _pct([j["elapsed_h"] for j in js], 100),
                     "med_wait_h": _pct([j["wait_h"] for j in js], 50),
                     "p90_wait_h": _pct([j["wait_h"] for j in js], 90),
                     "med_cpu_eff": _pct([j["cpu_eff"] for j in js], 50),
                     "med_maxrss_GB": _pct([j.get("maxrss_GB") for j in js], 50),
                     "med_req_mem_GB": _pct([j["mem_GB"] for j in js], 50)})
    table(rows[:20], ["family", "jobs", "states", "gpus", "med_elapsed_h", "max_elapsed_h", "med_wait_h", "p90_wait_h",
                      "med_cpu_eff", "med_maxrss_GB", "med_req_mem_GB"])
    R.p("most recent 25 jobs:")
    recent = sorted(jobs.values(), key=lambda j: j["submit"])[-25:]
    table(recent, ["job", "name", "state", "gpu", "cpus", "mem_GB", "elapsed_h", "limit_h", "wait_h", "cpu_eff", "maxrss_GB"])
    R.put("sacct_families", rows)


# ── history ──────────────────────────────────────────────────────────────────

_BAR = re.compile(rb"Inference:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[(\d[\d:]*)<")


def _tqdm_bars(err_path):
    """[(n, total, elapsed_s)] of every 'Inference' tqdm bar in a log, last state of each."""
    bars, last, buf = [], None, b""
    with open(err_path, "rb") as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            parts = re.split(rb"[\r\n]", buf + chunk)
            buf = parts.pop()
            for seg in parts:
                m = _BAR.search(seg)
                if not m:
                    continue
                cur = (int(m[1]), int(m[2]), _dur(m[3].decode()))
                if last and (cur[1] != last[1] or cur[0] < last[0]):
                    bars.append(last)
                last = cur
    m = _BAR.search(buf)
    if m:
        cur = (int(m[1]), int(m[2]), _dur(m[3].decode()))
        if last and (cur[1] != last[1] or cur[0] < last[0]):
            bars.append(last)
        last = cur
    if last:
        bars.append(last)
    return bars


def _fits(cells, total):
    return any(math.ceil(cells / bs) == total for bs in (8, 16, 32, 64, 128, 256, 512, 1024))


def sec_history(args, ctx):
    with block("past extraction / sync jobs (from *_embeddings_*.out.txt + .err.txt)"):
        dirs = [REPO_ROOT] + [Path(d) for d in args.log_dirs]
        outs = sorted({p for d in dirs for p in glob.glob(str(d / "*embeddings_*.out.txt"))})
        rows, jobs = [], []
        for out in outs:
            text = Path(out).read_text(errors="replace")
            gm = re.search(r"\|\s+\d+\s+((?:NVIDIA|Tesla|Quadro)[^|]*?)\s+(?:On|Off)\s+\|", text)
            gpu = re.sub(r"NVIDIA |GeForce ", "", gm[1]).strip() if gm else None
            jid = (re.search(r"_(\d+)\.out\.txt$", out) or [None, Path(out).name])[1]
            runs = re.findall(r"Running inference \[(\w+)\] on (\S+), cells: (\d+)", text)
            err = out[: -len(".out.txt")] + ".err.txt"
            bars = _tqdm_bars(err) if os.path.exists(err) else []
            jobs.append({"job": jid, "gpu": gpu, "runs": len(runs),
                         "read_failures": text.count("WARNING: failed to read region")})
            bi = 0
            for model, wsi, cells in runs:
                cells = int(cells)
                while bi < len(bars) and not _fits(cells, bars[bi][1]):
                    bi += 1
                if bi >= len(bars):
                    continue
                n, total, el = bars[bi]
                bi += 1
                done = cells if n == total else round(cells * n / total)
                rows.append({"job": jid, "gpu": gpu, "model": model,
                             "slide": re.sub(r"(_he_image)?(\.ome)?\.tiff?$", "", Path(wsi).name)[:48],
                             "cells": cells, "done": n == total, "hours": el / 3600 if el else None,
                             "cells_s": done / el if el else None})
        R.p(f"{len(outs)} log(s), {len(rows)} slide run(s) matched to a progress bar")
        table([j for j in jobs if j["runs"]][-30:], ["job", "gpu", "runs", "read_failures"])
        summ = {}
        for r in rows:
            if r["done"] and r["hours"]:
                s = summ.setdefault((r["gpu"], r["model"]), {"gpu": r["gpu"], "model": r["model"], "runs": 0,
                                                            "cells": 0, "hours": 0.0, "rates": []})
                s["runs"] += 1
                s["cells"] += r["cells"]
                s["hours"] += r["hours"]
                s["rates"].append(r["cells_s"])
        srows = [{**s, "cells_s": s["cells"] / (s["hours"] * 3600), "p10_cells_s": _pct(s["rates"], 10),
                  "p90_cells_s": _pct(s["rates"], 90)} for s in summ.values()]
        R.p("completed slides per GPU type:")
        table(srows, ["gpu", "model", "runs", "cells", "hours", "cells_s", "p10_cells_s", "p90_cells_s"])
        R.p("last 20 slide runs:")
        table(rows[-20:], ["job", "gpu", "slide", "cells", "done", "hours", "cells_s"])
        R.put("extraction_history", {"per_gpu": [{k: v for k, v in s.items() if k != "rates"} for s in srows],
                                     "runs": rows})

    with block("past classifier runs (results_summary.yaml timing)"):
        import yaml
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        files = []
        for root in args.results_roots:
            files += glob.glob(str((REPO_ROOT / root) / "**" / "results_summary.yaml"), recursive=True)
        files = sorted(files, key=os.path.getmtime)[-400:]
        rows = []
        for p in files:
            try:
                with open(p) as f:
                    t = (yaml.load(f, Loader=loader) or {}).get("timing") or {}
            except Exception:  # noqa: BLE001
                continue
            if t:
                rows.append({"run": os.path.relpath(p, REPO_ROOT)[-90:], **{k: t.get(k) for k in [
                    "n_cells_loaded", "n_embedding_datasets", "load_seconds", "grid_seconds",
                    "seconds_per_cell_epoch", "load_seconds_per_cell_dataset"]}})
        R.p(f"{len(files)} results_summary.yaml, {len(rows)} with timing")
        if rows:
            R.p("medians: " + ", ".join(f"{k}={_fmt(_pct([r[k] for r in rows], 50))}" for k in [
                "n_cells_loaded", "load_seconds", "grid_seconds", "seconds_per_cell_epoch",
                "load_seconds_per_cell_dataset"]))
            table(rows[-10:], ["run", "n_cells_loaded", "load_seconds", "grid_seconds", "seconds_per_cell_epoch"])
        R.put("classifier_history", rows)


# ── inventory ────────────────────────────────────────────────────────────────

def _ensure_runs(args, ctx):
    """{sample: run} for the extraction configs, resolved by the real build_run_configs."""
    if "runs" not in ctx:
        from src.python.extract_embeddings.extract_embeddings import build_run_configs, load_run_config
        runs = {}
        for cp in args.configs:
            cfg = load_run_config(REPO_ROOT / cp)
            for model, mc in build_run_configs(cfg).items():
                for run in mc["inference_runs"]:
                    runs.setdefault(Path(run["output_path"]).name, {
                        **run, "model": model, "mode": cfg["mode"], "batch_size": cfg.get("batch_size", 16),
                        "save_cls": cfg.get("save_cls", True)})
        ctx["runs"] = runs
    return ctx["runs"]


def _wsi_meta(path):
    import openslide
    meta = {"wsi_gb": os.path.getsize(path) / 1e9}
    s = openslide.OpenSlide(path)
    try:
        meta["W"], meta["H"] = s.dimensions
        p = s.properties
        meta["tile"] = int(p.get("openslide.level[0].tile-width") or 0)
        meta["tile_h"] = int(p.get("openslide.level[0].tile-height") or 0)
        meta["vendor"] = p.get("openslide.vendor")
        meta["levels"] = s.level_count
    finally:
        s.close()
    try:
        import tifffile
        with tifffile.TiffFile(path) as t:
            pg = t.pages[0]
            meta["compression"] = getattr(pg.compression, "name", str(pg.compression))
            if not meta["tile"] and pg.is_tiled:
                meta["tile"], meta["tile_h"] = pg.tilewidth, pg.tilelength
            meta["tiled"] = bool(pg.is_tiled)
    except Exception:  # noqa: BLE001
        meta.setdefault("compression", None)
    return meta


def _tile_reuse(x, y, sx, sy, tw, th, batch):
    """WSI tiles decoded per cell when each batch of `batch` cells decodes every tile its crops
    touch once (OpenSlide's cache is per worker, and a worker reads whole batches): in the file
    order, in a tile-sorted order, and the floor (every tile decoded once for the slide)."""
    n = len(x)
    tx0, tx1 = x // tw, (x + sx - 1) // tw
    ty0, ty1 = y // th, (y + sy - 1) // th
    ncols = int(tx1.max()) + 1
    ntile = int(ty1.max() + 1) * ncols
    b = np.arange(n, dtype=np.int64) // batch

    def per_cell(order):
        keys = np.concatenate([b * ntile + ty[order] * ncols + tx[order] for tx in (tx0, tx1) for ty in (ty0, ty1)])
        return len(np.unique(keys)) / n

    floor = len(np.unique(np.concatenate([ty * ncols + tx for tx in (tx0, tx1) for ty in (ty0, ty1)]))) / n
    return {"tiles_per_cell_file": per_cell(np.arange(n)), "tiles_per_cell_sorted": per_cell(np.lexsort((x, tx0, ty0))),
            "tiles_per_cell_floor": floor,
            "median_step_px": float(np.median(np.hypot(np.diff(x), np.diff(y)))) if n > 1 else 0.0}


def _slide_row(name, run):
    dc = run["dataset_configs"]
    sx, sy = dc.get("x_size", 224), dc.get("y_size", 224)
    row = {"sample": name}
    try:
        row.update(_wsi_meta(dc["wsi_path"]))
    except Exception as e:  # noqa: BLE001
        row["error"] = f"wsi: {type(e).__name__}: {e}"[:100]
    try:
        with h5py.File(dc["cells_info_path"], "r") as f:
            x = f["x_start"][:].astype(np.int64)
            y = f["y_start"][:].astype(np.int64)
        row["cells"] = len(x)
        if row.get("W") and row.get("tile") and len(x):
            x = np.clip(x + dc.get("offset_x", 0), 0, row["W"] - sx)
            y = np.clip(y + dc.get("offset_y", 0), 0, row["H"] - sy)
            row.update(_tile_reuse(x, y, sx, sy, row["tile"], row.get("tile_h") or row["tile"], run["batch_size"]))
    except Exception as e:  # noqa: BLE001
        row["error"] = f"coords: {type(e).__name__}: {e}"[:100]
    emb = Path(run["output_path"]) / "embeddings_dataset.h5"
    if emb.exists():
        row["emb_gb"] = emb.stat().st_size / 1e9
        try:
            with h5py.File(emb, "r") as f:
                keys = [k for k in f if k.startswith("embeddings_")]
                row["emb_rows"] = f["cell_ids"].shape[0] if "cell_ids" in f else None
                row["emb_keys"] = len(keys)
                if keys:
                    d = f[keys[0]]
                    row["emb_layout"] = (f"{d.dtype} " + (f"chunks{d.chunks}" if d.chunks else "contiguous")
                                         + (f" {d.compression}" if d.compression else ""))
        except Exception as e:  # noqa: BLE001
            row["error"] = f"emb: {type(e).__name__}: {e}"[:100]
    return row


def _ensure_inventory(args, ctx):
    if "inventory" not in ctx:
        runs = _ensure_runs(args, ctx)
        t = _now()
        with ThreadPoolExecutor(8) as ex:
            rows = list(ex.map(lambda kv: _slide_row(*kv), runs.items()))
        ctx["inventory"] = rows
        ctx["inventory_seconds"] = _now() - t
    return ctx["inventory"]


def _counts(values):
    out = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return " | ".join(f"{k}: {v}" for k, v in sorted(out.items(), key=lambda kv: -kv[1]))


def sec_inventory(args, ctx):
    rows = _ensure_inventory(args, ctx)
    runs = ctx["runs"]
    any_run = next(iter(runs.values()))
    R.p(f"{len(rows)} slides from {args.configs} (model {any_run['model']}, mode {any_run['mode']}, "
        f"batch {any_run['batch_size']}), scanned in {ctx['inventory_seconds']:.0f}s")
    cells = [r["cells"] for r in rows if r.get("cells")]
    R.p(f"cells: total {_fmt(sum(cells))}, min {_fmt(min(cells))}, median {_fmt(_pct(cells, 50))}, "
        f"p90 {_fmt(_pct(cells, 90))}, max {_fmt(max(cells))}")
    emb_rows = [r.get("emb_rows") or 0 for r in rows]
    R.p(f"extracted rows {_fmt(sum(emb_rows))} in {sum(1 for r in rows if r.get('emb_rows'))} files, "
        f"{_fmt(sum(r.get('emb_gb') or 0 for r in rows))} GB of embeddings h5; "
        f"WSIs {_fmt(sum(r.get('wsi_gb') or 0 for r in rows))} GB")
    R.p("WSI tile size :", _counts(f"{r.get('tile')}x{r.get('tile_h')}" for r in rows))
    R.p("WSI compress. :", _counts(r.get("compression") for r in rows))
    R.p("WSI vendor    :", _counts(r.get("vendor") for r in rows))
    R.p("embeddings h5 :", _counts(r.get("emb_layout") for r in rows))
    tr = [r for r in rows if r.get("tiles_per_cell_file")]
    if tr:
        f_, s_, fl = ([r[k] for r in tr] for k in ("tiles_per_cell_file", "tiles_per_cell_sorted", "tiles_per_cell_floor"))
        R.p(f"WSI tiles decoded per cell (per-batch cache): file order median {_fmt(_pct(f_, 50))} "
            f"(p10 {_fmt(_pct(f_, 10))}, p90 {_fmt(_pct(f_, 90))}); tile-sorted median {_fmt(_pct(s_, 50))}; "
            f"floor median {_fmt(_pct(fl, 50))}; median step between consecutive cells "
            f"{_fmt(_pct([r['median_step_px'] for r in tr], 50))} px")
    errs = [r for r in rows if r.get("error")]
    if errs:
        R.p(f"{len(errs)} slide(s) with errors:")
        table(errs[:15], ["sample", "error"])
    R.p("largest 15 slides:")
    big = sorted([r for r in rows if r.get("cells")], key=lambda r: -r["cells"])[:15]
    table(big, ["sample", "cells", "emb_rows", "wsi_gb", "tile", "compression", "tiles_per_cell_file",
                "tiles_per_cell_sorted", "median_step_px"])
    path = ctx["out_dir"] / f"inventory_{ctx['job']}.csv"
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k != "sample", k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    R.p(f"full table: {path}")
    R.put("inventory_summary", {"slides": len(rows), "cells": sum(cells), "emb_rows": sum(emb_rows)})


# ── extract ──────────────────────────────────────────────────────────────────

class GpuMon:
    """Samples GPU utilisation with nvidia-smi every 0.5 s while active."""

    def __init__(self, uuid=None):
        self.uuid, self.util, self.mem_mib, self.p = uuid, None, None, None

    def __enter__(self):
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", "500"]
        if self.uuid:
            cmd += ["-i", self.uuid]
        try:
            self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.p = None
        return self

    def __exit__(self, *exc):
        if self.p is None:
            return False
        self.p.terminate()
        try:
            out = self.p.communicate(timeout=5)[0]
        except Exception:  # noqa: BLE001
            self.p.kill()
            out = ""
        vals = []
        for line in out.splitlines():
            try:
                u, m = line.split(",")
                vals.append((float(u), float(m)))
            except ValueError:
                pass
        if vals:
            self.util = float(np.mean([v[0] for v in vals]))
            self.mem_mib = max(v[1] for v in vals)
        return False


def _gpu_uuid():
    import torch
    try:
        u = str(torch.cuda.get_device_properties(0).uuid)
        return u if u.startswith("GPU-") else f"GPU-{u}"
    except Exception:  # noqa: BLE001
        return None


def _sync(dev):
    import torch
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _crc_image(im):
    return zlib.crc32(np.asarray(im).tobytes())


def _reopen_wsi(_):
    """worker_init_fn: give this DataLoader worker its own OpenSlide handle."""
    import openslide
    import torch
    base = torch.utils.data.get_worker_info().dataset
    while hasattr(base, "dataset"):
        base = base.dataset
    base.wsi = openslide.OpenSlide(base.wsi_path)


def _split_transform(tf):
    """(PIL-side steps, Normalize) of a timm eval transform [*PIL steps, (Maybe)ToTensor, Normalize]."""
    steps = list(getattr(tf, "transforms", [tf]))
    names = [type(t).__name__ for t in steps]
    k = next(i for i, n in enumerate(names) if n in ("ToTensor", "MaybeToTensor"))
    if names[k + 1:] != ["Normalize"]:
        raise ValueError(f"unexpected transform layout {names}")
    return steps[:k], steps[k + 1]


def _make_fast_dataset():
    import openslide
    import torch

    class FastPatches(torch.utils.data.Dataset):
        """Prototype reader: OpenSlide handle opened in each worker, the transform's PIL steps on
        the CPU, uint8 CHW out (ToTensor + Normalize happen on the GPU)."""

        def __init__(self, wsi_path, xs, ys, size, pil_steps, cache_bytes=0):
            self.wsi_path, self.xs, self.ys, self.size = wsi_path, xs, ys, size
            self.pil_steps, self.cache_bytes = pil_steps, cache_bytes
            self._slide, self._pid = None, None

        def __len__(self):
            return len(self.xs)

        def _handle(self):
            if self._pid != os.getpid():
                self._slide = openslide.OpenSlide(self.wsi_path)
                if self.cache_bytes and hasattr(openslide, "OpenSlideCache"):
                    self._slide.set_cache(openslide.OpenSlideCache(self.cache_bytes))
                self._pid = os.getpid()
            return self._slide

        def __getitem__(self, i):
            im = self._handle().read_region((int(self.xs[i]), int(self.ys[i])), 0, (self.size, self.size)).convert("RGB")
            for t in self.pil_steps:
                im = t(im)
            return torch.from_numpy(np.array(im)).permute(2, 0, 1), i

    return FastPatches


def _read_rate(path, xs, ys, size, cache_bytes=0):
    import openslide
    s = openslide.OpenSlide(path)
    if cache_bytes:
        s.set_cache(openslide.OpenSlideCache(cache_bytes))
    t = _now()
    for x, y in zip(xs, ys):
        s.read_region((int(x), int(y)), 0, (size, size)).convert("RGB")
    dt = _now() - t
    s.close()
    return len(xs) / dt


def _loader_rate(dl, n_batches, workers):
    """(steady-state items/s after the first round of `workers` batches, seconds to first batch)."""
    t0 = _now()
    times, sizes = [], []
    it = iter(dl)
    for i, b in enumerate(it):
        times.append(_now())
        sizes.append(len(b[0]))
        if i + 1 >= n_batches:
            break
    del it
    k = min(max(workers, 1), len(times) - 2)
    rate = sum(sizes[k + 1:]) / (times[-1] - times[k]) if len(times) > k + 1 else None
    return rate, times[0] - t0


def sec_extract(args, ctx):
    import openslide
    import torch
    from torch.utils.data import DataLoader, Subset

    from src.python.extract_embeddings.data.patch_dataset import PatchDataset
    from src.python.extract_embeddings.extract_embeddings import select_inference_provider

    runs = _ensure_runs(args, ctx)
    inv = {r["sample"]: r for r in _ensure_inventory(args, ctx)}
    name = args.slide or _pick_slide(list(inv.values()))
    run, meta = runs[name], inv.get(name, {})
    dc, bs = run["dataset_configs"], run["batch_size"]
    ncpu = _ncpu()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ex = ctx.setdefault("extract", {})
    R.p(f"slide {name}: {_fmt(meta.get('cells'))} cells, WSI {_fmt(meta.get('wsi_gb'))} GB, tile "
        f"{meta.get('tile')}x{meta.get('tile_h')} {meta.get('compression')}, model {run['model']}, batch {bs}, "
        f"{ncpu} cpus, device {dev}")
    if run["mode"] != "patch":
        R.p("only mode 'patch' is profiled here")
        return

    ds = PatchDataset(**dc)
    wsi_path, size = dc["wsi_path"], ds.x_size
    X = np.clip(ds.min_x_dataset.astype(np.int64) + ds.offset_x, 0, ds.wsi_w - ds.x_size)
    Y = np.clip(ds.min_y_dataset.astype(np.int64) + ds.offset_y, 0, ds.wsi_h - ds.y_size)
    n = len(X)
    tw = meta.get("tile") or 512
    th = meta.get("tile_h") or tw
    order_sorted = np.lexsort((X, X // tw, Y // th))
    rng = np.random.default_rng(args.seed)
    sweep = sorted({w for w in (4, 8, 16, 32, ncpu) if w <= ncpu})
    win = min(n, max(3 * max(sweep) * bs, args.e2e_cells, 4096))
    s_file = int(rng.integers(0, n - win + 1))
    idx_file = np.arange(s_file, s_file + win)
    p_sorted = int(rng.integers(0, n - win + 1))
    idx_sorted = order_sorted[p_sorted:p_sorted + win]

    with block("E1 single-process read_region rate (cells/s; cold = first touch of that region)"):
        N1 = min(args.read_cells, n)
        c_rand = rng.choice(n, N1, replace=False)
        s1 = int(rng.integers(0, n - N1 + 1))
        c_file = np.arange(s1, s1 + N1)
        p1 = int(rng.integers(0, n - N1 + 1))
        c_sort = order_sorted[p1:p1 + N1]
        rows = []
        for label, c, cache in [("random cells, cold", c_rand, 0), ("file order, cold", c_file, 0),
                                ("tile-sorted, cold", c_sort, 0), ("file order, warm", c_file, 0),
                                ("tile-sorted, warm", c_sort, 0),
                                ("file order, warm, 1 GB OpenSlideCache", c_file, 1 << 30)]:
            if cache and not hasattr(openslide, "OpenSlideCache"):
                continue
            try:
                r = _read_rate(wsi_path, X[c], Y[c], size, cache)
                rows.append({"variant": label, "cells": len(c), "cells_s": r, "ms_cell": 1000 / r})
            except Exception as e:  # noqa: BLE001
                rows.append({"variant": label, "cells": len(c), "cells_s": f"failed: {e}"[:60]})
        table(rows, ["variant", "cells", "cells_s", "ms_cell"])
        ex["read"] = rows

    with block("E2 fork-safety: workers using the OpenSlide handle opened before the fork (current PatchDataset)"):
        idx = idx_file[:args.fork_cells]
        s = openslide.OpenSlide(wsi_path)
        ref = np.array([zlib.crc32(np.asarray(s.read_region((int(X[j]), int(Y[j])), 0, (size, size)).convert("RGB")).tobytes())
                        for j in idx], dtype=np.int64)
        s.close()
        black = zlib.crc32(bytes(size * size * 3))
        old_tf = ds.transform
        ds.transform = _crc_image
        rows = []
        try:
            for label, wif in [("inherited handle (current)", None), ("handle per worker", _reopen_wsi)]:
                dl = DataLoader(Subset(ds, idx), batch_size=64, num_workers=min(8, ncpu), shuffle=False,
                                worker_init_fn=wif)
                got = np.concatenate([b[0].numpy() for b in dl])
                rows.append({"variant": label, "cells": len(idx), "mismatch_vs_single_process": int((got != ref).sum()),
                             "black_placeholders": int((got == black).sum() - (ref == black).sum())})
        finally:
            ds.transform = old_tf
        table(rows, ["variant", "cells", "mismatch_vs_single_process", "black_placeholders"])
        ex["fork"] = rows

    prov = select_inference_provider(run["model"])
    t = _now()
    prov.load_model()
    R.p(f"\nmodel {run['model']} loaded in {_now() - t:.0f}s; transform: "
        + " -> ".join(str(s).replace("\n", " ") for s in getattr(prov.transforms, "transforms", [prov.transforms])))
    ds.transform = prov.transforms
    FastPatches = _make_fast_dataset()
    try:
        pil_steps, norm = _split_transform(prov.transforms)
    except Exception as e:  # noqa: BLE001
        pil_steps, norm = None, None
        R.p(f"prototype reader unavailable: {e}")

    with block("warming the page cache for the E3/E7 windows"):
        if pil_steps is not None:
            for idx in (idx_file, idx_sorted):
                dl = DataLoader(FastPatches(wsi_path, X[idx], Y[idx], size, pil_steps), batch_size=bs,
                                num_workers=ncpu, shuffle=False)
                for _ in dl:
                    pass
            R.p(f"read {2 * win} cells")

    with block(f"E3 DataLoader supply rate, batch {bs} (images/s, CPU only; warm page cache)"):
        rows = []

        def run_loader(label, dataset, workers, kw=None):
            nb = min(len(dataset) // bs, max(12, 3 * workers))
            dl = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=dev.type == "cuda",
                            **(kw or {}))
            rate, first = _loader_rate(dl, nb, workers)
            rows.append({"variant": label, "workers": workers, "batches": nb, "images_s": rate, "first_batch_s": first})

        for w in sweep:
            run_loader("current: PatchDataset + timm transform, file order", Subset(ds, idx_file), w)
        if pil_steps is not None:
            for w in sweep:
                run_loader("prototype: own handle + uint8, tile-sorted", FastPatches(
                    wsi_path, X[idx_sorted], Y[idx_sorted], size, pil_steps), w, {"prefetch_factor": 4})
            run_loader("prototype: own handle + uint8, file order", FastPatches(
                wsi_path, X[idx_file], Y[idx_file], size, pil_steps), max(sweep), {"prefetch_factor": 4})
        run_loader("current transform, tile-sorted", Subset(ds, idx_sorted), max(sweep))
        table(rows, ["variant", "workers", "batches", "images_s", "first_batch_s"])
        ex["loader"] = rows

        R.p("CPU cost per patch (single process, 200 patches):")
        s = openslide.OpenSlide(wsi_path)
        ims = [s.read_region((int(X[j]), int(Y[j])), 0, (size, size)).convert("RGB") for j in idx_file[:200]]
        s.close()
        costs = {}
        t = _now()
        for im in ims:
            prov.transforms(im)
        costs["full timm transform"] = (_now() - t) / len(ims) * 1000
        if pil_steps is not None:
            t = _now()
            for im in ims:
                for st in pil_steps:
                    im = st(im)
                torch.from_numpy(np.array(im)).permute(2, 0, 1)
            costs["PIL steps + uint8 tensor"] = (_now() - t) / len(ims) * 1000
        R.p("  " + ", ".join(f"{k} {v:.2f} ms" for k, v in costs.items()))

    if dev.type != "cuda" and not args.allow_cpu:
        R.p("no CUDA device: skipping the GPU measurements")
        return
    model = prov.model
    uuid = _gpu_uuid()
    real = next(iter(DataLoader(Subset(ds, idx_file[:64]), batch_size=64, num_workers=0)))[0]

    with block("E4 forward_features throughput (images/s) by precision and batch size; peak GPU memory"):
        if dev.type != "cuda":
            raise RuntimeError("needs CUDA")
        rows = []
        bf16_ok = _bf16_ok(dev)
        half = torch.bfloat16 if bf16_ok else torch.float16
        configs = [("fp32 (current)", b) for b in (64, 128, 256, 512)]
        configs += [(f"{'bf16' if bf16_ok else 'fp16'} autocast", b) for b in (64, 128, 256, 512)]
        configs += [("tf32", 256), ("fp16 autocast" if bf16_ok else None, 256), (f"{str(half)[6:]} weights", 256)]
        for label, b in configs:
            if label is None:
                continue
            x = real.repeat(math.ceil(b / 64), 1, 1, 1)[:b].to(dev)
            m = model
            if "weights" in label:
                m = copy.deepcopy(model).to(half)
                x = x.to(half)
            try:
                rate, peak = _bench_forward(m.forward_features, x, _precision_ctx(label), dev)
                rows.append({"precision": label, "batch": b, "images_s": rate, "peak_GiB": peak})
            except Exception as e:  # noqa: BLE001 -- OOM or an unsupported dtype
                rows.append({"precision": label, "batch": b, "images_s": type(e).__name__})
            finally:
                del x
                if m is not model:
                    del m
                torch.cuda.empty_cache()
        table(rows, ["precision", "batch", "images_s", "peak_GiB"])
        ex["model"] = rows
        xb = torch.empty((bs, 3, 224, 224), dtype=torch.float32).pin_memory()
        xu = torch.empty((bs, 3, 224, 224), dtype=torch.uint8).pin_memory()
        res = {}
        for label, t_ in [("fp32", xb), ("uint8", xu)]:
            t_.to(dev, non_blocking=True)
            torch.cuda.synchronize()
            t = _now()
            for _ in range(10):
                t_.to(dev, non_blocking=True)
            torch.cuda.synchronize()
            res[label] = (_now() - t) / 10 * 1000
        R.p(f"host->GPU copy of one batch of {bs} (pinned): fp32 {res['fp32']:.1f} ms, uint8 {res['uint8']:.1f} ms")

    with block("E5 numerical drift vs fp32 (64 real patches of this slide)"):
        _numerics(args, ctx, prov, ds, idx_file[:64], real, dev, FastPatches, pil_steps, norm, X, Y, size, run)

    with block("E6 h5 write cost per batch (5 x 1536 float32 per cell + ids)"):
        rows = []
        dests = [("output filesystem", _tmp_dir(ctx, args.tmp_root or os.environ["DATASETS_ROOT"]))]
        loc = _local_scratch()
        if loc:
            dests.append((f"local {loc}", _tmp_dir(ctx, loc)))
        for label, d in dests:
            rows += [{"destination": label, **r} for r in _bench_writes(d / "w.h5", bs)]
        table(rows, ["destination", "method", "ms_batch"])
        ex["writes"] = rows

    with block(f"E7 end to end on {args.e2e_cells} cells (cells/s; GPU util sampled by nvidia-smi)"):
        rows = []
        N = min(args.e2e_cells, win)
        out_root = _tmp_dir(ctx, args.tmp_root or os.environ["DATASETS_ROOT"])
        # current code, untouched: PatchDataset -> provider.inference (fp32, 8 workers, h5 reopened per batch)
        d2 = copy.copy(ds)
        for attr in ("min_x_dataset", "min_y_dataset", "cell_ids_dataset", "labels_dataset"):
            setattr(d2, attr, getattr(ds, attr)[idx_file[:N]])
        d2.wsi = openslide.OpenSlide(wsi_path)
        d2.transform = None
        torch.backends.cuda.matmul.allow_tf32 = False
        with GpuMon(uuid) as mon:
            t = _now()
            prov.inference(d2, str(out_root / "current"), batch_size=bs, save_cls=run["save_cls"])
            dt = _now() - t
        rows.append({"variant": "current code (fp32, 8 workers, file order)", "short": "current", "cells": N,
                     "cells_s": N / dt, "gpu_util_%": mon.util})
        if pil_steps is not None and hasattr(prov, "tokens_to_remove"):
            fast = "bf16" if _bf16_ok(dev) else "fp16"
            pw = args.proto_workers or max(2, ncpu - 2)
            for label, idx, prec in [(f"prototype {fast}, tile-sorted", idx_sorted[:N], fast),
                                     ("prototype fp32, tile-sorted", idx_sorted[:N], "fp32"),
                                     (f"prototype {fast}, file order", idx_file[:N], fast)]:
                r = _prototype(prov, FastPatches, wsi_path, X, Y, idx, size, pil_steps, norm, bs, pw, prec,
                               out_root / "proto.h5", dev, uuid)
                rows.append({"variant": f"{label}, {pw} workers", "short": label, "cells": N, **r})
        table(rows, ["variant", "cells", "cells_s", "gpu_util_%", "write_wait_s"])
        ex["e2e"] = rows
        total = sum(r.get("cells") or 0 for r in inv.values())
        if total and rows:
            R.p(f"=> all {_fmt(total)} cells of the configs on this GPU: " + "; ".join(
                f"{r['short']} {total / r['cells_s'] / 3600:.1f} GPU-h"
                for r in rows if isinstance(r.get("cells_s"), float)))

    if not args.skip_staging:
        with block("E8 staging the WSI to node-local scratch"):
            loc = _local_scratch()
            wsi_fs = sh(f"df -PT {wsi_path} | tail -1").split()
            loc_fs = sh(f"df -PT {loc} | tail -1").split() if loc else []
            free = shutil.disk_usage(loc).free if loc else 0
            if not loc or (wsi_fs and loc_fs and wsi_fs[0] == loc_fs[0]):
                R.p(f"no separate local scratch ({loc})")
            elif free < 1.5 * os.path.getsize(wsi_path):
                R.p(f"not enough space in {loc}: {free / 1e9:.0f} GB free")
            else:
                d = _tmp_dir(ctx, loc)
                dst = d / Path(wsi_path).name
                t = _now()
                shutil.copyfile(wsi_path, dst)
                with open(dst, "rb+") as f:
                    os.fsync(f.fileno())
                    if hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                dt = _now() - t
                R.p(f"copied {os.path.getsize(wsi_path) / 1e9:.2f} GB to {loc_fs[1] if len(loc_fs) > 1 else loc} "
                    f"in {dt:.1f}s ({os.path.getsize(wsi_path) / 1e6 / dt:.0f} MB/s sequential read from the WSI filesystem)")
                N8 = min(400, n)
                ca, cb = rng.choice(n, N8, replace=False), rng.choice(n, N8, replace=False)
                r_net = _read_rate(wsi_path, X[ca], Y[ca], size)
                r_loc = _read_rate(str(dst), X[cb], Y[cb], size)
                R.p(f"random cold reads: shared filesystem {r_net:.0f} cells/s, local copy {r_loc:.0f} cells/s")
                ex["staging"] = {"copy_MB_s": os.path.getsize(wsi_path) / 1e6 / dt, "net_cells_s": r_net, "local_cells_s": r_loc}
                dst.unlink()

    if args.compile and dev.type == "cuda":
        with block("E9 torch.compile(forward_features), batch 256, fastest autocast"):
            label = "bf16 autocast" if _bf16_ok(dev) else "fp16 autocast"
            x = real.repeat(4, 1, 1, 1).to(dev)
            cf = torch.compile(model.forward_features)
            t = _now()
            with torch.inference_mode(), _precision_ctx(label)():
                cf(x)
            torch.cuda.synchronize()
            R.p(f"compile (first call) {_now() - t:.0f}s")
            rate, peak = _bench_forward(cf, x, _precision_ctx(label), dev)
            R.p(f"compiled {label}: {rate:.0f} images/s, peak {peak:.1f} GiB")
            ex["compile"] = {"images_s": rate}


def _bf16_ok(dev):
    import torch
    return dev.type != "cuda" or torch.cuda.is_bf16_supported()


def _pick_slide(rows):
    cand = [r for r in rows if (r.get("cells") or 0) >= 100_000 and r.get("emb_rows") == r.get("cells") and r.get("tile")]
    cand = cand or [r for r in rows if r.get("cells") and r.get("tile")]
    cand.sort(key=lambda r: (r["cells"], r["sample"]))
    return cand[len(cand) // 2]["sample"]


def _precision_ctx(label):
    import torch
    if "autocast" in label:
        dt = torch.bfloat16 if label.startswith("bf16") else torch.float16
        return lambda: torch.autocast("cuda", dtype=dt)
    if label.startswith("tf32"):
        @contextlib.contextmanager
        def tf32():
            old = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                yield
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old
        return tf32
    return contextlib.nullcontext


def _bench_forward(fwd, x, ctx_fn, dev, min_s=5.0, max_iters=20):
    import torch
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), ctx_fn():
        for _ in range(2):
            fwd(x)
        _sync(dev)
        t, k = _now(), 0
        while True:
            fwd(x)
            _sync(dev)
            k += 1
            if k >= 3 and (_now() - t >= min_s or k >= max_iters):
                break
    return k * len(x) / (_now() - t), torch.cuda.max_memory_allocated() / 2**30


def _tokens(prov, out):
    import torch
    S = prov.num_patches_per_side
    idx = torch.tensor([prov.tokens_to_remove + r * S + c for r, c in prov.patches_to_save.values()], device=out.device)
    return out[:, idx].float(), out[:, 0].float()


def _numerics(args, ctx, prov, ds, idx, real, dev, FastPatches, pil_steps, norm, X, Y, size, run):
    import torch
    import torch.nn.functional as F
    model = prov.model
    x = real.to(dev)
    with torch.inference_mode():
        ref_t, ref_c = _tokens(prov, model.forward_features(x))
    ref_mean = ref_t.mean(1)

    def drift(t, c):
        mean = t.mean(1)
        cos_tok = F.cosine_similarity(t, ref_t, dim=-1)
        return {"min_cos_token": float(cos_tok.min()), "mean_cos_token": float(cos_tok.mean()),
                "min_cos_central_mean": float(F.cosine_similarity(mean, ref_mean, dim=-1).min()),
                "rel_l2_central_mean": float(((mean - ref_mean).norm(dim=-1) / ref_mean.norm(dim=-1)).max()),
                "min_cos_cls": float(F.cosine_similarity(c, ref_c, dim=-1).min())}

    rows = []
    emb = Path(run["output_path"]) / "embeddings_dataset.h5"
    if emb.exists():
        want = {(c.decode() if isinstance(c, bytes) else str(c)): i for i, c in enumerate(ds.cell_ids_dataset[idx])}
        with h5py.File(emb, "r") as f:
            stored_ids = f["cell_ids"][()]
            pos = {}
            for r, c in enumerate(stored_ids):
                c = c.decode() if isinstance(c, bytes) else str(c)
                if c in want:
                    pos[want[c]] = r
            if len(pos) == len(idx):
                keys = list(prov.patches_to_save)
                st = np.stack([np.stack([f[f"embeddings_{k}"][pos[i]] for k in keys]) for i in range(len(idx))])
                sc = np.stack([f["embeddings_cls"][pos[i]] for i in range(len(idx))]) if "embeddings_cls" in f else None
                st_t = torch.from_numpy(st).to(dev)
                sc_t = torch.from_numpy(sc).to(dev) if sc is not None else ref_c
                rows.append({"vs": "stored embeddings (recomputed fp32 here)",
                             "max_abs_diff": float((st_t - ref_t).abs().max()), **drift(st_t, sc_t)})
            else:
                R.p(f"stored embeddings: only {len(pos)}/{len(idx)} of these cells found")
    for label in ["tf32", "fp16 autocast", "bf16 autocast", "bf16 weights", "fp16 weights"]:
        if label.startswith("bf16") and not _bf16_ok(dev):
            continue
        m, xx = model, x
        if "weights" in label:
            dt = torch.bfloat16 if label.startswith("bf16") else torch.float16
            m, xx = copy.deepcopy(model).to(dt), x.to(dt)
        with torch.inference_mode(), _precision_ctx(label)():
            t, c = _tokens(prov, m.forward_features(xx))
        rows.append({"vs": label, "max_abs_diff": float((t - ref_t).abs().max()), **drift(t, c)})
        if m is not model:
            del m
            torch.cuda.empty_cache()
    table(rows, ["vs", "max_abs_diff", "min_cos_token", "mean_cos_token", "min_cos_central_mean",
                 "rel_l2_central_mean", "min_cos_cls"])
    ctx["extract"]["numerics"] = rows
    if pil_steps is not None:
        fp = FastPatches(ds.wsi_path, X[idx], Y[idx], size, pil_steps)
        u = torch.stack([fp[i][0] for i in range(len(idx))]).to(dev)
        mean = torch.tensor(norm.mean, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(norm.std, device=dev).view(1, 3, 1, 1)
        g = u.float().div_(255).sub_(mean).div_(std)
        R.p(f"uint8 + normalise on GPU vs current CPU transform: max abs input difference {float((g - x).abs().max()):.3g}")


def _bench_writes(fn, bs, n_batches=24, D=1536, keys=5):
    data = np.random.default_rng(0).random((bs, D), dtype=np.float32)
    ids = [f"cell{i:08d}-1" for i in range(bs)]

    def create(nrows):
        with h5py.File(fn, "w") as f:
            for k in range(keys):
                f.create_dataset(f"e{k}", (nrows, D), "f4")
            f.create_dataset("ids", (nrows,), dtype=h5py.string_dtype("utf-8"))

    def write(f, sel):
        for k in range(keys):
            f[f"e{k}"][sel] = data
        f["ids"][sel] = ids

    rows = []
    create(n_batches * bs)
    t = _now()
    for b in range(n_batches):
        with h5py.File(fn, "a") as f:
            write(f, slice(b * bs, (b + 1) * bs))
    rows.append({"method": "reopen file per batch (current)", "ms_batch": (_now() - t) / n_batches * 1000})
    create(n_batches * bs)
    with h5py.File(fn, "a") as f:
        t = _now()
        for b in range(n_batches):
            write(f, slice(b * bs, (b + 1) * bs))
        f.flush()
        rows.append({"method": "file kept open, contiguous rows", "ms_batch": (_now() - t) / n_batches * 1000})
    nrows = 8 * n_batches * bs
    create(nrows)
    rng = np.random.default_rng(1)
    with h5py.File(fn, "a") as f:
        t = _now()
        for b in range(n_batches):
            write(f, np.sort(rng.choice(nrows, bs, replace=False)))
        f.flush()
        rows.append({"method": "file kept open, scattered rows (tile-sorted reads)", "ms_batch": (_now() - t) / n_batches * 1000})
    os.remove(fn)
    return rows


def _prototype(prov, FastPatches, wsi_path, X, Y, idx, size, pil_steps, norm, bs, workers, prec, out_file, dev, uuid):
    """The suggested loop: per-worker handle, uint8 batches normalised on the GPU, autocast, rows
    written back to their original order by a writer thread into an h5 file kept open."""
    import torch
    from torch.utils.data import DataLoader
    n = len(idx)
    write_pos = np.argsort(np.argsort(idx, kind="stable"), kind="stable")  # rank = row in an order-preserving file
    keys = list(prov.patches_to_save) + ["cls"]
    D = prov.embedding_dim
    mean = torch.tensor(norm.mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(norm.std, device=dev).view(1, 3, 1, 1)
    dl = DataLoader(FastPatches(wsi_path, X[idx], Y[idx], size, pil_steps), batch_size=bs, shuffle=False,
                    num_workers=workers, pin_memory=True, prefetch_factor=4)
    f = h5py.File(out_file, "w")
    dsets = [f.create_dataset(f"embeddings_{k}", (n, D), "f4") for k in keys]
    q: queue.Queue = queue.Queue(maxsize=8)
    waited = [0.0]

    def writer():
        while True:
            item = q.get()
            if item is None:
                return
            pos, arr = item
            o = np.argsort(pos)
            p = pos[o]
            for j, d in enumerate(dsets):
                d[p] = arr[o, j]

    th = threading.Thread(target=writer, daemon=True)
    th.start()
    ctx_fn = _precision_ctx("fp32" if prec == "fp32" else f"{prec} autocast")
    try:
        with GpuMon(uuid) as mon:
            t = _now()
            with torch.inference_mode():
                for xb, ib in dl:
                    x = xb.to(dev, non_blocking=True).float().div_(255).sub_(mean).div_(std)
                    with ctx_fn():
                        out = prov.model.forward_features(x)
                    tok, cls = _tokens(prov, out)
                    arr = torch.cat([tok, cls[:, None]], 1).cpu().numpy()
                    tw = _now()
                    q.put((write_pos[ib.numpy()], arr))
                    waited[0] += _now() - tw
            q.put(None)
            th.join()
            dt = _now() - t
    finally:
        f.close()
        os.remove(out_file)
    return {"cells_s": n / dt, "gpu_util_%": mon.util, "write_wait_s": waited[0]}


# ── classify ─────────────────────────────────────────────────────────────────

def _proc_read(a):
    from src.python.utils import loading_functions as lf
    wsi, base, keys, nrows, dim = a
    out = np.empty((nrows, dim), np.float32)
    lf._read_mean_rows((wsi, base, keys, np.arange(nrows), np.arange(nrows), out, 8192))
    return 0


def _clf_splits(args, base):
    import yaml
    with open(REPO_ROOT / args.clf_config) as f:
        cfg = yaml.safe_load(f)
    cap = (cfg.get("data") or {}).get("max_cells_per_wsi")
    nrows = {}

    def rows(w):
        if w not in nrows:
            try:
                with h5py.File(base / w / "embeddings_dataset.h5", "r") as f:
                    nrows[w] = f["cell_labels"].shape[0]
            except Exception:  # noqa: BLE001
                nrows[w] = None
        return nrows[w]

    out = []
    for i, sp in enumerate(cfg.get("splits") or []):
        r = {"split": i}
        missing = 0
        for side in ("train", "val", "test"):
            tot = 0
            for w, v in (sp.get(side) or {}).items():
                if w == "cell_type_proportions" or not isinstance(v, (int, float)):
                    continue
                n = rows(w)
                missing += n is None
                tot += int(min(n or 0, cap or 1 << 62) * float(v))
            r[side] = tot
        r["missing_wsis"] = missing
        r["train_val_GB_fp32"] = (r["train"] + r["val"]) * 1536 * 4 / 1e9
        out.append(r)
    hp = cfg.get("hparams") or {}
    trials = int(np.prod([len(v["values"]) if v.get("tune") and isinstance(v.get("values"), list) else 1
                          for v in hp.values()])) if hp else 1
    tr = cfg.get("training") or {}
    return out, trials, int(tr.get("num_epochs", 30)), int(tr.get("batch_size", 256)), nrows


def sec_classify(args, ctx):
    import torch

    from src.python.classifier_training import trainer as tr
    from src.python.utils import loading_functions as lf

    base = Path(os.environ["DATASETS_ROOT"]) / f"{args.clf_model}_h5" / args.clf_correction
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cl = ctx.setdefault("classify", {})
    splits, trials, epochs, cbs, nrows = [], 1, 30, 256, {}
    with block(f"C0 split sizes of {args.clf_config} (embeddings in {base})"):
        splits, trials, epochs, cbs, nrows = _clf_splits(args, base)
        table(splits, ["split", "train", "val", "test", "train_val_GB_fp32", "missing_wsis"])
        R.p(f"{trials} grid trial(s) per split, up to {epochs} epochs, batch {cbs}")
        if dev.type == "cuda":
            R.p(f"this GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
        cl["splits"] = splits

    with block(f"C1 embeddings read throughput (first {args.read_rows} rows per slide, mean of {len(args.clf_datasets)} datasets)"):
        avail = sorted([w for w in (os.listdir(base) if base.exists() else [])
                        if (base / w / "embeddings_dataset.h5").exists()], key=lambda w: zlib.crc32(w.encode()))
        big = []
        for w in avail:
            n = nrows.get(w)
            if n is None:
                try:
                    with h5py.File(base / w / "embeddings_dataset.h5", "r") as f:
                        n = f["cell_labels"].shape[0]
                except Exception:  # noqa: BLE001
                    n = 0
            if n >= args.read_rows:
                big.append(w)
        k = min(args.read_slides, len(big) // 4)
        if k == 0:
            raise RuntimeError(f"need 4 x slides with >= {args.read_rows} rows, found {len(big)}")
        groups = [big[i * k:(i + 1) * k] for i in range(4)]
        with h5py.File(base / groups[0][0] / "embeddings_dataset.h5", "r") as f:
            dim = f[f"embeddings_{args.clf_datasets[0]}"].shape[1]
        Rr = args.read_rows
        rows = []

        def go(label, slides, keys, workers, mode):
            t = _now()
            if mode == "thread":
                outs = [np.empty((Rr, dim), np.float32) for _ in slides]
                with ThreadPoolExecutor(workers) as ex:
                    list(ex.map(lf._read_mean_rows, [(w, str(base), keys, np.arange(Rr), np.arange(Rr), o, 8192)
                                                     for w, o in zip(slides, outs)]))
            else:
                with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
                    list(ex.map(_proc_read, [(w, str(base), keys, Rr, dim) for w in slides]))
            dt = _now() - t
            mb = len(slides) * Rr * len(keys) * dim * 4 / 1e6
            rows.append({"variant": label, "slides": len(slides), "cells_s": len(slides) * Rr / dt, "MB_s": mb / dt})

        keys = list(args.clf_datasets)
        go("1 thread (cold)", groups[0], keys, 1, "thread")
        go(f"{k} threads, one per slide (current: ThreadPool(9)) (cold)", groups[1], keys, k, "thread")
        go(f"{k} processes (cold)", groups[2], keys, k, "process")
        go(f"{k} threads, 1 dataset instead of {len(keys)} (cold)", groups[3], keys[:1], k, "thread")
        go(f"{k} threads, warm re-read", groups[0], keys, k, "thread")
        table(rows, ["variant", "slides", "cells_s", "MB_s"])
        cl["read"] = rows

        from src.python.code_configs.mappings import mapping_factory
        mapping = mapping_factory(args.clf_mapping)
        t = _now()
        tot = 0
        for w in groups[0]:
            r_, _ = lf._labelled_rows((w, str(base), mapping, False, None, 0))
            tot += nrows.get(w) or len(r_)
        R.p(f"label pass (_labelled_rows, warm): {(_now() - t) / tot * 1e6:.2f} s per million cells")

    with block(f"C2 training-loop variants on {args.clf_cells:,} synthetic cells x 1536 (MLP hidden 32)"):
        _train_variants(args, ctx, tr, dev, cbs)

    with block("C3 extrapolation per split (from C0-C2)"):
        _extrapolate(ctx, splits, trials, epochs)


def _train_variants(args, ctx, tr, dev, cbs):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    N, D, C = args.clf_cells, 1536, 12
    X = np.random.default_rng(0).standard_normal((N, D), dtype=np.float32)
    t = _now()
    Xc = torch.tensor(X)
    copy_s = _now() - t
    del Xc
    Xt = torch.from_numpy(X)
    y = torch.randint(0, C, (N,))
    w = torch.ones(N)
    crit = torch.nn.CrossEntropyLoss(reduction="none")
    R.p(f"torch.tensor(numpy) copy (what the former trainer did): {copy_s:.1f}s and +{N * D * 4 / 1e9:.1f} GB; "
        f"torch.from_numpy shares the memory")

    def models(k=1):
        out = []
        for i in range(k):
            torch.manual_seed(i)
            m = tr.MLP(D, 32, C, 0.2).to(dev)
            out.append((m, torch.optim.AdamW(m.parameters(), lr=5e-4)))
        return out

    def run(label, batches, k=1, sync_each=True):
        ms = models(k)
        acc = torch.zeros((), device=dev)
        t0 = _now()
        first, seen, t1 = None, 0, None
        for xb, yb, wb in batches:
            xb, yb, wb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True), wb.to(dev, non_blocking=True)
            for m, opt in ms:
                loss = (crit(m(xb), yb) * wb).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if sync_each:
                    loss.item()
                else:
                    acc += loss.detach()
            if first is None:
                _sync(dev)
                first, t1 = _now() - t0, _now()
            else:
                seen += len(yb)
            if _now() - t0 > args.clf_seconds:
                break
        _sync(dev)
        rate = seen / (_now() - t1) if seen else None
        rows.append({"variant": label, "models": k, "cells_s": rate, "first_batch_s": first,
                     "epoch_s_per_M_cells": 1e6 / rate if rate else None})
        return rate

    def gpu_batches(Xg, yg, wg, b):
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, b):
            j = perm[i:i + b]
            yield Xg[j], yg[j], wg[j]

    def cpu_chunk_batches(b, chunk=64):
        perm = torch.randperm(N)
        for c in range(0, N, b * chunk):
            j = perm[c:c + b * chunk]
            xc = Xt.index_select(0, j)
            if dev.type == "cuda":
                xc = xc.pin_memory()
            xc, yc, wc = xc.to(dev, non_blocking=True), y[j].to(dev), w[j].to(dev)
            for i in range(0, len(j), b):
                yield xc[i:i + b], yc[i:i + b], wc[i:i + b]

    def trainer_batches(b):
        # what trainer.train_and_evaluate does (chunked index_select into pinned memory)
        order = tr._epoch_order(N, None)
        for _, (xc, yc, wc) in tr._device_chunks((Xt, y, w), str(dev), b * tr._CHUNK_BATCHES, order):
            for i in range(0, len(yc), b):
                yield xc[i:i + b], yc[i:i + b], wc[i:i + b]

    rows = []
    cl = ctx.setdefault("classify", {})
    cl["train_dataloader"] = run(f"former trainer: TensorDataset + DataLoader(num_workers=16), batch {cbs}",
                                 DataLoader(TensorDataset(Xt, y, w), batch_size=cbs, shuffle=True, num_workers=16))
    run("same, num_workers=0", DataLoader(TensorDataset(Xt, y, w), batch_size=cbs, shuffle=True, num_workers=0))
    cl["train_cpu_chunks"] = run(f"CPU tensor, batched index_select in chunks of 64 batches, batch {cbs}",
                                 cpu_chunk_batches(cbs), sync_each=False)
    cl["train_current"] = run(f"current trainer (chunks into pinned memory), batch {cbs}",
                              trainer_batches(cbs), sync_each=False)
    fits = dev.type == "cuda" and N * D * 4 < 0.6 * torch.cuda.mem_get_info()[0]
    if fits:
        Xg, yg, wg = Xt.to(dev), y.to(dev), w.to(dev)
        run(f"data on GPU, batch {cbs}, loss.item() every step", gpu_batches(Xg, yg, wg, cbs))
        cl["train_gpu"] = run(f"data on GPU, batch {cbs}, no per-step sync", gpu_batches(Xg, yg, wg, cbs), sync_each=False)
        run(f"data on GPU, batch {cbs}, 3 grid trials share each batch", gpu_batches(Xg, yg, wg, cbs), k=3, sync_each=False)
        run("data on GPU, batch 4096 (changes the optimisation)", gpu_batches(Xg, yg, wg, 4096), sync_each=False)
    table(rows, ["variant", "models", "cells_s", "first_batch_s", "epoch_s_per_M_cells"])

    # eval pass, as run once per epoch on val
    M = min(N, 200_000)
    m = models()[0][0].eval()
    t = _now()
    with torch.no_grad():
        for xb, yb, wb in DataLoader(TensorDataset(Xt[:M], y[:M], w[:M]), batch_size=cbs, num_workers=16):
            lo = m(xb.to(dev))
            (crit(lo, yb.to(dev)) * wb.to(dev)).sum().item()
            lo.argmax(1).cpu()
            torch.softmax(lo, 1).cpu()
    cl["eval_dataloader"] = M / (_now() - t)
    t = _now()
    with torch.no_grad():
        tr._run_forward_pass(m, Xt, y, w, crit, str(dev), cbs, return_probs=False)
    cl["eval_current"] = N / (_now() - t)
    msg = (f"eval pass: former DataLoader {cl['eval_dataloader']:,.0f} cells/s, "
           f"current trainer (batch {cbs}) {cl['eval_current']:,.0f} cells/s")
    if fits:
        t = _now()
        with torch.no_grad():
            for i in range(0, N, 65536):
                lo = m(Xg[i:i + 65536])
                crit(lo, yg[i:i + 65536]).sum()
                lo.argmax(1).cpu()
                torch.softmax(lo, 1).cpu()
        _sync(dev)
        cl["eval_gpu"] = N / (_now() - t)
        msg += f", data on GPU in batches of 65536 {cl['eval_gpu']:,.0f} cells/s"
        del Xg, yg, wg
        torch.cuda.empty_cache()
    R.p(msg)
    cl["train_rows"] = rows


def _extrapolate(ctx, splits, trials, epochs):
    import torch
    cl = ctx.get("classify", {})
    cur_t, cur_e = cl.get("train_current"), cl.get("eval_current")
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    reads = {r["variant"]: r["cells_s"] for r in cl.get("read", [])}
    read_cur = next((v for k, v in reads.items() if "current" in k), None)
    read_one = next((v for k, v in reads.items() if "1 dataset" in k), None)
    if not (splits and cur_t and cur_e):
        R.p("not enough measurements")
        return
    rows = []
    for s in splits:
        n_load = s["train"] + s["val"] + s["test"]
        # data on the GPU if train+val fit in 80% of it (fp32, else fp16), otherwise chunked CPU gather
        mode = ("gpu fp32" if s["train_val_GB_fp32"] < 0.8 * gpu_gb else
                "gpu fp16" if s["train_val_GB_fp32"] / 2 < 0.8 * gpu_gb else "cpu chunks")
        best_t = cl.get("train_gpu") if mode.startswith("gpu") and cl.get("train_gpu") else cl.get("train_cpu_chunks")
        best_e = cl.get("eval_gpu") if mode.startswith("gpu") and cl.get("eval_gpu") else cur_e
        r = {"split": s["split"], "train": s["train"], "val": s["val"], "best_mode": mode,
             "epoch_min_now": (s["train"] / cur_t + s["val"] / cur_e) / 60,
             "epoch_min_best": (s["train"] / best_t + s["val"] / best_e) / 60 if best_t else None,
             "load_min_now": n_load / read_cur / 60 if read_cur else None,
             "load_min_1dataset": n_load / read_one / 60 if read_one else None}
        r["job_h_now"] = (r["load_min_now"] or 0) / 60 + r["epoch_min_now"] * epochs * trials / 60
        if r["epoch_min_best"]:
            r["job_h_best"] = (r["load_min_1dataset"] or r["load_min_now"] or 0) / 60 + r["epoch_min_best"] * epochs * trials / 60
        rows.append(r)
    table(rows, ["split", "train", "val", "best_mode", "load_min_now", "load_min_1dataset", "epoch_min_now",
                 "epoch_min_best", "job_h_now", "job_h_best"])
    R.p(f"(job_h = load + {trials} trials x {epochs} epochs, i.e. without early stopping; cold reads)")
    cl["extrapolation"] = rows


# ── main ─────────────────────────────────────────────────────────────────────

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", choices=SECTIONS, default=SECTIONS)
    ap.add_argument("--configs", nargs="+", default=["configs/extract_embeddings/uni2_224.json"],
                    help="extraction JSON configs (relative to the repo root)")
    ap.add_argument("--slide", help="slide to profile in 'extract' (default: a median-sized extracted slide)")
    ap.add_argument("--read-cells", type=int, default=800, help="cells per single-process read measurement")
    ap.add_argument("--fork-cells", type=int, default=4096, help="cells for the fork-safety check")
    ap.add_argument("--e2e-cells", type=int, default=6144, help="cells per end-to-end run")
    ap.add_argument("--proto-workers", type=int, help="DataLoader workers of the prototype (default cpus - 2)")
    ap.add_argument("--no-compile", dest="compile", action="store_false", help="skip the torch.compile test")
    ap.add_argument("--skip-staging", action="store_true", help="skip copying the WSI to local scratch")
    ap.add_argument("--tmp-root", help="where temporary h5 files go (default DATASETS_ROOT)")
    ap.add_argument("--clf-config", default="configs/train_cross_cohort.yaml")
    ap.add_argument("--clf-model", default="UNI2")
    ap.add_argument("--clf-correction", default="raw")
    ap.add_argument("--clf-mapping", default="simplified_broad")
    ap.add_argument("--clf-datasets", nargs="+", default=CENTRAL)
    ap.add_argument("--clf-cells", type=int, default=1_000_000, help="synthetic cells for the training benchmark")
    ap.add_argument("--clf-seconds", type=float, default=20.0, help="time cap per training variant")
    ap.add_argument("--read-rows", type=int, default=65536, help="rows read per slide in C1")
    ap.add_argument("--read-slides", type=int, default=4, help="slides per read variant in C1")
    ap.add_argument("--history-days", type=int, default=90)
    ap.add_argument("--log-dirs", nargs="*", default=[], help="extra directories with *_embeddings_*.out.txt logs")
    ap.add_argument("--results-roots", nargs="*", default=["outputs_classifier"])
    ap.add_argument("--out-dir", default="profile_reports")
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)  # local testing only
    args = ap.parse_args(argv)

    job = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = {"job": job, "out_dir": out_dir}
    gpu = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = re.sub(r"[^a-z0-9]+", "", torch.cuda.get_device_name(0).lower().replace("nvidia", "").replace("geforce", ""))
    except Exception:  # noqa: BLE001
        pass
    R.p(f"profile_pipeline | sections {args.sections} | gpu {gpu} | job {job}")
    funcs = {"env": sec_env, "cluster": sec_cluster, "history": sec_history, "inventory": sec_inventory,
             "extract": sec_extract, "classify": sec_classify}
    t0 = _now()
    try:
        for s in SECTIONS:
            if s in args.sections:
                with block(s, top=True):
                    funcs[s](args, ctx)
    finally:
        if not args.keep_tmp:
            for d in ctx.get("tmp_dirs", []):
                shutil.rmtree(d, ignore_errors=True)
        R.p(f"\ntotal {_now() - t0:.0f}s")
        R.put("extract", ctx.get("extract"))
        R.put("classify", ctx.get("classify"))
        stem = out_dir / f"{gpu}_{job}"
        Path(f"{stem}.txt").write_text("\n".join(R.lines) + "\n")
        with open(f"{stem}.json", "w") as f:
            json.dump(R.data, f, indent=1, default=_json_default)
        print(f"report: {stem}.txt (+ .json)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
