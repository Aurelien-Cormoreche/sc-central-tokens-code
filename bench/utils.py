"""Small shared helpers: atomic writes, JSON with numpy, logging, formatting."""
from __future__ import annotations

import contextlib
import fnmatch
import json
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np

LOG = logging.getLogger("bench")


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname).1s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger("bench")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_bench", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(fmt)
        h._bench = True  # type: ignore[attr-defined]
        root.addHandler(h)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)


# ── atomic writes ──────────────────────────────────────────────────────────────

def tmp_path_for(path: Path, tag: Optional[str] = None) -> Path:
    tag = tag or f"{socket.gethostname()}-{os.getpid()}"
    return path.with_name(f"{path.name}.tmp-{tag}")


def fsync_and_replace(tmp: Path, final: Path) -> None:
    """fsync tmp, rename it over final (atomic on POSIX within a filesystem), fsync the dir."""
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    with contextlib.suppress(OSError):
        dfd = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


@contextlib.contextmanager
def atomic_path(final: os.PathLike, tag: Optional[str] = None) -> Iterator[Path]:
    """Yield a temporary path next to `final`; on success it replaces `final` atomically, on
    error it is removed. Readers therefore only ever see a complete file or none."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path_for(final, tag)
    try:
        yield tmp
        fsync_and_replace(tmp, final)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


class _NpEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _clean_nan(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    return obj


def dumps(obj: Any, indent: Optional[int] = None) -> str:
    return json.dumps(_clean_nan(obj), cls=_NpEncoder, indent=indent, allow_nan=False)


def write_json(path: os.PathLike, obj: Any, indent: Optional[int] = 1) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(dumps(obj, indent=indent))


def read_json(path: os.PathLike, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default


def append_jsonl(path: os.PathLike, record: dict) -> None:
    """One O_APPEND write per record: concurrent writers interleave whole lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (dumps(record) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def read_jsonl(path: os.PathLike) -> list[dict]:
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    with contextlib.suppress(json.JSONDecodeError):
                        out.append(json.loads(line))
    except FileNotFoundError:
        pass
    return out


def write_npy(path: os.PathLike, arr: np.ndarray) -> None:
    with atomic_path(path) as tmp:
        with open(tmp, "wb") as fh:
            np.save(fh, arr)


# ── matching / formatting ──────────────────────────────────────────────────────

def matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def fmt_hours(h: float) -> str:
    return f"{h:.1f}h" if h >= 0.1 else f"{h * 60:.0f}m"


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}TB"


def print_table(rows: list[dict], columns: Optional[list[str]] = None, file=None) -> None:
    file = file or sys.stdout
    if not rows:
        print("(no rows)", file=file)
        return
    columns = columns or list(rows[0].keys())
    cells = [[str(r.get(c, "")) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)), file=file)
    print("  ".join("-" * w for w in widths), file=file)
    for row in cells:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)), file=file)


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.t0
