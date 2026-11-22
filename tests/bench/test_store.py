"""Atomic, validated embedding files."""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from bench.extract.store import H5Writer, is_valid  # noqa: E402
from bench.utils import atomic_path, read_json, write_json  # noqa: E402


def _meta(n):
    return {"cell_ids": np.array([f"c{i}".encode() for i in range(n)]), "manifest_row": np.arange(n)}


def test_write_validate(tmp_path):
    p = tmp_path / "a" / "embeddings_dataset.h5"
    w = H5Writer(p, 10, {"cls": 4, "central": 3}, _meta(10))
    w.write(0, {"cls": np.ones((6, 4)), "central": np.ones((6, 3))})
    assert not p.exists()                         # nothing visible until finalize
    w.write(6, {"cls": np.ones((4, 4)), "central": np.ones((4, 3))})
    w.finalize({"manifest_hash": "h1", "model": "m"})
    assert is_valid(p, 10, "h1", ("cls", "central")) == (True, "ok")
    assert not is_valid(p, 11, "h1", ("cls",))[0]          # manifest has more cells now
    assert not is_valid(p, 10, "h2", ("cls",))[0]          # manifest regenerated
    assert not is_valid(p, 10, "h1", ("nucleus",))[0]      # readout not extracted
    assert list(tmp_path.glob("**/*.tmp*")) == []


def test_incomplete_and_nonfinite_never_land(tmp_path):
    p = tmp_path / "b.h5"
    w = H5Writer(p, 5, {"cls": 2}, _meta(5))
    w.write(0, {"cls": np.ones((3, 2))})
    with pytest.raises(RuntimeError):
        w.finalize({"manifest_hash": "h"})
    w.abort()
    assert not p.exists()
    w = H5Writer(p, 2, {"cls": 2}, _meta(2))
    w.write(0, {"cls": np.array([[1.0, np.nan], [1.0, 1.0]])})
    with pytest.raises(RuntimeError, match="non-finite"):
        w.finalize({"manifest_hash": "h"})
    assert not p.exists()
    assert list(tmp_path.glob("*.tmp*")) == []


def test_corrupt_file_is_invalid(tmp_path):
    p = tmp_path / "c.h5"
    p.write_bytes(b"not an h5 file")
    ok, why = is_valid(p, 1, "h", ("cls",))
    assert not ok and why.startswith("unreadable")


def test_atomic_json(tmp_path):
    p = tmp_path / "x" / "r.json"
    write_json(p, {"a": float("nan"), "b": [1, 2]})
    assert read_json(p) == {"a": None, "b": [1, 2]}
    with pytest.raises(ZeroDivisionError):
        with atomic_path(tmp_path / "y.txt") as tmp:
            tmp.write_text("partial")
            1 / 0
    assert not (tmp_path / "y.txt").exists()
