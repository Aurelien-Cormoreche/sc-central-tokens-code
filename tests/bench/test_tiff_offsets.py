"""scripts/check_tiff_offsets.py: undoing 32-bit offset wraps (tested with a small wrap size)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")


def _mod():
    p = Path(__file__).resolve().parents[2] / "scripts" / "check_tiff_offsets.py"
    spec = importlib.util.spec_from_file_location("check_tiff_offsets", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("layout", [{"rowsperstrip": 16}, {"tile": (64, 64)}])
def test_wrapped_offsets_rotate_colours_and_unwrap_recovers(tmp_path, layout):
    m = _mod()
    rng = np.random.default_rng(0)
    img = np.zeros((600, 500, 3), np.uint8)
    img[...] = (98, 55, 108)                                    # purple tissue
    img += rng.integers(0, 20, img.shape, dtype=np.uint8)
    p = tmp_path / "s.tif"
    tifffile.imwrite(p, img, photometric="rgb", **layout)
    size = p.stat().st_size
    wrap = 2 ** 18                                              # 256 KiB, like 2**32 not a multiple of 3
    with tifffile.TiffFile(p) as tf:
        page = tf.pages[0]
        true = np.asarray(page.dataoffsets, np.int64)
        cnt = np.asarray(page.databytecounts, np.int64)
        stored = true % wrap                                    # what a classic TIFF > 4 GiB would hold
        assert (np.diff(stored) < 0).sum() >= 2                 # several wraps
        fixed = m.unwrap(stored, cnt, size, wrap)
        np.testing.assert_array_equal(fixed, true)
        with open(p, "rb") as fh:
            ok = m.read_region(fh, page, fixed, 37, 410, 300, 150)
            bad = m.read_region(fh, page, stored, 37, 410, 300, 150)
    np.testing.assert_array_equal(ok, img[410:560, 37:337])
    assert not np.array_equal(bad, ok)
    k = m.first_wrapped_row(page, stored, fixed)
    assert k is not None and 0 < k < 600
