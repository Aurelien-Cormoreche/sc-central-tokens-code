"""bench overview: strip-wise block averaging == a direct block average, on single-level and pyramid files."""
import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("openslide")

import bench.data.overview as ov  # noqa: E402


def _img(h=1500, w=2000):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _block_mean(a, g):
    h, w = a.shape[0] // g * g, a.shape[1] // g * g
    return (a[:h, :w].reshape(h // g, g, w // g, g, 3).sum(axis=(1, 3), dtype=np.uint32) // (g * g)).astype(np.uint8)


def test_strips_equal_direct_block_mean(tmp_path, monkeypatch):
    img = _img()
    p = tmp_path / "s.tif"
    tifffile.imwrite(p, img, tile=(256, 256), photometric="rgb")
    monkeypatch.setattr(ov, "STRIP_BYTES", 2000 * 4 * 7)        # ~7 rows per strip -> many strips
    out, (w, h), k, level, g = ov.downscaled(str(p), 300)
    assert (w, h, level, g) == (2000, 1500, 0, 7) and k == pytest.approx(1 / 7)
    np.testing.assert_array_equal(out, _block_mean(img, 7))


def test_pyramid_level_is_used(tmp_path):
    img = _img(2048, 2048)
    p = tmp_path / "pyr.tif"
    with tifffile.TiffWriter(p, bigtiff=True) as tw:
        tw.write(img, tile=(256, 256), photometric="rgb")
        tw.write(img[::4, ::4].copy(), tile=(256, 256), photometric="rgb", subfiletype=1)
    out, (w, h), k, level, g = ov.downscaled(str(p), 256)
    assert level == 1 and g == 2 and k == pytest.approx(1 / 8) and out.shape == (256, 256, 3)
    np.testing.assert_array_equal(out, _block_mean(img[::4, ::4], 2))
