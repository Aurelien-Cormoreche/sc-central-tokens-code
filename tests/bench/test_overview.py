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
    out, (w, h), k, level, g, bad = ov.downscaled(str(p), 300)
    assert (w, h, level, g) == (2000, 1500, 0, 7) and k == pytest.approx(1 / 7)
    np.testing.assert_array_equal(out, _block_mean(img, 7))
    assert not bad.any()


def test_pyramid_level_is_used(tmp_path):
    img = _img(2048, 2048)
    p = tmp_path / "pyr.tif"
    with tifffile.TiffWriter(p, bigtiff=True) as tw:
        tw.write(img, tile=(256, 256), photometric="rgb")
        tw.write(img[::4, ::4].copy(), tile=(256, 256), photometric="rgb", subfiletype=1)
    out, (w, h), k, level, g, bad = ov.downscaled(str(p), 256)
    assert level == 1 and g == 2 and k == pytest.approx(1 / 8) and out.shape == (256, 256, 3)
    np.testing.assert_array_equal(out, _block_mean(img[::4, ::4], 2))


def test_corrupt_tile_only_marks_its_blocks(tmp_path, monkeypatch):
    """A read that touches the corrupt area fails (as TIFFRGBAImageGet does); the rest of the slide is drawn."""
    import openslide
    img = _img()
    p = tmp_path / "c.tif"
    tifffile.imwrite(p, img, tile=(256, 256), photometric="rgb")
    real = openslide.OpenSlide.read_region
    box = (1100, 300, 1200, 400)                                  # x0, y0, x1, y1 level-0 px

    def flaky(self, loc, level, size):
        (x, y), (w, h) = loc, size
        if x < box[2] and x + w > box[0] and y < box[3] and y + h > box[1]:
            raise openslide.OpenSlideError("TIFFRGBAImageGet failed")
        return real(self, loc, level, size)
    monkeypatch.setattr(openslide.OpenSlide, "read_region", flaky)
    monkeypatch.setattr(ov, "SUB_PX", 140)
    out, _, _, _, g, bad = ov.downscaled(str(p), 300)
    assert g == 7 and bad.any() and bad.mean() < 0.1
    ys, xs = np.nonzero(bad)                                      # bad blocks cover the box, and only near it
    assert xs.min() * g <= box[0] and (xs.max() + 1) * g >= box[2] and ys.min() * g <= box[1] and (ys.max() + 1) * g >= box[3]
    assert xs.min() * g > box[0] - 140 and ys.max() * g < box[3] + 140
    assert (out[bad] == ov.BAD_RGB).all()
    ok = _block_mean(img, 7)
    np.testing.assert_array_equal(out[~bad], ok[~bad])
