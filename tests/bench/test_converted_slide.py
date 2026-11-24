"""scripts/check_converted_slide.py and scripts/reconvert_slide.py: reading raw TIFFs segment by segment
(uncompressed with wrapped 32-bit offsets, tested with a small wrap size; JPEG), finding where a converted
copy stops matching its raw file, and rebuilding the copy."""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import check_converted_slide as cc  # noqa: E402

TISSUE = (98, 55, 108)


def _image(H=700, W=520, seed=1):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 240, np.uint8)
    t = img[100:H - 50, 40:W - 40]
    t[...] = TISSUE
    t += rng.integers(0, 30, t.shape, dtype=np.uint8)
    return img


def _wrap_offsets(path, layout, wrap):
    """Store the offsets modulo `wrap`, as a 32-bit writer does past 4 GiB."""
    with tifffile.TiffFile(path) as tf:
        tag = tf.pages[0].tags["StripOffsets" if "rowsperstrip" in layout else "TileOffsets"]
        true, vo = list(tag.value), tag.valueoffset
    with open(path, "r+b") as fh:
        fh.seek(vo)
        fh.write(struct.pack("<%dI" % len(true), *[v % wrap for v in true]))
    return np.asarray(true, np.int64)


def _corrupt(img, row, rows_back=37):
    """From `row` on, bytes taken 3 * W * rows_back - 1 bytes earlier: rows shifted, R,G,B -> G,B,R (as seen)."""
    H, W, _ = img.shape
    flat = img.reshape(-1)
    bad = flat.copy()
    b0, shift = row * W * 3, 3 * W * rows_back - 1
    bad[b0:] = flat[b0 - shift: flat.size - shift]
    return bad.reshape(img.shape)


def _write_copy(path, img):
    """The notebook's conversion: uncompressed tiled OME BigTIFF."""
    tifffile.imwrite(path, img, bigtiff=True, tile=(256, 256), photometric="rgb", planarconfig="contig",
                     ome=True, metadata={"axes": "YXS"})


@pytest.mark.parametrize("layout", [{"rowsperstrip": 16}, {"tile": (64, 64)}])
def test_wrapped_offsets_misread_and_unwrap_recovers(tmp_path, layout):
    img = _image(600, 500)
    p = tmp_path / "s.tif"
    tifffile.imwrite(p, img, photometric="rgb", **layout)
    size = p.stat().st_size
    wrap = 2 ** 18                                              # like 2**32, not a multiple of 3
    true = _wrap_offsets(p, layout, wrap)
    with tifffile.TiffFile(p) as tf, open(p, "rb") as fh:
        page = tf.pages[0]
        stored = np.asarray(page.dataoffsets, np.int64)
        assert (np.diff(stored) < 0).sum() >= 2                 # several wraps
        fixed = cc.unwrap(stored, page.databytecounts, size, wrap)
        np.testing.assert_array_equal(fixed, true)
        ok = cc.read_region(fh, page, fixed, 37, 410, 300, 150)
        bad = cc.read_region(fh, page, stored, 37, 410, 300, 150)
        k = cc.first_wrapped_row(page, stored, fixed)
    np.testing.assert_array_equal(ok, img[410:560, 37:337])
    assert not np.array_equal(bad, ok)
    assert k is not None and 0 < k < 600


@pytest.mark.parametrize("layout", [{"tile": (256, 256)}, {"rowsperstrip": 64}])
def test_jpeg_segments_decode_as_tifffile_does(tmp_path, layout):
    pytest.importorskip("imagecodecs")
    p = tmp_path / "j.tif"
    tifffile.imwrite(p, _image(), photometric="rgb", compression="jpeg", **layout)
    want = tifffile.imread(p)
    with tifffile.TiffFile(p) as tf, open(p, "rb") as fh:
        page = tf.pages[0]
        got = cc.read_region(fh, page, page.dataoffsets, 13, 250, 480, 449)
    np.testing.assert_array_equal(got, want[250:699, 13:493])


def test_find_boundary_locates_the_first_bad_row():
    img = _image()
    bad = _corrupt(img, 400)
    a = lambda x, y, w, h: bad[y:y + h, x:x + w]                # noqa: E731
    b = lambda x, y, w, h: img[y:y + h, x:x + w]                # noqa: E731
    row, prof = cc.find_boundary(a, b, img.shape[1], img.shape[0])
    assert row == 400
    assert cc.find_boundary(b, b, img.shape[1], img.shape[0])[0] is None


def _openslide():
    return pytest.importorskip("openslide")


@pytest.mark.parametrize("layout", [{"rowsperstrip": 16}, {"tile": (64, 64)}])
def test_reconvert_undoes_wrapped_offsets(tmp_path, layout):
    """Classic TIFF with wrapped offsets, converted as the notebook does (copies the misread pixels)."""
    openslide = _openslide()
    import reconvert_slide as rs
    img, wrap = _image(), 2 ** 18
    raw, conv = tmp_path / "S_he_image.tif", tmp_path / "S_he_image.ome.tif"
    tifffile.imwrite(raw, img, photometric="rgb", **layout)
    _wrap_offsets(raw, layout, wrap)
    misread = tifffile.imread(raw)
    assert not np.array_equal(misread, img)
    _write_copy(conv, misread)
    rs.reconvert(str(raw), str(conv), wrap=wrap)
    with openslide.OpenSlide(str(conv)) as sl:
        got = np.asarray(sl.read_region((0, 0), 0, (img.shape[1], img.shape[0])))[..., :3]
    np.testing.assert_array_equal(got, img)
    assert (tmp_path / "S_he_image.ome.tif.old").is_file()
    assert not (tmp_path / "S_he_image.partial.ome.tif").exists()


def test_reconvert_rebuilds_a_copy_broken_by_the_conversion(tmp_path):
    """JPEG raw that decodes fine; the copy went wrong from row 400 (bytes one position off, rows shifted)."""
    openslide = _openslide()
    pytest.importorskip("imagecodecs")
    import reconvert_slide as rs
    raw, conv = tmp_path / "S_he_image.ome.tif", tmp_path / "conv" / "S_he_image.ome.tif"
    conv.parent.mkdir()
    tifffile.imwrite(raw, _image(), photometric="rgb", compression="jpeg", tile=(256, 256))
    dec = tifffile.imread(raw)
    _write_copy(conv, _corrupt(dec, 400))
    rs.reconvert(str(raw), str(conv))
    with openslide.OpenSlide(str(conv)) as sl:
        got = np.asarray(sl.read_region((0, 0), 0, (dec.shape[1], dec.shape[0])))[..., :3]
    np.testing.assert_array_equal(got, dec)
    assert (conv.parent / "S_he_image.ome.tif.old").is_file()


def test_reconvert_refuses_when_the_copy_already_matches(tmp_path):
    _openslide()
    pytest.importorskip("imagecodecs")
    import reconvert_slide as rs
    raw, conv = tmp_path / "S_he_image.ome.tif", tmp_path / "conv" / "S_he_image.ome.tif"
    conv.parent.mkdir()
    tifffile.imwrite(raw, _image(), photometric="rgb", compression="jpeg", tile=(256, 256))
    _write_copy(conv, tifffile.imread(raw))
    before = conv.read_bytes()
    with pytest.raises(SystemExit, match="would change nothing"):
        rs.reconvert(str(raw), str(conv))
    assert conv.read_bytes() == before
    assert sorted(p.name for p in conv.parent.iterdir()) == ["S_he_image.ome.tif"]
