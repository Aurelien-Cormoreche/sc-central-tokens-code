"""A region whose box holds a corrupt tile that no crop touches (the manifest keeps crops clear of unreadable
blocks) is read in parts around its crops; a crop on the corrupt tile still fails."""
import numpy as np
import pytest

from bench.data.regions import crop_origins, plan_regions, read_crops


class _Bad(Exception):
    pass


def test_read_crops_goes_around_a_bad_area():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
    bad = (300, 200, 400, 300)                               # x0, y0, x1, y1

    def read(x, y, w, h):
        if x < bad[2] and x + w > bad[0] and y < bad[3] and y + h > bad[1]:
            raise _Bad(f"({x},{y},{w},{h})")
        return img[y:y + h, x:x + w]

    n = 40
    dx = np.array([10, 100, 250, 420, 500, 700, 320, 330, 10, 600])
    dy = np.array([10, 150, 230, 210, 50, 500, 320, 140, 520, 330])
    out = np.zeros_like(img)
    failed = read_crops(read, dx, dy, n, out, _Bad)
    assert failed > 0
    for x, y in zip(dx, dy):
        np.testing.assert_array_equal(out[y:y + n, x:x + n], img[y:y + n, x:x + n])
    with pytest.raises(_Bad):                                # a crop on the bad area
        read_crops(read, np.r_[dx, 320], np.r_[dy, 230], n, np.zeros_like(img), _Bad)


def test_extraction_reads_a_region_with_a_corrupt_tile_between_its_crops(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("imagecodecs")
    openslide = pytest.importorskip("openslide")
    tifffile = pytest.importorskip("tifffile")
    from bench.data.reader import ReadError, WSIReader
    from bench.extract.worker import _Regions

    p = tmp_path / "corrupt.tif"
    rng = np.random.default_rng(1)
    tifffile.imwrite(p, rng.integers(0, 255, (2048, 2048, 3), dtype=np.uint8), tile=(256, 256),
                     photometric="rgb", compression="jpeg")
    with tifffile.TiffFile(p) as tf:
        off, cnt = tf.pages[0].dataoffsets, tf.pages[0].databytecounts
    k = 3 * 8 + 4                                            # tile x 1024..1280, y 768..1024
    with open(p, "r+b") as fh:
        fh.seek(off[k])
        fh.write(bytes(cnt[k]))

    n = 64
    ex = np.array([700.0, 900.0, 1500.0, 1600.0, 1150.0, 1100.0])
    ey = np.array([900.0, 600.0, 800.0, 1000.0, 600.0, 1200.0])   # all > n/2 from the corrupt tile
    ox, oy = crop_origins(ex, ey, n)
    regions = plan_regions(ox, oy, n, 2048)
    assert len(regions) == 1
    r = regions[0]
    box = (r.x0, r.y0, r.w, r.h)
    with pytest.raises(ReadError):                           # the region box holds the corrupt tile
        _Regions(str(p), 1.0, [box], pad_outside=False)[0]
    _, arr = _Regions(str(p), 1.0, [box], pad_outside=False, crops=[(r.dx, r.dy)], n_px=n)[0]
    arr = arr.numpy()
    rd = WSIReader(str(p))
    try:
        for x, y, ddx, ddy in zip(ox[r.cells], oy[r.cells], r.dx, r.dy):
            np.testing.assert_array_equal(arr[ddy:ddy + n, ddx:ddx + n], rd.read_ext(int(x), int(y), n, n))
    finally:
        rd.close()
    ox2, oy2 = crop_origins(np.r_[ex, 1150.0], np.r_[ey, 900.0], n)       # one crop on the corrupt tile
    r2 = plan_regions(ox2, oy2, n, 2048)[0]
    with pytest.raises(ReadError):
        _Regions(str(p), 1.0, [(r2.x0, r2.y0, r2.w, r2.h)], pad_outside=False, crops=[(r2.dx, r2.dy)], n_px=n)[0]
