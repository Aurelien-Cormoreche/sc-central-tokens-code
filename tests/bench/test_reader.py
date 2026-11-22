"""Region reads == direct reads (pixel-exact at native resolution; <= 1 after resampling)."""
import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("openslide")

from bench.data.reader import WSIReader  # noqa: E402
from bench.data.regions import crop_origins, plan_regions  # noqa: E402


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(1536, 1280, 3), dtype=np.uint8)
    p = tmp_path_factory.mktemp("wsi") / "s_he_image.tif"
    tifffile.imwrite(p, img, tile=(256, 256), photometric="rgb", resolution=(1e4 / 0.125, 1e4 / 0.125),
                     resolutionunit="CENTIMETER")
    return p, img


def test_native_read_matches_array(slide):
    p, img = slide
    r = WSIReader(p, scale=1.0)
    assert (r.width, r.height) == (1280, 1536)
    np.testing.assert_array_equal(r.read_ext(100, 200, 300, 250), img[200:450, 100:400])
    r.close()


def test_region_crops_equal_direct_crops(slide):
    p, _ = slide
    rng = np.random.default_rng(1)
    ex, ey = rng.uniform(120, 1150, 200), rng.uniform(120, 1400, 200)
    n = 64
    ox, oy = crop_origins(ex, ey, n)
    r = WSIReader(p, scale=1.0)
    for reg in plan_regions(ox, oy, n, 512):
        arr = r.read_ext(reg.x0, reg.y0, reg.w, reg.h)
        for c, dx, dy in zip(reg.cells, reg.dx, reg.dy):
            np.testing.assert_array_equal(arr[dy:dy + n, dx:dx + n], r.read_ext(int(ox[c]), int(oy[c]), n, n))
    r.close()


def test_resampled_region_crops_match_direct(slide):
    p, img = slide
    s = 0.5
    r = WSIReader(p, scale=s)
    assert (r.ext_width, r.ext_height) == (640, 768)
    # 2x2 box mean reference
    ref = img.reshape(768, 2, 640, 2, 3).astype(np.float64).mean(axis=(1, 3))
    got = r.read_ext(10, 20, 100, 90).astype(np.float64)
    assert np.abs(got - ref[20:110, 10:110]).max() <= 1.0
    rng = np.random.default_rng(2)
    ex, ey = rng.uniform(40, 600, 80), rng.uniform(40, 720, 80)
    n = 32
    ox, oy = crop_origins(ex, ey, n)
    for reg in plan_regions(ox, oy, n, 256):
        arr = r.read_ext(reg.x0, reg.y0, reg.w, reg.h).astype(np.int16)
        for c, dx, dy in zip(reg.cells, reg.dx, reg.dy):
            direct = r.read_ext(int(ox[c]), int(oy[c]), n, n).astype(np.int16)
            assert np.abs(arr[dy:dy + n, dx:dx + n] - direct).max() <= 1
    r.close()
