"""Crop centring and token readouts for every (input size, patch) the benchmark uses."""
import numpy as np
import pytest

from bench.data.regions import crop_origins, emission_order, plan_regions
from bench.models.encoders import round_to_patch
from bench.models.readouts import central_flat, central_pixel_box, central_tokens, nearest_2x2_flat

# stage 1 (model input sizes) + stage 2a sizes after rounding up to the patch
SIZES = [(224, 14), (448, 16), (224, 16), (40, 8),
         *[(round_to_patch(f, p, "up"), p) for f in (56, 112, 224, 448) for p in (14, 16)],
         *[(round_to_patch(f, 8, "up"), 8) for f in (56, 112, 224, 448)]]


@pytest.mark.parametrize("n,p", sorted(set(SIZES)))
def test_central_box_is_centred(n, p):
    g = n // p
    lo, hi = central_pixel_box(g, p)
    assert (lo + hi) / 2 == n / 2, (n, p, lo, hi)
    assert len(central_tokens(g)) == (1 if g % 2 else 4)
    flat = central_flat(g).tolist()
    assert all(0 <= i < g * g for i in flat)


def test_crop_origin_puts_centroid_at_crop_centre():
    rng = np.random.default_rng(0)
    e = rng.uniform(100, 10000, 1000)
    for n in (40, 224, 448):
        o, _ = crop_origins(e, e, n)
        off = e - o                      # centroid position inside the crop
        assert np.all(off >= n / 2 - 0.5) and np.all(off < n / 2 + 0.5)


def test_crop_size_must_be_even():
    with pytest.raises(ValueError):
        crop_origins(np.zeros(1), np.zeros(1), 41)


def test_round_to_patch():
    assert round_to_patch(56, 16, "up") == 64
    assert round_to_patch(56, 14, "up") == 56
    assert round_to_patch(56, 16, "down") == 48
    assert round_to_patch(4, 16, "down") == 16


def test_nearest_2x2():
    g, p = 16, 14
    # exactly on the corner between tokens (3,4)/(4,5): rows 3-4, cols 4-5
    idx = nearest_2x2_flat(np.array([5 * p]), np.array([4 * p]), p, g)[0]
    assert sorted(idx.tolist()) == sorted([3 * g + 4, 3 * g + 5, 4 * g + 4, 4 * g + 5])
    # near the image corner: clipped so all four tokens exist
    idx = nearest_2x2_flat(np.array([0.0]), np.array([0.0]), p, g)[0]
    assert sorted(idx.tolist()) == [0, 1, g, g + 1]
    idx = nearest_2x2_flat(np.array([g * p - 1.0]), np.array([g * p - 1.0]), p, g)[0]
    assert max(idx.tolist()) == g * g - 1


def test_regions_cover_every_cell_once():
    rng = np.random.default_rng(1)
    ex, ey = rng.uniform(300, 9000, 5000), rng.uniform(300, 9000, 5000)
    ox, oy = crop_origins(ex, ey, 224)
    regs = plan_regions(ox, oy, 224, 2048)
    order = emission_order(regs)
    assert sorted(order.tolist()) == list(range(5000))
    for r in regs:
        assert np.all(r.dx >= 0) and np.all(r.dy >= 0)
        assert np.all(r.dx + 224 <= r.w) and np.all(r.dy + 224 <= r.h)
        assert np.array_equal(ox[r.cells] - r.x0, r.dx)
