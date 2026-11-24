"""scripts/find_copy_offset.py on a slide written by the faulty writer (BGR buffer, byte index wrapping; a small
wrap size stands in for 2**32: both are 1 more than a multiple of 3)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import find_copy_offset as f  # noqa: E402

H, W, M = 1600, 1400, 2 ** 22


@pytest.fixture(scope="module")
def slide():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]
    T = np.full((H, W, 3), 238, np.uint8)
    blob = ((xx - 700) / 650) ** 2 + ((yy - 800) / 780) ** 2 < 1
    T[blob] = (215, 150, 200)                                   # eosin
    cx, cy = rng.uniform(0, W, 6000), rng.uniform(0, H, 6000)
    keep = blob[cy.astype(int), cx.astype(int)]
    cx, cy = cx[keep], cy[keep]
    for x, y in zip(cx, cy):                                    # a nucleus under every cell
        r = 4.5
        y0, y1, x0, x1 = int(max(0, y - r)), int(min(H, y + r + 1)), int(max(0, x - r)), int(min(W, x + r + 1))
        m = (yy[y0:y1, x0:x1] - y) ** 2 + (xx[y0:y1, x0:x1] - x) ** 2 <= r * r
        T[y0:y1, x0:x1][m] = (95, 60, 140)
    bgr = T[..., ::-1].reshape(-1)
    stored = bgr[np.arange(bgr.size) % M].reshape(H, W, 3)[..., ::-1].copy()   # the faulty writer
    return T, stored, cx, cy


def test_mapping_points_at_the_copied_pixel(slide):
    T, stored, _, _ = slide
    rng = np.random.default_rng(1)
    dy, dx = divmod(f.copy_shift(1, M), W)
    ys, xs = rng.integers(dy + 1, H, 3000), rng.integers(0, W, 3000)
    mx, my, k = f.to_original(xs, ys, W, M)
    assert (k == 1).all()
    s, t = stored[ys, xs].astype(int), T[my, mx].astype(int)
    np.testing.assert_array_equal(s[:, :2], t[:, 1:])           # stored (R, G) = original (G, B)
    x, y, k = f.to_original([5, W - 1], [dy - 1, dy - 1], W, M)  # above the copy: unchanged
    assert list(k) == [0, 0] and list(x) == [5, W - 1] and list(y) == [dy - 1, dy - 1]


def test_overview_and_full_resolution_find_both_shifts(slide):
    _, stored, _, _ = slide
    g = 8
    thumb = stored.reshape(H // g, g, W // g, g, 3).mean(axis=(1, 3)).astype(np.uint8)
    read = lambda x, y, w, h: stored[y:y + h, x:x + w]        # noqa: E731
    dy, dx = divmod(f.copy_shift(1, M), W)
    found = set()
    for py, px, _ in f.thumb_shift(thumb)[:2]:
        for b in f.refine(read, W, H, (round(py * g), round(px * g)), thumb, 1 / g, n_boxes=2, box=128, margin=32):
            assert b[4] > 0.9 and b[5] == (2, 0, 1) and b[6] < 3
            found.add((b[2], b[3]))
    assert found == {(dy, dx), (dy + 1, dx - W)}


def test_nuclear_contrast_tells_whether_cells_belong_to_the_image(slide):
    _, stored, cx, cy = slide
    read = lambda x, y, w, h: stored[y:y + h, x:x + w]        # noqa: E731
    r = f.cell_report(read, cx, cy, W, H, M)
    assert r["n_green_part"] > 500
    assert r["nuclear_contrast_purple_part"] > 30
    assert abs(r["nuclear_contrast_purple_part_moved_9px"]) < 0.5 * r["nuclear_contrast_purple_part"]
    assert r["nuclear_contrast_green_part_as_placed"] < 0.5 * r["nuclear_contrast_purple_part"]
