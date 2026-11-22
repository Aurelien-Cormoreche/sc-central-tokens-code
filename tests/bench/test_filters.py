"""Dark-spot block maps and the L-inf exclusion filter."""
import numpy as np

from bench.data.darkspots import _block_reduce_nodata, needed_tiles, remove_small_blobs
from bench.data.manifest import linf_hits


def test_block_reduce_nodata():
    rgba = np.full((64, 64, 4), 200, dtype=np.uint8)
    rgba[:16, :16, :3] = 5          # black block
    rgba[16:32, 16:32, 3] = 0       # transparent (outside the scan) block
    rgba[32:48, 32:40, :3] = 0      # half-black block
    frac = _block_reduce_nodata(rgba, 16, 8)
    assert frac[0, 0] == 1.0 and frac[1, 1] == 1.0 and frac[2, 2] == 0.5 and frac[3, 3] == 0.0


def test_linf_hits_radius():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10, 10] = True             # native pixels [160, 176) on both axes
    r = 224.0
    cy = np.full(4, 168.0)
    cx = np.array([176 + r - 2,     # box reaches into the block -> hit
                   176 + r + 2,     # box stops short -> no hit
                   160 - r + 2,     # left side, reaches -> hit
                   160 - r - 20])   # far left -> no hit
    assert linf_hits(mask, 16, cx, cy, r).tolist() == [True, False, True, False]
    # vertical distance counts the same way (L-inf box, not a disk)
    assert linf_hits(mask, 16, np.array([168.0]), np.array([176 + r - 2]), r).tolist() == [True]
    # diagonal: inside the L-inf box although farther than r in Euclidean distance
    d = r - 2
    assert linf_hits(mask, 16, np.array([176 + d]), np.array([176 + d]), r).tolist() == [True]


def test_needed_tiles_only_near_cells():
    need = needed_tiles(np.array([100.0]), np.array([100.0]), 50, 4096, 4096, 1024)
    assert need.sum() == 1 and need[0, 0]
    need = needed_tiles(np.array([1020.0]), np.array([100.0]), 50, 4096, 4096, 1024)
    assert need[0, 0] and need[0, 1] and need.sum() == 2


def test_remove_small_blobs():
    m = np.zeros((20, 20), dtype=bool)
    m[2, 2] = True
    m[10:14, 10:14] = True
    out = remove_small_blobs(m, 4)
    assert not out[2, 2] and out[10:14, 10:14].all()
    assert remove_small_blobs(m, 1).sum() == m.sum()
