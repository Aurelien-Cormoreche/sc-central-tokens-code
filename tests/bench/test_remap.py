"""bench/data/remap.py: cells of a slide's copied part are read from the part the copy repeats."""
from types import SimpleNamespace

import numpy as np
import pytest

from bench.data.remap import read_positions, remap_drop, remap_key

RM = SimpleNamespace(from_row=600, dy=500, dx=20)


def test_cells_in_the_copy_are_read_from_the_original_part():
    cx = np.array([100.0, 100.0, 100.0, 300.0, 50.0])
    cy = np.array([599.0, 600.0, 1099.0, 1150.0, 700.0])
    qx, qy, steps = read_positions(RM, cx, cy)
    assert qx.tolist() == [100, 80, 80, 260, 30] and qy.tolist() == [599, 100, 599, 150, 200]
    assert steps.tolist() == [0, 1, 1, 2, 1]                 # 1150 -> 650 (still in the copy) -> 150
    assert (qy < RM.from_row).all()
    qx, qy, steps = read_positions(None, cx, cy)
    assert qx.tolist() == cx.tolist() and qy.tolist() == cy.tolist() and not steps.any()


def test_crops_reaching_the_copy_or_leaving_the_image_are_dropped():
    b, W = 40, 1100
    rx = np.array([500.0, 500.0, 500.0, 30.0, 30.0])
    ry = np.array([560.0, 561.0, 300.0, 300.0, 300.0])
    moved = np.array([False, False, True, True, False])
    out = remap_drop(RM, rx, ry, moved, 1.0, b, W)
    # crop ends exactly at row 600 | crosses it | moved, inside | moved, off the left edge | in place (border's job)
    assert out.tolist() == [False, True, False, True, False]
    assert not remap_drop(None, rx, ry, moved, 1.0, b, W).any()
    assert remap_drop(RM, np.array([500.0]), np.array([1130.0]), np.array([False]), 0.5, b, W).tolist() == [True]


def test_config_rejects_a_shift_that_does_not_point_up():
    from bench.config import Remap
    with pytest.raises(Exception):
        Remap(from_row=600, dy=0, dx=5)
    assert Remap(from_row=600, dy=500, dx=-7).dx == -7


def test_darkspot_map_redone_when_the_remap_changes(tmp_path):
    from bench.data.darkspots import job_done
    fl = SimpleNamespace(block_px=16, scan_tile_px=4096, black_max=8, block_full_frac=0.9, min_blob_blocks=1,
                         linf_px=224)
    wsi = tmp_path / "s.tif"
    wsi.write_bytes(b"image")
    base = dict(block=16, tile=4096, black_max=8, full_frac=0.9, min_blob_blocks=1, linf_px=224, r_native=224.0,
                scale=1.0, smoke=False)
    job = {"slide": "S", "out": str(tmp_path / "s.npz"), "row": {"slide": "S", "scale": 1.0, "wsi_path": str(wsi)}}
    np.savez(job["out"], meta=np.array([repr(base)]))                       # made before remap existed
    plain = SimpleNamespace(filters=fl, is_smoke=False, remap={})
    with_rm = SimpleNamespace(filters=fl, is_smoke=False, remap={"S": RM})
    assert job_done(plain, job) and not job_done(with_rm, job)
    np.savez(job["out"], meta=np.array([repr({**base, "remap": remap_key(RM)})]))
    assert job_done(with_rm, job) and not job_done(plain, job)
    other = SimpleNamespace(filters=fl, is_smoke=False, remap={"S": SimpleNamespace(from_row=600, dy=500, dx=21)})
    assert not job_done(other, job)
