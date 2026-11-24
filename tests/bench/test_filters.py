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


def test_darkspot_map_redone_when_scale_changes(tmp_path):
    """A map made at another scale has another L-inf radius in native px: it must not count as done."""
    from types import SimpleNamespace
    from bench.data.darkspots import job_done
    fl = SimpleNamespace(block_px=16, scan_tile_px=4096, black_max=8, block_full_frac=0.9, min_blob_blocks=1,
                         linf_px=224)
    cfg = SimpleNamespace(filters=fl, is_smoke=False)
    wsi = tmp_path / "s.tif"
    wsi.write_bytes(b"image")
    p = tmp_path / "s.npz"
    meta = dict(block=16, tile=4096, black_max=8, full_frac=0.9, min_blob_blocks=1, linf_px=224,
                r_native=224 / 0.5, scale=0.5, smoke=False)
    np.savez(p, meta=np.array([repr(meta)]))
    assert job_done(cfg, {"out": str(p), "row": {"scale": 0.5, "wsi_path": str(wsi)}})
    assert not job_done(cfg, {"out": str(p), "row": {"scale": 1.0, "wsi_path": str(wsi)}})


def test_darkspot_map_redone_when_image_replaced(tmp_path):
    """A map made from another version of the image file (e.g. before a re-conversion) is not done, and the
    manifest refuses it; maps from before fingerprints fall back on the file times."""
    import os
    from types import SimpleNamespace
    from bench.data.darkspots import job_done, map_matches_image
    from bench.utils import file_fingerprint
    fl = SimpleNamespace(block_px=16, scan_tile_px=4096, black_max=8, block_full_frac=0.9, min_blob_blocks=1,
                         linf_px=224)
    cfg = SimpleNamespace(filters=fl, is_smoke=False)
    wsi = tmp_path / "s.tif"
    rng = np.random.default_rng(0)
    wsi.write_bytes(rng.integers(0, 255, 1 << 20, dtype=np.uint8).tobytes())
    base = dict(block=16, tile=4096, black_max=8, full_frac=0.9, min_blob_blocks=1, linf_px=224,
                r_native=224.0, scale=1.0, smoke=False)
    job = {"out": str(tmp_path / "s.npz"), "row": {"scale": 1.0, "wsi_path": str(wsi)}}
    np.savez(job["out"], meta=np.array([repr({**base, "wsi_fp": file_fingerprint(wsi)})]))
    assert job_done(cfg, job)
    np.savez(tmp_path / "legacy.npz", meta=np.array([repr(base)]))            # no fingerprint, newer than image
    assert map_matches_image(base, tmp_path / "legacy.npz", str(wsi))

    data = bytearray(wsi.read_bytes())                                       # same size, other pixels
    data[len(data) // 2: len(data) // 2 + 4096] = bytes(4096)
    wsi.write_bytes(bytes(data))
    t = os.path.getmtime(tmp_path / "legacy.npz") + 10
    os.utime(wsi, (t, t))                                                    # replaced after the legacy map
    assert not job_done(cfg, job)
    assert not map_matches_image(base, tmp_path / "legacy.npz", str(wsi))


def test_manifest_hash_follows_the_image():
    import pandas as pd
    from bench.data.manifest import manifest_hash
    df = pd.DataFrame({"cell_id": ["a", "b"], "label": ["T", "B"], "ex": [1.0, 2.0], "ey": [3.0, 4.0],
                       "extract": [True, True]})
    assert manifest_hash(df) == manifest_hash(df, wsi_fp="")
    assert manifest_hash(df, wsi_fp="f1") != manifest_hash(df, wsi_fp="f2")
