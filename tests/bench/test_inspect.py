"""bench inspect's pixel-size sources: file > raw original of a converted copy > alignment matrix > metadata."""
import h5py
import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("openslide")

from bench.config import load_config  # noqa: E402
from bench.data.inspect import _inspect_one, mpp_from_alignment  # noqa: E402
from bench.registry import Slide  # noqa: E402

W, H = 640, 512


def _tif(path, mpp=None, ome_mpp=None, size=(H, W)):
    img = np.full(size + (3,), 200, np.uint8)
    kw = {}
    if mpp:
        kw = dict(resolution=(1e4 / mpp, 1e4 / mpp), resolutionunit="CENTIMETER")
    if ome_mpp:
        kw = dict(ome=True, metadata={"axes": "YXS", "PhysicalSizeX": ome_mpp, "PhysicalSizeY": ome_mpp})
    tifffile.imwrite(path, img, tile=(256, 256), photometric="rgb", **kw)


def _align(path, mpp, ratio=1.0):
    s = mpp / 0.2125
    m = np.array([[s * np.cos(0.3) * ratio, -s * np.sin(0.3), 12.0], [s * np.sin(0.3) * ratio, s * np.cos(0.3), -5.0],
                  [0, 0, 1]])
    np.savetxt(path, m, delimiter=",")


@pytest.fixture
def env(tmp_path):
    d = {k: tmp_path / k for k in ("raw", "conv", "xen", "align", "root")}
    for p in d.values():
        p.mkdir()
    cfg = load_config(overrides={"paths": {"bench_root": str(d["root"]), "wsi_raw": str(d["raw"]),
                                           "wsi_converted": str(d["conv"]), "xenium_processed": str(d["xen"]),
                                           "alignment": str(d["align"]), "wsi_manifest": None}})

    def slide(name, conv=None, raw=None, align=None, override=None):
        (d["xen"] / name).mkdir()
        with h5py.File(d["xen"] / name / "patch_coordinates.h5", "w") as f:
            f["cell_id"] = np.arange(3)
        if raw is not None:
            _tif(d["raw"] / f"{name}_he_image.tif", **raw)
        if conv is not None:
            _tif(d["conv"] / f"{name}_he_image.ome.tif", **conv)
        if align is not None:
            _align(d["align"] / f"{name}_he_imagealignment.csv", *align)
        c = cfg if override is None else cfg.model_copy(update={"mpp_overrides": {name: override}})
        return _inspect_one(c, Slide(name, "C", "d", False), {}, {})
    return slide


def test_alignment_matrix_scale(tmp_path):
    p = tmp_path / "a.csv"
    _align(p, 0.3638)
    mpp, ratio = mpp_from_alignment(p)
    assert mpp == pytest.approx(0.3638) and ratio == pytest.approx(1.0)


def test_file_mpp_cross_checked_against_alignment(env):
    r = env("a", raw={"mpp": 0.2738}, align=(0.2738,))
    assert r["status"] == "ok" and r["mpp_source"] in ("openslide", "tiff_cm") and r["mpp"] == pytest.approx(0.2738, rel=1e-4)
    r = env("b", raw={"mpp": 0.2738}, align=(0.3638,))
    assert r["status"] == "mpp_mismatch" and "alignment 0.36380" in r["error"]


def test_converted_copy_takes_raw_originals_pixel_size(env):
    r = env("c", conv={}, raw={"ome_mpp": 0.3638}, align=(0.3638,))
    assert r["converted"] and r["status"] == "ok" and r["mpp_source"] == "raw:ome"
    assert r["mpp"] == pytest.approx(0.3638)


def test_raw_original_of_another_size_is_not_used(env):
    r = env("d", conv={}, raw={"mpp": 0.25, "size": (H // 2, W // 2)}, align=(0.5,))
    assert r["mpp_source"] == "alignment" and r["mpp"] == pytest.approx(0.5) and "not used" in r["mpp_note"]


def test_no_pixel_size_anywhere(env):
    r = env("e", conv={})
    assert r["status"] == "unknown_mpp" and r["mpp"] is None
    r = env("f", conv={}, override=0.3)
    assert (r["status"], r["mpp_source"], r["mpp"]) == ("ok", "override", 0.3)


def test_anisotropic_alignment_is_reported(env):
    r = env("g", conv={}, align=(0.25, 1.05))
    assert r["mpp_source"] == "alignment" and "anisotropic" in r
