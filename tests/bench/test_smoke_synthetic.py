"""The whole CLI chain on generated slides with the tiny random ViT (CPU, ~30 s)."""
import pytest

pytest.importorskip("openslide")
pytest.importorskip("tifffile")


def test_synthetic_smoke(tmp_path):
    from bench.smoke import run_synthetic
    assert run_synthetic(str(tmp_path / "work")) == 0
