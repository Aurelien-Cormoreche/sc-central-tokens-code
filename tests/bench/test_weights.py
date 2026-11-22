"""`bench fetch-weights --offline`: weights put in the HF cache by `hf download` (no network here)."""
import json
import os

import pytest

from bench import cli
from bench.config import ModelCfg
from bench.models.encoders import cached_snapshot, hf_download_command, missing_weight_files, weight_spec

SHA = "a" * 40


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from huggingface_hub import constants
    d = tmp_path / "hub"
    d.mkdir()
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(d))
    return d


def put(cache, repo, files, sha=SHA, ref=True, broken=()):
    """Lay out one repo the way `hf download` does: blobs, snapshots/<sha>/<file> -> blob symlinks, refs/main."""
    d = cache / f"models--{repo.replace('/', '--')}"
    (d / "blobs").mkdir(parents=True, exist_ok=True)
    snap = d / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        blob = d / "blobs" / f"blob{i}"
        if f not in broken:
            blob.write_bytes(b"x")
        os.symlink(os.path.relpath(blob, snap), snap / f)
    if ref:
        (d / "refs").mkdir(exist_ok=True)
        (d / "refs" / "main").write_text(sha)
    return snap


def test_cached_snapshot_and_required_files(cache):
    m = ModelCfg(input_px=40, patch=8, arch="vits8")
    assert missing_weight_files("lemon", m, cached_snapshot("aliceblondel/LEMON"))   # nothing there yet
    snap = put(cache, "aliceblondel/LEMON", ["mean_std.json", "lemon_vits8.pth.tar"])
    assert cached_snapshot("aliceblondel/LEMON") == snap
    assert cached_snapshot("aliceblondel/LEMON", SHA) == snap
    assert cached_snapshot("aliceblondel/LEMON", "b" * 40) is None
    assert missing_weight_files("lemon", m, snap) == []
    # timm models: config.json + either weights format; a broken symlink (incomplete copy) is missing
    snap = put(cache, "MahmoodLab/UNI2-h", ["config.json", "pytorch_model.bin"], ref=False, broken=("pytorch_model.bin",))
    assert cached_snapshot("MahmoodLab/UNI2-h") == snap                               # only snapshot, no ref
    u = ModelCfg(input_px=224, patch=14)
    assert missing_weight_files("uni2", u, snap) == ["model.safetensors or pytorch_model.bin"]
    (snap.parent.parent / "blobs" / "blob1").write_bytes(b"x")
    assert missing_weight_files("uni2", u, snap) == []


def test_download_commands():
    m = ModelCfg(input_px=448, patch=16)
    assert hf_download_command("MahmoodLab/conchv1_5", None, weight_spec("conch15", m)[0]) == \
        "hf download MahmoodLab/conchv1_5 pytorch_model_vision.bin"
    assert hf_download_command("paige-ai/Virchow2", SHA, weight_spec("virchow2", m)[0]) == \
        f"hf download paige-ai/Virchow2 --revision {SHA}"


def test_cli_offline_records_shas(cache, tmp_path, monkeypatch, capsys):
    for v in ("WSI_RAW_ROOT", "WSI_CONVERTED_ROOT", "XENIUM_PROCESSED_OUTPUT_ROOT"):
        monkeypatch.setenv(v, str(tmp_path))
    monkeypatch.setenv("BENCH_ROOT", str(tmp_path / "out"))
    argv = ["fetch-weights", "--offline", "--models", "lemon", "conch15"]
    assert cli.main(argv) == 1
    out = capsys.readouterr().out
    assert "hf download MahmoodLab/conchv1_5 pytorch_model_vision.bin" in out
    assert "hf download aliceblondel/LEMON mean_std.json lemon_vits8.pth.tar" in out
    put(cache, "aliceblondel/LEMON", ["mean_std.json", "lemon_vits8.pth.tar"])
    put(cache, "MahmoodLab/conchv1_5", ["pytorch_model_vision.bin"], sha="c" * 40)
    dirs = [d for d in cache.rglob("*") if d.is_dir() and not d.is_symlink()] + [cache]
    for d in dirs:                       # the cluster sees the cache read-only (customapps)
        d.chmod(0o555)
    try:
        assert cli.main(argv) == 0
    finally:
        for d in dirs:
            d.chmod(0o755)
    revs = json.loads((tmp_path / "out" / "weights_revisions.json").read_text())
    assert revs["lemon"]["sha"] == SHA and revs["conch15"]["sha"] == "c" * 40


def test_download_script_matches_weight_spec():
    """scripts/cluster/download_weights.sh fetches exactly what the loaders read."""
    from bench import REPO
    from bench.models.encoders import ENCODERS
    text = (REPO / "scripts" / "cluster" / "download_weights.sh").read_text()
    lines = [" ".join(ln.split("--cache-dir")[0].split()) for ln in text.splitlines() if ln.startswith("hf download")]
    cfg = {"lemon": ModelCfg(input_px=40, patch=8, arch="vits8")}
    expected = [hf_download_command(ENCODERS[m].repo, None, weight_spec(m, cfg.get(m, ModelCfg(input_px=224, patch=14)))[0])
                for m in ("uni2", "virchow2", "hoptimus1", "conch15", "lemon", "genbio")]
    assert lines == expected
