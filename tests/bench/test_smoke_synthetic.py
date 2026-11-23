"""The whole CLI chain on generated slides with the tiny random ViT (CPU, ~30 s), then the login-node check."""
import json
import subprocess
import sys

import pytest

pytest.importorskip("openslide")
pytest.importorskip("tifffile")

# Every command run on a login node: plan, submit (--dry-run here), status, report, weights. Torch must not be
# imported there: its CUDA libraries fail to load on login nodes ("failed to map segment from shared object").
LOGIN_NODE_COMMANDS = [
    ["inspect", "--dry-run"], ["crops", "--cells", "2"], ["darkspots", "--dry-run"], ["manifest", "--dry-run"],
    ["fetch-weights", "--offline", "--dry-run"],
    ["extract", "--stage", "1", "--dry-run"], ["classify", "--stage", "1", "--dry-run"], ["genexp", "--dry-run"],
    ["status"], ["status", "--deletable"], ["report"], ["calibrate", "--from-status", "--dry-run"],
]
GATED_COMMANDS = [
    ["extract", "--stage", "2a", "--dry-run"], ["extract", "--stage", "2b", "--dry-run"],
    ["classify", "--stage", "2", "--dry-run"], ["scaling", "--dry-run"],
]
RUNNER = """
import json, sys
from bench.cli import main
for argv in json.loads(sys.argv[1]):
    try:
        rc = main(argv)
    except SystemExit as e:
        rc = e.code
    if rc not in (0, None):
        print("FAILED: bench " + " ".join(argv) + f" -> {rc!r}")
        sys.exit(1)
assert "torch" not in sys.modules, "torch was imported"
print("ALL OK")
"""


def test_synthetic_smoke(tmp_path):
    from bench.smoke import run_synthetic
    work = tmp_path / "work"
    assert run_synthetic(str(work)) == 0
    _check_login_node(tmp_path, work)


def _check_login_node(tmp_path, work):
    import yaml
    from bench import REPO
    fake = tmp_path / "login_node" / "torch"                 # importable like the real one, fails like it
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(
        'raise ImportError("libcublasLt.so.12: failed to map segment from shared object (simulated login node)")\n')
    cfg = work / "bench_synthetic.yaml"
    d = yaml.safe_load(cfg.read_text())
    d["gate1"] = {"chosen": {"model": "tiny", "readout": "central"}}
    d.setdefault("stage2", {})["multicell"] = {"tile_px": 224}
    gated = work / "bench_gated.yaml"
    gated.write_text(yaml.safe_dump(d, sort_keys=False))
    runs = [a + ["--config", str(cfg)] for a in LOGIN_NODE_COMMANDS] + [a + ["--config", str(gated)] for a in GATED_COMMANDS]
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(fake.parent)
    r = subprocess.run([sys.executable, "-c", RUNNER, json.dumps(runs)], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "ALL OK" in r.stdout, f"{r.stdout[-2500:]}\n{r.stderr[-4000:]}"
