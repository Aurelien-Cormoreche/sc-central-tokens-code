"""bench: single-cell benchmark of pathology foundation models on Xenium + H&E.

One config (configs/bench.yaml), one CLI (`bench <command>`, see bench/cli.py), one storage
format (bench/extract/store.py) and one static website (`bench report`). README_bench.md lists
the exact commands in order.
"""
from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"

REPO = Path(__file__).resolve().parents[1]

# The reused training code lives under src/python and is imported as `src.python...` (the
# repo's convention), which needs the repo root on sys.path -- true for `python -m bench`
# from the repo root, not for an installed `bench` console script run from elsewhere.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
