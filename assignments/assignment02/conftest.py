"""Pytest configuration for Assignment 2: make ``src/`` and this folder importable."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parents[1] / "src"
for folder in (SRC_DIR, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
