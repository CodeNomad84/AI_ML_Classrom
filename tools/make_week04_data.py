"""Regenerate the synthetic fuel-flow table used in Week 4 and Practice 4.

Provenance: new content (Week 4).

``week04_fuel_flow.csv``: 400 seeded measurements of a small piston UAV
(airspeed, altitude, climb rate, mass -> fuel flow in g/h with 3 % noise),
written by the ``fuel_flow_synthetic`` loader of
:mod:`aerospace_ai.datasets`, so the CSV and the in-memory loader agree.

Usage:
    python tools/make_week04_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerospace_ai.datasets import DatasetFactory  # noqa: E402

SAMPLE_DIR: Final[Path] = REPO_ROOT / "data" / "sample"
FUEL_CSV: Final[Path] = SAMPLE_DIR / "week04_fuel_flow.csv"
N_SAMPLES: Final[int] = 400
SEED: Final[int] = 4
DECIMALS: Final[int] = 3


def main() -> None:
    """Write the CSV and print a short summary."""
    dataset = DatasetFactory.load("fuel_flow_synthetic", n_samples=N_SAMPLES, seed=SEED)
    frame = dataset.to_frame().round(DECIMALS)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FUEL_CSV, index=False)
    print("wrote", FUEL_CSV.relative_to(REPO_ROOT), frame.shape)


if __name__ == "__main__":
    main()
