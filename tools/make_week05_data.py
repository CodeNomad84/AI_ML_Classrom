"""Regenerate the raw rover logs used in Week 5 and Assignment 2.

Provenance: new content (Week 5).

* ``week05_rover_log.csv``: 16 test drives, seed 5 (lecture, lab, slides).
* ``assignment02_rover_log.csv``: 16 test drives, seed 52 (Assignment 2).

Both are written by :class:`aerospace_ai.rover.RoverLogGenerator`, so the
CSV files and the ``rover_range_forecast`` loader agree. The injected defect
counts are printed (they are the answer key for the audit tasks).

Usage:
    python tools/make_week05_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerospace_ai.rover import RoverLogConfig, RoverLogGenerator  # noqa: E402

SAMPLE_DIR: Final[Path] = REPO_ROOT / "data" / "sample"
LOGS: Final[dict[str, int]] = {
    "week05_rover_log.csv": 5,
    "assignment02_rover_log.csv": 52,
}
N_RUNS: Final[int] = 16


def main() -> None:
    """Write both CSV files and print the defect counts."""
    for name, seed in LOGS.items():
        generator = RoverLogGenerator(RoverLogConfig(n_runs=N_RUNS, seed=seed))
        path = generator.to_csv(SAMPLE_DIR / name)
        print("wrote", path.relative_to(REPO_ROOT))
        print(generator.defects)


if __name__ == "__main__":
    main()
