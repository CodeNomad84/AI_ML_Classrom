"""Regenerate the synthetic flight logs shipped in data/sample/.

Provenance: new content (Week 1).

Usage:
    python tools/make_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerospace_ai.utils import (  # noqa: E402
    FlightProfileConfig,
    Maneuver,
    SyntheticFlightLogGenerator,
)

SAMPLE_DIR: Final[Path] = REPO_ROOT / "data" / "sample"

# Week 1 lecture/lab log: default profile (30 deg turn, 1.8 g pull-up).
LECTURE_CONFIG: Final[FlightProfileConfig] = FlightProfileConfig()

# Assignment 0 log: a different flight with a 45 deg turn and a 2.25 g
# pull-up that exceeds the 2.0 g structural limit used in the assignment.
ASSIGNMENT_CONFIG: Final[FlightProfileConfig] = FlightProfileConfig(
    seed=2026,
    maneuvers=(
        Maneuver("steep turn", 140.0, 25.0, 1.414, yaw_rate_dps=20.1),
        Maneuver("pull-up", 200.0, 5.0, 2.25, pitch_rate_dps=12.0),
        Maneuver("touchdown", 270.0, 0.6, 1.5),
    ),
    gap_start_s=175.0,
    gap_duration_s=2.0,
    spike_count=4,
    dropout_fraction=0.01,
)

OUTPUTS: Final[dict[str, FlightProfileConfig]] = {
    "week01_uav_flight_log.csv": LECTURE_CONFIG,
    "assignment00_uav_flight_log.csv": ASSIGNMENT_CONFIG,
}


def main() -> None:
    """Write every sample log listed in OUTPUTS."""
    for name, config in OUTPUTS.items():
        path = SyntheticFlightLogGenerator(config).to_csv(SAMPLE_DIR / name)
        print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
