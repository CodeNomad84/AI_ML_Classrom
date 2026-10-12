"""Unit tests for aerospace_ai.utils (Week 1). Provenance: new content."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aerospace_ai.utils import (
    LOG_COLUMNS,
    STANDARD_GRAVITY_MPS2,
    FlightLog,
    FlightProfileConfig,
    SyntheticFlightLogGenerator,
    load_factor_from_accel,
    smoothstep,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = REPO_ROOT / "data" / "sample" / "week01_uav_flight_log.csv"


@pytest.fixture(scope="module")
def log() -> FlightLog:
    """Return the default synthetic flight log."""
    return FlightLog(SyntheticFlightLogGenerator().generate(), source="fixture")


def test_smoothstep_endpoints() -> None:
    t = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(smoothstep(t, 0.0, 1.0), [0.0, 0.0, 0.5, 1.0, 1.0])


def test_load_factor_hand_example() -> None:
    # Worked example from the lecture notes: |(1.2, -0.4, 19.6)| = 19.641 m/s^2.
    n = load_factor_from_accel(np.array([[1.2, -0.4, 19.6]]))
    assert n[0] == pytest.approx(math.sqrt(385.76) / STANDARD_GRAVITY_MPS2)
    assert n[0] == pytest.approx(2.003, abs=1e-3)


def test_config_rejects_unordered_phases() -> None:
    with pytest.raises(ValueError):
        FlightProfileConfig(climb_end_s=20.0)


def test_generator_is_reproducible() -> None:
    first = SyntheticFlightLogGenerator(FlightProfileConfig(seed=7)).generate()
    second = SyntheticFlightLogGenerator(FlightProfileConfig(seed=7)).generate()
    pd.testing.assert_frame_equal(first, second)


def test_columns_and_rate(log: FlightLog) -> None:
    assert tuple(log.data.columns) == LOG_COLUMNS
    assert log.sample_rate_hz == pytest.approx(50.0)
    assert log.duration_s == pytest.approx(299.98)


def test_flight_log_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        FlightLog(pd.DataFrame({"time_s": [0.0, 0.02]}))


def test_gap_detection(log: FlightLog) -> None:
    gaps = log.time_gaps()
    assert len(gaps) == 1
    assert gaps.loc[0, "missing_samples"] == 60
    assert gaps.loc[0, "gap_start_s"] == pytest.approx(99.98)


def test_missing_values(log: FlightLog) -> None:
    missing = log.missing_values()
    assert missing["baro_alt_m"] > 0
    assert missing["accel_z_mps2"] == 0


def test_rolling_median_removes_spikes(log: FlightLog) -> None:
    assert log.load_factor().max() > 4.0  # injected glitches
    assert log.robust_load_factor().max() == pytest.approx(1.8, abs=0.05)


def test_robust_load_factor_rejects_even_window(log: FlightLog) -> None:
    with pytest.raises(ValueError):
        log.robust_load_factor(window=4)


def test_ground_segment_is_one_g(log: FlightLog) -> None:
    ground = log.segment(0.0, 25.0)
    assert ground.robust_load_factor().median() == pytest.approx(1.0, abs=0.01)
    assert ground.data["gyro_z_dps"].std() == pytest.approx(0.3, abs=0.03)


def test_climb_rate_sign(log: FlightLog) -> None:
    rate = log.climb_rate(window_s=2.0)
    climbing = log.time.between(60.0, 110.0)
    descending = log.time.between(225.0, 255.0)
    assert rate[climbing].mean() > 5.0
    assert rate[descending].mean() < -5.0


def test_sample_csv_matches_generator() -> None:
    if not SAMPLE_CSV.exists():
        pytest.skip("sample CSV not generated")
    loaded = FlightLog.from_csv(SAMPLE_CSV)
    assert len(loaded) == len(SyntheticFlightLogGenerator().generate())


def test_plot_overview_has_three_panels(log: FlightLog) -> None:
    import matplotlib

    matplotlib.use("Agg")
    fig = log.plot_overview()
    assert len(fig.axes) == 3
