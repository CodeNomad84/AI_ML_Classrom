"""Unit tests for aerospace_ai.rover and the rover loader (Week 5). Provenance: new content."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerospace_ai.datasets import (
    ROVER_FEATURES,
    ROVER_TARGET,
    Dataset,
    DatasetFactory,
    RoverRangeForecastLoader,
    rover_cleaning_pipeline,
)
from aerospace_ai.preprocessing import audit_frame, count_duplicate_keys
from aerospace_ai.rover import (
    LIDAR_MAX_RANGE_M,
    RANGE_COLUMN,
    ROVER_COLUMNS,
    RoverLogConfig,
    RoverLogGenerator,
)

SMALL = RoverLogConfig(n_runs=4, seed=3, stuck_runs=1)


@pytest.fixture(scope="module")
def generator() -> RoverLogGenerator:
    """A small generator that has produced its log once."""
    gen = RoverLogGenerator(SMALL)
    gen.generate()
    return gen


def test_truth_is_physical(generator: RoverLogGenerator) -> None:
    truth = generator.generate_truth()
    assert list(truth.columns) == list(ROVER_COLUMNS)
    for _, run in truth.groupby("run_id"):
        assert run["wheel_speed_mps"].min() >= 0.0
        assert run["lidar_range_m"].is_monotonic_decreasing  # never drives backwards
        assert run["wheel_speed_mps"].iloc[-1] == 0.0  # ends at rest
        # Stops near the chosen distance; the drive-train lag costs a few cm.
        assert 0.5 <= run["lidar_range_m"].iloc[-1] <= 1.5


def test_generation_is_reproducible() -> None:
    a = RoverLogGenerator(SMALL).generate()
    b = RoverLogGenerator(SMALL).generate()
    assert a.equals(b)
    c = RoverLogGenerator(RoverLogConfig(n_runs=4, seed=4, stuck_runs=1)).generate()
    assert not a.equals(c)


def test_defect_log_matches_the_data(generator: RoverLogGenerator) -> None:
    raw = generator.generate()
    log = generator.defects
    table = audit_frame(raw, {RANGE_COLUMN: 0.0})
    truth = generator.generate_truth()
    assert log.no_return_out_of_range == int(
        (truth[RANGE_COLUMN] > LIDAR_MAX_RANGE_M).sum()
    )
    # Duplicated rows repeat sentinels and NaN, so compare after de-duplication.
    unique = raw.drop_duplicates(subset=["run_id", "time_s"])
    assert int((unique[RANGE_COLUMN] == 0.0).sum()) == (
        log.no_return_out_of_range + log.no_return_dropout
    )
    assert int(unique["wheel_speed_mps"].isna().sum()) == log.speed_nan
    assert count_duplicate_keys(raw) == log.duplicate_rows
    assert table.loc["gyro_z_dps", "longest_repeat"] >= SMALL.stuck_samples
    assert len(log.stuck_segments) == SMALL.stuck_runs
    assert log.as_series()["duplicate_rows"] == log.duplicate_rows


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        RoverLogConfig(n_runs=0)
    with pytest.raises(ValueError):
        RoverLogConfig(n_runs=2, stuck_runs=3)
    with pytest.raises(ValueError):
        RoverLogConfig(dropout_fraction=1.5)


def test_cleaning_pipeline_output(generator: RoverLogGenerator) -> None:
    clean = rover_cleaning_pipeline().fit_transform(generator.generate())
    assert clean[[*ROVER_FEATURES[:3], ROVER_TARGET]].notna().all().all()
    assert clean[RANGE_COLUMN].between(0.3, LIDAR_MAX_RANGE_M).all()
    assert clean["accel_x_mps2"].abs().max() < 5.0  # spikes removed
    assert not clean.duplicated(subset=["run_id", "time_s"]).any()


def test_loader_registered_and_grouped() -> None:
    assert (
        DatasetFactory.loader_class("rover_range_forecast") is RoverRangeForecastLoader
    )
    data = DatasetFactory.load("rover_range_forecast", n_runs=4, seed=3)
    assert isinstance(data, Dataset) and data.feature_names == ROVER_FEATURES
    assert set(np.unique(data.groups)) == {1, 2, 3, 4}
    # Target = range one second later in the same run: never more than 1 s of travel.
    assert np.all(data.X[:, 0] - data.y > -0.2)
    assert data.metadata["rows_dropped"] > 0


def test_loader_reads_csv(tmp_path: Path) -> None:
    path = RoverLogGenerator(SMALL).to_csv(tmp_path / "log.csv")
    from_file = DatasetFactory.load("rover_range_forecast", path=path)
    generated = DatasetFactory.load("rover_range_forecast", n_runs=4, seed=3)
    assert len(from_file) == len(generated)
    np.testing.assert_allclose(from_file.y, generated.y)


def test_loader_horizon_validation() -> None:
    with pytest.raises(ValueError):
        RoverRangeForecastLoader(horizon_s=0.05)
    longer = DatasetFactory.load(
        "rover_range_forecast", n_runs=4, seed=3, horizon_s=2.0
    )
    shorter = DatasetFactory.load("rover_range_forecast", n_runs=4, seed=3)
    assert len(longer) < len(shorter)
