"""Your pytest tests for Assignment 1 (Task 8).

Provenance: new content (Week 2). Pytest usage follows (rewritten, not copied)
GeeksforGeeks "Getting Started with Pytest".

Run from the repository root:

    python -m pytest assignments/assignment01 -q

Rules for Task 8 (see the README):
* at least 8 test functions in total (the two examples below count),
* use at least one ``pytest.raises``, one ``@pytest.mark.parametrize`` and one
  fixture,
* tests must be self-contained: small hand-made numbers, no CSV files,
* every test must pass on a correct implementation. The instructor also runs
  your tests against three subtly broken implementations; each one your tests
  catch earns points.
"""

from __future__ import annotations

import math  # noqa: F401  (handy for the tests you will write)

import numpy as np
import pytest

from sensor_hierarchy import GPSSensor, LidarSensor, sensor_report  # noqa: F401

# Tehran Mehrabad airport, used as the local frame origin in the examples.
ORIGIN_LAT_DEG = 35.6892
ORIGIN_LON_DEG = 51.3134
ORIGIN_ALT_M = 1190.0


@pytest.fixture
def gps() -> GPSSensor:
    """Return a fresh GPS receiver with its origin at Mehrabad."""
    return GPSSensor("gps0", 5.0, ORIGIN_LAT_DEG, ORIGIN_LON_DEG, ORIGIN_ALT_M)


# --------------------------------------------------------------- examples
def test_origin_fix_maps_to_zero(gps: GPSSensor) -> None:
    """A fix exactly at the origin must be (0, 0, 0) in the local frame."""
    reading = gps.add_reading(0.0, [ORIGIN_LAT_DEG, ORIGIN_LON_DEG, ORIGIN_ALT_M, 1.0])
    assert reading.is_valid
    np.testing.assert_allclose(reading.values[:3], [0.0, 0.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("n_beams", [1, 4, 16, 360])
def test_lidar_has_one_channel_per_beam(n_beams: int) -> None:
    """channel_names and angles_deg both have n_beams entries."""
    lidar = LidarSensor("lidar0", 10.0, n_beams=n_beams)
    assert len(lidar.channel_names) == n_beams
    assert lidar.angles_deg.shape == (n_beams,)


# ------------------------------------------------------- your tests below
# TODO (Task 8): add at least 6 more tests. Ideas (pick your own too):
#   * one hundredth of a degree of latitude north of the origin is ~1112 m north
#   * the same longitude step gives less distance east than north (cos(lat0))
#   * a fix with HDOP above max_hdop is stored but marked invalid
#   * a latitude of 135 deg is rejected
#   * a beam reading 0.0, inf or below min_range_m becomes NaN after preprocess
#   * a scan with too few usable beams is rejected
#   * nearest_obstacle ignores NaN beams and raises ValueError with no scan
#   * sensor_report raises ValueError for duplicate ids
