"""Unit tests for aerospace_ai.sensors (Week 2). Provenance: new content."""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import numpy as np
import pytest

from aerospace_ai.sensors import (
    ISA_SEA_LEVEL_PRESSURE_PA,
    BarometerSensor,
    BaseSensor,
    IMUSensor,
    SensorReading,
    pressure_to_altitude,
)
from aerospace_ai.utils import STANDARD_GRAVITY_MPS2

LEVEL_RAW = [0.0, 0.0, STANDARD_GRAVITY_MPS2, 0.0, 0.0, 0.0]


@pytest.fixture
def imu() -> IMUSensor:
    """Return a 50 Hz IMU with known biases."""
    return IMUSensor(
        "imu0", 50.0, accel_bias=(0.1, -0.2, 0.3), gyro_bias=(0.5, 0.0, -0.5)
    )


# ---------------------------------------------------------------- SensorReading
def test_reading_is_immutable_and_comparable() -> None:
    a = SensorReading(1.0, (1.0, 2.0), "s")
    b = SensorReading(1.0, (1.0, 2.0), "s")
    assert a == b and hash(a) == hash(b)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.timestamp = 2.0  # type: ignore[misc]


def test_reading_has_no_instance_dict() -> None:
    assert not hasattr(SensorReading(0.0, (1.0,), "s"), "__dict__")


@pytest.mark.parametrize(
    ("timestamp", "values"), [(math.nan, (1.0,)), (math.inf, (1.0,)), (0.0, ())]
)
def test_reading_rejects_impossible_records(
    timestamp: float, values: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError):
        SensorReading(timestamp, values, "s")


# ------------------------------------------------------------------ BaseSensor
def test_base_sensor_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseSensor("x", 1.0)  # type: ignore[abstract]


def test_subclass_with_mismatched_units_is_rejected_at_definition() -> None:
    with pytest.raises(TypeError):

        class _Broken(BaseSensor):  # pylint: disable=unused-variable
            CHANNELS: ClassVar[tuple[str, ...]] = ("a", "b")
            UNITS: ClassVar[tuple[str, ...]] = ("m",)


@pytest.mark.parametrize("rate", [0.0, -5.0, math.nan, math.inf])
def test_rate_must_be_positive_and_finite(rate: float) -> None:
    with pytest.raises(ValueError):
        IMUSensor("imu0", rate)


def test_rate_setter_validates(imu: IMUSensor) -> None:
    imu.rate_hz = 100.0
    assert imu.period_s == pytest.approx(0.01)
    with pytest.raises(ValueError):
        imu.rate_hz = -1.0


def test_sensor_id_is_read_only(imu: IMUSensor) -> None:
    with pytest.raises(AttributeError):
        imu.sensor_id = "other"  # type: ignore[misc]


def test_add_reading_checks_shape_and_time(imu: IMUSensor) -> None:
    imu.add_reading(0.0, LEVEL_RAW)
    with pytest.raises(ValueError):
        imu.add_reading(0.02, [1.0, 2.0])
    with pytest.raises(ValueError):
        imu.add_reading(0.0, LEVEL_RAW)


def test_buffer_drops_oldest() -> None:
    imu = IMUSensor("imu0", 50.0, buffer_size=3)
    for k in range(5):
        imu.add_reading(0.02 * k, LEVEL_RAW)
    assert len(imu) == 3
    assert imu.timestamps()[0] == pytest.approx(0.04)


def test_to_frame_columns(imu: IMUSensor) -> None:
    imu.add_reading(0.0, LEVEL_RAW)
    imu.add_reading(0.02, [math.nan, *LEVEL_RAW[1:]])
    frame = imu.to_frame()
    assert list(frame.columns) == ["timestamp", *IMUSensor.CHANNELS, "is_valid"]
    assert frame["is_valid"].tolist() == [True, False]
    assert imu.to_array().shape == (1, 6)


def test_repr_names_the_class(imu: IMUSensor) -> None:
    assert repr(imu).startswith("IMUSensor(sensor_id='imu0'")


# ------------------------------------------------------------------- IMUSensor
def test_imu_removes_bias(imu: IMUSensor) -> None:
    reading = imu.add_reading(0.0, [0.1, -0.2, 10.1, 1.5, 0.0, -0.5])
    np.testing.assert_allclose(reading.values, [0.0, 0.0, 9.8, 1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "raw",
    [
        [0.0, 0.0, 16.0 * STANDARD_GRAVITY_MPS2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 9.8, 0.0, -2000.0, 0.0],
        [0.0, math.nan, 9.8, 0.0, 0.0, 0.0],
    ],
)
def test_imu_rejects_saturated_or_nan(imu: IMUSensor, raw: list[float]) -> None:
    reading = imu.add_reading(0.0, raw)
    assert not reading.is_valid
    np.testing.assert_array_equal(reading.as_array(), raw)  # stored unprocessed
    assert imu.n_rejected == 1


def test_imu_calibrate_recovers_bias() -> None:
    rng = np.random.default_rng(0)
    true_bias = np.array([0.05, -0.03, 0.12, 0.4, -0.2, 0.1])
    rest = np.array(LEVEL_RAW) + true_bias + rng.normal(0, 1e-3, (500, 6))
    imu = IMUSensor("imu0", 50.0)
    imu.calibrate(rest)
    np.testing.assert_allclose(imu.accel_bias, true_bias[:3], atol=1e-3)
    np.testing.assert_allclose(imu.gyro_bias, true_bias[3:], atol=1e-3)


def test_imu_bias_property_returns_copy(imu: IMUSensor) -> None:
    bias = imu.accel_bias
    bias[0] = 99.0
    assert imu.accel_bias[0] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        imu.accel_bias = (1.0, 2.0)


def test_imu_load_factor_hand_example() -> None:
    # Lecture example: |(1.2, -0.4, 19.6)| / g0 = 2.003.
    imu = IMUSensor("imu0", 50.0)
    imu.add_reading(0.0, [1.2, -0.4, 19.6, 0.0, 0.0, 0.0])
    assert imu.load_factor()[0] == pytest.approx(2.003, abs=1e-3)


# ------------------------------------------------------------- BarometerSensor
def test_isa_altitude_hand_example() -> None:
    # Lecture example: 95 000 Pa -> 540.3 m in the standard atmosphere.
    assert pressure_to_altitude(95_000.0) == pytest.approx(540.34, abs=0.01)
    assert pressure_to_altitude(ISA_SEA_LEVEL_PRESSURE_PA) == pytest.approx(0.0)


def test_barometer_uses_reference_pressure() -> None:
    baro = BarometerSensor("baro0", 25.0, reference_pressure_pa=95_000.0)
    assert baro.add_reading(0.0, [95_000.0]).values[0] == pytest.approx(0.0)
    assert not baro.add_reading(0.04, [5_000.0]).is_valid
    with pytest.raises(ValueError):
        baro.reference_pressure_pa = 0.0


def test_polymorphism_one_loop_many_sensors() -> None:
    sensors: list[BaseSensor] = [IMUSensor("imu0", 50.0), BarometerSensor("b", 25.0)]
    raws = [LEVEL_RAW, [101_325.0]]
    for sensor, raw in zip(sensors, raws):
        assert sensor.add_reading(0.0, raw).is_valid
    assert [len(s) for s in sensors] == [1, 1]
