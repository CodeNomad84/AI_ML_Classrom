"""Sensor hierarchy: one interface for every sensor on the vehicle.

Provenance: new content (Week 2). The OOP concepts follow, but are rewritten
from, W3Schools (Python Classes, ``__init__``, Inheritance, Polymorphism) and
GeeksforGeeks (Abstract Classes in Python, Understanding Python Dataclasses,
Python ``property()``). The barometric formula is the International Standard
Atmosphere troposphere model (adapted from Anderson, *Introduction to Flight*,
Ch. 3).

Design (the "anchor" of Block 0):

* :class:`SensorReading` is an immutable record: timestamp, values, sensor id
  and a validity flag.
* :class:`BaseSensor` is an abstract base class. It owns everything that is the
  same for every sensor (buffering, timestamp checks, conversion to arrays and
  DataFrames) and leaves two decisions to subclasses:
  :meth:`BaseSensor.validate` (is this raw sample physically plausible?) and
  :meth:`BaseSensor.preprocess` (turn a raw sample into engineering units).
* :meth:`BaseSensor.add_reading` composes the two in a fixed order. This is the
  *template method* pattern: the algorithm is fixed in the base class, the
  steps are filled in by subclasses.

Concrete sensors shipped here: :class:`IMUSensor` and :class:`BarometerSensor`.
The LiDAR and GPS sensors are Assignment 1; the instructor's reference versions
join this module after the assignment has been graded.

Convention: ``CHANNELS`` names the *preprocessed* values that a sensor stores,
in order. A raw sample must have the same number of values.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, Final

import numpy as np
import pandas as pd

from aerospace_ai.utils import STANDARD_GRAVITY_MPS2, load_factor_from_accel

DEFAULT_BUFFER_SIZE: Final[int] = 100_000

# International Standard Atmosphere, troposphere (sea level to 11 km).
ISA_SEA_LEVEL_PRESSURE_PA: Final[float] = 101_325.0
ISA_SEA_LEVEL_TEMPERATURE_K: Final[float] = 288.15
ISA_LAPSE_RATE_K_PER_M: Final[float] = 0.0065
DRY_AIR_GAS_CONSTANT: Final[float] = 287.05287
ISA_EXPONENT: Final[float] = (
    DRY_AIR_GAS_CONSTANT * ISA_LAPSE_RATE_K_PER_M / STANDARD_GRAVITY_MPS2
)

# Typical MEMS IMU full-scale ranges (for example +-16 g, +-2000 deg/s).
IMU_ACCEL_RANGE_MPS2: Final[float] = 16.0 * STANDARD_GRAVITY_MPS2
IMU_GYRO_RANGE_DPS: Final[float] = 2000.0
N_AXES: Final[int] = 3

# Plausible static pressure for a small UAV (about -500 m to 9 km altitude).
BARO_MIN_PRESSURE_PA: Final[float] = 30_000.0
BARO_MAX_PRESSURE_PA: Final[float] = 110_000.0


def pressure_to_altitude(
    pressure_pa: float | np.ndarray,
    reference_pressure_pa: float = ISA_SEA_LEVEL_PRESSURE_PA,
) -> float | np.ndarray:
    """Convert static pressure to ISA pressure altitude (troposphere only).

    h = (T0 / L) * (1 - (p / p_ref) ** (R * L / g0))

    Args:
        pressure_pa: Static pressure in Pa (scalar or array).
        reference_pressure_pa: Pressure at h = 0 (101 325 Pa, or the local QNH).

    Returns:
        Altitude in metres above the reference level.
    """
    ratio = np.asarray(pressure_pa, dtype=float) / reference_pressure_pa
    altitude = (ISA_SEA_LEVEL_TEMPERATURE_K / ISA_LAPSE_RATE_K_PER_M) * (
        1.0 - ratio**ISA_EXPONENT
    )
    return float(altitude) if altitude.ndim == 0 else altitude


@dataclass(frozen=True, slots=True)
class SensorReading:
    """One time-stamped sample from one sensor.

    ``frozen=True`` makes a reading immutable (and hashable): once logged, a
    measurement must never change. ``slots=True`` removes the per-object
    ``__dict__``, which saves memory when a sensor keeps 100 000 readings.
    The dataclass decorator writes ``__init__``, ``__repr__`` and ``__eq__``.

    Attributes:
        timestamp: Time of the sample in seconds.
        values: Sample values, one per channel (a tuple, so it is immutable).
        sensor_id: Identifier of the sensor that produced the sample.
        is_valid: False if the sensor's ``validate`` rejected the raw sample.
    """

    timestamp: float
    values: tuple[float, ...]
    sensor_id: str
    is_valid: bool = True

    def __post_init__(self) -> None:
        """Reject readings that cannot be correct whatever the sensor.

        Raises:
            ValueError: If the timestamp is not finite or there are no values.
        """
        if not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be finite, got {self.timestamp}")
        if len(self.values) == 0:
            raise ValueError("a reading needs at least one value")

    def __len__(self) -> int:
        """Return the number of channels in the reading."""
        return len(self.values)

    def as_array(self) -> np.ndarray:
        """Return the values as a 1-D float NumPy array."""
        return np.asarray(self.values, dtype=float)


class BaseSensor(ABC):
    """Abstract base class shared by every sensor on the vehicle.

    Subclasses must define the class attribute ``CHANNELS`` (or override the
    :attr:`channel_names` property) and implement :meth:`validate` and
    :meth:`preprocess`. Everything else is inherited.

    Example:
        >>> imu = IMUSensor("imu0", rate_hz=50.0)
        >>> r = imu.add_reading(0.0, [0.1, 0.0, 9.8, 0.0, 0.0, 0.0])
        >>> r.is_valid, len(imu)
        (True, 1)
    """

    CHANNELS: ClassVar[tuple[str, ...]] = ()
    UNITS: ClassVar[tuple[str, ...]] = ()

    __slots__ = ("_sensor_id", "_rate_hz", "_buffer", "_n_rejected")

    def __init__(
        self,
        sensor_id: str,
        rate_hz: float,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        """Initialise the parts every sensor has.

        Args:
            sensor_id: Unique, non-empty name such as ``"imu0"``.
            rate_hz: Nominal sampling rate in Hz (must be positive).
            buffer_size: Maximum number of readings kept; the oldest are
                dropped first.

        Raises:
            ValueError: If ``sensor_id`` is empty or a number is not positive.
        """
        if not sensor_id:
            raise ValueError("sensor_id must be a non-empty string")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self._sensor_id = sensor_id
        self.rate_hz = rate_hz  # goes through the validating property setter
        self._buffer: deque[SensorReading] = deque(maxlen=buffer_size)
        self._n_rejected = 0

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Check, when a subclass is *defined*, that CHANNELS and UNITS agree.

        Raises:
            TypeError: If a subclass declares a different number of units and
                channels.
        """
        super().__init_subclass__(**kwargs)
        if cls.UNITS and len(cls.UNITS) != len(cls.CHANNELS):
            raise TypeError(
                f"{cls.__name__}: {len(cls.CHANNELS)} CHANNELS but "
                f"{len(cls.UNITS)} UNITS"
            )

    # ------------------------------------------------------------------
    # The two steps every concrete sensor must provide.
    # ------------------------------------------------------------------
    @abstractmethod
    def validate(self, raw: np.ndarray) -> bool:
        """Return True if a raw sample is physically plausible.

        Args:
            raw: Raw sample with shape ``(n_channels,)``.
        """

    @abstractmethod
    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Convert a *valid* raw sample to calibrated engineering units.

        Args:
            raw: Raw sample with shape ``(n_channels,)``.

        Returns:
            Array with shape ``(n_channels,)``.
        """

    # ------------------------------------------------------------------
    # Read-only and validated attributes.
    # ------------------------------------------------------------------
    @property
    def sensor_id(self) -> str:
        """Return the sensor's identifier (read-only)."""
        return self._sensor_id

    @property
    def rate_hz(self) -> float:
        """Return the nominal sampling rate in Hz."""
        return self._rate_hz

    @rate_hz.setter
    def rate_hz(self, value: float) -> None:
        """Set the sampling rate after checking that it is positive and finite."""
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"rate_hz must be positive and finite, got {value}")
        self._rate_hz = float(value)

    @property
    def period_s(self) -> float:
        """Return the nominal time between samples, 1 / rate_hz."""
        return 1.0 / self._rate_hz

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return the names of the preprocessed channels, in order."""
        return self.CHANNELS

    @property
    def n_channels(self) -> int:
        """Return the number of values per sample."""
        return len(self.channel_names)

    @property
    def n_rejected(self) -> int:
        """Return how many raw samples failed :meth:`validate`."""
        return self._n_rejected

    @property
    def latest(self) -> SensorReading | None:
        """Return the most recent reading, or None if there is none yet."""
        return self._buffer[-1] if self._buffer else None

    # ------------------------------------------------------------------
    # The template method: fixed algorithm, sensor-specific steps.
    # ------------------------------------------------------------------
    def add_reading(self, timestamp: float, raw: Sequence[float]) -> SensorReading:
        """Validate, preprocess and store one raw sample.

        Steps: check the shape, check that time increases, call
        :meth:`validate`, call :meth:`preprocess` if valid, store the result.
        Invalid samples are kept *unprocessed* with ``is_valid=False`` so they
        can be inspected later; they are excluded from :meth:`to_array`.

        Args:
            timestamp: Sample time in seconds; must exceed the previous one.
            raw: Raw values, one per channel.

        Returns:
            The stored :class:`SensorReading`.

        Raises:
            ValueError: If the sample has the wrong length or the timestamp
                does not increase.
        """
        values = np.asarray(raw, dtype=float)
        if values.shape != (self.n_channels,):
            raise ValueError(
                f"{self.sensor_id}: expected {self.n_channels} values, "
                f"got shape {values.shape}"
            )
        last = self.latest
        if last is not None and timestamp <= last.timestamp:
            raise ValueError(
                f"{self.sensor_id}: timestamp {timestamp} is not after "
                f"{last.timestamp}"
            )
        is_valid = bool(self.validate(values))
        if is_valid:
            values = np.asarray(self.preprocess(values), dtype=float)
        else:
            self._n_rejected += 1
        reading = SensorReading(
            timestamp=float(timestamp),
            values=tuple(float(v) for v in values),
            sensor_id=self.sensor_id,
            is_valid=is_valid,
        )
        self._buffer.append(reading)
        return reading

    def add_readings(self, timestamps: Sequence[float], raw: np.ndarray) -> int:
        """Add a block of samples (one row per sample).

        Args:
            timestamps: Sample times, shape ``(N,)``.
            raw: Raw values, shape ``(N, n_channels)``.

        Returns:
            Number of samples that passed validation.
        """
        block = np.asarray(raw, dtype=float)
        if len(timestamps) != len(block):
            raise ValueError("timestamps and raw must have the same length")
        return sum(
            self.add_reading(float(t), row).is_valid
            for t, row in zip(timestamps, block)
        )

    # ------------------------------------------------------------------
    # Containers and conversions.
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of stored readings (valid and invalid)."""
        return len(self._buffer)

    def __iter__(self) -> Iterator[SensorReading]:
        """Iterate over stored readings, oldest first."""
        return iter(self._buffer)

    def __repr__(self) -> str:
        """Return an unambiguous one-line description."""
        return (
            f"{type(self).__name__}(sensor_id={self.sensor_id!r}, "
            f"rate_hz={self.rate_hz}, readings={len(self)}, "
            f"rejected={self.n_rejected})"
        )

    def readings(self, valid_only: bool = False) -> list[SensorReading]:
        """Return the stored readings as a list.

        Args:
            valid_only: If True, drop readings that failed validation.
        """
        return [r for r in self._buffer if r.is_valid or not valid_only]

    def timestamps(self, valid_only: bool = True) -> np.ndarray:
        """Return reading times as an array of shape ``(N,)``."""
        return np.array([r.timestamp for r in self.readings(valid_only)], dtype=float)

    def to_array(self, valid_only: bool = True) -> np.ndarray:
        """Return reading values as an array of shape ``(N, n_channels)``."""
        rows = [r.values for r in self.readings(valid_only)]
        return np.array(rows, dtype=float).reshape(len(rows), self.n_channels)

    def to_frame(self) -> pd.DataFrame:
        """Return all readings as a DataFrame (one column per channel)."""
        frame = pd.DataFrame(
            self.to_array(valid_only=False), columns=list(self.channel_names)
        )
        frame.insert(0, "timestamp", self.timestamps(valid_only=False))
        frame["is_valid"] = [r.is_valid for r in self._buffer]
        return frame

    def clear(self) -> None:
        """Forget all readings and reset the rejection counter."""
        self._buffer.clear()
        self._n_rejected = 0


class IMUSensor(BaseSensor):
    """Six-axis inertial measurement unit (accelerometer + gyroscope).

    Raw sample: ``(ax, ay, az, gx, gy, gz)`` in m/s^2 and deg/s, body axes with
    z up (a parked aircraft reads ``az = +g``). Preprocessing removes the
    constant biases. A sample is invalid if any value is not finite or any axis
    is at or beyond full scale (the sensor saturated, so the true value is
    unknown).

    Example:
        >>> imu = IMUSensor("imu0", 50.0, accel_bias=(0.1, 0.0, 0.0))
        >>> imu.add_reading(0.0, [0.3, 0.0, 9.8, 1.0, 0.0, 0.0]).values[0]
        0.19999999999999998
    """

    CHANNELS: ClassVar[tuple[str, ...]] = (
        "accel_x_mps2",
        "accel_y_mps2",
        "accel_z_mps2",
        "gyro_x_dps",
        "gyro_y_dps",
        "gyro_z_dps",
    )
    UNITS: ClassVar[tuple[str, ...]] = ("m/s^2",) * N_AXES + ("deg/s",) * N_AXES

    __slots__ = ("_accel_bias", "_gyro_bias", "accel_range_mps2", "gyro_range_dps")

    def __init__(
        self,
        sensor_id: str,
        rate_hz: float,
        accel_bias: Sequence[float] = (0.0, 0.0, 0.0),
        gyro_bias: Sequence[float] = (0.0, 0.0, 0.0),
        accel_range_mps2: float = IMU_ACCEL_RANGE_MPS2,
        gyro_range_dps: float = IMU_GYRO_RANGE_DPS,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        """Create an IMU.

        Args:
            sensor_id: Unique name, for example ``"imu0"``.
            rate_hz: Sampling rate in Hz.
            accel_bias: Constant accelerometer offsets (x, y, z) in m/s^2.
            gyro_bias: Constant gyroscope offsets (x, y, z) in deg/s.
            accel_range_mps2: Accelerometer full scale (absolute value).
            gyro_range_dps: Gyroscope full scale (absolute value).
            buffer_size: Maximum number of stored readings.
        """
        super().__init__(sensor_id, rate_hz, buffer_size)
        self.accel_bias = accel_bias
        self.gyro_bias = gyro_bias
        self.accel_range_mps2 = accel_range_mps2
        self.gyro_range_dps = gyro_range_dps

    @staticmethod
    def _as_axis_vector(value: Sequence[float], name: str) -> np.ndarray:
        """Return ``value`` as a float array of shape (3,) or raise ValueError."""
        vector = np.asarray(value, dtype=float)
        if vector.shape != (N_AXES,):
            raise ValueError(f"{name} must have {N_AXES} components")
        return vector

    @property
    def accel_bias(self) -> np.ndarray:
        """Return a copy of the accelerometer bias (x, y, z) in m/s^2."""
        return self._accel_bias.copy()

    @accel_bias.setter
    def accel_bias(self, value: Sequence[float]) -> None:
        """Set the accelerometer bias after checking it has three components."""
        self._accel_bias = self._as_axis_vector(value, "accel_bias")

    @property
    def gyro_bias(self) -> np.ndarray:
        """Return a copy of the gyroscope bias (x, y, z) in deg/s."""
        return self._gyro_bias.copy()

    @gyro_bias.setter
    def gyro_bias(self, value: Sequence[float]) -> None:
        """Set the gyroscope bias after checking it has three components."""
        self._gyro_bias = self._as_axis_vector(value, "gyro_bias")

    def validate(self, raw: np.ndarray) -> bool:
        """Reject non-finite or saturated samples."""
        accel, gyro = raw[:N_AXES], raw[N_AXES:]
        return bool(
            np.all(np.isfinite(raw))
            and np.all(np.abs(accel) < self.accel_range_mps2)
            and np.all(np.abs(gyro) < self.gyro_range_dps)
        )

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Subtract the accelerometer and gyroscope biases."""
        return raw - np.concatenate([self._accel_bias, self._gyro_bias])

    def calibrate(self, rest_samples: np.ndarray) -> None:
        """Estimate both biases from raw samples taken while parked and level.

        At rest the true specific force is (0, 0, +g0) and the true angular
        rate is zero, so the bias is the mean reading minus the true value.

        Args:
            rest_samples: Raw samples, shape ``(N, 6)``, vehicle stationary.
        """
        samples = np.asarray(rest_samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != self.n_channels:
            raise ValueError(f"rest_samples must have shape (N, {self.n_channels})")
        mean = samples.mean(axis=0)
        gravity = np.array([0.0, 0.0, STANDARD_GRAVITY_MPS2])
        self.accel_bias = mean[:N_AXES] - gravity
        self.gyro_bias = mean[N_AXES:]

    def load_factor(self) -> np.ndarray:
        """Return n = |f| / g0 for every valid stored reading."""
        return load_factor_from_accel(self.to_array()[:, :N_AXES])


class BarometerSensor(BaseSensor):
    """Static-pressure sensor that reports ISA pressure altitude.

    Raw sample: ``(pressure_pa,)``. Stored value: ``(altitude_m,)`` above the
    reference pressure level (standard 101 325 Pa, or the airfield QNH).

    Example:
        >>> baro = BarometerSensor("baro0", 25.0)
        >>> round(baro.add_reading(0.0, [95_000.0]).values[0], 1)
        540.3
    """

    CHANNELS: ClassVar[tuple[str, ...]] = ("altitude_m",)
    UNITS: ClassVar[tuple[str, ...]] = ("m",)

    __slots__ = ("_reference_pressure_pa",)

    def __init__(
        self,
        sensor_id: str,
        rate_hz: float,
        reference_pressure_pa: float = ISA_SEA_LEVEL_PRESSURE_PA,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        """Create a barometer.

        Args:
            sensor_id: Unique name, for example ``"baro0"``.
            rate_hz: Sampling rate in Hz.
            reference_pressure_pa: Pressure that corresponds to 0 m.
            buffer_size: Maximum number of stored readings.
        """
        super().__init__(sensor_id, rate_hz, buffer_size)
        self.reference_pressure_pa = reference_pressure_pa

    @property
    def reference_pressure_pa(self) -> float:
        """Return the pressure that corresponds to zero altitude."""
        return self._reference_pressure_pa

    @reference_pressure_pa.setter
    def reference_pressure_pa(self, value: float) -> None:
        """Set the reference pressure after a plausibility check."""
        if not BARO_MIN_PRESSURE_PA < value < BARO_MAX_PRESSURE_PA:
            raise ValueError(f"implausible reference pressure {value} Pa")
        self._reference_pressure_pa = float(value)

    def validate(self, raw: np.ndarray) -> bool:
        """Accept finite pressures inside the plausible static range."""
        pressure = raw[0]
        return bool(
            np.isfinite(pressure)
            and BARO_MIN_PRESSURE_PA < pressure < BARO_MAX_PRESSURE_PA
        )

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Convert pressure to altitude with the ISA troposphere model."""
        return np.atleast_1d(pressure_to_altitude(raw, self._reference_pressure_pa))
