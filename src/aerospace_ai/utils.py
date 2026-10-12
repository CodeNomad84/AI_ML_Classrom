"""Shared utilities for the course: flight-log generation, loading and inspection.

Provenance: new content (Week 1). Pandas/NumPy idioms follow the W3Schools
Pandas tutorial (https://www.w3schools.com/python/pandas/default.asp) and the
GeeksforGeeks ``DataFrame.describe()`` article; the load-factor relation is
standard flight mechanics (adapted from Anderson, *Introduction to Flight*,
Ch. 6).

The module provides two classes:

* :class:`SyntheticFlightLogGenerator` builds a physically motivated 50 Hz
  UAV flight log (IMU + barometer + airspeed + temperature) with realistic
  defects: random dropouts, a timestamp gap and single-sample spikes.
* :class:`FlightLog` wraps a loaded log and answers the engineering questions
  asked in Week 1 (sample rate, gaps, missing values, load factor, climb rate).

Sign convention: body z-axis points up, so a level, unaccelerated aircraft
(or one parked on the ground) reads ``accel_z = +g``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

STANDARD_GRAVITY_MPS2: Final[float] = 9.80665
SEA_LEVEL_TEMPERATURE_C: Final[float] = 15.0
LAPSE_RATE_C_PER_M: Final[float] = 0.0065
DEFAULT_GAP_TOLERANCE: Final[float] = 1.5

TIME_COLUMN: Final[str] = "time_s"
ACCEL_COLUMNS: Final[tuple[str, ...]] = ("accel_x_mps2", "accel_y_mps2", "accel_z_mps2")
GYRO_COLUMNS: Final[tuple[str, ...]] = ("gyro_x_dps", "gyro_y_dps", "gyro_z_dps")
ALTITUDE_COLUMN: Final[str] = "baro_alt_m"
AIRSPEED_COLUMN: Final[str] = "airspeed_mps"
TEMPERATURE_COLUMN: Final[str] = "temp_c"
LOG_COLUMNS: Final[tuple[str, ...]] = (
    TIME_COLUMN,
    *ACCEL_COLUMNS,
    *GYRO_COLUMNS,
    ALTITUDE_COLUMN,
    AIRSPEED_COLUMN,
    TEMPERATURE_COLUMN,
)


def smoothstep(t: np.ndarray, start: float, end: float) -> np.ndarray:
    """Return a C2-continuous ramp from 0 (t <= start) to 1 (t >= end).

    Uses the quintic 6x^5 - 15x^4 + 10x^3, whose first and second derivatives
    vanish at both ends, so velocity and acceleration have no jumps.

    Args:
        t: Time samples in seconds.
        start: Time at which the ramp leaves 0.
        end: Time at which the ramp reaches 1.

    Returns:
        Array with the same shape as ``t`` and values in [0, 1].
    """
    x = np.clip((t - start) / (end - start), 0.0, 1.0)
    return x**3 * (x * (6.0 * x - 15.0) + 10.0)


def sine_pulse(t: np.ndarray, start: float, duration: float) -> np.ndarray:
    """Return a sin^2 pulse that rises from 0 to 1 and back over ``duration``.

    Args:
        t: Time samples in seconds.
        start: Pulse start time in seconds.
        duration: Pulse length in seconds.

    Returns:
        Array with the same shape as ``t`` and values in [0, 1].
    """
    phase = np.clip((t - start) / duration, 0.0, 1.0)
    return np.sin(np.pi * phase) ** 2


def load_factor_from_accel(accel_xyz: np.ndarray) -> np.ndarray:
    """Compute the load factor n = |f| / g0 from accelerometer specific force.

    Args:
        accel_xyz: Array of shape (N, 3) with specific force in m/s^2.

    Returns:
        Array of shape (N,) with the dimensionless load factor.
    """
    return np.linalg.norm(accel_xyz, axis=1) / STANDARD_GRAVITY_MPS2


@dataclass(frozen=True, slots=True)
class Maneuver:
    """A smooth load-factor excursion added on top of 1 g flight.

    Attributes:
        name: Human-readable label (used only for documentation/plots).
        start_s: Time at which the maneuver starts.
        duration_s: Total length of the excursion.
        peak_load_factor: Load factor reached at the middle of the maneuver.
        yaw_rate_dps: Peak yaw rate during the maneuver (turns).
        pitch_rate_dps: Peak pitch rate during the maneuver (pull-ups).
    """

    name: str
    start_s: float
    duration_s: float
    peak_load_factor: float
    yaw_rate_dps: float = 0.0
    pitch_rate_dps: float = 0.0


# A 30 deg coordinated turn at 28 m/s: n = 1/cos(30 deg) = 1.155,
# yaw rate = g * tan(30 deg) / V = 0.202 rad/s = 11.6 deg/s.
DEFAULT_MANEUVERS: Final[tuple[Maneuver, ...]] = (
    Maneuver("coordinated turn", 150.0, 20.0, 1.155, yaw_rate_dps=11.6),
    Maneuver("pull-up", 190.0, 4.0, 1.8, pitch_rate_dps=8.0),
    Maneuver("touchdown", 270.0, 0.6, 1.4),
)


@dataclass(frozen=True, slots=True)
class FlightProfileConfig:
    """Parameters of a synthetic take-off / climb / cruise / descent / landing log.

    Attributes:
        sample_rate_hz: IMU sampling frequency.
        duration_s: Total log length.
        takeoff_s: Start of the take-off roll (aircraft parked before this).
        climb_end_s: Time at which cruise altitude is reached.
        descent_start_s: Time at which the descent starts.
        landing_s: Touchdown time.
        cruise_altitude_m: Altitude above the runway during cruise.
        cruise_airspeed_mps: True airspeed during cruise.
        accel_noise_std_mps2: Accelerometer white-noise standard deviation.
        gyro_noise_std_dps: Gyroscope white-noise standard deviation.
        baro_noise_std_m: Barometric altitude noise standard deviation.
        airspeed_noise_std_mps: Pitot airspeed noise standard deviation.
        temperature_noise_std_c: Temperature sensor noise standard deviation.
        dropout_fraction: Fraction of baro/airspeed samples replaced by NaN.
        gap_start_s: Start of a logger outage (rows removed); None disables it.
        gap_duration_s: Length of the logger outage.
        spike_count: Number of single-sample spikes injected into accel_z.
        spike_magnitude_mps2: Amplitude added to accel_z by each spike.
        maneuvers: Load-factor excursions flown during the flight.
        seed: Seed of the random generator (makes the log reproducible).
    """

    sample_rate_hz: float = 50.0
    duration_s: float = 300.0
    takeoff_s: float = 30.0
    climb_end_s: float = 120.0
    descent_start_s: float = 210.0
    landing_s: float = 270.0
    cruise_altitude_m: float = 600.0
    cruise_airspeed_mps: float = 28.0
    accel_noise_std_mps2: float = 0.05
    gyro_noise_std_dps: float = 0.3
    baro_noise_std_m: float = 0.8
    airspeed_noise_std_mps: float = 0.4
    temperature_noise_std_c: float = 0.1
    dropout_fraction: float = 0.005
    gap_start_s: float | None = 100.0
    gap_duration_s: float = 1.2
    spike_count: int = 3
    spike_magnitude_mps2: float = 40.0
    maneuvers: tuple[Maneuver, ...] = field(default=DEFAULT_MANEUVERS)
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate that the flight phases are ordered and parameters positive."""
        phases = (
            0.0,
            self.takeoff_s,
            self.climb_end_s,
            self.descent_start_s,
            self.landing_s,
            self.duration_s,
        )
        if any(a >= b for a, b in zip(phases, phases[1:])):
            raise ValueError(f"Flight phases must be strictly increasing: {phases}")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if not 0.0 <= self.dropout_fraction < 1.0:
            raise ValueError("dropout_fraction must be in [0, 1)")


class SyntheticFlightLogGenerator:
    """Generate a reproducible UAV flight log with realistic sensor defects.

    Example:
        >>> log = SyntheticFlightLogGenerator(FlightProfileConfig(seed=1)).generate()
        >>> list(log.columns) == list(LOG_COLUMNS)
        True
    """

    # Time constant of the take-off acceleration and landing roll-out.
    _SPEED_RAMP_S: Final[float] = 20.0
    # Delay between start of take-off roll and lift-off.
    _LIFTOFF_DELAY_S: Final[float] = 15.0

    def __init__(self, config: FlightProfileConfig | None = None) -> None:
        """Create a generator.

        Args:
            config: Flight profile; the default profile is used when None.
        """
        self.config = config or FlightProfileConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def __repr__(self) -> str:
        """Return a short description of the generator."""
        return f"{type(self).__name__}(seed={self.config.seed})"

    def generate(self) -> pd.DataFrame:
        """Build the flight log.

        Returns:
            DataFrame with columns :data:`LOG_COLUMNS`, one row per sample.
        """
        cfg = self.config
        t = np.arange(0.0, cfg.duration_s, 1.0 / cfg.sample_rate_hz)
        clean = self._clean_signals(t)
        noisy = self._add_noise(clean)
        frame = pd.DataFrame(noisy, columns=list(LOG_COLUMNS))
        frame = self._inject_defects(frame)
        return self._round(frame)

    def to_csv(self, path: str | Path) -> Path:
        """Generate the log and write it to ``path`` as CSV.

        Args:
            path: Destination file; parent folders are created if needed.

        Returns:
            The resolved output path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.generate().to_csv(out, index=False)
        return out.resolve()

    def _clean_signals(self, t: np.ndarray) -> np.ndarray:
        """Return noise-free signals stacked as columns in LOG_COLUMNS order."""
        cfg = self.config
        g0 = STANDARD_GRAVITY_MPS2
        airborne_start = cfg.takeoff_s + self._LIFTOFF_DELAY_S

        airspeed = cfg.cruise_airspeed_mps * (
            smoothstep(t, cfg.takeoff_s, cfg.takeoff_s + self._SPEED_RAMP_S)
            - smoothstep(t, cfg.landing_s, cfg.landing_s + self._SPEED_RAMP_S)
        )
        altitude = cfg.cruise_altitude_m * (
            smoothstep(t, airborne_start, cfg.climb_end_s)
            - smoothstep(t, cfg.descent_start_s, cfg.landing_s)
        )
        climb_rate = np.gradient(altitude, t)
        vertical_accel = np.gradient(climb_rate, t)
        longitudinal_accel = np.gradient(airspeed, t)

        load_factor = np.ones_like(t)
        yaw_rate = np.zeros_like(t)
        pitch_rate = np.zeros_like(t)
        for m in cfg.maneuvers:
            pulse = sine_pulse(t, m.start_s, m.duration_s)
            load_factor += (m.peak_load_factor - 1.0) * pulse
            yaw_rate += m.yaw_rate_dps * pulse
            pitch_rate += m.pitch_rate_dps * pulse

        # Flight-path angle from climb rate; its derivative is the pitch rate
        # of a stable climb (no angle-of-attack change assumed).
        safe_speed = np.maximum(airspeed, 1.0)
        path_angle_deg = np.degrees(np.arcsin(np.clip(climb_rate / safe_speed, -1, 1)))
        pitch_rate += np.gradient(path_angle_deg, t)
        # Bank angle of a coordinated turn: n = 1 / cos(phi). Only maneuvers
        # with a yaw rate are turns; pull-ups and touchdowns do not bank.
        bank_deg = np.zeros_like(t)
        for m in cfg.maneuvers:
            if m.yaw_rate_dps != 0.0:
                turn_n = 1.0 + (m.peak_load_factor - 1.0) * sine_pulse(
                    t, m.start_s, m.duration_s
                )
                bank_deg += np.degrees(np.arccos(1.0 / turn_n))
        roll_rate = np.gradient(bank_deg, t)

        temperature = SEA_LEVEL_TEMPERATURE_C - LAPSE_RATE_C_PER_M * altitude
        return np.column_stack(
            [
                t,
                longitudinal_accel,
                np.zeros_like(t),
                g0 * load_factor + vertical_accel,
                roll_rate,
                pitch_rate,
                yaw_rate,
                altitude,
                airspeed,
                temperature,
            ]
        )

    def _add_noise(self, clean: np.ndarray) -> np.ndarray:
        """Add white Gaussian noise to every sensor column (not to time)."""
        cfg = self.config
        stds = np.array(
            [
                0.0,
                *(cfg.accel_noise_std_mps2,) * len(ACCEL_COLUMNS),
                *(cfg.gyro_noise_std_dps,) * len(GYRO_COLUMNS),
                cfg.baro_noise_std_m,
                cfg.airspeed_noise_std_mps,
                cfg.temperature_noise_std_c,
            ]
        )
        return clean + self._rng.normal(0.0, 1.0, clean.shape) * stds

    def _inject_defects(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Inject dropouts (NaN), single-sample spikes and a logger gap."""
        cfg = self.config
        if cfg.gap_start_s is not None:
            t = frame[TIME_COLUMN]
            in_gap = (t >= cfg.gap_start_s) & (t < cfg.gap_start_s + cfg.gap_duration_s)
            frame = frame.loc[~in_gap].reset_index(drop=True)

        n_rows = len(frame)
        for column in (ALTITUDE_COLUMN, AIRSPEED_COLUMN):
            n_drop = int(round(cfg.dropout_fraction * n_rows))
            rows = self._rng.choice(n_rows, size=n_drop, replace=False)
            frame.loc[rows, column] = np.nan

        spike_rows = self._rng.choice(n_rows, size=cfg.spike_count, replace=False)
        frame.loc[spike_rows, "accel_z_mps2"] += cfg.spike_magnitude_mps2
        return frame

    @staticmethod
    def _round(frame: pd.DataFrame) -> pd.DataFrame:
        """Round columns to realistic logger resolution to keep the CSV small."""
        decimals = {
            TIME_COLUMN: 2,
            ALTITUDE_COLUMN: 2,
            AIRSPEED_COLUMN: 2,
            TEMPERATURE_COLUMN: 2,
        }
        decimals |= {c: 4 for c in (*ACCEL_COLUMNS, *GYRO_COLUMNS)}
        return frame.round(decimals)


class FlightLog:
    """A loaded flight log with the Week 1 engineering queries as methods.

    The wrapped DataFrame is sorted by time and never modified in place;
    methods return new Series/DataFrames.

    Example:
        >>> log = FlightLog(SyntheticFlightLogGenerator().generate())
        >>> round(log.sample_rate_hz)
        50
    """

    __slots__ = ("_data", "source")

    def __init__(self, data: pd.DataFrame, source: str = "<memory>") -> None:
        """Wrap a DataFrame after checking that all expected columns exist.

        Args:
            data: Flight log with at least the columns in :data:`LOG_COLUMNS`.
            source: Where the data came from (file path or description).

        Raises:
            ValueError: If required columns are missing or the log is empty.
        """
        missing = set(LOG_COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"Flight log is missing columns: {sorted(missing)}")
        if data.empty:
            raise ValueError("Flight log is empty")
        self._data = data.sort_values(TIME_COLUMN).reset_index(drop=True)
        self.source = source

    @classmethod
    def from_csv(cls, path: str | Path) -> FlightLog:
        """Load a flight log from a CSV file.

        Args:
            path: CSV file with a header row containing :data:`LOG_COLUMNS`.

        Returns:
            A new :class:`FlightLog`.
        """
        return cls(pd.read_csv(path), source=str(path))

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return a one-line summary useful in notebooks."""
        return (
            f"FlightLog(source={Path(self.source).name!r}, samples={len(self)}, "
            f"duration={self.duration_s:.1f} s, rate={self.sample_rate_hz:.1f} Hz)"
        )

    @property
    def data(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame (safe to modify)."""
        return self._data.copy()

    @property
    def time(self) -> pd.Series:
        """Return the time column in seconds."""
        return self._data[TIME_COLUMN]

    @property
    def sample_rate_hz(self) -> float:
        """Estimate the sampling rate as 1 / median time step (robust to gaps)."""
        return float(1.0 / self.time.diff().median())

    @property
    def duration_s(self) -> float:
        """Return the time between the first and the last sample."""
        return float(self.time.iloc[-1] - self.time.iloc[0])

    def missing_values(self) -> pd.Series:
        """Count NaN values per column.

        Returns:
            Series indexed by column name, sorted descending.
        """
        return self._data.isna().sum().sort_values(ascending=False)

    def time_gaps(self, tolerance: float = DEFAULT_GAP_TOLERANCE) -> pd.DataFrame:
        """Find logger outages: time steps longer than ``tolerance`` nominal steps.

        Args:
            tolerance: A step counts as a gap if dt > tolerance / sample_rate.

        Returns:
            DataFrame with columns ``gap_start_s``, ``gap_end_s`` and
            ``missing_samples`` (one row per gap).
        """
        dt = self.time.diff()
        nominal = 1.0 / self.sample_rate_hz
        is_gap = dt > tolerance * nominal
        ends = self.time[is_gap]
        starts = self.time.shift(1)[is_gap]
        missing = (np.round(dt[is_gap] / nominal) - 1).astype(int)
        return pd.DataFrame(
            {
                "gap_start_s": starts.to_numpy(),
                "gap_end_s": ends.to_numpy(),
                "missing_samples": missing.to_numpy(),
            }
        )

    def load_factor(self) -> pd.Series:
        """Return the raw load factor n = |accel| / g0 for every sample."""
        values = load_factor_from_accel(self._data[list(ACCEL_COLUMNS)].to_numpy())
        return pd.Series(values, index=self._data.index, name="load_factor")

    def robust_load_factor(self, window: int = 5) -> pd.Series:
        """Return the load factor after a centred rolling median.

        A rolling median removes single-sample spikes (sensor glitches) while
        keeping real maneuvers, which last many samples.

        Args:
            window: Odd number of samples in the median window.

        Returns:
            Smoothed load factor series.
        """
        if window < 1 or window % 2 == 0:
            raise ValueError("window must be a positive odd integer")
        smoothed = (
            self.load_factor().rolling(window, center=True, min_periods=1).median()
        )
        return smoothed.rename("load_factor_median")

    def climb_rate(self, window_s: float = 1.0) -> pd.Series:
        """Estimate vertical speed from barometric altitude.

        Altitude is resampled onto a uniform time grid (filling NaN dropouts
        and logger gaps by linear interpolation), smoothed with a centred
        rolling mean, then differentiated with central differences.

        Args:
            window_s: Length of the smoothing window in seconds.

        Returns:
            Climb rate in m/s (positive up).
        """
        # Resample onto a uniform grid so that gaps and NaN dropouts are filled
        # by linear interpolation; a centred window is then symmetric everywhere.
        nominal_dt = 1.0 / self.sample_rate_hz
        t = self.time.to_numpy()
        n_grid = int(round((t[-1] - t[0]) / nominal_dt)) + 1
        grid = t[0] + nominal_dt * np.arange(n_grid)
        altitude = self._data[ALTITUDE_COLUMN].to_numpy()
        valid = ~np.isnan(altitude)
        altitude_grid = np.interp(grid, t[valid], altitude[valid])
        window = max(1, int(round(window_s / nominal_dt)))
        smooth = (
            pd.Series(altitude_grid)
            .rolling(window, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        rate = np.interp(t, grid, np.gradient(smooth, nominal_dt))
        return pd.Series(rate, index=self._data.index, name="climb_rate_mps")

    def segment(self, start_s: float, end_s: float) -> FlightLog:
        """Return the part of the log with start_s <= t < end_s.

        Args:
            start_s: Segment start time in seconds.
            end_s: Segment end time in seconds.

        Returns:
            A new :class:`FlightLog` holding only that time window.
        """
        mask = (self.time >= start_s) & (self.time < end_s)
        return FlightLog(
            self._data.loc[mask], source=f"{self.source}[{start_s}:{end_s}]"
        )

    def summary(self) -> pd.DataFrame:
        """Return ``describe()`` transposed: one row per channel."""
        return self._data.describe().T

    def plot_overview(self) -> Figure:
        """Plot altitude, airspeed and load factor on a shared time axis.

        Returns:
            The Matplotlib figure (call ``fig.savefig`` to export it).
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(self.time, self._data[ALTITUDE_COLUMN], lw=0.8)
        axes[0].set_ylabel("Altitude [m]")
        axes[1].plot(self.time, self._data[AIRSPEED_COLUMN], lw=0.8, color="tab:orange")
        axes[1].set_ylabel("Airspeed [m/s]")
        axes[2].plot(
            self.time, self.load_factor(), lw=0.6, color="tab:gray", label="raw"
        )
        axes[2].plot(
            self.time,
            self.robust_load_factor(),
            lw=1.0,
            color="tab:red",
            label="rolling median",
        )
        axes[2].set_ylabel("Load factor n [-]")
        axes[2].set_xlabel("Time [s]")
        axes[2].legend(loc="upper left")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.suptitle(f"Flight log overview: {Path(self.source).name}")
        fig.tight_layout()
        return fig
