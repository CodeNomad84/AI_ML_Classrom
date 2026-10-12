"""Synthetic test drives of a small autonomous rover with realistic sensor defects.

Provenance: new content (Week 5). The vehicle is the final-project rover
(module M3 needs the obstacle range to decide when to brake). Kinematics are a
plain constant-acceleration model with a first-order actuator lag; the LiDAR
defects (no-return zeros, out-of-range returns, dust echoes) follow the error
taxonomy of Thrun, Burgard and Fox, *Probabilistic Robotics* (2005), Sec. 6.3
(beam model: max-range readings, random measurements, short readings). The
missing-data mechanisms (MCAR, MNAR) follow Little and Rubin, *Statistical
Analysis with Missing Data*, 3rd ed. (2019), Ch. 1.

Each *run* is one approach to a wall: the rover waits, accelerates to a cruise
speed, brakes so that it stops a little before the wall, and waits again. The
logger writes one row every 0.1 s with the columns in :data:`ROVER_COLUMNS`.

Defects injected by :meth:`RoverLogGenerator.generate` (all seeded):

* LiDAR beyond its 12 m maximum range returns ``0.0`` (missing *because* the
  value is large: MNAR).
* Random LiDAR dropouts also return ``0.0`` (MCAR).
* Dust echoes: short, plausible-looking ranges while the wall is far away.
* Wheel-speed NaN bursts of 1 to 4 samples (CAN-bus packet loss, MCAR).
* Single-sample accelerometer spikes of 15 to 30 m/s^2 (connector glitches).
* A frozen gyroscope: the yaw rate repeats one value for 3 s in some runs.
* Duplicated rows (the logger re-sends a packet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

RUN_COLUMN: Final[str] = "run_id"
TIME_COLUMN: Final[str] = "time_s"
RANGE_COLUMN: Final[str] = "lidar_range_m"
SPEED_COLUMN: Final[str] = "wheel_speed_mps"
ACCEL_COLUMN: Final[str] = "accel_x_mps2"
GYRO_COLUMN: Final[str] = "gyro_z_dps"
ROVER_COLUMNS: Final[tuple[str, ...]] = (
    RUN_COLUMN,
    TIME_COLUMN,
    RANGE_COLUMN,
    SPEED_COLUMN,
    ACCEL_COLUMN,
    GYRO_COLUMN,
)
SENSOR_COLUMNS: Final[tuple[str, ...]] = ROVER_COLUMNS[2:]

LIDAR_MIN_RANGE_M: Final[float] = 0.3
LIDAR_MAX_RANGE_M: Final[float] = 12.0
LIDAR_NO_RETURN: Final[float] = 0.0
SAMPLE_PERIOD_S: Final[float] = 0.1
DECIMALS: Final[int] = 3


@dataclass(frozen=True, slots=True)
class RoverLogConfig:
    """Everything that defines one synthetic set of test drives.

    Attributes:
        n_runs: Number of approaches to a wall.
        seed: Random seed (same seed, same log).
        start_range_m: Range of initial distances to the wall.
        cruise_speed_mps: Range of cruise speeds.
        brake_decel_mps2: Range of braking decelerations.
        stop_distance_m: Range of distances at which the rover comes to rest.
        accel_mps2: Acceleration while speeding up.
        actuator_lag_s: Time constant of the drive train (first-order lag).
        idle_before_s: Standing time before the rover starts.
        idle_after_s: Standing time after it stops.
        lidar_noise_m: Standard deviation of the LiDAR noise.
        speed_noise_mps: Standard deviation of the wheel-speed noise.
        accel_noise_mps2: Accelerometer noise at standstill.
        accel_vibration_per_mps: Extra accelerometer noise per m/s of speed.
        gyro_noise_dps: Gyroscope noise.
        gyro_bias_dps: Constant gyroscope bias.
        steering_amplitude_dps: Amplitude of the small steering corrections.
        dropout_fraction: Fraction of LiDAR samples lost at random.
        dust_fraction: Fraction of far LiDAR samples replaced by dust echoes.
        dust_range_m: Range of dust-echo distances.
        dust_min_true_range_m: Dust echoes only appear when the wall is farther.
        speed_bursts: Number of wheel-speed NaN bursts in the whole log.
        speed_burst_max: Longest burst in samples.
        spike_fraction: Fraction of accelerometer samples that spike.
        spike_size_mps2: Range of spike magnitudes (sign is random).
        stuck_runs: Number of runs with a frozen gyroscope segment.
        stuck_samples: Length of each frozen segment.
        duplicate_fraction: Fraction of rows sent twice.
    """

    n_runs: int = 16
    seed: int = 5
    start_range_m: tuple[float, float] = (6.0, 18.0)
    cruise_speed_mps: tuple[float, float] = (0.8, 2.0)
    brake_decel_mps2: tuple[float, float] = (0.6, 1.2)
    stop_distance_m: tuple[float, float] = (0.8, 1.5)
    accel_mps2: float = 0.5
    actuator_lag_s: float = 0.3
    idle_before_s: float = 2.0
    idle_after_s: float = 3.0
    lidar_noise_m: float = 0.02
    speed_noise_mps: float = 0.02
    accel_noise_mps2: float = 0.1
    accel_vibration_per_mps: float = 0.1
    gyro_noise_dps: float = 0.3
    gyro_bias_dps: float = 0.5
    steering_amplitude_dps: float = 3.0
    dropout_fraction: float = 0.02
    dust_fraction: float = 0.01
    dust_range_m: tuple[float, float] = (0.5, 3.0)
    dust_min_true_range_m: float = 4.0
    speed_bursts: int = 10
    speed_burst_max: int = 4
    spike_fraction: float = 0.005
    spike_size_mps2: tuple[float, float] = (15.0, 30.0)
    stuck_runs: int = 2
    stuck_samples: int = 30
    duplicate_fraction: float = 0.01

    def __post_init__(self) -> None:
        """Check the settings that would make the generator fail.

        Raises:
            ValueError: If a count or fraction is out of range.
        """
        if self.n_runs < 1:
            raise ValueError("n_runs must be at least 1")
        if self.stuck_runs > self.n_runs:
            raise ValueError("stuck_runs cannot exceed n_runs")
        fractions = (
            self.dropout_fraction,
            self.dust_fraction,
            self.spike_fraction,
            self.duplicate_fraction,
        )
        if any(not 0.0 <= f < 1.0 for f in fractions):
            raise ValueError("defect fractions must be in [0, 1)")


@dataclass
class DefectLog:
    """Counts of every defect the generator injected (the answer key).

    Attributes:
        no_return_out_of_range: LiDAR zeros because the wall was beyond 12 m.
        no_return_dropout: LiDAR zeros lost at random (wall within range).
        dust_echoes: LiDAR samples replaced by dust echoes.
        speed_nan: Wheel-speed samples set to NaN.
        accel_spikes: Accelerometer spikes.
        gyro_stuck: Gyroscope samples that repeat a frozen value.
        duplicate_rows: Rows written twice.
        stuck_segments: ``(run_id, first_time_s)`` of each frozen segment.
    """

    no_return_out_of_range: int = 0
    no_return_dropout: int = 0
    dust_echoes: int = 0
    speed_nan: int = 0
    accel_spikes: int = 0
    gyro_stuck: int = 0
    duplicate_rows: int = 0
    stuck_segments: list[tuple[int, float]] = field(default_factory=list)

    def as_series(self) -> pd.Series:
        """Return the counts (not the segment list) as a Series."""
        counts = {k: v for k, v in vars(self).items() if isinstance(v, int)}
        return pd.Series(counts, name="injected")


class RoverLogGenerator:
    """Generate reproducible rover test drives, clean and with defects.

    Example:
        >>> gen = RoverLogGenerator(RoverLogConfig(n_runs=2, seed=1))
        >>> list(gen.generate().columns) == list(ROVER_COLUMNS)
        True
    """

    def __init__(self, config: RoverLogConfig | None = None) -> None:
        """Store the configuration.

        Args:
            config: Generator settings; the course default if None.
        """
        self.config = config or RoverLogConfig()
        self.defects = DefectLog()

    def __repr__(self) -> str:
        """Return the class name with the run count and seed."""
        return (
            f"RoverLogGenerator(n_runs={self.config.n_runs}, seed={self.config.seed})"
        )

    # ------------------------------------------------------------------ truth
    def _simulate_run(self, run_id: int, rng: np.random.Generator) -> pd.DataFrame:
        """Integrate one approach to the wall and return the noise-free signals."""
        cfg = self.config
        start = rng.uniform(*cfg.start_range_m)
        cruise = rng.uniform(*cfg.cruise_speed_mps)
        decel = rng.uniform(*cfg.brake_decel_mps2)
        stop_at = rng.uniform(*cfg.stop_distance_m)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        steer_period_s = rng.uniform(5.0, 9.0)
        dt = SAMPLE_PERIOD_S
        rows: list[tuple[float, float, float, float, float]] = []
        t, r, v, a = 0.0, start, 0.0, 0.0
        stopped_at: float | None = None
        while True:
            if t < cfg.idle_before_s or stopped_at is not None:
                command = 0.0
            elif r - stop_at <= v**2 / (2.0 * decel) + v * dt:
                command = -decel
            elif v < cruise:
                command = cfg.accel_mps2
            else:
                command = 0.0
            a += (command - a) * dt / cfg.actuator_lag_s
            v_new = v + a * dt
            if v_new <= 0.0 and t > cfg.idle_before_s and command < 0.0:
                v_new, a, stopped_at = 0.0, 0.0, t
            v = max(v_new, 0.0)
            r -= v * dt
            moving = v > 0.0
            yaw = (
                cfg.steering_amplitude_dps
                * np.sin(2.0 * np.pi * t / steer_period_s + phase)
                * moving
            )
            rows.append((t, r, v, a if moving else 0.0, yaw))
            t = round(t + dt, 10)
            if stopped_at is not None and t >= stopped_at + cfg.idle_after_s:
                break
        frame = pd.DataFrame(rows, columns=list(ROVER_COLUMNS[1:]))
        frame.insert(0, RUN_COLUMN, run_id)
        return frame

    def generate_truth(self) -> pd.DataFrame:
        """Return the noise-free signals of all runs (rows ordered by run, time)."""
        rng = np.random.default_rng(self.config.seed)
        runs = [self._simulate_run(i + 1, rng) for i in range(self.config.n_runs)]
        return pd.concat(runs, ignore_index=True)

    # ----------------------------------------------------------- measurement
    def generate(self) -> pd.DataFrame:
        """Return the raw log: truth + noise + defects, rounded like the logger.

        :attr:`defects` is reset and filled with the injected counts.
        """
        truth = self.generate_truth()
        rng = np.random.default_rng(self.config.seed + 1)
        noisy = self._add_noise(truth, rng)
        frame = self._inject_defects(noisy, truth, rng)
        return frame.round(DECIMALS)

    def _add_noise(self, truth: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Add white noise, vibration and the gyro bias."""
        cfg = self.config
        n = len(truth)
        out = truth.copy()
        out[RANGE_COLUMN] += cfg.lidar_noise_m * rng.standard_normal(n)
        moving = truth[SPEED_COLUMN].to_numpy() > 0.0
        out[SPEED_COLUMN] += cfg.speed_noise_mps * rng.standard_normal(n) * moving
        vibration = (
            cfg.accel_noise_mps2
            + cfg.accel_vibration_per_mps * truth[SPEED_COLUMN].to_numpy()
        )
        out[ACCEL_COLUMN] += vibration * rng.standard_normal(n)
        out[
            GYRO_COLUMN
        ] += cfg.gyro_bias_dps + cfg.gyro_noise_dps * rng.standard_normal(n)
        return out

    def _inject_defects(
        self, frame: pd.DataFrame, truth: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Apply the defects listed in the module docstring and log their counts."""
        cfg = self.config
        log = DefectLog()
        out = frame.copy()
        n = len(out)
        true_range = truth[RANGE_COLUMN].to_numpy()

        far = true_range > LIDAR_MAX_RANGE_M
        out.loc[far, RANGE_COLUMN] = LIDAR_NO_RETURN
        log.no_return_out_of_range = int(far.sum())

        dropout = ~far & (rng.random(n) < cfg.dropout_fraction)
        out.loc[dropout, RANGE_COLUMN] = LIDAR_NO_RETURN
        log.no_return_dropout = int(dropout.sum())

        dust = (
            ~far
            & ~dropout
            & (true_range > cfg.dust_min_true_range_m)
            & (rng.random(n) < cfg.dust_fraction)
        )
        out.loc[dust, RANGE_COLUMN] = rng.uniform(*cfg.dust_range_m, int(dust.sum()))
        log.dust_echoes = int(dust.sum())

        speed = out[SPEED_COLUMN].to_numpy(copy=True)
        for start in rng.choice(
            n - cfg.speed_burst_max, cfg.speed_bursts, replace=False
        ):
            length = int(rng.integers(1, cfg.speed_burst_max + 1))
            speed[start : start + length] = np.nan
        out[SPEED_COLUMN] = speed
        log.speed_nan = int(np.isnan(speed).sum())

        spikes = rng.random(n) < cfg.spike_fraction
        signs = rng.choice([-1.0, 1.0], int(spikes.sum()))
        out.loc[spikes, ACCEL_COLUMN] += signs * rng.uniform(
            *cfg.spike_size_mps2, int(spikes.sum())
        )
        log.accel_spikes = int(spikes.sum())

        gyro = out[GYRO_COLUMN].to_numpy(copy=True)
        stuck_ids = rng.choice(
            np.arange(1, cfg.n_runs + 1), cfg.stuck_runs, replace=False
        )
        for run_id in sorted(int(i) for i in stuck_ids):
            rows = np.flatnonzero(out[RUN_COLUMN].to_numpy() == run_id)
            first = int(rng.integers(rows[0] + 1, rows[-1] - cfg.stuck_samples))
            gyro[first : first + cfg.stuck_samples] = gyro[first - 1]
            log.stuck_segments.append((run_id, float(out[TIME_COLUMN].iloc[first])))
        out[GYRO_COLUMN] = gyro
        log.gyro_stuck = cfg.stuck_runs * cfg.stuck_samples

        duplicated = np.flatnonzero(rng.random(n) < cfg.duplicate_fraction)
        order = np.sort(np.concatenate([np.arange(n), duplicated]))
        out = out.iloc[order].reset_index(drop=True)
        log.duplicate_rows = len(duplicated)

        self.defects = log
        return out

    def to_csv(self, path: str | Path) -> Path:
        """Write the raw log to ``path`` and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.generate().to_csv(target, index=False)
        return target
