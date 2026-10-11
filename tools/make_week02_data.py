"""Regenerate the synthetic GPS and LiDAR logs used in Week 2 / Assignment 1.

Provenance: new content (Week 2).

* ``assignment01_gps_track.csv``: 60 s of 5 Hz fixes from a ground robot driving
  one 40 m-radius circle at Tehran Mehrabad airport, with bad fixes (high HDOP,
  NaN coordinates, one corrupted latitude).
* ``assignment01_lidar_scans.csv``: 5 s of 10 Hz, 16-beam scans from a robot
  driving along a 10 m x 6 m hall with one pillar, with beam dropouts (0.0 or
  inf), dust echoes below the minimum range and fog-blinded scans.

Usage:
    python tools/make_week02_data.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SAMPLE_DIR: Final[Path] = REPO_ROOT / "data" / "sample"
EARTH_MEAN_RADIUS_M: Final[float] = 6_371_000.0


@dataclass(frozen=True, slots=True)
class GpsTrackConfig:
    """Parameters of the synthetic GPS track."""

    origin_lat_deg: float = 35.6892
    origin_lon_deg: float = 51.3134
    origin_alt_m: float = 1190.0
    rate_hz: float = 5.0
    duration_s: float = 60.0
    radius_m: float = 40.0
    position_noise_m: float = 0.3
    altitude_noise_m: float = 0.8
    hdop_nominal: float = 0.9
    hdop_spread: float = 0.15
    n_high_hdop: int = 8
    high_hdop_range: tuple[float, float] = (5.5, 9.0)
    n_nan_fixes: int = 3
    corrupt_latitude_offset_deg: float = 100.0
    seed: int = 202602


@dataclass(frozen=True, slots=True)
class LidarLogConfig:
    """Parameters of the synthetic LiDAR log."""

    rate_hz: float = 10.0
    duration_s: float = 5.0
    n_beams: int = 16
    hall_length_m: float = 10.0
    hall_width_m: float = 6.0
    start_x_m: float = 2.0
    y_m: float = 3.0
    speed_mps: float = 0.4
    pillar_center: tuple[float, float] = (6.0, 2.2)
    pillar_radius_m: float = 0.3
    range_noise_m: float = 0.02
    dropout_probability: float = 0.06
    n_dust_echoes: int = 5
    dust_range_m: float = 0.05
    fog_scans: tuple[int, ...] = (12, 13, 31, 44)
    fog_blind_fraction: float = 0.7
    seed: int = 202603


def make_gps_track(cfg: GpsTrackConfig) -> pd.DataFrame:
    """Return the GPS log as a DataFrame (timestamp_s, lat_deg, lon_deg, alt_m, hdop)."""
    rng = np.random.default_rng(cfg.seed)
    t = np.arange(0.0, cfg.duration_s, 1.0 / cfg.rate_hz)
    # One full circle that starts at the origin, centre 40 m east.
    phase = 2.0 * np.pi * t / cfg.duration_s
    east = cfg.radius_m * (1.0 - np.cos(phase)) + rng.normal(
        0, cfg.position_noise_m, t.size
    )
    north = cfg.radius_m * np.sin(phase) + rng.normal(0, cfg.position_noise_m, t.size)
    lat = cfg.origin_lat_deg + np.degrees(north / EARTH_MEAN_RADIUS_M)
    lon = cfg.origin_lon_deg + np.degrees(
        east / (EARTH_MEAN_RADIUS_M * np.cos(np.radians(cfg.origin_lat_deg)))
    )
    alt = cfg.origin_alt_m + rng.normal(0, cfg.altitude_noise_m, t.size)
    hdop = np.abs(rng.normal(cfg.hdop_nominal, cfg.hdop_spread, t.size))

    # Defects on distinct rows (never the first fix).
    rows = rng.choice(
        np.arange(1, t.size), cfg.n_high_hdop + cfg.n_nan_fixes + 1, replace=False
    )
    high, nan_rows, corrupt = (
        rows[: cfg.n_high_hdop],
        rows[cfg.n_high_hdop : -1],
        rows[-1],
    )
    hdop[high] = rng.uniform(*cfg.high_hdop_range, high.size)
    lat[nan_rows] = np.nan
    lon[nan_rows] = np.nan
    lat[corrupt] += cfg.corrupt_latitude_offset_deg

    return pd.DataFrame(
        {
            "timestamp_s": np.round(t, 2),
            "lat_deg": np.round(lat, 8),
            "lon_deg": np.round(lon, 8),
            "alt_m": np.round(alt, 2),
            "hdop": np.round(hdop, 2),
        }
    )


def ray_range(origin: np.ndarray, angle_rad: float, cfg: LidarLogConfig) -> float:
    """Return the distance from ``origin`` along ``angle_rad`` to the first hit."""
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    hits: list[float] = []
    # Hall walls x = 0, x = L, y = 0, y = W.
    for axis, wall in (
        (0, 0.0),
        (0, cfg.hall_length_m),
        (1, 0.0),
        (1, cfg.hall_width_m),
    ):
        if abs(direction[axis]) > 1e-12:
            s = (wall - origin[axis]) / direction[axis]
            if s > 0:
                hits.append(s)
    # Pillar: solve |o + s d - c|^2 = r^2 for the smallest positive s.
    offset = origin - np.asarray(cfg.pillar_center)
    b = float(offset @ direction)
    c = float(offset @ offset) - cfg.pillar_radius_m**2
    disc = b * b - c
    if disc >= 0:
        s = -b - np.sqrt(disc)
        if s > 0:
            hits.append(s)
    return min(hits)


def make_lidar_scans(cfg: LidarLogConfig) -> pd.DataFrame:
    """Return the LiDAR log (timestamp_s, range_00_m ... range_15_m)."""
    rng = np.random.default_rng(cfg.seed)
    t = np.arange(0.0, cfg.duration_s, 1.0 / cfg.rate_hz)
    angles = np.radians(-180.0 + (360.0 / cfg.n_beams) * np.arange(cfg.n_beams))
    ranges = np.empty((t.size, cfg.n_beams))
    for k, tk in enumerate(t):
        position = np.array([cfg.start_x_m + cfg.speed_mps * tk, cfg.y_m])
        ranges[k] = [ray_range(position, a, cfg) for a in angles]
    ranges += rng.normal(0.0, cfg.range_noise_m, ranges.shape)

    dropouts = rng.random(ranges.shape) < cfg.dropout_probability
    no_echo = np.where(rng.random(ranges.shape) < 0.5, 0.0, np.inf)
    ranges = np.where(dropouts, no_echo, ranges)
    for k in cfg.fog_scans:
        blind = rng.random(cfg.n_beams) < cfg.fog_blind_fraction
        ranges[k, blind] = 0.0
    dust_rows = rng.integers(0, t.size, cfg.n_dust_echoes)
    dust_cols = rng.integers(0, cfg.n_beams, cfg.n_dust_echoes)
    ranges[dust_rows, dust_cols] = cfg.dust_range_m

    frame = pd.DataFrame(
        np.round(ranges, 3), columns=[f"range_{i:02d}_m" for i in range(cfg.n_beams)]
    )
    frame.insert(0, "timestamp_s", np.round(t, 2))
    return frame


def main() -> None:
    """Write both Week 2 sample files."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "assignment01_gps_track.csv": make_gps_track(GpsTrackConfig()),
        "assignment01_lidar_scans.csv": make_lidar_scans(LidarLogConfig()),
    }
    for name, frame in outputs.items():
        path = SAMPLE_DIR / name
        frame.to_csv(path, index=False)
        print(f"wrote {path.relative_to(REPO_ROOT)} {frame.shape}")


if __name__ == "__main__":
    main()
