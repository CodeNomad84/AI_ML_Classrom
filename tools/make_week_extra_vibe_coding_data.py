"""Regenerate the synthetic static-fire log of the bonus-week practice assignment.

Provenance: new content (bonus week "Vibe Coding in Engineering").

The thrust model is a manufactured solution: a half-sine pulse
F(t) = F_peak sin(pi (t - t_ign) / t_b) during the burn, zero outside, so the
exact total impulse is 2 F_peak t_b / pi. The logger adds what a real
load-cell channel adds: Gaussian noise (so some samples near zero are
negative), timestamp jitter around a nominal rate, and dropped packets that
leave gaps in the time axis. Nothing here is measured data.

Usage:
    python tools/make_week_extra_vibe_coding_data.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
OUTPUT: Final[Path] = REPO_ROOT / "data" / "sample" / "practice_vc_static_fire.csv"

SEED: Final[int] = 2026
PEAK_THRUST_N: Final[float] = 120.0
BURN_TIME_S: Final[float] = 1.6
IGNITION_S: Final[float] = 0.5
RECORD_LENGTH_S: Final[float] = 2.6
NOMINAL_RATE_HZ: Final[float] = 200.0
JITTER_FRACTION: Final[float] = 0.2  # each interval is (1 +- 0.2) / rate
NOISE_STD_N: Final[float] = 0.4
# Dropped packets: (start time s, length s); all three fall inside the burn.
GAPS_S: Final[tuple[tuple[float, float], ...]] = (
    (0.80, 0.04),
    (1.20, 0.04),
    (1.65, 0.04),
)
DECIMALS: Final[int] = 4


def exact_thrust(time_s: np.ndarray) -> np.ndarray:
    """Noise-free half-sine thrust (N) of the manufactured solution."""
    phase = (time_s - IGNITION_S) / BURN_TIME_S
    burning = (phase >= 0.0) & (phase <= 1.0)
    return np.where(
        burning, PEAK_THRUST_N * np.sin(np.pi * np.clip(phase, 0.0, 1.0)), 0.0
    )


def exact_total_impulse() -> float:
    """Exact impulse of the half-sine pulse: 2 F_peak t_b / pi (N s)."""
    return 2.0 * PEAK_THRUST_N * BURN_TIME_S / math.pi


def sample_times(rng: np.random.Generator) -> np.ndarray:
    """Jittered, strictly increasing timestamps with dropped-packet gaps."""
    nominal_dt = 1.0 / NOMINAL_RATE_HZ
    n_max = int(2 * RECORD_LENGTH_S * NOMINAL_RATE_HZ)
    steps = nominal_dt * (1.0 + rng.uniform(-JITTER_FRACTION, JITTER_FRACTION, n_max))
    times = np.concatenate([[0.0], np.cumsum(steps)])
    times = times[times <= RECORD_LENGTH_S]
    keep = np.ones(times.size, dtype=bool)
    for start, length in GAPS_S:
        keep &= ~((times > start) & (times < start + length))
    return times[keep]


def build_log(seed: int = SEED) -> pd.DataFrame:
    """Return the synthetic static-fire log (columns ``time_s``, ``thrust_n``)."""
    rng = np.random.default_rng(seed)
    times = sample_times(rng)
    thrust = exact_thrust(times) + rng.normal(0.0, NOISE_STD_N, times.size)
    return pd.DataFrame(
        {"time_s": times.round(DECIMALS), "thrust_n": thrust.round(DECIMALS)}
    )


def main() -> None:
    """Write the CSV and print the facts students can check."""
    log = build_log()
    log.to_csv(OUTPUT, index=False)
    dt = np.diff(log["time_s"].to_numpy())
    print("wrote", OUTPUT.relative_to(REPO_ROOT), "rows:", len(log))
    print(f"exact impulse of the noise-free pulse: {exact_total_impulse():.4f} N s")
    print(
        f"interval min/median/max: {dt.min():.4f} / {np.median(dt):.4f}"
        f" / {dt.max():.4f} s"
    )
    print("negative thrust samples:", int((log["thrust_n"] < 0).sum()))


if __name__ == "__main__":
    main()
