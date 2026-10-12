"""Practice Assignment 3: stages for a reproducible air-data pipeline.

Provenance: new content (Week 3). Dataclass and ABC usage follows (rewritten,
not copied) W3Schools "Python dataclasses Module" and GeeksforGeeks
"Understanding Python Dataclasses" and "Abstract Classes in Python"; min-max
scaling adapted from Géron, *Hands-On Machine Learning*, 3rd ed., Ch. 2.

Fill in every ``# TODO`` block. Do not rename classes, methods, fields or
constants: the notebook and the instructor's tests import them by name.
Write all comments in English.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Final

import numpy as np
import pandas as pd

from aerospace_ai.pipeline import (
    DataPipeline,
    HampelFilter,
    NotFittedError,
    PipelineStage,
    RollingMean,
    SelectColumns,
    TimeDerivative,
)
from aerospace_ai.utils import STANDARD_GRAVITY_MPS2

# ---------------------------------------------------------------------------
# Constants (use these; do not type the numbers again in the code below).
# ---------------------------------------------------------------------------
TIME_COLUMN: Final[str] = "time_s"
ALTITUDE_COLUMN: Final[str] = "baro_alt_m"
AIRSPEED_COLUMN: Final[str] = "airspeed_mps"
ACCEL_Z_COLUMN: Final[str] = "accel_z_mps2"
CLIMB_COLUMN: Final[str] = "climb_mps"
IMU_RANGE_G: Final[float] = 4.0
RESAMPLE_RATE_HZ: Final[float] = 10.0
HAMPEL_WINDOW: Final[int] = 7
SMOOTH_WINDOW: Final[int] = 5
TRAIN_FRACTION: Final[float] = 0.7
DEFAULT_FEATURE_RANGE: Final[tuple[float, float]] = (0.0, 1.0)
# Grid points closer than this fraction of a period to the last sample count.
GRID_TOLERANCE: Final[float] = 1e-9


def _column_tuple(columns: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return ``columns`` as a non-empty tuple (one string = one column)."""
    result = (columns,) if isinstance(columns, str) else tuple(columns)
    if not result:
        raise ValueError("at least one column is required")
    return result


# ---------------------------------------------------------------------------
# Task 1
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlausibleRange(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Replace physically impossible values by NaN (never clip them).

    Attributes:
        columns: Columns to check.
        lower: Smallest plausible value (inclusive).
        upper: Largest plausible value (inclusive).
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    columns: tuple[str, ...]
    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Normalise ``columns`` and check the bounds.

        Raises:
            ValueError: If a bound is not finite or ``lower >= upper``.
        """
        # TODO: store columns as a tuple with object.__setattr__ and
        #       _column_tuple (the dataclass is frozen), then validate the bounds.
        raise NotImplementedError

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with out-of-range values set to NaN."""
        # TODO: copy the frame; in each column keep values in [lower, upper]
        #       (inclusive) and replace the others by NaN. Do not modify ``data``.
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 2
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Resample(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Interpolate every numeric column onto a uniform time grid.

    The grid starts at the first time stamp and steps by ``1 / rate_hz`` up to
    the last time stamp. Each column is interpolated linearly through its own
    non-NaN samples, so dropouts, NaN from earlier stages and logger gaps are
    all bridged.

    Attributes:
        rate_hz: Output sampling rate in Hz.
        time_column: Column holding time in seconds.
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    rate_hz: float
    time_column: str = TIME_COLUMN

    def __post_init__(self) -> None:
        """Reject a rate that is not positive and finite."""
        # TODO: raise ValueError unless rate_hz is finite and > 0
        raise NotImplementedError

    @property
    def period_s(self) -> float:
        """Return the grid spacing in seconds."""
        # TODO: one line
        raise NotImplementedError

    def grid(self, t_start: float, t_end: float) -> np.ndarray:
        """Return the uniform time grid from ``t_start`` up to ``t_end``."""
        # TODO: t_start, t_start + period, ... as long as the point is <= t_end
        #       (use GRID_TOLERANCE against round-off; avoid np.arange with a
        #       float step, whose length is unreliable)
        raise NotImplementedError

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a new frame on the uniform grid (numeric columns only)."""
        # TODO: raise ValueError for fewer than two rows; build the grid from the
        #       first and last time stamp; interpolate every numeric column
        #       (except the time column) through its non-NaN samples with
        #       np.interp; an all-NaN column stays all-NaN.
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 3
# ---------------------------------------------------------------------------
@dataclass
class MinMaxScale(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Map each column linearly so that its training range becomes ``feature_range``.

    Attributes:
        columns: Columns to scale.
        feature_range: (low, high) of the scaled training data.
        data_min_: Learned minima (None before fitting).
        data_max_: Learned maxima (None before fitting).
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame
    REQUIRES_FIT: ClassVar[bool] = True

    columns: tuple[str, ...]
    feature_range: tuple[float, float] = DEFAULT_FEATURE_RANGE
    # TODO: two learned fields, data_min_ and data_max_, of type
    #       dict[str, float] | None, default None, that are NOT constructor
    #       arguments, NOT printed and NOT compared (use dataclasses.field).

    def __post_init__(self) -> None:
        """Normalise ``columns`` and ``feature_range``; check the range."""
        # TODO: columns -> tuple; feature_range -> tuple of two floats;
        #       ValueError unless low < high
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        """Return True once ``fit`` has stored the minima and maxima."""
        # TODO
        raise NotImplementedError

    def fit(self, data: pd.DataFrame) -> MinMaxScale:
        """Learn the minimum and maximum of every column (NaN ignored).

        Raises:
            ValueError: If a column is constant.
        """
        # TODO: learn min and max per column (NaN ignored), reject constant
        #       columns, store dicts in data_min_ / data_max_, return self
        raise NotImplementedError

    def _check_fitted(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return the learned minima and maxima or raise NotFittedError."""
        # TODO
        raise NotImplementedError

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with x' = low + (x - min) (high - low) / (max - min)."""
        # TODO: use _check_fitted(); do not modify ``data``
        raise NotImplementedError

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Undo :meth:`transform` (back to engineering units)."""
        # TODO: the algebraic inverse of transform
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 4
# ---------------------------------------------------------------------------
def build_airdata_pipeline() -> DataPipeline[pd.DataFrame, pd.DataFrame]:
    """Return the (unfitted) air-data pipeline specified in the README."""
    # TODO: seven stages, in the order given in the README, built with >>
    raise NotImplementedError


def split_fit_transform(
    pipeline: DataPipeline[pd.DataFrame, pd.DataFrame],
    log: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time, fit on the early part only, transform both parts.

    Args:
        pipeline: An unfitted pipeline (it is fitted in place).
        log: The raw flight log, sorted by time.
        train_fraction: Fraction of rows (the earliest ones) used for fitting.

    Returns:
        (train_output, test_output).

    Raises:
        ValueError: If ``train_fraction`` is not in (0, 1).
    """
    # TODO: validate train_fraction; the first int(len(log) * train_fraction)
    #       rows are the training part; fit on it only; transform both parts
    raise NotImplementedError


def save_pipeline(pipeline: DataPipeline[object, object], path: str | Path) -> Path:
    """Write ``pipeline.to_config()`` as indented JSON and return the path."""
    # TODO
    raise NotImplementedError


def load_pipeline(path: str | Path) -> DataPipeline[object, object]:
    """Rebuild an unfitted pipeline from a JSON file written by save_pipeline."""
    # TODO
    raise NotImplementedError
