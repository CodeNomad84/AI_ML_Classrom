"""Assignment 2 (Week 5): leak-free cleaning of a noisy rover log.

Provenance: new content (Week 5). Missing-data mechanisms after Little and
Rubin, *Statistical Analysis with Missing Data* (2019), Ch. 1; Tukey's fences
from Tukey, *Exploratory Data Analysis* (1977), Ch. 2, as rewritten by
GeeksforGeeks ("Detect and Remove the Outliers using Python"); leakage rules
rewritten from GeeksforGeeks ("Data Leakage").

Fill in every ``# TODO``. Do not change class names, signatures or constants:
the hidden tests import them. Use the Week 3 and Week 5 library
(``aerospace_ai.pipeline``, ``aerospace_ai.preprocessing``,
``aerospace_ai.splits``); do not edit anything in ``src/``. Run your tests with
``python -m pytest assignments/assignment02 -q``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final

import numpy as np
import pandas as pd

from aerospace_ai.datasets import (  # noqa: F401 (you will need these)
    ROVER_FEATURES,
    ROVER_TARGET,
    rover_cleaning_pipeline,
)
from aerospace_ai.pipeline import DataPipeline, PipelineStage, _as_column_tuple
from aerospace_ai.preprocessing import (  # noqa: F401
    LOWER_QUARTILE,
    UPPER_QUARTILE,
    ColumnStatStage,
    ImputeMedian,
    PerGroup,
    RobustScale,
    count_duplicate_keys,
)
from aerospace_ai.rover import (  # noqa: F401
    ACCEL_COLUMN,
    GYRO_COLUMN,
    LIDAR_NO_RETURN,
    RANGE_COLUMN,
    RUN_COLUMN,
    SAMPLE_PERIOD_S,
    SPEED_COLUMN,
    TIME_COLUMN,
)
from aerospace_ai.splits import (  # noqa: F401
    GroupHoldoutSplit,
    LeakageReport,
    check_split,
)

KEYS: Final[tuple[str, str]] = (RUN_COLUMN, TIME_COLUMN)
DEFAULT_MIN_RUN: Final[int] = 10
TUKEY_K: Final[float] = 1.5
# "Far out" fences: with k = 1.5 the fences would clip genuine braking.
FAR_OUT_K: Final[float] = 3.0
TEST_FRACTION: Final[float] = 0.25


# ---------------------------------------------------------------- Task 1
@dataclass(frozen=True, slots=True)
class DefectCounts:
    """Defects found in the raw log (counted after removing duplicate rows).

    Attributes:
        duplicate_rows: Rows whose (run_id, time_s) repeats an earlier row.
        lidar_leading_zeros: LiDAR 0.0 readings *before the first valid
            (non-zero) reading of their run*: the wall is beyond 12 m (MNAR).
        lidar_isolated_zeros: All other LiDAR 0.0 readings (random dropouts).
        speed_nan: NaN cells in the wheel-speed column.
    """

    duplicate_rows: int
    lidar_leading_zeros: int
    lidar_isolated_zeros: int
    speed_nan: int

    @property
    def lidar_zeros(self) -> int:
        """All LiDAR no-return samples."""
        return self.lidar_leading_zeros + self.lidar_isolated_zeros


def audit_report(raw: pd.DataFrame) -> DefectCounts:
    """Count the defects of a raw rover log.

    Count duplicates on the raw table; count everything else after sorting by
    (run_id, time_s) and dropping the duplicates, so a re-sent row is not
    counted twice.

    Args:
        raw: The raw log as read from the CSV file.

    Returns:
        The counts.
    """
    # TODO: count duplicates with count_duplicate_keys(raw, KEYS)
    # TODO: sort, drop duplicates, then split the LiDAR zeros of each run into
    #       "leading" (before the run's first non-zero reading) and "isolated".
    #       Hint: groupby(RUN_COLUMN) and cummax() on a "valid so far" flag.
    raise NotImplementedError


# ---------------------------------------------------------------- Task 2
@dataclass(frozen=True)
class FlagStuck(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Set runs of at least ``min_run`` identical consecutive values to NaN.

    A frozen sensor repeats its last value exactly; real signals with noise
    never do for long. Every sample of such a run becomes NaN, including the
    first one. NaN values never form a stuck run. Stateless: wrap it in
    :class:`~aerospace_ai.preprocessing.PerGroup` to apply it per run.

    Attributes:
        columns: Columns to check.
        min_run: Shortest run of equal values that counts as stuck (>= 2).
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    columns: tuple[str, ...]
    min_run: int = DEFAULT_MIN_RUN

    def __post_init__(self) -> None:
        """Normalise ``columns`` and check ``min_run``.

        Raises:
            ValueError: If ``min_run`` < 2.
        """
        object.__setattr__(self, "columns", _as_column_tuple(self.columns))
        # TODO: raise ValueError if self.min_run < 2

    def stuck_mask(self, values: np.ndarray) -> np.ndarray:
        """Return True for every sample inside a run of >= ``min_run`` equal values.

        Example:
            min_run=3, values [1, 2, 2, 2, 3, 3] -> [F, T, T, T, F, F]
        """
        # TODO: label runs of equal values (np.diff or a comparison with the
        #       previous value, then np.cumsum), measure each run's length
        #       (np.bincount) and compare with min_run. Exclude NaN.
        raise NotImplementedError

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with every stuck sample replaced by NaN."""
        # TODO: copy the frame, then for each column set stuck samples to NaN
        raise NotImplementedError


# ---------------------------------------------------------------- Task 3
@dataclass
class IQRClipper(ColumnStatStage):
    """Clip each column to Tukey's fences learned from the training split.

    Fences: [Q1 - k * IQR, Q3 + k * IQR] with IQR = Q3 - Q1, from the
    training data only. Values outside are set to the nearest fence
    (winsorising), values inside are unchanged, NaN stays NaN.

    Attributes:
        k: Fence width in IQRs (1.5: Tukey's "outside"; 3.0: "far out").
    """

    k: float = TUKEY_K

    def __post_init__(self) -> None:
        """Normalise ``columns`` and check ``k``.

        Raises:
            ValueError: If ``k`` <= 0.
        """
        super().__post_init__()
        # TODO: raise ValueError if self.k <= 0

    def learn(self, values: pd.Series) -> tuple[float, ...]:
        """Return (lower fence, upper fence) of one training column."""
        # TODO: quartiles with values.quantile([LOWER_QUARTILE, UPPER_QUARTILE])
        raise NotImplementedError

    def apply(self, values: pd.Series, stats: tuple[float, ...]) -> pd.Series:
        """Clip the column to the stored fences."""
        # TODO: one line with Series.clip
        raise NotImplementedError


# ---------------------------------------------------------------- Task 4
def build_cleaning_pipeline(horizon_steps: int = 10) -> DataPipeline:
    """Return the stateless cleaning recipe for the raw rover log.

    Start from the lecture recipe (:func:`rover_cleaning_pipeline`) and insert
    a per-run ``FlagStuck`` on the gyroscope column *directly after* the first
    stage (sorting and de-duplication) and before anything that changes values.
    Pipelines can be sliced (``p[:1]``) and joined (``a >> b``).

    Args:
        horizon_steps: Forecast horizon in samples (10 = 1 s).
    """
    # TODO
    raise NotImplementedError


def build_learned_pipeline() -> DataPipeline:
    """Return the learned recipe (fit it on the training split only).

    Three stages, in this order: median imputation of the gyroscope column,
    far-out (k = FAR_OUT_K) IQR clipping of the acceleration column, robust
    scaling of all four ROVER_FEATURES.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Task 5
@dataclass(frozen=True)
class PreparedData:
    """Model-ready arrays and the evidence that they are leak-free.

    Attributes:
        X_train: Scaled training features, columns in ROVER_FEATURES order.
        X_test: Scaled test features (same learned statistics).
        y_train: Training targets (metres, not scaled).
        y_test: Test targets.
        train_runs: Run ids in the training set, in order.
        test_runs: Run ids in the test set, in order.
        learned: The learned pipeline, fitted on the training rows only.
        leakage: :func:`~aerospace_ai.splits.check_split` of the split used.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_runs: tuple[int, ...]
    test_runs: tuple[int, ...]
    learned: DataPipeline
    leakage: LeakageReport


def prepare(path: str | Path, horizon_s: float = 1.0) -> PreparedData:
    """Raw CSV -> clean -> split by run -> fit learned stages on train -> arrays.

    Steps: read the CSV; run the cleaning pipeline; hold out the last
    TEST_FRACTION of the runs with GroupHoldoutSplit; fit the learned pipeline
    on the training rows; transform both parts; audit the split with
    check_split (groups = run ids, horizon = horizon in samples).

    Args:
        path: Raw rover log (CSV).
        horizon_s: Forecast horizon in seconds (a multiple of 0.1 s).
    """
    # TODO
    raise NotImplementedError
