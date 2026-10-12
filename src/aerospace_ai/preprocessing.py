"""Data-engineering stages for noisy sensor logs: invalid values, gaps, scaling.

Provenance: new content (Week 5), building on the Week 3 pipeline. Missing-data
mechanisms follow Little and Rubin, *Statistical Analysis with Missing Data*,
3rd ed. (2019), Ch. 1; imputation, outlier fences and scaling are rewritten
from GeeksforGeeks ("Handling Missing Values in Machine Learning", "Detect and
Remove the Outliers using Python", "Data Leakage") and W3Schools ("Pandas -
Cleaning Data", "Python Machine Learning Scaling"); the robust scaler follows
Huber and Ronchetti, *Robust Statistics*, 2nd ed. (2009), Ch. 5.

Two kinds of stage, and the rule that keeps a pipeline leak-free:

* **Stateless** stages look only at the rows they are given (and, wrapped in
  :class:`PerGroup`, only at one run at a time). They may run on the whole raw
  log *before* the train/test split: :class:`SortAndDeduplicate`,
  :class:`MarkInvalid`, :class:`CausalRollingMean`, :class:`ForecastTarget`.
* **Learned** stages compute statistics from data (medians, minima,
  quartiles). They subclass :class:`ColumnStatStage` and must be fitted on
  the *training* split only, then applied unchanged to validation and test
  data: :class:`ImputeMedian`, :class:`MinMaxScale`, :class:`RobustScale`.

:func:`audit_frame` produces the data-quality table that starts every cleaning
job: what is missing, what is a sentinel, what never changes.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final

import numpy as np
import pandas as pd

from aerospace_ai.pipeline import (
    NotFittedError,
    PipelineStage,
    _as_column_tuple,
    _require_columns,
)

DEFAULT_GROUP_COLUMN: Final[str] = "run_id"
DEFAULT_KEYS: Final[tuple[str, ...]] = ("run_id", "time_s")
MISSING_SUFFIX: Final[str] = "_was_missing"
LOWER_QUARTILE: Final[float] = 0.25
UPPER_QUARTILE: Final[float] = 0.75

FrameStage = PipelineStage[pd.DataFrame, pd.DataFrame]


def longest_true_run(mask: np.ndarray) -> int:
    """Return the length of the longest run of consecutive True values.

    Example:
        >>> longest_true_run(np.array([True, False, True, True, True, False]))
        3
    """
    best = current = 0
    for flag in np.asarray(mask, dtype=bool):
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def audit_frame(
    frame: pd.DataFrame,
    sentinels: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Return a data-quality table with one row per numeric column.

    Columns of the result: ``n_nan`` (NaN cells), ``n_inf`` (+-inf),
    ``n_sentinel`` (cells equal to the column's sentinel, e.g. a LiDAR 0.0
    "no return"), ``pct_unusable`` (all three as a percentage),
    ``longest_nan_run`` (samples), ``longest_repeat`` (longest run of exactly
    repeated values, a frozen-sensor hint), ``min`` and ``max`` of the usable
    values.

    Args:
        frame: The raw table.
        sentinels: Optional ``{column: sentinel value}``.

    Returns:
        A DataFrame indexed by column name.
    """
    sentinels = dict(sentinels or {})
    rows: dict[str, dict[str, float]] = {}
    for col in frame.select_dtypes("number").columns:
        values = frame[col].to_numpy(dtype=float)
        is_nan = np.isnan(values)
        is_inf = np.isinf(values)
        is_sentinel = (
            values == sentinels[col] if col in sentinels else np.zeros_like(is_nan)
        )
        usable = values[~(is_nan | is_inf | is_sentinel)]
        repeats = np.concatenate([[False], np.diff(values) == 0.0])
        rows[col] = {
            "n_nan": int(is_nan.sum()),
            "n_inf": int(is_inf.sum()),
            "n_sentinel": int(is_sentinel.sum()),
            "pct_unusable": 100.0 * float((is_nan | is_inf | is_sentinel).mean()),
            "longest_nan_run": longest_true_run(is_nan),
            # A run of k repeated samples has k - 1 zero differences.
            "longest_repeat": longest_true_run(repeats) + 1,
            "min": float(usable.min()) if usable.size else np.nan,
            "max": float(usable.max()) if usable.size else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def count_duplicate_keys(
    frame: pd.DataFrame, keys: tuple[str, ...] = DEFAULT_KEYS
) -> int:
    """Return how many rows repeat an earlier row's key (e.g. run and time)."""
    _require_columns(frame, keys, "count_duplicate_keys")
    return int(frame.duplicated(subset=list(keys)).sum())


# ----------------------------------------------------------------------
# Stateless stages.
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SortAndDeduplicate(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Sort by the key columns and keep the first row of each key.

    Loggers re-send packets, so the same (run, time) can appear twice. Sorting
    first makes every later per-run, time-based stage safe.

    Attributes:
        keys: Columns that identify a sample, in sort order.
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    keys: tuple[str, ...] = DEFAULT_KEYS

    def __post_init__(self) -> None:
        """Normalise ``keys`` to a tuple."""
        object.__setattr__(self, "keys", _as_column_tuple(self.keys))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return the sorted rows without repeated keys, re-indexed."""
        _require_columns(data, self.keys, self.name)
        ordered = data.sort_values(list(self.keys), kind="stable")
        return ordered.drop_duplicates(subset=list(self.keys)).reset_index(drop=True)


@dataclass(frozen=True)
class MarkInvalid(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Turn sentinels, +-inf and physically impossible values into NaN.

    A LiDAR that receives no echo reports ``0.0``; a model would read that as
    "obstacle touching the sensor". Making it NaN states the truth: *we do not
    know*. Deciding what to do with the NaN is a separate, later step.

    Attributes:
        columns: Columns to check.
        sentinels: Values that mean "no measurement".
        valid_min: Smallest physically possible value (None: no lower bound).
        valid_max: Largest physically possible value (None: no upper bound).
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    columns: tuple[str, ...]
    sentinels: tuple[float, ...] = ()
    valid_min: float | None = None
    valid_max: float | None = None

    def __post_init__(self) -> None:
        """Normalise the tuples and check the bounds."""
        object.__setattr__(self, "columns", _as_column_tuple(self.columns))
        object.__setattr__(self, "sentinels", tuple(float(s) for s in self.sentinels))
        if (
            self.valid_min is not None
            and self.valid_max is not None
            and self.valid_min >= self.valid_max
        ):
            raise ValueError(f"{self.name}: valid_min must be below valid_max")

    def invalid_mask(self, values: np.ndarray) -> np.ndarray:
        """Return True where a value is a sentinel, non-finite or out of bounds."""
        bad = ~np.isfinite(values) | np.isin(values, self.sentinels)
        with np.errstate(invalid="ignore"):
            if self.valid_min is not None:
                bad |= values < self.valid_min
            if self.valid_max is not None:
                bad |= values > self.valid_max
        return bad

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with every invalid cell replaced by NaN."""
        _require_columns(data, self.columns, self.name)
        out = data.copy()
        for col in self.columns:
            values = data[col].to_numpy(dtype=float, copy=True)
            values[self.invalid_mask(values)] = np.nan
            out[col] = values
        return out


@dataclass(frozen=True)
class CausalRollingMean(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Add a *trailing* moving average: only the current and past samples.

    Week 3's :class:`~aerospace_ai.pipeline.RollingMean` is centred, so each
    value uses samples from the future. For a forecasting model that is
    leakage: the feature already "knows" part of what it should predict. A
    trailing window can be computed on the vehicle in real time.

    Attributes:
        columns: Columns to average.
        window: Window length in samples (>= 2).
        suffix: Appended to the column name for the new feature.
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    columns: tuple[str, ...]
    window: int = 5
    suffix: str = "_trailing_mean"

    def __post_init__(self) -> None:
        """Normalise ``columns`` and check the window."""
        object.__setattr__(self, "columns", _as_column_tuple(self.columns))
        if self.window < 2:
            raise ValueError(f"{self.name}: window must be at least 2")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with one new ``<column><suffix>`` per column."""
        _require_columns(data, self.columns, self.name)
        out = data.copy()
        for col in self.columns:
            out[f"{col}{self.suffix}"] = (
                data[col].rolling(self.window, min_periods=1).mean()
            )
        return out


@dataclass(frozen=True)
class ForecastTarget(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Add the value of ``column`` ``horizon`` samples in the future as a target.

    Wrap it in :class:`PerGroup` so that the last samples of one run never take
    their target from the first samples of the next run. Rows near the end
    get NaN (there is no future yet) and are dropped later.

    Attributes:
        column: Signal to forecast.
        horizon: Steps ahead (>= 1).
        output_column: Name of the target column.
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    column: str
    horizon: int
    output_column: str

    def __post_init__(self) -> None:
        """Check the horizon."""
        if self.horizon < 1:
            raise ValueError(f"{self.name}: horizon must be at least 1")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with ``output_column`` = ``column`` shifted back in time."""
        _require_columns(data, (self.column,), self.name)
        out = data.copy()
        out[self.output_column] = data[self.column].shift(-self.horizon)
        return out


@dataclass(frozen=True)
class PerGroup(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Apply a stateless frame stage to each group (run, flight) separately.

    A composite: it *is* a stage and it *contains* a stage. Interpolation,
    despiking, rolling windows and shifts must not reach across the boundary
    between two test drives; this wrapper makes any Week 3 stage respect it,
    without editing that stage.

    Attributes:
        inner: A stateless DataFrame-to-DataFrame stage (or its config dict
            ``{"stage": name, "params": {...}}`` as written by ``to_config``).
        group_column: Column that identifies the group.
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    inner: Any
    group_column: str = DEFAULT_GROUP_COLUMN

    def __post_init__(self) -> None:
        """Rebuild ``inner`` from a config dict if needed and check it.

        Raises:
            TypeError: If ``inner`` is not a stateless frame-to-frame stage.
        """
        inner = self.inner
        if isinstance(inner, Mapping):
            inner = PipelineStage.registry[inner["stage"]](**inner["params"])
            object.__setattr__(self, "inner", inner)
        if not isinstance(inner, PipelineStage):
            raise TypeError(f"{self.name}: inner must be a PipelineStage")
        if not (
            issubclass(inner.OUTPUT_TYPE, pd.DataFrame)
            and issubclass(pd.DataFrame, inner.INPUT_TYPE)
        ):
            raise TypeError(f"{self.name}: inner must map DataFrame to DataFrame")
        if inner.REQUIRES_FIT:
            raise TypeError(
                f"{self.name}: {inner.name} learns from data; fit it on the "
                "training split instead of per group"
            )

    def get_params(self) -> dict[str, Any]:
        """Return a JSON-ready configuration (the inner stage as a nested dict)."""
        return {
            "inner": {"stage": self.inner.name, "params": self.inner.get_params()},
            "group_column": self.group_column,
        }

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run ``inner`` on every group and put the pieces back together.

        If the inner stage keeps every row, the original index and row order
        are restored; if it drops rows, the result is re-indexed from 0.
        """
        _require_columns(data, (self.group_column,), self.name)
        pieces: list[pd.DataFrame] = []
        keeps_rows = True
        for positions in data.groupby(self.group_column, sort=False).indices.values():
            part = self.inner(data.iloc[positions].reset_index(drop=True))
            if len(part) == len(positions):
                part.index = data.index[positions]
            else:
                keeps_rows = False
            pieces.append(part)
        if not pieces:
            return data.copy()
        if keeps_rows:
            return pd.concat(pieces).loc[data.index]
        return pd.concat(pieces, ignore_index=True)


# ----------------------------------------------------------------------
# Learned stages: fit on the training split only.
# ----------------------------------------------------------------------
@dataclass
class ColumnStatStage(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Abstract base for stages that learn a few numbers per column.

    Template method: :meth:`fit` checks the columns, calls :meth:`learn` on
    each training column and stores the tuples in ``stats_``;
    :meth:`transform` calls :meth:`apply` with those stored numbers. A
    subclass only says *what* to learn and *how* to use it. NaN cells are
    ignored when learning.

    Attributes:
        columns: Columns to process.
        stats_: Learned numbers per column (None before fitting).
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame
    REQUIRES_FIT: ClassVar[bool] = True

    columns: tuple[str, ...]
    stats_: dict[str, tuple[float, ...]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Normalise ``columns`` to a tuple."""
        self.columns = _as_column_tuple(self.columns)

    @property
    def is_fitted(self) -> bool:
        """Return True once :meth:`fit` has stored the statistics."""
        return self.stats_ is not None

    @abstractmethod
    def learn(self, values: pd.Series) -> tuple[float, ...]:
        """Return the statistics of one training column (NaN already removed).

        Raises:
            ValueError: If the column does not allow the statistic (e.g. constant).
        """

    @abstractmethod
    def apply(self, values: pd.Series, stats: tuple[float, ...]) -> pd.Series:
        """Return the transformed column using the stored statistics."""

    def fit(self, data: pd.DataFrame) -> ColumnStatStage:
        """Learn the statistics of every column from ``data`` (the training split).

        Raises:
            ValueError: If a column has no usable value.
        """
        _require_columns(data, self.columns, self.name)
        stats: dict[str, tuple[float, ...]] = {}
        for col in self.columns:
            values = data[col].astype(float).dropna()
            if values.empty:
                raise ValueError(f"{self.name}: column {col!r} has no usable value")
            stats[col] = tuple(float(s) for s in self.learn(values))
        self.stats_ = stats
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with every column passed through :meth:`apply`.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        if self.stats_ is None:
            raise NotFittedError(f"{self.name} must be fitted before transform")
        _require_columns(data, self.columns, self.name)
        out = data.copy()
        for col in self.columns:
            out[col] = self.apply(data[col].astype(float), self.stats_[col])
        return out


@dataclass
class ImputeMedian(ColumnStatStage):
    """Fill NaN with the training median, optionally flagging what was filled.

    The flag column ``<column>_was_missing`` (0.0 or 1.0) keeps the
    information that a value was missing. When missingness is informative
    (MNAR: the LiDAR is silent *because* the wall is far), the flag is often
    worth more than the filled value.

    Attributes:
        add_indicator: Also add the 0/1 flag columns.
    """

    add_indicator: bool = False

    def learn(self, values: pd.Series) -> tuple[float, ...]:
        """Return the training median."""
        return (float(values.median()),)

    def apply(self, values: pd.Series, stats: tuple[float, ...]) -> pd.Series:
        """Fill NaN with the stored median."""
        return values.fillna(stats[0])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill the gaps and, if requested, add the missing-value flags."""
        out = super().transform(data)
        if self.add_indicator:
            for col in self.columns:
                out[f"{col}{MISSING_SUFFIX}"] = data[col].isna().astype(float)
        return out


@dataclass
class MinMaxScale(ColumnStatStage):
    """Scale to [0, 1] with the training minimum and maximum.

    Test values outside the training range map outside [0, 1]; that is
    correct, not a bug (clipping them would hide a distribution shift).
    """

    def learn(self, values: pd.Series) -> tuple[float, ...]:
        """Return (min, max); a constant column cannot be scaled."""
        low, high = float(values.min()), float(values.max())
        if not high > low:
            raise ValueError(f"{self.name}: constant column")
        return low, high

    def apply(self, values: pd.Series, stats: tuple[float, ...]) -> pd.Series:
        """Return (x - min) / (max - min)."""
        low, high = stats
        return (values - low) / (high - low)


@dataclass
class RobustScale(ColumnStatStage):
    """Centre on the training median and divide by the interquartile range.

    Medians and quartiles barely move when a few spikes are present, so the
    scale is not inflated by the very outliers we are trying to tame.
    """

    def learn(self, values: pd.Series) -> tuple[float, ...]:
        """Return (median, IQR); a column with zero IQR cannot be scaled."""
        q1, median, q3 = values.quantile([LOWER_QUARTILE, 0.5, UPPER_QUARTILE])
        if not q3 > q1:
            raise ValueError(f"{self.name}: interquartile range is zero")
        return float(median), float(q3 - q1)

    def apply(self, values: pd.Series, stats: tuple[float, ...]) -> pd.Series:
        """Return (x - median) / IQR."""
        median, iqr = stats
        return (values - median) / iqr
