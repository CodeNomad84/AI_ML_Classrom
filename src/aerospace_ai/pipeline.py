"""Composable, typed data pipelines: one reproducible recipe for cleaning sensor data.

Provenance: new content (Week 3). The Python concepts follow, but are rewritten
from, W3Schools (Python abc Module, Python dataclasses Module, Python typing
Module, ``__repr__()`` and ``__eq__()`` magic methods, Python Iterators) and
GeeksforGeeks (Abstract Classes in Python, Understanding Python Dataclasses,
Python Generics, str() vs repr() in Python). The Hampel filter is adapted from
Pearson, "Outliers in process modeling and identification", IEEE Trans.
Control Systems Technology 10(1), 2002; the MAD consistency constant 1.4826
from Rousseeuw and Croux, JASA 88(424), 1993.

Design (the Week 3 "anchor"):

* :class:`PipelineStage` is an abstract, generic base class
  ``PipelineStage[InT, OutT]``. A stage turns one object into another with
  :meth:`PipelineStage.transform`; stages that learn something from data
  (for example a mean and standard deviation) also override
  :meth:`PipelineStage.fit`. Every stage declares ``INPUT_TYPE`` and
  ``OUTPUT_TYPE`` so that type mistakes are caught *when the pipeline is
  built*, before any data flows.
* Concrete stages are dataclasses. Their fields are their configuration, so
  ``__init__``, ``__repr__`` and ``__eq__`` are generated: two stages with the
  same settings compare equal and print the same. Learned state is declared
  with ``field(init=False, repr=False, compare=False)``.
* :class:`DataPipeline` is an immutable, ordered chain of stages built with
  ``stage_a >> stage_b``. It writes its own ``__repr__``, ``__eq__`` and
  container methods, checks type compatibility in ``__init__``, records a
  :class:`StageReport` per stage, and round-trips through a plain
  ``to_config``/``from_config`` dictionary so a result can always be traced
  to the exact recipe that produced it.

The pipeline works with *any* :class:`~aerospace_ai.sensors.BaseSensor`
through :class:`SensorToFrame`; it does not depend on a particular sensor
class.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, Final, Generic, TypeVar, overload

import numpy as np
import pandas as pd

from aerospace_ai.sensors import BaseSensor

InT = TypeVar("InT")
OutT = TypeVar("OutT")
MidT = TypeVar("MidT")

DEFAULT_TIME_COLUMN: Final[str] = "timestamp"
# Scale factor that makes the median absolute deviation (MAD) a consistent
# estimator of the standard deviation for Gaussian noise: 1 / Phi^-1(3/4).
MAD_TO_SIGMA: Final[float] = 1.4826
DEFAULT_HAMPEL_WINDOW: Final[int] = 7
DEFAULT_HAMPEL_SIGMAS: Final[float] = 3.0
DEFAULT_SMOOTHING_WINDOW: Final[int] = 5
DEFAULT_MAX_GAP_SAMPLES: Final[int] = 5


class NotFittedError(RuntimeError):
    """Raised when a stage that must learn from data is used before ``fit``."""


# ----------------------------------------------------------------------
# Small helpers shared by the stages.
# ----------------------------------------------------------------------
def _as_column_tuple(columns: str | Iterable[str]) -> tuple[str, ...]:
    """Return ``columns`` as a non-empty tuple of strings.

    A single string is treated as one column name (not as a sequence of
    characters). Lists are converted to tuples so that frozen dataclasses stay
    hashable and JSON configurations (which only know lists) round-trip.

    Raises:
        ValueError: If no column is given.
    """
    result = (columns,) if isinstance(columns, str) else tuple(columns)
    if not result:
        raise ValueError("at least one column is required")
    return result


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], stage: str) -> None:
    """Raise a KeyError that names the stage if a column is missing."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{stage}: missing column(s) {missing}; have {list(frame)}")


def _require_odd_window(window: int, stage: str) -> None:
    """Raise ValueError unless ``window`` is an odd integer >= 3 (centred windows)."""
    if window < 3 or window % 2 == 0:
        raise ValueError(f"{stage}: window must be an odd integer >= 3, got {window}")


def _count_rows(data: object) -> int | None:
    """Return the number of rows of a frame, array or sensor, else None."""
    try:
        return len(data)  # type: ignore[arg-type]
    except TypeError:
        return None


def _count_nan(data: object) -> int:
    """Return the number of NaN values in a frame or float array (0 otherwise)."""
    if isinstance(data, pd.DataFrame):
        return int(data.select_dtypes("number").isna().to_numpy().sum())
    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
        return int(np.isnan(data).sum())
    return 0


def _count_changed(before: object, after: object) -> int | None:
    """Return how many numeric cells a frame-to-frame stage changed.

    Only defined when both are DataFrames with the same index; cells that are
    NaN in both count as unchanged. Returns None otherwise.
    """
    if not (isinstance(before, pd.DataFrame) and isinstance(after, pd.DataFrame)):
        return None
    if not before.index.equals(after.index):
        return None
    common = [
        c
        for c in after.select_dtypes("number").columns
        if c in before.columns and pd.api.types.is_numeric_dtype(before[c])
    ]
    old = before[common].to_numpy(dtype=float)
    new = after[common].to_numpy(dtype=float)
    differ = ~((old == new) | (np.isnan(old) & np.isnan(new)))
    return int(differ.sum())


# ----------------------------------------------------------------------
# The abstract stage.
# ----------------------------------------------------------------------
class PipelineStage(ABC, Generic[InT, OutT]):
    """Abstract base class for one step of a data pipeline.

    Subclasses set ``INPUT_TYPE`` and ``OUTPUT_TYPE`` (the runtime types the
    stage accepts and returns; generic parameters are erased at runtime, so
    the pipeline needs these to check a chain) and implement
    :meth:`transform`. Stages that learn from data also set
    ``REQUIRES_FIT = True`` and override :meth:`fit` and :attr:`is_fitted`.

    Every subclass is registered by class name in ``registry`` when it is
    defined, which is what lets :meth:`DataPipeline.from_config` rebuild a
    pipeline from plain text.

    Example:
        >>> stage = SelectColumns(("timestamp", "accel_z_mps2"))
        >>> stage.name, stage.get_params()
        ('SelectColumns', {'columns': ('timestamp', 'accel_z_mps2')})
    """

    INPUT_TYPE: ClassVar[type] = object
    OUTPUT_TYPE: ClassVar[type] = object
    REQUIRES_FIT: ClassVar[bool] = False
    registry: ClassVar[dict[str, type[PipelineStage[Any, Any]]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register every subclass by name when its class statement runs.

        Redefining a class with the same name (for example re-running a
        notebook cell) replaces the earlier entry.
        """
        super().__init_subclass__(**kwargs)
        PipelineStage.registry[cls.__name__] = cls

    # ------------------------------------------------------------------
    # What a concrete stage must or may provide.
    # ------------------------------------------------------------------
    @abstractmethod
    def transform(self, data: InT) -> OutT:
        """Apply the stage to ``data`` and return a *new* object.

        Implementations must not modify ``data`` in place.
        """

    def fit(self, data: InT) -> PipelineStage[InT, OutT]:
        """Learn whatever the stage needs from ``data``; stateless by default.

        Returns:
            The stage itself, so that ``stage.fit(x).transform(x)`` chains.
        """
        return self

    @property
    def is_fitted(self) -> bool:
        """Return True if the stage can transform (always True if stateless)."""
        return not self.REQUIRES_FIT

    # ------------------------------------------------------------------
    # Concrete behaviour shared by every stage.
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Return the stage's registry name (its class name)."""
        return type(self).__name__

    def get_params(self) -> dict[str, Any]:
        """Return the configuration: the dataclass fields that ``__init__`` accepts.

        Learned state (fields with ``init=False``) is not configuration and is
        left out, so ``type(stage)(**stage.get_params())`` builds an unfitted
        copy.
        """
        if not is_dataclass(self):
            return {}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    def check_input(self, data: object) -> None:
        """Raise TypeError if ``data`` is not an instance of ``INPUT_TYPE``."""
        if not isinstance(data, self.INPUT_TYPE):
            raise TypeError(
                f"{self.name} expects {self.INPUT_TYPE.__name__}, "
                f"got {type(data).__name__}"
            )

    def __call__(self, data: InT) -> OutT:
        """Check the input type, refuse if unfitted, transform, check the output.

        Raises:
            TypeError: If the input or output has the wrong type.
            NotFittedError: If the stage must be fitted first.
        """
        self.check_input(data)
        if not self.is_fitted:
            raise NotFittedError(f"{self.name} must be fitted before transform")
        result = self.transform(data)
        if not isinstance(result, self.OUTPUT_TYPE):
            raise TypeError(
                f"{self.name}.transform returned {type(result).__name__}, "
                f"declared {self.OUTPUT_TYPE.__name__}"
            )
        return result

    def __rshift__(
        self, other: PipelineStage[OutT, MidT] | DataPipeline[OutT, MidT]
    ) -> DataPipeline[InT, MidT]:
        """Return the two-step pipeline ``self`` then ``other`` (``a >> b``)."""
        return DataPipeline([self]) >> other


# ----------------------------------------------------------------------
# Concrete stages. Each is a dataclass: fields = configuration.
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SensorToFrame(PipelineStage[BaseSensor, pd.DataFrame]):
    """Turn any sensor's stored readings into a DataFrame.

    Works for every :class:`~aerospace_ai.sensors.BaseSensor` subclass, which
    is polymorphism at work: the stage only uses the shared interface.

    Attributes:
        valid_only: Drop readings that failed ``validate`` (and the
            ``is_valid`` column) if True.
        time_column: Name given to the time column.
    """

    INPUT_TYPE: ClassVar[type] = BaseSensor
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    valid_only: bool = True
    time_column: str = DEFAULT_TIME_COLUMN

    def transform(self, data: BaseSensor) -> pd.DataFrame:
        """Return one row per reading: time, one column per channel."""
        frame = data.to_frame()
        if self.valid_only:
            frame = frame.loc[frame["is_valid"]].drop(columns="is_valid")
            frame = frame.reset_index(drop=True)
        return frame.rename(columns={"timestamp": self.time_column})


@dataclass(frozen=True)
class SelectColumns(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Keep only the listed columns, in the listed order.

    Attributes:
        columns: Column names to keep (a list is accepted and stored as a tuple).
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalise ``columns`` to a tuple (frozen, so use object.__setattr__)."""
        object.__setattr__(self, "columns", _as_column_tuple(self.columns))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy that holds only ``columns``."""
        _require_columns(data, self.columns, self.name)
        return data.loc[:, list(self.columns)].copy()


@dataclass(frozen=True)
class DropNonFinite(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Drop rows that contain NaN or +-inf in the given columns.

    Attributes:
        columns: Columns to check; None checks every numeric column.
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    columns: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Normalise ``columns`` to a tuple unless it is None."""
        if self.columns is not None:
            object.__setattr__(self, "columns", _as_column_tuple(self.columns))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return the rows whose checked values are all finite, re-indexed."""
        if self.columns is None:
            checked = data.select_dtypes("number")
        else:
            _require_columns(data, self.columns, self.name)
            checked = data.loc[:, list(self.columns)]
        keep = np.isfinite(checked.to_numpy(dtype=float)).all(axis=1)
        return data.loc[keep].reset_index(drop=True)


@dataclass(frozen=True)
class FillGaps(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Linearly interpolate short runs of NaN inside a signal.

    Runs of at most ``max_gap`` consecutive NaN are filled; longer runs are
    left untouched (interpolating across a long dropout invents data). Leading
    and trailing NaN are never filled.

    Attributes:
        columns: Columns to fill.
        max_gap: Longest run of consecutive NaN that is filled, in samples.
        time_column: Column used as the interpolation abscissa.
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    columns: tuple[str, ...]
    max_gap: int = DEFAULT_MAX_GAP_SAMPLES
    time_column: str = DEFAULT_TIME_COLUMN

    def __post_init__(self) -> None:
        """Normalise ``columns`` and check ``max_gap``."""
        object.__setattr__(self, "columns", _as_column_tuple(self.columns))
        if self.max_gap < 1:
            raise ValueError(f"{self.name}: max_gap must be >= 1")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with short interior NaN runs interpolated against time."""
        _require_columns(data, (*self.columns, self.time_column), self.name)
        out = data.copy()
        t = data[self.time_column].to_numpy(dtype=float)
        for col in self.columns:
            values = data[col].to_numpy(dtype=float, copy=True)
            missing = np.isnan(values)
            if not missing.any() or missing.all():
                continue
            # Label each run of NaN, measure its length, keep the short interior ones.
            run_id = np.cumsum(~missing)
            run_length = pd.Series(missing).groupby(run_id).transform("sum").to_numpy()
            first, last = np.flatnonzero(~missing)[[0, -1]]
            interior = (np.arange(len(values)) > first) & (
                np.arange(len(values)) < last
            )
            fill = missing & interior & (run_length <= self.max_gap)
            values[fill] = np.interp(t[fill], t[~missing], values[~missing])
            out[col] = values
        return out


@dataclass(frozen=True)
class HampelFilter(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Replace isolated outliers (spikes) by the local median.

    For every sample x_k, take the centred window of ``window`` samples,
    its median m_k and its median absolute deviation
    MAD_k = median(|x_j - m_k|). The sample is an outlier if

        |x_k - m_k| > n_sigmas * 1.4826 * MAD_k

    and is then replaced by m_k. Near the ends the window is truncated. NaN
    values are ignored by the medians and left as NaN.

    Attributes:
        columns: Columns to despike.
        window: Odd window length in samples (>= 3).
        n_sigmas: Threshold in robust standard deviations.
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    columns: tuple[str, ...]
    window: int = DEFAULT_HAMPEL_WINDOW
    n_sigmas: float = DEFAULT_HAMPEL_SIGMAS

    def __post_init__(self) -> None:
        """Normalise ``columns`` and check the window and threshold."""
        object.__setattr__(self, "columns", _as_column_tuple(self.columns))
        _require_odd_window(self.window, self.name)
        if self.n_sigmas <= 0:
            raise ValueError(f"{self.name}: n_sigmas must be positive")

    @property
    def half_window(self) -> int:
        """Return the number of samples on each side of the centre sample."""
        return self.window // 2

    def outlier_mask(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Return (is_outlier, local_median) for one column.

        Args:
            series: The signal to test.

        Returns:
            Boolean array of outliers and the rolling median, both shape (N,).
        """
        values = series.to_numpy(dtype=float)
        # Pad with NaN so every sample gets a full-width row of neighbours;
        # nanmedian then ignores the padding, which truncates the window at
        # the ends. Shape of ``windows``: (N, window).
        padded = np.pad(values, self.half_window, constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, self.window)
        with warnings.catch_warnings():
            # A window that is all NaN has no median; NaN is the right answer.
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(windows, axis=1)
            mad = np.nanmedian(np.abs(windows - median[:, None]), axis=1)
        deviation = np.abs(values - median)
        with np.errstate(invalid="ignore"):
            is_outlier = deviation > self.n_sigmas * MAD_TO_SIGMA * mad
        return is_outlier, median

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with outliers replaced by the local median."""
        _require_columns(data, self.columns, self.name)
        out = data.copy()
        for col in self.columns:
            is_outlier, median = self.outlier_mask(data[col].astype(float))
            out[col] = np.where(is_outlier, median, data[col].to_numpy(dtype=float))
        return out


@dataclass(frozen=True)
class RollingMean(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Smooth columns with a centred moving average (truncated at the ends).

    Attributes:
        columns: Columns to smooth.
        window: Odd window length in samples (>= 3).
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    columns: tuple[str, ...]
    window: int = DEFAULT_SMOOTHING_WINDOW

    def __post_init__(self) -> None:
        """Normalise ``columns`` and check the window."""
        object.__setattr__(self, "columns", _as_column_tuple(self.columns))
        _require_odd_window(self.window, self.name)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with each column replaced by its centred moving mean."""
        _require_columns(data, self.columns, self.name)
        out = data.copy()
        for col in self.columns:
            out[col] = data[col].rolling(self.window, center=True, min_periods=1).mean()
        return out


@dataclass(frozen=True)
class TimeDerivative(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Add the time derivative of one column (for example climb rate).

    Uses second-order central differences inside the record and one-sided
    differences at the ends (``numpy.gradient`` with the actual time stamps,
    so a non-uniform sample rate is handled).

    Attributes:
        column: Column to differentiate.
        output_column: Name of the new column.
        time_column: Column holding time in seconds.
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame

    column: str
    output_column: str
    time_column: str = DEFAULT_TIME_COLUMN

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with ``output_column`` = d(column)/d(time)."""
        _require_columns(data, (self.column, self.time_column), self.name)
        if len(data) < 2:
            raise ValueError(f"{self.name}: need at least two rows")
        out = data.copy()
        out[self.output_column] = np.gradient(
            data[self.column].to_numpy(dtype=float),
            data[self.time_column].to_numpy(dtype=float),
        )
        return out


@dataclass
class Standardize(PipelineStage[pd.DataFrame, pd.DataFrame]):
    """Scale columns to zero mean and unit standard deviation.

    The mean and standard deviation are *learned* by :meth:`fit` and then
    reused by :meth:`transform`. Fit on training data only: fitting on the
    test data leaks information into the model (Week 5).

    This dataclass is not frozen, because ``fit`` changes it. The learned
    fields use ``init=False`` (not a constructor argument), ``repr=False``
    (keep the printout short) and ``compare=False`` (two stages with the same
    settings are the same recipe, fitted or not).

    Attributes:
        columns: Columns to scale.
        mean_: Learned means (None before fitting).
        std_: Learned population standard deviations (None before fitting).
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = pd.DataFrame
    REQUIRES_FIT: ClassVar[bool] = True

    columns: tuple[str, ...]
    mean_: dict[str, float] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    std_: dict[str, float] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Normalise ``columns`` to a tuple."""
        self.columns = _as_column_tuple(self.columns)

    @property
    def is_fitted(self) -> bool:
        """Return True once ``fit`` has stored the statistics."""
        return self.mean_ is not None and self.std_ is not None

    def fit(self, data: pd.DataFrame) -> Standardize:
        """Learn the mean and (population, ddof=0) standard deviation per column.

        Raises:
            ValueError: If a column is constant (its standard deviation is 0).
        """
        _require_columns(data, self.columns, self.name)
        means = data[list(self.columns)].mean()
        stds = data[list(self.columns)].std(ddof=0)
        constant = [c for c in self.columns if not stds[c] > 0]
        if constant:
            raise ValueError(f"{self.name}: constant column(s) {constant}")
        self.mean_ = {c: float(means[c]) for c in self.columns}
        self.std_ = {c: float(stds[c]) for c in self.columns}
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with z = (x - mean_) / std_ in each column.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        if self.mean_ is None or self.std_ is None:
            raise NotFittedError(f"{self.name} must be fitted before transform")
        _require_columns(data, self.columns, self.name)
        out = data.copy()
        for col in self.columns:
            out[col] = (data[col] - self.mean_[col]) / self.std_[col]
        return out


@dataclass(frozen=True)
class ToArray(PipelineStage[pd.DataFrame, np.ndarray]):
    """Convert selected columns to a float NumPy array (model input).

    After this stage the data is no longer a DataFrame, so a DataFrame stage
    cannot follow it: the pipeline refuses such a chain when it is built.

    Attributes:
        columns: Columns to export, in order.
    """

    INPUT_TYPE: ClassVar[type] = pd.DataFrame
    OUTPUT_TYPE: ClassVar[type] = np.ndarray

    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalise ``columns`` to a tuple."""
        object.__setattr__(self, "columns", _as_column_tuple(self.columns))

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Return an array of shape (n_rows, n_columns)."""
        _require_columns(data, self.columns, self.name)
        return data.loc[:, list(self.columns)].to_numpy(dtype=float)


# ----------------------------------------------------------------------
# Run records.
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StageReport:
    """What one stage did during one run.

    Attributes:
        stage: Stage name.
        rows_in: Rows (or readings) received, None if not countable.
        rows_out: Rows returned, None if not countable.
        nan_out: NaN values in the output.
        cells_changed: Numeric cells modified (frame-to-frame stages only).
        seconds: Wall-clock time of the stage.
    """

    stage: str
    rows_in: int | None
    rows_out: int | None
    nan_out: int
    cells_changed: int | None
    seconds: float


@dataclass(frozen=True, slots=True)
class PipelineResult(Generic[OutT]):
    """The output of :meth:`DataPipeline.run` together with its audit trail.

    Attributes:
        output: The final data.
        reports: One :class:`StageReport` per stage, in order.
        config: The pipeline configuration that produced ``output``.
    """

    output: OutT
    reports: tuple[StageReport, ...]
    config: dict[str, Any]

    def report_frame(self) -> pd.DataFrame:
        """Return the stage reports as a DataFrame indexed by step number."""
        frame = pd.DataFrame(
            [
                {f.name: getattr(r, f.name) for f in fields(StageReport)}
                for r in self.reports
            ]
        )
        frame.index.name = "step"
        return frame


# ----------------------------------------------------------------------
# The pipeline.
# ----------------------------------------------------------------------
class DataPipeline(Generic[InT, OutT]):
    """An immutable, type-checked chain of :class:`PipelineStage` objects.

    Build one from a list of stages or with ``>>``. The constructor checks
    that each stage's ``OUTPUT_TYPE`` is accepted by the next stage's
    ``INPUT_TYPE``. ``>>`` and slicing return *new* pipelines; the stage
    tuple itself never changes. Stage objects are shared, not copied, so
    fitting a pipeline fits its stages.

    Example:
        >>> p = SelectColumns(["timestamp", "x"]) >> RollingMean(["x"], window=3)
        >>> len(p), p.stage_names
        (2, ('SelectColumns', 'RollingMean'))
    """

    __slots__ = ("_stages",)
    CONFIG_VERSION: ClassVar[int] = 1

    def __init__(self, stages: Iterable[PipelineStage[Any, Any]]) -> None:
        """Store the stages after checking that they fit together.

        Args:
            stages: One or more stages, in execution order.

        Raises:
            ValueError: If no stage is given.
            TypeError: If an item is not a stage, or two neighbours do not fit.
        """
        stage_tuple = tuple(stages)
        if not stage_tuple:
            raise ValueError("a pipeline needs at least one stage")
        for position, stage in enumerate(stage_tuple):
            if not isinstance(stage, PipelineStage):
                raise TypeError(
                    f"item {position} is {type(stage).__name__}, not a PipelineStage"
                )
        for position, (left, right) in enumerate(zip(stage_tuple, stage_tuple[1:])):
            if not issubclass(left.OUTPUT_TYPE, right.INPUT_TYPE):
                raise TypeError(
                    f"step {position} {left.name} outputs "
                    f"{left.OUTPUT_TYPE.__name__}, but step {position + 1} "
                    f"{right.name} expects {right.INPUT_TYPE.__name__}"
                )
        self._stages = stage_tuple

    # ------------------------------------------------------------------
    # Read-only views.
    # ------------------------------------------------------------------
    @property
    def stages(self) -> tuple[PipelineStage[Any, Any], ...]:
        """Return the stages (a tuple, so it cannot be modified)."""
        return self._stages

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Return the stage names in order."""
        return tuple(s.name for s in self._stages)

    @property
    def input_type(self) -> type:
        """Return the type the first stage accepts."""
        return self._stages[0].INPUT_TYPE

    @property
    def output_type(self) -> type:
        """Return the type the last stage produces."""
        return self._stages[-1].OUTPUT_TYPE

    @property
    def is_fitted(self) -> bool:
        """Return True if every stage is ready to transform."""
        return all(s.is_fitted for s in self._stages)

    # ------------------------------------------------------------------
    # Container protocol and operators.
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of stages."""
        return len(self._stages)

    def __iter__(self) -> Iterator[PipelineStage[Any, Any]]:
        """Iterate over the stages in execution order."""
        return iter(self._stages)

    @overload
    def __getitem__(self, index: int) -> PipelineStage[Any, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> DataPipeline[Any, Any]: ...

    def __getitem__(
        self, index: int | slice
    ) -> PipelineStage[Any, Any] | DataPipeline[Any, Any]:
        """Return one stage, or a sub-pipeline for a slice (``p[:2]``)."""
        if isinstance(index, slice):
            return DataPipeline(self._stages[index])
        return self._stages[index]

    def __rshift__(
        self, other: PipelineStage[OutT, MidT] | DataPipeline[OutT, MidT]
    ) -> DataPipeline[InT, MidT]:
        """Return a new pipeline with ``other`` (a stage or pipeline) appended."""
        if isinstance(other, DataPipeline):
            return DataPipeline((*self._stages, *other.stages))
        if isinstance(other, PipelineStage):
            return DataPipeline((*self._stages, other))
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        """Two pipelines are equal if they have equal stages in the same order.

        Stage equality comes from the dataclass-generated ``__eq__`` (same
        class, same configuration). Comparing with a non-pipeline returns
        NotImplemented so Python can try the other operand.
        """
        if not isinstance(other, DataPipeline):
            return NotImplemented
        return self._stages == other._stages

    # Equal pipelines may contain mutable (fittable) stages, so pipelines are
    # deliberately unhashable: defining __eq__ without __hash__ means this.
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        """Return a multi-line, copy-pasteable description of the chain."""
        body = "\n    >> ".join(repr(s) for s in self._stages)
        return f"DataPipeline(\n    {body}\n)"

    def __str__(self) -> str:
        """Return a one-line summary for reports: ``A -> B -> C``."""
        return " -> ".join(self.stage_names)

    # ------------------------------------------------------------------
    # Running.
    # ------------------------------------------------------------------
    def fit(self, data: InT) -> DataPipeline[InT, OutT]:
        """Fit every stage in order on the output of the stage before it.

        Returns:
            The pipeline itself (its stages are now fitted).
        """
        current: Any = data
        for stage in self._stages:
            stage.check_input(current)
            stage.fit(current)
            current = stage(current)
        return self

    def transform(self, data: InT) -> OutT:
        """Run the data through every stage and return the final output."""
        return self.run(data).output

    def fit_transform(self, data: InT) -> OutT:
        """Fit on ``data`` and return the transformed result."""
        return self.fit(data).transform(data)

    def __call__(self, data: InT) -> OutT:
        """Same as :meth:`transform`, so a pipeline can be used like a stage."""
        return self.transform(data)

    def run(self, data: InT) -> PipelineResult[OutT]:
        """Run every stage and record what each one did.

        Returns:
            A :class:`PipelineResult` with the output, one report per stage
            and the configuration used.

        Raises:
            NotFittedError: If a stage that must learn has not been fitted.
            TypeError: If the input is not of the pipeline's input type.
        """
        reports: list[StageReport] = []
        current: Any = data
        for stage in self._stages:
            start = time.perf_counter()
            result = stage(current)
            reports.append(
                StageReport(
                    stage=stage.name,
                    rows_in=_count_rows(current),
                    rows_out=_count_rows(result),
                    nan_out=_count_nan(result),
                    cells_changed=_count_changed(current, result),
                    seconds=time.perf_counter() - start,
                )
            )
            current = result
        return PipelineResult(
            output=current, reports=tuple(reports), config=self.to_config()
        )

    # ------------------------------------------------------------------
    # Reproducibility.
    # ------------------------------------------------------------------
    def to_config(self) -> dict[str, Any]:
        """Return a JSON-ready description: stage names and parameters.

        Tuples become lists so ``json.dumps`` output reads naturally.
        """

        def plain(value: Any) -> Any:
            """Convert tuples (recursively) to lists."""
            if isinstance(value, tuple):
                return [plain(v) for v in value]
            return value

        return {
            "version": self.CONFIG_VERSION,
            "stages": [
                {
                    "stage": s.name,
                    "params": {k: plain(v) for k, v in s.get_params().items()},
                }
                for s in self._stages
            ],
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DataPipeline[Any, Any]:
        """Rebuild an (unfitted) pipeline from :meth:`to_config` output.

        Raises:
            ValueError: If the version is unknown.
            KeyError: If a stage name is not registered.
        """
        if config.get("version") != cls.CONFIG_VERSION:
            raise ValueError(f"unsupported config version {config.get('version')}")
        stages = []
        for entry in config["stages"]:
            name = entry["stage"]
            if name not in PipelineStage.registry:
                raise KeyError(f"unknown stage {name!r}; is its module imported?")
            stages.append(PipelineStage.registry[name](**entry["params"]))
        return cls(stages)
