"""Leak-free train/test splitting for time-ordered, grouped sensor data.

Provenance: new content (Week 5). The leakage causes and remedies are
rewritten from GeeksforGeeks ("Data Leakage") and W3Schools ("Python Machine
Learning - Train/Test"); blocked and grouped validation for autocorrelated
data follows Roberts et al., "Cross-validation strategies for data with
temporal, spatial, hierarchical, or phylogenetic structure", *Ecography*
40(8), 2017; the gap between training and test blocks ("purging") follows
Lopez de Prado, *Advances in Financial Machine Learning* (2018), Ch. 7.

Every splitter is a Strategy (Week 4): it has one job, :meth:`Splitter.split`,
which returns row indices, never data. :func:`check_split` audits a split and
:func:`split_dataset` applies one to a :class:`~aerospace_ai.datasets.Dataset`.

Why it matters: neighbouring sensor samples are almost identical (10 Hz,
smooth motion). A random row split puts the sample at t = 4.1 s in the test
set and its twins at 4.0 s and 4.2 s in the training set, so a nearest-
neighbour model "predicts" the test set by looking it up. The score is
excellent and meaningless.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from aerospace_ai.datasets import Dataset

DEFAULT_TEST_FRACTION: Final[float] = 0.25


@dataclass(frozen=True, eq=False)
class SplitIndices:
    """Row indices of one train/test split.

    Attributes:
        train: Sorted training row positions.
        test: Sorted test row positions.
    """

    train: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        """Store sorted integer copies and refuse empty or overlapping parts.

        Raises:
            ValueError: If a part is empty or a row is in both.
        """
        train = np.sort(np.asarray(self.train, dtype=int))
        test = np.sort(np.asarray(self.test, dtype=int))
        if train.size == 0 or test.size == 0:
            raise ValueError("a split needs at least one training and one test row")
        if np.intersect1d(train, test).size:
            raise ValueError("a row cannot be in both the training and the test set")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "test", test)

    def __repr__(self) -> str:
        """Return the sizes only."""
        return f"SplitIndices(n_train={self.train.size}, n_test={self.test.size})"


def _check_fraction(test_fraction: float) -> None:
    """Raise ValueError unless 0 < test_fraction < 1."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")


class Splitter(ABC):
    """Strategy interface: turn a sample count (and groups) into splits."""

    @abstractmethod
    def split(
        self, n_samples: int, groups: np.ndarray | None = None
    ) -> list[SplitIndices]:
        """Return one or more train/test splits as row positions.

        Args:
            n_samples: Number of rows, in time order (per group).
            groups: Optional group label per row (e.g. run id).
        """


@dataclass(frozen=True)
class RandomRowSplit(Splitter):
    """Shuffle the rows, then cut. Correct for i.i.d. rows, leaky for time series.

    Attributes:
        test_fraction: Share of rows in the test set.
        seed: Random seed.
    """

    test_fraction: float = DEFAULT_TEST_FRACTION
    seed: int = 0

    def __post_init__(self) -> None:
        """Check the fraction."""
        _check_fraction(self.test_fraction)

    def split(
        self, n_samples: int, groups: np.ndarray | None = None
    ) -> list[SplitIndices]:
        """Return one random split (``groups`` is ignored)."""
        order = np.random.default_rng(self.seed).permutation(n_samples)
        n_test = round(self.test_fraction * n_samples)
        return [SplitIndices(order[n_test:], order[:n_test])]


@dataclass(frozen=True)
class ChronologicalSplit(Splitter):
    """Train on the past, test on the future, with an optional gap between.

    The ``gap`` rows just before the test block are dropped from training.
    Use a gap of at least the forecast horizon: a training row near the
    boundary has a target taken from inside the test period.

    Attributes:
        test_fraction: Share of rows (at the end) in the test set.
        gap: Rows removed between the training and the test block.
    """

    test_fraction: float = DEFAULT_TEST_FRACTION
    gap: int = 0

    def __post_init__(self) -> None:
        """Check the fraction and the gap."""
        _check_fraction(self.test_fraction)
        if self.gap < 0:
            raise ValueError("gap must be non-negative")

    def split(
        self, n_samples: int, groups: np.ndarray | None = None
    ) -> list[SplitIndices]:
        """Return one split: rows [0, start - gap) train, [start, n) test."""
        start = n_samples - round(self.test_fraction * n_samples)
        return [SplitIndices(np.arange(start - self.gap), np.arange(start, n_samples))]


@dataclass(frozen=True)
class GroupHoldoutSplit(Splitter):
    """Hold out whole groups (the last runs recorded) for testing.

    Groups are taken in order of first appearance, which is chronological for
    a log sorted by run. No run contributes rows to both sides, so nothing
    learned from one run can be "looked up" in the test set.

    Attributes:
        test_fraction: Share of *groups* (rounded up) in the test set.
    """

    test_fraction: float = DEFAULT_TEST_FRACTION

    def __post_init__(self) -> None:
        """Check the fraction."""
        _check_fraction(self.test_fraction)

    def split(
        self, n_samples: int, groups: np.ndarray | None = None
    ) -> list[SplitIndices]:
        """Return one split with the last ``ceil(fraction * n_groups)`` groups as test.

        Raises:
            ValueError: If ``groups`` is missing, has the wrong length, or has
                fewer than two groups.
        """
        if groups is None or len(groups) != n_samples:
            raise ValueError("GroupHoldoutSplit needs one group label per row")
        labels = pd.unique(np.asarray(groups))
        if labels.size < 2:
            raise ValueError("need at least two groups")
        n_test = min(math.ceil(self.test_fraction * labels.size), labels.size - 1)
        is_test = np.isin(groups, labels[labels.size - n_test :])
        return [SplitIndices(np.flatnonzero(~is_test), np.flatnonzero(is_test))]


@dataclass(frozen=True)
class ExpandingWindowCV(Splitter):
    """Time-series cross-validation: train on blocks 1..k, test on block k + 1.

    The rows are cut into ``n_splits + 1`` equal blocks. Fold k trains on all
    blocks before block k + 1 (minus ``gap`` rows) and tests on block k + 1,
    so the model is never trained on the future of its test data.

    Attributes:
        n_splits: Number of folds (>= 2).
        gap: Rows dropped at the end of each training window.
    """

    n_splits: int = 4
    gap: int = 0

    def __post_init__(self) -> None:
        """Check the settings."""
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.gap < 0:
            raise ValueError("gap must be non-negative")

    def split(
        self, n_samples: int, groups: np.ndarray | None = None
    ) -> list[SplitIndices]:
        """Return ``n_splits`` expanding-window folds."""
        block = n_samples // (self.n_splits + 1)
        if block - self.gap < 1:
            raise ValueError("too few samples for this many folds and this gap")
        folds = []
        for k in range(1, self.n_splits + 1):
            test_start = n_samples - (self.n_splits - k + 1) * block
            folds.append(
                SplitIndices(
                    np.arange(test_start - self.gap),
                    np.arange(test_start, test_start + block),
                )
            )
        return folds


# ----------------------------------------------------------------------
# Auditing and applying splits.
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LeakageReport:
    """What a split shares between its training and test rows.

    Attributes:
        shared_rows: Row positions in both parts (must be 0).
        shared_groups: Group labels with rows in both parts.
        min_distance: Smallest |position(test) - position(train)| within one
            group, in rows (None if the parts share no group).
        horizon: Forecast horizon in rows that the distance must exceed.
    """

    shared_rows: int
    shared_groups: tuple[object, ...]
    min_distance: int | None
    horizon: int

    @property
    def ok(self) -> bool:
        """Return True if no row is shared and no test row is within the horizon."""
        close = self.min_distance is not None and self.min_distance <= self.horizon
        return self.shared_rows == 0 and not close

    def __str__(self) -> str:
        """Return a one-line verdict."""
        verdict = "OK" if self.ok else "LEAKY"
        return (
            f"{verdict}: {self.shared_rows} shared rows, "
            f"{len(self.shared_groups)} shared groups, "
            f"closest train/test rows {self.min_distance} apart "
            f"(horizon {self.horizon})"
        )


def check_split(
    split: SplitIndices, groups: np.ndarray | None = None, horizon: int = 0
) -> LeakageReport:
    """Audit a split for the two common leaks: shared rows and close neighbours.

    Args:
        split: The split to audit.
        groups: Group label per row; without it, all rows form one group.
        horizon: Forecast horizon in rows; a training row closer than this to a
            test row of the same group has a target from the test period.

    Returns:
        A :class:`LeakageReport`.
    """
    shared_rows = int(np.intersect1d(split.train, split.test).size)
    if groups is None:
        groups = np.zeros(max(split.train.max(), split.test.max()) + 1, dtype=int)
    groups = np.asarray(groups)
    shared = np.intersect1d(groups[split.train], groups[split.test])
    distance: int | None = None
    for label in shared:
        train_rows = split.train[groups[split.train] == label]
        test_rows = split.test[groups[split.test] == label]
        positions = np.searchsorted(train_rows, test_rows)
        left = train_rows[np.clip(positions - 1, 0, train_rows.size - 1)]
        right = train_rows[np.clip(positions, 0, train_rows.size - 1)]
        nearest = np.minimum(np.abs(test_rows - left), np.abs(test_rows - right))
        distance = (
            int(nearest.min())
            if distance is None
            else min(distance, int(nearest.min()))
        )
    return LeakageReport(shared_rows, tuple(shared.tolist()), distance, horizon)


def split_dataset(
    dataset: Dataset, splitter: Splitter
) -> list[tuple[Dataset, Dataset]]:
    """Apply a splitter to a Dataset (its ``groups`` are passed on).

    Returns:
        One ``(train, test)`` pair per fold.
    """
    return [
        (dataset.subset(s.train, "train"), dataset.subset(s.test, "test"))
        for s in splitter.split(len(dataset), dataset.groups)
    ]


def split_frame(
    frame: pd.DataFrame, splitter: Splitter, group_column: str | None = None
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Apply a splitter to a DataFrame (rows must already be in time order).

    Returns:
        One ``(train, test)`` pair of re-indexed frames per fold.
    """
    groups = None if group_column is None else frame[group_column].to_numpy()
    return [
        (
            frame.iloc[s.train].reset_index(drop=True),
            frame.iloc[s.test].reset_index(drop=True),
        )
        for s in splitter.split(len(frame), groups)
    ]
