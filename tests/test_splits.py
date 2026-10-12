"""Unit tests for aerospace_ai.splits (Week 5). Provenance: new content."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerospace_ai.datasets import Dataset
from aerospace_ai.splits import (
    ChronologicalSplit,
    ExpandingWindowCV,
    GroupHoldoutSplit,
    RandomRowSplit,
    SplitIndices,
    Splitter,
    check_split,
    split_dataset,
    split_frame,
)

GROUPS = np.repeat([1, 2, 3, 4], 5)  # four runs of five rows


def test_split_indices_validation() -> None:
    with pytest.raises(ValueError):
        SplitIndices(np.array([0, 1]), np.array([1, 2]))
    with pytest.raises(ValueError):
        SplitIndices(np.array([], dtype=int), np.array([1]))
    s = SplitIndices(np.array([3, 1]), np.array([2]))
    assert s.train.tolist() == [1, 3] and repr(s) == "SplitIndices(n_train=2, n_test=1)"


@pytest.mark.parametrize(
    "splitter",
    [
        RandomRowSplit(0.25, seed=1),
        ChronologicalSplit(0.25, gap=2),
        GroupHoldoutSplit(0.25),
        ExpandingWindowCV(3, gap=1),
    ],
)
def test_every_splitter_contract(splitter: Splitter) -> None:
    folds = splitter.split(20, GROUPS)
    assert folds and all(isinstance(f, SplitIndices) for f in folds)
    for fold in folds:
        assert np.intersect1d(fold.train, fold.test).size == 0
        assert fold.train.max() < 20 and fold.test.max() < 20


def test_chronological_split_with_gap() -> None:
    (fold,) = ChronologicalSplit(0.25, gap=3).split(20)
    assert fold.train.tolist() == list(range(12)) and fold.test.tolist() == list(
        range(15, 20)
    )


def test_group_holdout_keeps_runs_whole() -> None:
    (fold,) = GroupHoldoutSplit(0.25).split(20, GROUPS)
    assert set(GROUPS[fold.test]) == {4} and 4 not in set(GROUPS[fold.train])
    (fold,) = GroupHoldoutSplit(0.3).split(20, GROUPS)  # ceil(1.2) = 2 runs
    assert set(GROUPS[fold.test]) == {3, 4}


def test_group_holdout_errors() -> None:
    with pytest.raises(ValueError):
        GroupHoldoutSplit().split(20)
    with pytest.raises(ValueError):
        GroupHoldoutSplit().split(5, np.ones(5))


def test_expanding_window_folds() -> None:
    folds = ExpandingWindowCV(n_splits=3, gap=1).split(20)
    assert [f.test.tolist() for f in folds] == [
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
    ]
    assert [f.train.max() for f in folds] == [3, 8, 13]
    with pytest.raises(ValueError):
        ExpandingWindowCV(n_splits=10, gap=2).split(20)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_fraction_validation(bad: float) -> None:
    for cls in (RandomRowSplit, ChronologicalSplit, GroupHoldoutSplit):
        with pytest.raises(ValueError):
            cls(bad)


def test_check_split_detects_close_neighbours() -> None:
    (random_fold,) = RandomRowSplit(0.25, seed=0).split(20)
    report = check_split(random_fold, horizon=2)
    assert not report.ok and report.min_distance == 1
    (gapped,) = ChronologicalSplit(0.25, gap=2).split(20)
    assert check_split(gapped, horizon=2).ok
    assert check_split(gapped, horizon=3).ok is False
    (grouped,) = GroupHoldoutSplit(0.25).split(20, GROUPS)
    report = check_split(grouped, GROUPS, horizon=10)
    assert report.ok and report.shared_groups == () and report.min_distance is None
    assert str(report).startswith("OK")


def test_split_dataset_carries_groups() -> None:
    data = Dataset(
        np.arange(40.0).reshape(20, 2), np.arange(20.0), ("a", "b"), "y", groups=GROUPS
    )
    ((train, test),) = split_dataset(data, GroupHoldoutSplit(0.25))
    assert len(train) == 15 and set(test.groups) == {4}
    assert test.name == "dataset/test"


def test_split_frame() -> None:
    frame = pd.DataFrame({"run_id": GROUPS, "x": np.arange(20)})
    ((train, test),) = split_frame(frame, GroupHoldoutSplit(0.5), "run_id")
    assert test["run_id"].unique().tolist() == [3, 4] and train.index[0] == 0
