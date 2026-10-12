"""Unit tests for aerospace_ai.preprocessing (Week 5). Provenance: new content."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aerospace_ai.pipeline import (
    DataPipeline,
    FillGaps,
    HampelFilter,
    NotFittedError,
    RollingMean,
    Standardize,
)
from aerospace_ai.preprocessing import (
    CausalRollingMean,
    ColumnStatStage,
    ForecastTarget,
    ImputeMedian,
    MarkInvalid,
    MinMaxScale,
    PerGroup,
    RobustScale,
    SortAndDeduplicate,
    audit_frame,
    count_duplicate_keys,
    longest_true_run,
)


@pytest.fixture
def two_runs() -> pd.DataFrame:
    """Two short runs whose time restarts at 0; x jumps between the runs."""
    return pd.DataFrame(
        {
            "run_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "time_s": [0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3],
            "x": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )


# --------------------------------------------------------------- helpers
@pytest.mark.parametrize(
    ("mask", "expected"),
    [([], 0), ([False, False], 0), ([True], 1), ([1, 1, 0, 1, 1, 1, 0], 3)],
)
def test_longest_true_run(mask: list[int], expected: int) -> None:
    assert longest_true_run(np.array(mask, dtype=bool)) == expected


def test_audit_frame_counts() -> None:
    frame = pd.DataFrame(
        {"r": [0.0, 5.0, np.inf, np.nan, 7.0, 7.0, 7.0], "s": [1.0, 2, 3, 4, 5, 6, 7]}
    )
    table = audit_frame(frame, {"r": 0.0})
    assert table.loc["r", ["n_nan", "n_inf", "n_sentinel"]].tolist() == [1, 1, 1]
    assert table.loc["r", "pct_unusable"] == pytest.approx(300 / 7)
    assert table.loc["r", "longest_repeat"] == 3
    assert (table.loc["r", "min"], table.loc["r", "max"]) == (5.0, 7.0)
    assert table.loc["s", "longest_repeat"] == 1


def test_count_duplicate_keys(two_runs: pd.DataFrame) -> None:
    doubled = pd.concat([two_runs, two_runs.iloc[[1, 5]]])
    assert count_duplicate_keys(two_runs) == 0
    assert count_duplicate_keys(doubled) == 2


# --------------------------------------------------------- stateless stages
def test_sort_and_deduplicate(two_runs: pd.DataFrame) -> None:
    messy = pd.concat([two_runs.iloc[::-1], two_runs.iloc[[2]]])
    clean = SortAndDeduplicate()(messy)
    pd.testing.assert_frame_equal(clean, two_runs)


def test_mark_invalid() -> None:
    frame = pd.DataFrame({"r": [0.0, 0.2, 5.0, 13.0, np.inf, -np.inf, 6.0]})
    out = MarkInvalid(("r",), sentinels=(0.0,), valid_min=0.3, valid_max=12.0)(frame)
    assert out["r"].isna().tolist() == [True, True, False, True, True, True, False]
    assert frame["r"].iloc[0] == 0.0  # input untouched


def test_mark_invalid_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError):
        MarkInvalid(("r",), valid_min=5.0, valid_max=1.0)


def test_causal_rolling_mean_uses_no_future() -> None:
    frame = pd.DataFrame({"x": [0.0, 0.0, 0.0, 9.0]})
    trailing = CausalRollingMean(("x",), window=3)(frame)["x_trailing_mean"]
    centred = RollingMean(("x",), window=3)(frame)["x"]
    assert trailing.tolist() == [0.0, 0.0, 0.0, 3.0]
    assert centred.iloc[2] == 3.0  # the centred window already sees the 9


def test_forecast_target_and_validation() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    out = ForecastTarget("x", 2, "x_ahead")(frame)
    assert out["x_ahead"].iloc[0] == 3.0 and out["x_ahead"].iloc[1:].isna().all()
    with pytest.raises(ValueError):
        ForecastTarget("x", 0, "y")


def test_per_group_keeps_runs_apart(two_runs: pd.DataFrame) -> None:
    out = PerGroup(ForecastTarget("x", 1, "ahead"))(two_runs)
    # The last sample of run 1 must NOT take its target from run 2.
    assert np.isnan(out["ahead"].iloc[3])
    assert out["ahead"].iloc[0] == 2.0 and out["ahead"].iloc[4] == 20.0
    plain = ForecastTarget("x", 1, "ahead")(two_runs)
    assert plain["ahead"].iloc[3] == 10.0  # the leak PerGroup prevents


def test_per_group_restores_order_and_index() -> None:
    frame = pd.DataFrame(
        {
            "run_id": [2, 1, 2, 1],
            "time_s": [0.0, 0.0, 0.1, 0.1],
            "x": [1.0, 5, np.nan, 7],
        },
        index=[10, 11, 12, 13],
    )
    out = PerGroup(FillGaps(("x",), time_column="time_s"))(frame)
    assert out.index.tolist() == [10, 11, 12, 13]
    assert out["run_id"].tolist() == [2, 1, 2, 1]


def test_per_group_with_row_dropping_inner(two_runs: pd.DataFrame) -> None:
    from aerospace_ai.pipeline import DropNonFinite

    frame = two_runs.assign(x=[1.0, np.nan, 3, 4, 10, 20, np.nan, 40])
    out = PerGroup(DropNonFinite(("x",)))(frame)
    assert len(out) == 6 and out.index.tolist() == list(range(6))


def test_per_group_refuses_learned_or_wrong_stages() -> None:
    with pytest.raises(TypeError):
        PerGroup(Standardize(("x",)))
    with pytest.raises(TypeError):
        PerGroup("not a stage")


def test_per_group_config_round_trip() -> None:
    pipeline = DataPipeline([PerGroup(HampelFilter(("x",), window=5), "run_id")])
    config = json.loads(json.dumps(pipeline.to_config()))
    assert DataPipeline.from_config(config) == pipeline


# ----------------------------------------------------------- learned stages
@pytest.fixture
def train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and test columns with a gap and an outlier."""
    train = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 100.0]})
    test = pd.DataFrame({"x": [np.nan, 3.0, 200.0]})
    return train, test


@pytest.mark.parametrize("cls", [ImputeMedian, MinMaxScale, RobustScale])
def test_learned_stage_contract(
    cls: type[ColumnStatStage], train_test: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, test = train_test
    stage = cls(("x",))
    assert not stage.is_fitted
    with pytest.raises(NotFittedError):
        stage(test)
    assert stage.fit(train) is stage and stage.is_fitted
    before = test.copy()
    stage(test)
    pd.testing.assert_frame_equal(test, before)  # input untouched


def test_learned_stage_uses_training_statistics_only(
    train_test: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = train_test
    imputer = ImputeMedian(("x",), add_indicator=True).fit(train)
    out = imputer(test)
    assert imputer.stats_ == {"x": (3.0,)}  # median of 1, 2, 4, 100
    assert out["x"].tolist() == [3.0, 3.0, 200.0]
    assert out["x_was_missing"].tolist() == [1.0, 0.0, 0.0]
    scaled = MinMaxScale(("x",)).fit(train)(test)
    assert scaled["x"].iloc[2] == pytest.approx(199 / 99)  # outside [0, 1]: fine
    robust = RobustScale(("x",)).fit(train)
    median, iqr = robust.stats_["x"]
    assert median == 3.0 and iqr == pytest.approx(28.0 - 1.75)


@pytest.mark.parametrize("cls", [MinMaxScale, RobustScale])
def test_learned_stage_rejects_constant_column(cls: type[ColumnStatStage]) -> None:
    with pytest.raises(ValueError):
        cls(("x",)).fit(pd.DataFrame({"x": [2.0, 2.0, 2.0]}))


def test_learned_stage_rejects_empty_column() -> None:
    with pytest.raises(ValueError):
        ImputeMedian(("x",)).fit(pd.DataFrame({"x": [np.nan, np.nan]}))


def test_column_stat_stage_is_abstract() -> None:
    with pytest.raises(TypeError):
        ColumnStatStage(("x",))  # type: ignore[abstract]


def test_learned_stage_repr_and_equality() -> None:
    a, b = ImputeMedian(("x",)), ImputeMedian(["x"])
    a.fit(pd.DataFrame({"x": [1.0]}))
    assert a == b  # same recipe, fitted or not
    assert repr(a) == "ImputeMedian(columns=('x',), add_indicator=False)"
