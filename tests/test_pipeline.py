"""Unit tests for aerospace_ai.pipeline (Week 3). Provenance: new content."""

from __future__ import annotations

import dataclasses
import json
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from aerospace_ai.pipeline import (
    MAD_TO_SIGMA,
    DataPipeline,
    DropNonFinite,
    FillGaps,
    HampelFilter,
    NotFittedError,
    PipelineStage,
    RollingMean,
    SelectColumns,
    SensorToFrame,
    Standardize,
    TimeDerivative,
    ToArray,
)
from aerospace_ai.sensors import BarometerSensor, IMUSensor

# The worked example of the lecture: a barometric altitude record with one spike.
SPIKY_ALTITUDE = [100.0, 100.4, 99.8, 106.0, 100.3, 100.1, 99.9]


@pytest.fixture
def spiky() -> pd.DataFrame:
    """Return the seven-sample altitude record used in the lecture notes."""
    return pd.DataFrame(
        {"timestamp": np.arange(7, dtype=float), "alt_m": SPIKY_ALTITUDE}
    )


@pytest.fixture
def imu() -> IMUSensor:
    """Return an IMU with five level samples, one of them saturated."""
    sensor = IMUSensor("imu0", 50.0)
    for k in range(5):
        az = 1_000.0 if k == 2 else 9.8
        sensor.add_reading(0.02 * k, [0.0, 0.0, az, 0.0, 0.0, 0.0])
    return sensor


# ---------------------------------------------------------------- the ABC
def test_stage_base_class_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        PipelineStage()  # type: ignore[abstract]


def test_subclass_without_transform_stays_abstract() -> None:
    class Incomplete(PipelineStage[pd.DataFrame, pd.DataFrame]):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_subclasses_register_themselves() -> None:
    @dataclasses.dataclass(frozen=True)
    class Negate(PipelineStage[pd.DataFrame, pd.DataFrame]):
        INPUT_TYPE: ClassVar[type] = pd.DataFrame
        OUTPUT_TYPE: ClassVar[type] = pd.DataFrame
        column: str

        def transform(self, data: pd.DataFrame) -> pd.DataFrame:
            return data.assign(**{self.column: -data[self.column]})

    assert PipelineStage.registry["Negate"] is Negate
    config = DataPipeline([Negate("x")]).to_config()
    rebuilt = DataPipeline.from_config(config)
    assert rebuilt(pd.DataFrame({"x": [1.0]}))["x"].iloc[0] == -1.0


# ---------------------------------------------------------------- dataclass behaviour
def test_stages_compare_and_print_by_configuration() -> None:
    a = HampelFilter(["alt_m"], window=5)
    b = HampelFilter(("alt_m",), window=5)
    assert a == b and hash(a) == hash(b)
    assert a != HampelFilter(["alt_m"], window=7)
    assert repr(a) == "HampelFilter(columns=('alt_m',), window=5, n_sigmas=3.0)"


def test_frozen_stage_cannot_be_changed() -> None:
    stage = RollingMean(["alt_m"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        stage.window = 9  # type: ignore[misc]


def test_single_string_is_one_column() -> None:
    assert SelectColumns("alt_m").columns == ("alt_m",)


@pytest.mark.parametrize("window", [0, 1, 2, 4, -3])
def test_windows_must_be_odd_and_at_least_three(window: int) -> None:
    with pytest.raises(ValueError, match="odd"):
        HampelFilter(["x"], window=window)
    with pytest.raises(ValueError, match="odd"):
        RollingMean(["x"], window=window)


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        SelectColumns([])
    with pytest.raises(ValueError):
        HampelFilter(["x"], n_sigmas=0.0)
    with pytest.raises(ValueError):
        FillGaps(["x"], max_gap=0)


# ---------------------------------------------------------------- stages
def test_hampel_matches_the_hand_calculation(spiky: pd.DataFrame) -> None:
    stage = HampelFilter(["alt_m"], window=5)
    is_outlier, median = stage.outlier_mask(spiky["alt_m"])
    # Window around k = 3: {100.4, 99.8, 106.0, 100.3, 100.1} -> median 100.3,
    # MAD 0.2, threshold 3 * 1.4826 * 0.2 = 0.8896 < |106.0 - 100.3| = 5.7.
    assert median[3] == pytest.approx(100.3)
    assert list(np.flatnonzero(is_outlier)) == [3]
    out = stage(spiky)
    assert out["alt_m"].iloc[3] == pytest.approx(100.3)
    assert spiky["alt_m"].iloc[3] == 106.0, "input must not be modified"
    assert MAD_TO_SIGMA == pytest.approx(1.4826)


def test_hampel_leaves_nan_alone() -> None:
    frame = pd.DataFrame({"x": [1.0, np.nan, 1.1, 0.9, 1.0]})
    out = HampelFilter(["x"], window=3).transform(frame)
    assert np.isnan(out["x"].iloc[1])


def test_rolling_mean_is_centred(spiky: pd.DataFrame) -> None:
    out = RollingMean(["alt_m"], window=3)(spiky)
    assert out["alt_m"].iloc[3] == pytest.approx((99.8 + 106.0 + 100.3) / 3)
    assert out["alt_m"].iloc[0] == pytest.approx((100.0 + 100.4) / 2)


def test_order_of_stages_matters(spiky: pd.DataFrame) -> None:
    despike_first = HampelFilter(["alt_m"], window=5) >> RollingMean(["alt_m"], 3)
    smooth_first = RollingMean(["alt_m"], 3) >> HampelFilter(["alt_m"], window=5)
    a = despike_first(spiky)["alt_m"]
    b = smooth_first(spiky)["alt_m"]
    assert despike_first != smooth_first
    assert a.max() < 100.5 < b.max()


def test_fill_gaps_only_fills_short_interior_runs() -> None:
    t = np.arange(10, dtype=float)
    x = np.array([np.nan, 1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 8.0, np.nan])
    frame = pd.DataFrame({"timestamp": t, "x": x})
    out = FillGaps(["x"], max_gap=2)(frame)["x"].to_numpy()
    assert out[2] == pytest.approx(2.0)
    assert np.isnan(out[[0, 4, 5, 6, 9]]).all()
    assert np.isnan(frame["x"].iloc[2]), "input must not be modified"


def test_drop_non_finite(spiky: pd.DataFrame) -> None:
    frame = spiky.copy()
    frame.loc[[1, 4], "alt_m"] = [np.nan, np.inf]
    out = DropNonFinite()(frame)
    assert len(out) == 5 and list(out.index) == list(range(5))
    assert len(DropNonFinite(["timestamp"])(frame)) == 7


def test_time_derivative_handles_uneven_sampling() -> None:
    frame = pd.DataFrame({"timestamp": [0.0, 1.0, 3.0, 4.0]})
    frame["alt_m"] = 2.0 * frame["timestamp"] + 5.0
    out = TimeDerivative("alt_m", "climb_mps")(frame)
    assert np.allclose(out["climb_mps"], 2.0)


def test_select_columns_reports_missing_names(spiky: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="SelectColumns.*speed"):
        SelectColumns(["timestamp", "speed"])(spiky)


def test_standardize_learns_then_applies(spiky: pd.DataFrame) -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    stage = Standardize(["x"])
    assert not stage.is_fitted
    with pytest.raises(NotFittedError):
        stage(train)
    stage.fit(train)
    assert stage.mean_ == {"x": 2.0}
    assert stage.std_ == pytest.approx({"x": np.sqrt(2.0 / 3.0)})
    out = stage(pd.DataFrame({"x": [4.0]}))
    assert out["x"].iloc[0] == pytest.approx(2.0 / np.sqrt(2.0 / 3.0))


def test_standardize_learned_state_is_not_configuration() -> None:
    fitted = Standardize(["x"]).fit(pd.DataFrame({"x": [0.0, 1.0]}))
    assert fitted == Standardize(["x"]), "compare=False on learned fields"
    assert "mean_" not in repr(fitted)
    assert fitted.get_params() == {"columns": ("x",)}
    with pytest.raises(ValueError, match="constant"):
        Standardize(["x"]).fit(pd.DataFrame({"x": [3.0, 3.0]}))


def test_sensor_to_frame_works_for_any_sensor(imu: IMUSensor) -> None:
    frame = SensorToFrame()(imu)
    assert len(frame) == 4 and "is_valid" not in frame
    assert list(frame.columns[:2]) == ["timestamp", "accel_x_mps2"]
    assert len(SensorToFrame(valid_only=False)(imu)) == 5
    baro = BarometerSensor("baro0", 25.0)
    baro.add_reading(0.0, [95_000.0])
    assert SensorToFrame()(baro)["altitude_m"].iloc[0] == pytest.approx(540.3, abs=0.1)


def test_to_array_shape(spiky: pd.DataFrame) -> None:
    array = ToArray(["timestamp", "alt_m"])(spiky)
    assert isinstance(array, np.ndarray) and array.shape == (7, 2)


# ---------------------------------------------------------------- the pipeline
def test_pipeline_refuses_incompatible_neighbours() -> None:
    with pytest.raises(TypeError, match="ToArray outputs ndarray"):
        ToArray(["x"]) >> RollingMean(["x"])
    with pytest.raises(TypeError, match="expects BaseSensor"):
        SelectColumns(["x"]) >> SensorToFrame()


def test_pipeline_refuses_empty_and_non_stages() -> None:
    with pytest.raises(ValueError):
        DataPipeline([])
    with pytest.raises(TypeError, match="not a PipelineStage"):
        DataPipeline([RollingMean(["x"]), len])  # type: ignore[list-item]


def test_stage_checks_input_type_at_run_time(spiky: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="expects DataFrame"):
        RollingMean(["alt_m"])(spiky.to_numpy())


def test_pipeline_container_protocol() -> None:
    p = SelectColumns(["t", "x"]) >> RollingMean(["x"]) >> ToArray(["x"])
    assert len(p) == 3
    assert p.stage_names == ("SelectColumns", "RollingMean", "ToArray")
    assert isinstance(p[1], RollingMean)
    assert isinstance(p[:2], DataPipeline) and len(p[:2]) == 2
    assert p.input_type is pd.DataFrame and p.output_type is np.ndarray
    assert [s.name for s in p] == list(p.stage_names)
    assert str(p) == "SelectColumns -> RollingMean -> ToArray"


def test_rshift_returns_a_new_pipeline() -> None:
    base = DataPipeline([RollingMean(["x"])])
    longer = base >> RollingMean(["x"], window=7)
    assert len(base) == 1 and len(longer) == 2
    assert (base >> DataPipeline([ToArray(["x"])])).output_type is np.ndarray


def test_pipeline_equality_and_hash() -> None:
    a = RollingMean(["x"]) >> ToArray(["x"])
    b = RollingMean(["x"]) >> ToArray(["x"])
    assert a == b and a is not b
    assert a != RollingMean(["x"], window=7) >> ToArray(["x"])
    assert a != "not a pipeline"
    with pytest.raises(TypeError):
        hash(a)


def test_repr_round_trips_through_eval() -> None:
    p = HampelFilter(["x"], window=5) >> RollingMean(["x"])
    namespace = {"DataPipeline": DataPipeline, "HampelFilter": HampelFilter}
    namespace["RollingMean"] = RollingMean
    rebuilt = eval(repr(p).replace("DataPipeline(", "(", 1), namespace)  # noqa: S307
    assert rebuilt == p


def test_fit_transform_fits_on_upstream_output() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 30.0, 4.0, 5.0]})
    p = HampelFilter(["x"], window=5) >> Standardize(["x"]) >> ToArray(["x"])
    assert not p.is_fitted
    z = p.fit_transform(frame)
    assert p.is_fitted
    scaler = p[1]
    assert isinstance(scaler, Standardize) and scaler.mean_ is not None
    # The scaler learned from the despiked data (the 30.0 became the median 4.0).
    assert scaler.mean_["x"] == pytest.approx(np.mean([1.0, 2.0, 4.0, 4.0, 5.0]))
    assert z.mean() == pytest.approx(0.0)


def test_run_records_one_report_per_stage(spiky: pd.DataFrame) -> None:
    p = HampelFilter(["alt_m"], window=5) >> DropNonFinite() >> ToArray(["alt_m"])
    result = p.run(spiky)
    assert [r.stage for r in result.reports] == list(p.stage_names)
    assert result.reports[0].cells_changed == 1
    assert result.reports[2].cells_changed is None
    frame = result.report_frame()
    assert list(frame["rows_out"]) == [7, 7, 7]
    assert result.config == p.to_config()


def test_config_round_trip_through_json(spiky: pd.DataFrame) -> None:
    p = (
        SelectColumns(["timestamp", "alt_m"])
        >> FillGaps(["alt_m"], max_gap=3)
        >> HampelFilter(["alt_m"], window=5, n_sigmas=2.5)
        >> RollingMean(["alt_m"], window=3)
        >> TimeDerivative("alt_m", "climb_mps")
    )
    text = json.dumps(p.to_config())
    rebuilt = DataPipeline.from_config(json.loads(text))
    assert rebuilt == p
    pd.testing.assert_frame_equal(rebuilt(spiky), p(spiky))


def test_from_config_rejects_unknown_input() -> None:
    with pytest.raises(ValueError, match="version"):
        DataPipeline.from_config({"version": 99, "stages": []})
    with pytest.raises(KeyError, match="NoSuchStage"):
        DataPipeline.from_config(
            {"version": 1, "stages": [{"stage": "NoSuchStage", "params": {}}]}
        )


def test_full_sensor_pipeline(imu: IMUSensor) -> None:
    p = (
        SensorToFrame()
        >> SelectColumns(["timestamp", "accel_z_mps2"])
        >> ToArray(["accel_z_mps2"])
    )
    out = p(imu)
    assert out.shape == (4, 1) and np.allclose(out, 9.8)
