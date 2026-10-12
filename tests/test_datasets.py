"""Unit tests for aerospace_ai.datasets (Week 4, Factory). Provenance: new content."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aerospace_ai.datasets import (
    FUEL_FEATURES,
    CsvTableLoader,
    Dataset,
    DatasetFactory,
    DatasetLoader,
    FlightLogCsvLoader,
    FuelFlowAircraft,
    SyntheticFuelFlowLoader,
    UnknownDatasetError,
    isa_density,
)
from aerospace_ai.pipeline import DataPipeline, DropNonFinite
from aerospace_ai.utils import FlightProfileConfig, SyntheticFlightLogGenerator

BUILT_IN = ("csv", "flight_log_csv", "fuel_flow_synthetic")


@pytest.fixture
def tiny() -> Dataset:
    """Ten samples, two features, y = row index."""
    X = np.arange(20, dtype=float).reshape(10, 2)
    return Dataset(X, np.arange(10.0), ("a", "b"), "y", "tiny")


@pytest.fixture
def scratch_name() -> Iterator[str]:
    """A registry key that is removed again after the test."""
    name = "unit_test_loader"
    yield name
    if name in DatasetFactory.available():
        DatasetFactory.unregister(name)


# --------------------------------------------------------------------- Dataset
def test_dataset_repr_and_len(tiny: Dataset) -> None:
    assert len(tiny) == 10 and tiny.n_features == 2
    assert repr(tiny).startswith("Dataset(name='tiny', n_samples=10")


@pytest.mark.parametrize(
    ("X", "y", "names"),
    [
        (np.zeros(5), np.zeros(5), ("a",)),  # X not 2-D
        (np.zeros((5, 1)), np.zeros((5, 1)), ("a",)),  # y not 1-D
        (np.zeros((5, 1)), np.zeros(4), ("a",)),  # length mismatch
        (np.zeros((5, 2)), np.zeros(5), ("a",)),  # too few names
    ],
)
def test_dataset_rejects_bad_shapes(X: np.ndarray, y: np.ndarray, names: tuple) -> None:
    with pytest.raises(ValueError):
        Dataset(X, y, names, "y")


def test_chronological_split_keeps_order(tiny: Dataset) -> None:
    train, test = tiny.split(test_fraction=0.3)
    np.testing.assert_array_equal(train.y, np.arange(7.0))
    np.testing.assert_array_equal(test.y, [7.0, 8.0, 9.0])
    assert train.name == "tiny/train" and test.name == "tiny/test"


def test_shuffled_split_is_seeded_and_disjoint(tiny: Dataset) -> None:
    a_train, a_test = tiny.split(0.3, shuffle=True, seed=1)
    b_train, _ = tiny.split(0.3, shuffle=True, seed=1)
    np.testing.assert_array_equal(a_train.y, b_train.y)
    assert set(a_train.y).isdisjoint(a_test.y)
    assert len(a_train) + len(a_test) == len(tiny)


@pytest.mark.parametrize("fraction", [0.0, 1.0, 0.01])
def test_split_rejects_degenerate_fraction(tiny: Dataset, fraction: float) -> None:
    with pytest.raises(ValueError):
        tiny.split(fraction)


# --------------------------------------------------------------------- factory
def test_built_in_loaders_are_registered() -> None:
    assert set(BUILT_IN) <= set(DatasetFactory.available())
    assert DatasetFactory.loader_class("fuel_flow_synthetic") is SyntheticFuelFlowLoader


def test_factory_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        DatasetFactory()


def test_unknown_name_suggests_close_match() -> None:
    with pytest.raises(UnknownDatasetError, match="fuel_flow_synthetic"):
        DatasetFactory.create("fuel_flow_synthetc")


def test_names_are_case_insensitive() -> None:
    assert isinstance(
        DatasetFactory.create(" Fuel_Flow_Synthetic "), SyntheticFuelFlowLoader
    )


def test_register_new_loader_without_touching_factory(scratch_name: str) -> None:
    @DatasetFactory.register(scratch_name)
    class ConstantLoader(DatasetLoader):
        def __init__(self, n: int = 3) -> None:
            self.n = n

        def load(self) -> Dataset:
            return Dataset(np.ones((self.n, 1)), np.zeros(self.n), ("one",), "zero")

    assert ConstantLoader.registry_name == scratch_name
    assert len(DatasetFactory.load(scratch_name, n=4)) == 4


def test_register_rejects_duplicates_and_non_loaders(scratch_name: str) -> None:
    with pytest.raises(ValueError, match="already registered"):
        DatasetFactory.register("csv")(SyntheticFuelFlowLoader)
    with pytest.raises(TypeError):
        DatasetFactory.register(scratch_name)(dict)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DatasetFactory.register("   ")


# ---------------------------------------------------------- concrete loaders
def test_isa_density_sea_level_and_5000_ft() -> None:
    assert isa_density(np.array(0.0)) == pytest.approx(1.225)
    # Standard table value at 1524 m (5000 ft): 1.0556 kg/m^3.
    assert isa_density(np.array(1524.0)) == pytest.approx(1.0556, rel=1e-3)


def test_fuel_model_hand_calculation() -> None:
    """V = 25 m/s at sea level, level flight, 30 kg (worked example in the notes)."""
    fuel = FuelFlowAircraft().fuel_flow_gph(25.0, 0.0, 0.0, 30.0)
    # P_par = 430.66 W, P_ind = 149.94 W, shaft = 774.13 W, fuel = 270.95 g/h.
    assert float(fuel) == pytest.approx(270.95, abs=0.01)


def test_fuel_model_has_idle_floor() -> None:
    fuel = FuelFlowAircraft().fuel_flow_gph(25.0, 0.0, -5.0, 30.0)
    assert float(fuel) == pytest.approx(0.35 * 150.0)


def test_synthetic_loader_is_reproducible_and_in_range() -> None:
    a = DatasetFactory.load("fuel_flow_synthetic", n_samples=50, seed=7)
    b = DatasetFactory.load("fuel_flow_synthetic", n_samples=50, seed=7)
    np.testing.assert_array_equal(a.X, b.X)
    assert a.feature_names == FUEL_FEATURES and np.all(a.y > 0)
    lo, hi = SyntheticFuelFlowLoader.AIRSPEED_RANGE_MPS
    assert lo <= a.X[:, 0].min() and a.X[:, 0].max() <= hi


@pytest.mark.parametrize(("n", "noise"), [(1, 0.0), (10, -0.1)])
def test_synthetic_loader_validates_arguments(n: int, noise: float) -> None:
    with pytest.raises(ValueError):
        SyntheticFuelFlowLoader(n_samples=n, noise_fraction=noise)


def test_csv_loader_drops_incomplete_rows(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    pd.DataFrame({"x": [1.0, 2.0, None, 4.0], "y": [2.0, 4.0, 6.0, None]}).to_csv(
        path, index=False
    )
    ds = DatasetFactory.load("csv", path=path, features=["x"], target="y")
    assert len(ds) == 2 and ds.metadata["rows_dropped"] == 2


def test_csv_loader_reports_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    pd.DataFrame({"x": [1.0]}).to_csv(path, index=False)
    with pytest.raises(KeyError, match="nope"):
        CsvTableLoader(path, ["x"], "nope").load()
    with pytest.raises(ValueError):
        CsvTableLoader(path, ["x"], "x")


def test_flight_log_loader_adds_load_factor(tmp_path: Path) -> None:
    path = SyntheticFlightLogGenerator(FlightProfileConfig()).to_csv(
        tmp_path / "log.csv"
    )
    loader = DatasetFactory.create(
        "flight_log_csv", path=path, features=["airspeed_mps"], target="load_factor"
    )
    assert isinstance(loader, FlightLogCsvLoader)
    ds = loader.load()
    assert ds.target_name == "load_factor"
    assert 0.5 < float(np.median(ds.y)) < 1.5


def test_csv_loader_runs_week3_cleaning_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    pd.DataFrame({"x": [1.0, np.inf, 3.0], "y": [2.0, 4.0, 6.0]}).to_csv(
        path, index=False
    )
    assert np.isinf(CsvTableLoader(path, ["x"], "y").load().X).any()  # dropna keeps inf
    cleaned = CsvTableLoader(path, ["x"], "y", cleaning=DataPipeline([DropNonFinite()]))
    np.testing.assert_array_equal(cleaned.load().y, [2.0, 6.0])
