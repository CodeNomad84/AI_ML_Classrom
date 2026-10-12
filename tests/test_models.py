"""Unit tests for aerospace_ai.models (Week 4, Strategy). Provenance: new content.

The first block is a *contract test*: one parametrized test suite that every
strategy must pass. Adding a new model to ``STRATEGIES`` tests it for free.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from aerospace_ai.datasets import Dataset
from aerospace_ai.models import (
    GradientDescentRegressor,
    KNearestNeighbors,
    LinearLeastSquares,
    MeanBaseline,
    NotFittedError,
    RegressionExperiment,
    Regressor,
    SklearnRegressor,
    mae,
    r2_score,
    rmse,
)

STRATEGIES: dict[str, Callable[[], Regressor]] = {
    "mean": MeanBaseline,
    "ols": LinearLeastSquares,
    "ridge": lambda: LinearLeastSquares(alpha=1.0),
    "knn": lambda: KNearestNeighbors(k=3),
    "gd": lambda: GradientDescentRegressor(learning_rate=0.1, epochs=500),
}


@pytest.fixture
def line_data() -> tuple[np.ndarray, np.ndarray]:
    """y = 2 + 3 x1 - x2, no noise, 30 samples."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(30, 2))
    return X, 2.0 + 3.0 * X[:, 0] - X[:, 1]


@pytest.fixture(params=sorted(STRATEGIES), ids=sorted(STRATEGIES))
def strategy(request: pytest.FixtureRequest) -> Regressor:
    """Every registered strategy, one test run each."""
    return STRATEGIES[request.param]()


# ------------------------------------------------------------ contract tests
def test_contract_fit_returns_self_and_predicts_right_shape(
    strategy: Regressor, line_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = line_data
    assert strategy.fit(X, y) is strategy
    assert strategy.is_fitted and strategy.n_features == 2
    assert strategy.predict(X[:7]).shape == (7,)


def test_contract_predict_before_fit_raises(strategy: Regressor) -> None:
    with pytest.raises(NotFittedError):
        strategy.predict(np.zeros((2, 2)))


def test_contract_wrong_feature_count_raises(
    strategy: Regressor, line_data: tuple[np.ndarray, np.ndarray]
) -> None:
    strategy.fit(*line_data)
    with pytest.raises(ValueError, match="expected 2 features"):
        strategy.predict(np.zeros((3, 5)))


@pytest.mark.parametrize(
    ("X", "y"),
    [
        (np.zeros((0, 2)), np.zeros(0)),
        (np.zeros((3, 2)), np.zeros(4)),
        (np.array([[0.0, np.nan], [1.0, 1.0], [2.0, 2.0]]), np.zeros(3)),
    ],
    ids=["empty", "mismatch", "nan"],
)
def test_contract_bad_training_data_raises(
    strategy: Regressor, X: np.ndarray, y: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        strategy.fit(X, y)


def test_contract_beats_or_equals_baseline(
    strategy: Regressor, line_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = line_data
    baseline = MeanBaseline().fit(X, y).score(X, y)
    assert strategy.fit(X, y).score(X, y) <= baseline + 1e-9


# ------------------------------------------------------- strategy specifics
def test_metrics_hand_values() -> None:
    y, p = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 6.0])
    assert rmse(y, p) == pytest.approx(np.sqrt(3.0))
    assert mae(y, p) == pytest.approx(1.0)
    assert r2_score(y, p) == pytest.approx(1.0 - 9.0 / 2.0)
    assert r2_score(np.ones(3), np.ones(3)) == 0.0


def test_mean_baseline_hand_value() -> None:
    model = MeanBaseline().fit(np.zeros((4, 1)), [1.0, 2.0, 3.0, 6.0])
    np.testing.assert_allclose(model.predict(np.zeros((2, 1))), [3.0, 3.0])


def test_least_squares_recovers_exact_line(
    line_data: tuple[np.ndarray, np.ndarray],
) -> None:
    model = LinearLeastSquares().fit(*line_data)
    assert model.intercept_ == pytest.approx(2.0)
    np.testing.assert_allclose(model.coef_, [3.0, -1.0], atol=1e-10)


def test_ridge_shrinks_slopes(line_data: tuple[np.ndarray, np.ndarray]) -> None:
    ols = LinearLeastSquares().fit(*line_data)
    ridge = LinearLeastSquares(alpha=10.0).fit(*line_data)
    assert np.linalg.norm(ridge.coef_) < np.linalg.norm(ols.coef_)


def test_knn_hand_example() -> None:
    """In-class Problem 3: 1-D, k = 2, no scaling."""
    X = np.array([[18.0], [22.0], [26.0], [30.0]])
    y = np.array([170.0, 200.0, 280.0, 400.0])
    model = KNearestNeighbors(k=2, standardize=False).fit(X, y)
    np.testing.assert_allclose(model.predict([[25.0], [29.0]]), [240.0, 340.0])


def test_knn_validates_k() -> None:
    with pytest.raises(ValueError):
        KNearestNeighbors(k=0)
    with pytest.raises(ValueError, match="exceeds"):
        KNearestNeighbors(k=5).fit(np.zeros((3, 1)), np.zeros(3))


def test_gradient_descent_one_step_by_hand() -> None:
    """In-class Problem 1: three points, no scaling, eta = 0.1."""
    model = GradientDescentRegressor(learning_rate=0.1, epochs=1, standardize=False)
    X, y = np.array([[1.0], [2.0], [3.0]]), np.array([2.0, 4.0, 5.0])
    model.initialize(X, y)
    assert model.loss(X, y) == pytest.approx(15.0)
    np.testing.assert_allclose(model.gradient(X, y), [-22.0 / 3.0, -50.0 / 3.0])
    model.train_step(X, y)
    np.testing.assert_allclose(model.weights_, [22.0 / 30.0, 50.0 / 30.0])


def test_gradient_descent_converges_to_least_squares(
    line_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = line_data
    gd = GradientDescentRegressor(learning_rate=0.2, epochs=2000).fit(X, y)
    np.testing.assert_allclose(
        gd.predict(X), LinearLeastSquares().fit(X, y).predict(X), atol=1e-6
    )


def test_repr_shows_hyperparameters() -> None:
    assert repr(KNearestNeighbors(k=7)) == "KNearestNeighbors(k=7)"
    assert repr(LinearLeastSquares(alpha=0.5)) == "LinearLeastSquares(alpha=0.5)"


# ------------------------------------------------------------------- adapter
class FakeEstimator:
    """Test double with a scikit-learn style API (no sklearn needed)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> FakeEstimator:
        self.value = float(np.max(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), 1), self.value)  # 2-D on purpose


def test_adapter_wraps_duck_typed_estimator() -> None:
    model = SklearnRegressor(FakeEstimator()).fit(np.zeros((3, 1)), [1.0, 5.0, 2.0])
    np.testing.assert_allclose(model.predict(np.zeros((2, 1))), [5.0, 5.0])


def test_adapter_rejects_objects_without_fit_predict() -> None:
    with pytest.raises(TypeError):
        SklearnRegressor(object())  # type: ignore[arg-type]


def test_adapter_with_real_sklearn(line_data: tuple[np.ndarray, np.ndarray]) -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")
    model = SklearnRegressor(linear_model.LinearRegression()).fit(*line_data)
    assert model.score(*line_data) == pytest.approx(0.0, abs=1e-10)


# ------------------------------------------------------------------- context
@pytest.fixture
def split(line_data: tuple[np.ndarray, np.ndarray]) -> tuple[Dataset, Dataset]:
    X, y = line_data
    return Dataset(X, y, ("x1", "x2"), "y", "line").split(0.3)


def test_experiment_swaps_strategy_at_run_time(split: tuple[Dataset, Dataset]) -> None:
    exp = RegressionExperiment(MeanBaseline())
    worse = exp.run(*split)
    exp.strategy = LinearLeastSquares()
    better = exp.run(*split)
    assert better.test_rmse < worse.test_rmse and better.dataset == "line"


def test_experiment_rejects_non_strategy() -> None:
    with pytest.raises(TypeError):
        RegressionExperiment("linear")  # type: ignore[arg-type]


def test_compare_sorts_best_first(split: tuple[Dataset, Dataset]) -> None:
    results = RegressionExperiment(MeanBaseline()).compare(
        [MeanBaseline(), LinearLeastSquares(), KNearestNeighbors(k=3)], *split
    )
    assert results[0].model.startswith("LinearLeastSquares")
    assert [r.test_rmse for r in results] == sorted(r.test_rmse for r in results)
