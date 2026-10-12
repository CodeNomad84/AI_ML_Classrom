"""Regression models as interchangeable strategies.

Provenance: new content (Week 4). The Strategy pattern is rewritten from
GeeksforGeeks ("Strategy Design Pattern") and Gamma et al., *Design Patterns*
(1994), Ch. 5; Python polymorphism follows the W3Schools tutorial. Least
squares and k-nearest neighbours are adapted from Bishop, *Pattern Recognition
and Machine Learning* (2006), Ch. 3 and Sec. 2.5.2. The ``fit``/``predict``
interface deliberately mirrors scikit-learn's estimator API.

Design:

* :class:`Regressor` is the *strategy interface* (an ABC): ``fit`` learns from
  data, ``predict`` answers. Concrete strategies: :class:`MeanBaseline`,
  :class:`LinearLeastSquares`, :class:`KNearestNeighbors` and
  :class:`GradientDescentRegressor`.
* :class:`SklearnRegressor` is an *adapter*: it lets any object that has
  ``fit``/``predict`` (a scikit-learn estimator, typed by the
  :class:`SupportsFitPredict` protocol) act as a strategy.
* :class:`RegressionExperiment` is the *context*: it holds one strategy,
  delegates to it, and can swap it at run time without changing a line of its
  own code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Protocol, runtime_checkable

import numpy as np

from aerospace_ai.datasets import Dataset
from aerospace_ai.pipeline import NotFittedError

DEFAULT_LEARNING_RATE: Final[float] = 0.1
DEFAULT_EPOCHS: Final[int] = 200
DEFAULT_NEIGHBORS: Final[int] = 5
MIN_STD: Final[float] = 1e-12


# ------------------------------------------------------------------ metrics
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error, in the units of the target."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error, in the units of the target."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot.

    Returns 0.0 when the target is constant (SS_tot = 0) and the prediction
    is perfect, and -inf when the target is constant but the prediction is not.
    """
    y_true = np.asarray(y_true, dtype=float)
    ss_res = float(np.sum((y_true - np.asarray(y_pred)) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0 if ss_res == 0.0 else float("-inf")
    return 1.0 - ss_res / ss_tot


def as_2d(X: np.ndarray) -> np.ndarray:
    """Return ``X`` as a 2-D float array (a 1-D input becomes one column)."""
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


# --------------------------------------------------------- strategy interface
class Regressor(ABC):
    """Strategy interface: every regression model offers fit and predict.

    Subclasses implement :meth:`_fit` and :meth:`_predict`; the public
    :meth:`fit` and :meth:`predict` (template methods, as in Week 2) do the
    input checks once for everybody.
    """

    name: ClassVar[str] = "regressor"

    def __init__(self) -> None:
        """Start unfitted."""
        self._n_features: int | None = None

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self._n_features is not None

    @property
    def n_features(self) -> int:
        """Number of features seen during fit.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if self._n_features is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted yet")
        return self._n_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> Regressor:
        """Learn from training data.

        Args:
            X: Features, shape (n_samples, n_features) (1-D means one feature).
            y: Targets, shape (n_samples,).

        Returns:
            ``self``, so calls can be chained: ``model.fit(X, y).predict(X)``.

        Raises:
            ValueError: On empty data, mismatched lengths or non-finite values.
        """
        X, y = as_2d(X), np.asarray(y, dtype=float).ravel()
        if X.shape[0] == 0:
            raise ValueError("cannot fit on zero samples")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            raise ValueError("X and y must be finite (clean the data first)")
        self._fit(X, y)
        self._n_features = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for new samples.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            ValueError: If the number of features differs from training.
        """
        X = as_2d(X)
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        return self._predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the RMSE of the model on ``(X, y)`` (lower is better)."""
        return rmse(y, self.predict(X))

    def hyperparameters(self) -> dict[str, object]:
        """Return the settings chosen by the user (overridden by subclasses)."""
        return {}

    def __repr__(self) -> str:
        """Return the class name and settings, e.g. ``KNearestNeighbors(k=5)``."""
        args = ", ".join(f"{k}={v!r}" for k, v in self.hyperparameters().items())
        return f"{type(self).__name__}({args})"

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Model-specific learning (inputs already checked)."""

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Model-specific prediction (inputs already checked)."""


# ------------------------------------------------------- concrete strategies
class MeanBaseline(Regressor):
    """Always predict the training mean. Every real model must beat this."""

    name: ClassVar[str] = "mean_baseline"

    def __init__(self) -> None:
        """Start unfitted."""
        super().__init__()
        self.mean_: float = 0.0

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean_ = float(y.mean())

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.mean_)


class LinearLeastSquares(Regressor):
    """Linear model y = w0 + w . x fitted in closed form (normal equations).

    With ``alpha > 0`` this is ridge regression: the penalty alpha * |w|^2 is
    added to the squared error (the intercept is not penalised).
    """

    name: ClassVar[str] = "linear_least_squares"

    def __init__(self, alpha: float = 0.0) -> None:
        """Configure the model.

        Args:
            alpha: Ridge penalty strength (0 = ordinary least squares).

        Raises:
            ValueError: If ``alpha`` is negative.
        """
        super().__init__()
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.alpha = alpha
        self.intercept_: float = 0.0
        self.coef_: np.ndarray = np.empty(0)

    def hyperparameters(self) -> dict[str, object]:
        """Return ``{"alpha": ...}``."""
        return {"alpha": self.alpha}

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Centring removes the intercept from the linear system, so the
        # ridge penalty acts on the slopes only.
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        gram = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        # lstsq instead of inv: stable even when the Gram matrix is singular.
        self.coef_ = np.linalg.lstsq(gram, Xc.T @ yc, rcond=None)[0]
        self.intercept_ = float(y_mean - x_mean @ self.coef_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.intercept_ + X @ self.coef_


class KNearestNeighbors(Regressor):
    """Predict the mean target of the k closest training samples.

    Features are standardised with the training mean and standard deviation
    first, otherwise altitude (thousands of metres) would dominate airspeed
    (tens of m/s) in the Euclidean distance.
    """

    name: ClassVar[str] = "k_nearest_neighbors"

    def __init__(self, k: int = DEFAULT_NEIGHBORS, standardize: bool = True) -> None:
        """Configure the model.

        Args:
            k: Number of neighbours averaged.
            standardize: Scale features to zero mean, unit variance.

        Raises:
            ValueError: If ``k`` < 1.
        """
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = k
        self.standardize = standardize
        self._X = np.empty((0, 0))
        self._y = np.empty(0)
        self._mean = np.empty(0)
        self._std = np.empty(0)

    def hyperparameters(self) -> dict[str, object]:
        """Return ``{"k": ...}`` (plus ``standardize`` when it is switched off)."""
        params: dict[str, object] = {"k": self.k}
        if not self.standardize:
            params["standardize"] = False
        return params

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std if self.standardize else X

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.k > X.shape[0]:
            raise ValueError(f"k = {self.k} exceeds the {X.shape[0]} training samples")
        self._mean = X.mean(axis=0)
        self._std = np.maximum(X.std(axis=0), MIN_STD)
        self._X, self._y = self._scale(X), y.copy()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scale(X)
        # Squared distances from every query (rows) to every training sample.
        d2 = ((Xs[:, None, :] - self._X[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argpartition(d2, self.k - 1, axis=1)[:, : self.k]
        return self._y[nearest].mean(axis=1)


class GradientDescentRegressor(Regressor):
    """Linear model trained iteratively by full-batch gradient descent on MSE.

    Unlike :class:`LinearLeastSquares` it learns step by step, which makes it
    *trainable*: :class:`aerospace_ai.training.Trainer` can drive it one epoch
    at a time and notify callbacks after each epoch. Features are standardised
    internally so that one learning rate suits all of them.
    """

    name: ClassVar[str] = "gradient_descent"

    def __init__(
        self,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        epochs: int = DEFAULT_EPOCHS,
        standardize: bool = True,
    ) -> None:
        """Configure the model.

        Args:
            learning_rate: Step size eta.
            epochs: Number of steps used by :meth:`fit`.
            standardize: Scale features to zero mean, unit variance.

        Raises:
            ValueError: If ``learning_rate`` <= 0 or ``epochs`` < 1.
        """
        super().__init__()
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {epochs}")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.standardize = standardize
        self.weights_ = np.empty(0)  # weights_[0] is the bias
        self._mean = np.empty(0)
        self._std = np.empty(0)

    def hyperparameters(self) -> dict[str, object]:
        """Return the learning rate and number of epochs."""
        return {"learning_rate": self.learning_rate, "epochs": self.epochs}

    # ---- the "trainable" protocol used by Trainer
    def initialize(self, X: np.ndarray, y: np.ndarray) -> None:
        """Set the scaling from the training data and zero the weights."""
        X = as_2d(X)
        if self.standardize:
            self._mean = X.mean(axis=0)
            self._std = np.maximum(X.std(axis=0), MIN_STD)
        else:
            self._mean = np.zeros(X.shape[1])
            self._std = np.ones(X.shape[1])
        self.weights_ = np.zeros(X.shape[1] + 1)
        self._n_features = X.shape[1]

    def _design(self, X: np.ndarray) -> np.ndarray:
        """Return [1, scaled X] so that the bias is weights_[0]."""
        Xs = (as_2d(X) - self._mean) / self._std
        return np.column_stack([np.ones(Xs.shape[0]), Xs])

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean squared error of the current weights."""
        residual = self._design(X) @ self.weights_ - np.asarray(y, dtype=float)
        return float(np.mean(residual**2))

    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient of the MSE: (2 / n) A^T (A w - y)."""
        A = self._design(X)
        return 2.0 / A.shape[0] * A.T @ (A @ self.weights_ - np.asarray(y, dtype=float))

    def train_step(
        self, X: np.ndarray, y: np.ndarray, learning_rate: float | None = None
    ) -> float:
        """Take one gradient step and return the training loss after it.

        Args:
            X: Training features.
            y: Training targets.
            learning_rate: Step size for this step (default: the model's).

        Returns:
            Training MSE after the step.
        """
        eta = self.learning_rate if learning_rate is None else learning_rate
        self.weights_ = self.weights_ - eta * self.gradient(X, y)
        return self.loss(X, y)

    def get_state(self) -> np.ndarray:
        """Return a copy of the learnable parameters (for checkpoints)."""
        return self.weights_.copy()

    def set_state(self, state: np.ndarray) -> None:
        """Restore parameters saved by :meth:`get_state`."""
        self.weights_ = np.asarray(state, dtype=float).copy()

    # ---- Regressor hooks
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.initialize(X, y)
        for _ in range(self.epochs):
            self.train_step(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._design(X) @ self.weights_


# ------------------------------------------------------------------- adapter
@runtime_checkable
class SupportsFitPredict(Protocol):
    """Structural type: anything with scikit-learn style fit and predict."""

    def fit(self, X: Any, y: Any) -> Any:  # pragma: no cover - protocol
        """Learn from data."""

    def predict(self, X: Any) -> Any:  # pragma: no cover - protocol
        """Return predictions."""


class SklearnRegressor(Regressor):
    """Adapter: wrap any fit/predict estimator so it acts as a :class:`Regressor`.

    Example:
        >>> from sklearn.linear_model import Ridge  # doctest: +SKIP
        >>> model = SklearnRegressor(Ridge(alpha=1.0))  # doctest: +SKIP
    """

    name: ClassVar[str] = "sklearn"

    def __init__(self, estimator: SupportsFitPredict) -> None:
        """Wrap an estimator.

        Raises:
            TypeError: If the object has no ``fit``/``predict`` methods.
        """
        super().__init__()
        if not isinstance(estimator, SupportsFitPredict):
            raise TypeError(f"{type(estimator).__name__} has no fit/predict")
        self.estimator = estimator

    def __repr__(self) -> str:
        """Show the wrapped estimator."""
        return f"SklearnRegressor({self.estimator!r})"

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.estimator.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float).ravel()


# -------------------------------------------------------------------- context
@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Scores of one model on one train/test split.

    Attributes:
        model: Model description (``repr`` of the strategy).
        dataset: Dataset name.
        train_rmse: RMSE on the training set.
        test_rmse: RMSE on the test set.
        test_mae: MAE on the test set.
        test_r2: R^2 on the test set.
    """

    model: str
    dataset: str
    train_rmse: float
    test_rmse: float
    test_mae: float
    test_r2: float


class RegressionExperiment:
    """Context of the Strategy pattern: evaluate whatever model it is given.

    The experiment never asks *which* model it holds; it only calls the
    :class:`Regressor` interface. Swapping models is one assignment.

    Example:
        >>> exp = RegressionExperiment(MeanBaseline())
        >>> exp.strategy = KNearestNeighbors(k=3)   # swap at run time
    """

    def __init__(self, strategy: Regressor) -> None:
        """Store the initial strategy."""
        self.strategy = strategy

    @property
    def strategy(self) -> Regressor:
        """The model currently used."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: Regressor) -> None:
        if not isinstance(value, Regressor):
            raise TypeError(f"strategy must be a Regressor, got {type(value).__name__}")
        self._strategy = value

    def run(self, train: Dataset, test: Dataset) -> EvaluationResult:
        """Fit on ``train`` and score on both sets.

        Raises:
            ValueError: If the two sets have different features.
        """
        if train.feature_names != test.feature_names:
            raise ValueError("train and test sets have different features")
        model = self.strategy.fit(train.X, train.y)
        test_pred = model.predict(test.X)
        return EvaluationResult(
            model=repr(model),
            dataset=train.name.split("/")[0],
            train_rmse=model.score(train.X, train.y),
            test_rmse=rmse(test.y, test_pred),
            test_mae=mae(test.y, test_pred),
            test_r2=r2_score(test.y, test_pred),
        )

    def compare(
        self, strategies: list[Regressor], train: Dataset, test: Dataset
    ) -> list[EvaluationResult]:
        """Run every strategy on the same split; best (lowest test RMSE) first."""
        results = []
        for strategy in strategies:
            self.strategy = strategy
            results.append(self.run(train, test))
        return sorted(results, key=lambda r: r.test_rmse)
