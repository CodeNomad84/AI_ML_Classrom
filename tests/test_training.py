"""Unit tests for aerospace_ai.training (Week 4, Observer). Provenance: new content."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest

from aerospace_ai.models import GradientDescentRegressor, MeanBaseline
from aerospace_ai.training import (
    Callback,
    CallbackList,
    DivergenceGuard,
    EarlyStopping,
    History,
    ProgressLogger,
    StepDecay,
    Trainer,
    TrainingState,
)


class SpyCallback(Callback):
    """Test double: records every event it receives, in order."""

    def __init__(self, name: str = "spy", log: list[str] | None = None) -> None:
        self.name = name
        self.events: list[str] = [] if log is None else log

    def on_train_begin(self, state: TrainingState) -> None:
        self.events.append(f"{self.name}:begin")

    def on_epoch_end(self, state: TrainingState) -> None:
        self.events.append(f"{self.name}:epoch{state.epoch}")

    def on_train_end(self, state: TrainingState) -> None:
        self.events.append(f"{self.name}:end")


class ScriptedModel:
    """Test double: returns a fixed sequence of validation losses."""

    def __init__(self, val_losses: list[float]) -> None:
        self.learning_rate = 0.1
        self._val = iter(val_losses)
        self.params = np.array([-1.0])
        self.restored: np.ndarray | None = None

    def initialize(self, X: np.ndarray, y: np.ndarray) -> None:
        self.params = np.array([-1.0])

    def train_step(
        self, X: np.ndarray, y: np.ndarray, learning_rate: float | None = None
    ) -> float:
        self.params = self.params + 1.0  # params = epoch index after the step
        return 1.0

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return next(self._val)

    def get_state(self) -> np.ndarray:
        return self.params.copy()

    def set_state(self, state: np.ndarray) -> None:
        self.params = state.copy()


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    """y = 1 + 2 x with 40 samples."""
    X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    return X, 1.0 + 2.0 * X[:, 0]


DUMMY = (np.zeros((2, 1)), np.zeros(2))


# ---------------------------------------------------------------- subject
def test_events_arrive_in_order_for_every_observer(data: tuple) -> None:
    log: list[str] = []
    Trainer([SpyCallback("a", log), SpyCallback("b", log)]).fit(
        GradientDescentRegressor(), *data, max_epochs=2
    )
    assert log == [
        "a:begin",
        "b:begin",
        "a:epoch0",
        "b:epoch0",
        "a:epoch1",
        "b:epoch1",
        "a:end",
        "b:end",
    ]


def test_attach_detach() -> None:
    spy = SpyCallback()
    callbacks = CallbackList([spy])
    with pytest.raises(ValueError):
        callbacks.attach(spy)
    callbacks.detach(spy)
    assert len(callbacks) == 0
    with pytest.raises(ValueError):
        callbacks.detach(spy)
    with pytest.raises(TypeError):
        callbacks.attach(print)  # type: ignore[arg-type]


def test_trainer_rejects_untrainable_model_and_bad_args(data: tuple) -> None:
    with pytest.raises(TypeError):
        Trainer().fit(MeanBaseline(), *data)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Trainer().fit(GradientDescentRegressor(), *data, max_epochs=0)
    with pytest.raises(ValueError):
        Trainer().fit(GradientDescentRegressor(), *data, X_val=data[0])


def test_training_reduces_loss(data: tuple) -> None:
    history = History()
    Trainer([history]).fit(
        GradientDescentRegressor(learning_rate=0.1), *data, max_epochs=50
    )
    assert len(history) == 50
    assert history.train_loss[-1] < 1e-3 < history.train_loss[0]
    assert all(math.isnan(v) for v in history.val_loss)


# ---------------------------------------------------------------- observers
def test_early_stopping_hand_trace() -> None:
    """In-class Problem 2: patience 2, min_delta 0.05."""
    losses = [1.00, 0.80, 0.78, 0.70, 0.69, 0.68, 0.50]
    model = ScriptedModel(losses)
    stopper = EarlyStopping(patience=2, min_delta=0.05)
    state = Trainer([stopper]).fit(model, *DUMMY, *DUMMY, max_epochs=len(losses))
    # Improvements at epochs 0, 1, 3; epochs 4 and 5 fail -> stop after epoch 5.
    assert stopper.best_epoch == 3 and stopper.best == pytest.approx(0.70)
    assert stopper.stopped_epoch == 5 and state.epoch == 5 and state.stop_training
    np.testing.assert_array_equal(model.params, [3.0])  # best weights restored


def test_early_stopping_without_restore_keeps_last_weights() -> None:
    model = ScriptedModel([1.0, 1.0, 1.0])
    Trainer([EarlyStopping(patience=2, restore_best=False)]).fit(
        model, *DUMMY, *DUMMY, max_epochs=3
    )
    np.testing.assert_array_equal(model.params, [2.0])


def test_early_stopping_needs_validation_data(data: tuple) -> None:
    with pytest.raises(ValueError, match="val_loss is NaN"):
        Trainer([EarlyStopping()]).fit(GradientDescentRegressor(), *data, max_epochs=3)


def test_early_stopping_is_reusable() -> None:
    stopper = EarlyStopping(patience=1)
    Trainer([stopper]).fit(ScriptedModel([1.0, 2.0]), *DUMMY, *DUMMY, max_epochs=2)
    Trainer([stopper]).fit(ScriptedModel([5.0, 4.0]), *DUMMY, *DUMMY, max_epochs=2)
    assert stopper.best == 4.0 and stopper.stopped_epoch is None


def test_divergence_guard_stops_large_learning_rate(data: tuple) -> None:
    guard, history = DivergenceGuard(factor=10.0), History()
    state = Trainer([history, guard]).fit(
        GradientDescentRegressor(learning_rate=1.5), *data, max_epochs=100
    )
    assert guard.triggered and len(history) < 100
    assert "diverged" in state.stop_reason


def test_step_decay_halves_rate(data: tuple) -> None:
    history = History()
    Trainer([history, StepDecay(step_size=2, gamma=0.5)]).fit(
        GradientDescentRegressor(learning_rate=0.4), *data, max_epochs=5
    )
    assert history.learning_rate == pytest.approx([0.4, 0.4, 0.2, 0.2, 0.1])


@pytest.mark.parametrize(("every", "lines"), [(1, 4 + 1), (2, 3 + 1), (10, 1 + 1)])
def test_progress_logger_output(data: tuple, every: int, lines: int) -> None:
    stream = io.StringIO()
    Trainer([ProgressLogger(every=every, stream=stream)]).fit(
        GradientDescentRegressor(), *data, max_epochs=4
    )
    out = stream.getvalue().splitlines()
    assert len(out) == lines and out[-1].startswith("stopped after 4 epochs")


def test_progress_logger_defaults_to_stdout(
    data: tuple, capsys: pytest.CaptureFixture[str]
) -> None:
    Trainer([ProgressLogger(every=1)]).fit(
        GradientDescentRegressor(), *data, max_epochs=1
    )
    assert "epoch    1/1" in capsys.readouterr().out


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EarlyStopping(patience=0),
        lambda: EarlyStopping(min_delta=-1.0),
        lambda: DivergenceGuard(factor=1.0),
        lambda: StepDecay(step_size=0),
        lambda: StepDecay(step_size=1, gamma=1.5),
        lambda: ProgressLogger(every=0),
    ],
)
def test_callbacks_validate_configuration(factory) -> None:  # noqa: ANN001
    with pytest.raises(ValueError):
        factory()
