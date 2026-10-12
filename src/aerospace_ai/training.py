"""Training loop with callbacks: the Observer pattern applied to learning.

Provenance: new content (Week 4). The Observer pattern is rewritten from
GeeksforGeeks ("Observer Design Pattern") and Gamma et al., *Design Patterns*
(1994), Ch. 5. The callback hooks follow the design popularised by Keras and
PyTorch Lightning; early stopping is adapted from Prechelt, "Early Stopping -
But When?" (1998) and Goodfellow et al., *Deep Learning* (2016), Sec. 7.8.

Design:

* :class:`Trainer` is the *subject*. It runs the epoch loop and, at fixed
  moments (train begin, epoch end, train end), notifies every attached
  :class:`Callback`. The loop itself knows nothing about logging, stopping
  rules or learning-rate schedules.
* :class:`Callback` is the *observer* base class. Its hooks do nothing by
  default, so an observer overrides only the events it cares about.
* Observers talk back only through the shared :class:`TrainingState` (for
  example ``state.stop_training = True``), never by calling into each other.

The model must satisfy the :class:`TrainableModel` protocol;
:class:`aerospace_ai.models.GradientDescentRegressor` does. In Block 2 the
same Trainer idea drives PyTorch networks.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol, TextIO, runtime_checkable

import numpy as np

DEFAULT_PATIENCE: Final[int] = 5
DEFAULT_DIVERGENCE_FACTOR: Final[float] = 10.0
DEFAULT_LOG_EVERY: Final[int] = 10

Monitor = Literal["train_loss", "val_loss"]


@runtime_checkable
class TrainableModel(Protocol):
    """What the Trainer needs from a model (structural typing, no inheritance)."""

    learning_rate: float

    def initialize(self, X: np.ndarray, y: np.ndarray) -> None:
        """Prepare parameters for training on ``(X, y)``."""

    def train_step(
        self, X: np.ndarray, y: np.ndarray, learning_rate: float | None = None
    ) -> float:
        """Take one optimisation step; return the training loss."""

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the loss on ``(X, y)`` without changing the model."""

    def get_state(self) -> np.ndarray:
        """Return a copy of the parameters."""

    def set_state(self, state: np.ndarray) -> None:
        """Restore parameters returned by :meth:`get_state`."""


@dataclass(slots=True)
class TrainingState:
    """Everything an observer may read (and the few things it may change).

    Attributes:
        model: The model being trained.
        max_epochs: Epoch budget given to :meth:`Trainer.fit`.
        epoch: Index of the epoch that just finished (0-based; -1 before).
        learning_rate: Step size used for the next epoch (callbacks may set it).
        train_loss: Training loss after the latest epoch.
        val_loss: Validation loss after the latest epoch (NaN without val data).
        stop_training: Set to True by a callback to end training after this epoch.
        stop_reason: Human-readable reason recorded with ``stop_training``.
    """

    model: TrainableModel
    max_epochs: int
    epoch: int = -1
    learning_rate: float = 0.0
    train_loss: float = math.nan
    val_loss: float = math.nan
    stop_training: bool = False
    stop_reason: str = ""

    def monitored(self, name: Monitor) -> float:
        """Return the current value of ``"train_loss"`` or ``"val_loss"``."""
        return self.train_loss if name == "train_loss" else self.val_loss

    def request_stop(self, reason: str) -> None:
        """Ask the trainer to stop after the current epoch."""
        self.stop_training = True
        self.stop_reason = reason


# ------------------------------------------------------------------ observer
class Callback:
    """Observer base class. Override any subset of the three hooks.

    The hooks are deliberately *not* abstract: an early-stopping rule has no
    business at train begin, a logger may ignore train end. Forcing every
    observer to implement every hook would only produce empty methods.
    """

    def on_train_begin(self, state: TrainingState) -> None:
        """Called once, after the model is initialised and before epoch 0."""

    def on_epoch_end(self, state: TrainingState) -> None:
        """Called after every epoch, once the losses in ``state`` are updated."""

    def on_train_end(self, state: TrainingState) -> None:
        """Called once, after the last epoch (also after an early stop)."""

    def __repr__(self) -> str:
        """Return the class name."""
        return f"{type(self).__name__}()"


class CallbackList:
    """The subscriber list of the subject: attach, detach and notify.

    Observers are notified in the order they were attached. The order matters
    when one observer changes the state another reads (attach schedules
    before loggers if the logger should print the new learning rate).
    """

    def __init__(self, callbacks: Iterable[Callback] = ()) -> None:
        """Start with the given observers."""
        self._callbacks: list[Callback] = []
        for callback in callbacks:
            self.attach(callback)

    def attach(self, callback: Callback) -> None:
        """Subscribe an observer.

        Raises:
            TypeError: If ``callback`` is not a :class:`Callback`.
            ValueError: If the same object is already attached.
        """
        if not isinstance(callback, Callback):
            raise TypeError(f"expected a Callback, got {type(callback).__name__}")
        if any(c is callback for c in self._callbacks):
            raise ValueError(f"{callback!r} is already attached")
        self._callbacks.append(callback)

    def detach(self, callback: Callback) -> None:
        """Unsubscribe an observer.

        Raises:
            ValueError: If it was not attached.
        """
        for i, c in enumerate(self._callbacks):
            if c is callback:
                del self._callbacks[i]
                return
        raise ValueError(f"{callback!r} is not attached")

    def notify(self, event: str, state: TrainingState) -> None:
        """Call hook ``event`` (for example ``"on_epoch_end"``) on every observer."""
        for callback in self._callbacks:
            getattr(callback, event)(state)

    def __len__(self) -> int:
        """Return the number of observers."""
        return len(self._callbacks)

    def __iter__(self) -> Iterator[Callback]:
        """Iterate over the observers in notification order."""
        return iter(self._callbacks)


# ------------------------------------------------------------------- subject
class Trainer:
    """Subject of the Observer pattern: runs epochs and broadcasts events.

    Example:
        >>> from aerospace_ai.models import GradientDescentRegressor
        >>> history = History()
        >>> trainer = Trainer([history, EarlyStopping(patience=3)])
        >>> # trainer.fit(GradientDescentRegressor(), X, y, X_val, y_val)
    """

    def __init__(self, callbacks: Iterable[Callback] = ()) -> None:
        """Create a trainer with an initial list of observers."""
        self.callbacks = CallbackList(callbacks)

    def fit(
        self,
        model: TrainableModel,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        max_epochs: int = 100,
    ) -> TrainingState:
        """Train ``model`` for up to ``max_epochs`` epochs.

        Args:
            model: A model satisfying :class:`TrainableModel`.
            X: Training features.
            y: Training targets.
            X_val: Optional validation features.
            y_val: Optional validation targets (required with ``X_val``).
            max_epochs: Epoch budget.

        Returns:
            The final :class:`TrainingState`.

        Raises:
            TypeError: If the model is not trainable.
            ValueError: If ``max_epochs`` < 1 or only one of X_val/y_val is given.
        """
        if not isinstance(model, TrainableModel):
            raise TypeError(f"{type(model).__name__} does not satisfy TrainableModel")
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be at least 1, got {max_epochs}")
        if (X_val is None) != (y_val is None):
            raise ValueError("give both X_val and y_val, or neither")

        model.initialize(X, y)
        state = TrainingState(model, max_epochs, learning_rate=model.learning_rate)
        self.callbacks.notify("on_train_begin", state)
        for epoch in range(max_epochs):
            state.epoch = epoch
            state.train_loss = model.train_step(X, y, state.learning_rate)
            if X_val is not None and y_val is not None:
                state.val_loss = model.loss(X_val, y_val)
            self.callbacks.notify("on_epoch_end", state)
            if state.stop_training:
                break
        self.callbacks.notify("on_train_end", state)
        return state


# -------------------------------------------------------- concrete observers
@dataclass
class History(Callback):
    """Record the losses and learning rate of every epoch.

    Attributes:
        train_loss: One entry per epoch.
        val_loss: One entry per epoch (NaN without validation data).
        learning_rate: Learning rate used in each epoch.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)

    def on_train_begin(self, state: TrainingState) -> None:
        """Clear previous runs so one History can be reused."""
        self.train_loss.clear()
        self.val_loss.clear()
        self.learning_rate.clear()

    def on_epoch_end(self, state: TrainingState) -> None:
        """Append this epoch's numbers."""
        self.train_loss.append(state.train_loss)
        self.val_loss.append(state.val_loss)
        self.learning_rate.append(state.learning_rate)

    def __len__(self) -> int:
        """Return the number of recorded epochs."""
        return len(self.train_loss)

    def __repr__(self) -> str:
        """Return the number of recorded epochs."""
        return f"History(epochs={len(self)})"


class EarlyStopping(Callback):
    """Stop when the monitored loss has not improved for ``patience`` epochs.

    An epoch counts as an improvement only if the loss drops by more than
    ``min_delta`` below the best value so far. With ``restore_best`` the model
    gets back the parameters of its best epoch at train end.
    """

    def __init__(
        self,
        patience: int = DEFAULT_PATIENCE,
        min_delta: float = 0.0,
        monitor: Monitor = "val_loss",
        restore_best: bool = True,
    ) -> None:
        """Configure the rule.

        Raises:
            ValueError: If ``patience`` < 1 or ``min_delta`` < 0.
        """
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}")
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        self.patience = patience
        self.min_delta = min_delta
        self.monitor: Monitor = monitor
        self.restore_best = restore_best
        self.best = math.inf
        self.best_epoch = -1
        self.wait = 0
        self.stopped_epoch: int | None = None
        self._best_state: np.ndarray | None = None

    def __repr__(self) -> str:
        """Show the configuration."""
        return (
            f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, "
            f"monitor={self.monitor!r})"
        )

    def on_train_begin(self, state: TrainingState) -> None:
        """Reset the counters (the same object can serve several runs)."""
        self.best, self.best_epoch, self.wait = math.inf, -1, 0
        self.stopped_epoch, self._best_state = None, None

    def on_epoch_end(self, state: TrainingState) -> None:
        """Update the best value or the patience counter.

        Raises:
            ValueError: If the monitored loss is NaN because there is no
                validation data.
        """
        current = state.monitored(self.monitor)
        if math.isnan(current):
            raise ValueError(
                f"{self.monitor} is NaN: pass validation data or monitor train_loss"
            )
        if current < self.best - self.min_delta:
            self.best, self.best_epoch, self.wait = current, state.epoch, 0
            if self.restore_best:
                self._best_state = state.model.get_state()
            return
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = state.epoch
            state.request_stop(
                f"{self.monitor} did not improve for {self.patience} epochs "
                f"(best {self.best:.4g} at epoch {self.best_epoch})"
            )

    def on_train_end(self, state: TrainingState) -> None:
        """Restore the best parameters if asked to."""
        if self.restore_best and self._best_state is not None:
            state.model.set_state(self._best_state)


class DivergenceGuard(Callback):
    """Abort when the training loss becomes non-finite or explodes.

    The training equivalent of an engine over-temperature cut-off: a
    learning rate that is too large makes the loss grow without bound, and
    there is no point burning the remaining epochs.
    """

    def __init__(self, factor: float = DEFAULT_DIVERGENCE_FACTOR) -> None:
        """Stop if the loss exceeds ``factor`` times the first epoch's loss.

        Raises:
            ValueError: If ``factor`` <= 1.
        """
        if factor <= 1.0:
            raise ValueError(f"factor must be greater than 1, got {factor}")
        self.factor = factor
        self._reference = math.nan
        self.triggered = False

    def on_train_begin(self, state: TrainingState) -> None:
        """Forget the previous run."""
        self._reference, self.triggered = math.nan, False

    def on_epoch_end(self, state: TrainingState) -> None:
        """Compare the loss with the first epoch's value."""
        if state.epoch == 0:
            self._reference = state.train_loss
        if not math.isfinite(state.train_loss) or state.train_loss > (
            self.factor * self._reference
        ):
            self.triggered = True
            state.request_stop(
                f"training diverged at epoch {state.epoch} "
                f"(loss {state.train_loss:.4g}); lower the learning rate"
            )


class StepDecay(Callback):
    """Multiply the learning rate by ``gamma`` every ``step_size`` epochs."""

    def __init__(self, step_size: int, gamma: float = 0.5) -> None:
        """Configure the schedule.

        Raises:
            ValueError: If ``step_size`` < 1 or ``gamma`` is not in (0, 1].
        """
        if step_size < 1:
            raise ValueError(f"step_size must be at least 1, got {step_size}")
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self.step_size = step_size
        self.gamma = gamma

    def on_epoch_end(self, state: TrainingState) -> None:
        """Decay the rate used from the next epoch on."""
        if (state.epoch + 1) % self.step_size == 0:
            state.learning_rate *= self.gamma


class ProgressLogger(Callback):
    """Print a progress line every ``every`` epochs and a summary at the end."""

    def __init__(
        self, every: int = DEFAULT_LOG_EVERY, stream: TextIO | None = None
    ) -> None:
        """Configure the logger.

        Args:
            every: Print every this many epochs (the first epoch always prints).
            stream: Where to write; ``sys.stdout`` (looked up at print time) if None.

        Raises:
            ValueError: If ``every`` < 1.
        """
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")
        self.every = every
        self.stream = stream

    def _write(self, text: str) -> None:
        print(text, file=self.stream or sys.stdout)

    def on_epoch_end(self, state: TrainingState) -> None:
        """Print epoch, losses and learning rate."""
        if state.epoch == 0 or (state.epoch + 1) % self.every == 0:
            self._write(
                f"epoch {state.epoch + 1:4d}/{state.max_epochs}  "
                f"train {state.train_loss:10.4f}  val {state.val_loss:10.4f}  "
                f"lr {state.learning_rate:.4g}"
            )

    def on_train_end(self, state: TrainingState) -> None:
        """Print why and when training ended."""
        reason = state.stop_reason or "epoch budget used"
        self._write(f"stopped after {state.epoch + 1} epochs: {reason}")
