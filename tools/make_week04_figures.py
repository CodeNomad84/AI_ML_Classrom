"""Render the figures used by the Week 4 slides and lecture notes.

Provenance: new content (Week 4). Output: slides/assets/week04/*.png

Usage:
    python tools/make_week04_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerospace_ai.datasets import (  # noqa: E402
    DatasetFactory,
    FuelFlowAircraft,
    isa_density,
)
from aerospace_ai.models import (  # noqa: E402
    GradientDescentRegressor,
    KNearestNeighbors,
    LinearLeastSquares,
    MeanBaseline,
    RegressionExperiment,
)
from aerospace_ai.utils import STANDARD_GRAVITY_MPS2  # noqa: E402

OUT_DIR: Final[Path] = REPO_ROOT / "slides" / "assets" / "week04"
DPI: Final[int] = 150
REFERENCE_MASS_KG: Final[float] = 30.0
EARLY_STOP_LOSSES: Final[tuple[float, ...]] = (1.00, 0.80, 0.78, 0.70, 0.69, 0.68, 0.50)
EARLY_STOP_PATIENCE: Final[int] = 2
EARLY_STOP_MIN_DELTA: Final[float] = 0.05

NAVY: Final[str] = "#1f3a5f"
ACCENT: Final[str] = "#e07a1f"
GREEN: Final[str] = "#3a8f4b"
BASE_FILL: Final[str] = "#dbe6f3"
SUB_FILL: Final[str] = "#fbe3cc"
GREY_FILL: Final[str] = "#eeeeee"
OK_FILL: Final[str] = "#d6efd9"


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    text: str,
    fill: str,
    dashed: bool = False,
    fontsize: float = 10.0,
) -> None:
    """Draw a rounded box with centred, multi-line text."""
    ax.add_patch(
        FancyBboxPatch(
            xy,
            *size,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            fc=fill,
            ec=NAVY,
            lw=1.4,
            ls="--" if dashed else "-",
        )
    )
    ax.text(
        xy[0] + size[0] / 2,
        xy[1] + size[1] / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        family="DejaVu Sans Mono",
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    style: str = "-|>",
    color: str = NAVY,
    dashed: bool = False,
) -> None:
    """Draw an arrow from ``start`` to ``end``."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=16,
            color=color,
            lw=1.4,
            ls="--" if dashed else "-",
        )
    )


class Week04Figures:
    """Build every Week 4 figure."""

    def __init__(self, out_dir: Path) -> None:
        """Make sure the output folder exists."""
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig: plt.Figure, name: str) -> None:
        """Save ``fig`` as ``name`` in the output folder and close it."""
        fig.savefig(self.out_dir / name, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("wrote", (self.out_dir / name).relative_to(REPO_ROOT))

    def spaghetti_vs_patterns(self) -> None:
        """Grid: which parts of the code change when one thing is added."""
        parts = ["load data", "build model", "train loop", "report"]
        changes = ["+ model", "+ dataset", "+ callback"]
        touched_procedural = {
            "+ model": {"build model", "train loop", "report"},
            "+ dataset": {"load data", "report"},
            "+ callback": {"train loop", "report"},
        }
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        panels = (
            ("One run_experiment() with if/elif", True),
            ("Strategy + Factory + Observer", False),
        )
        for ax, (title, procedural) in zip(axes, panels):
            for i, part in enumerate(parts):
                for j, change in enumerate(changes):
                    edited = procedural and part in touched_procedural[change]
                    ax.add_patch(
                        plt.Rectangle(
                            (j, -i),
                            0.92,
                            0.85,
                            fc=ACCENT if edited else BASE_FILL,
                            ec="white",
                        )
                    )
                ax.text(
                    -0.1,
                    -i + 0.42,
                    part,
                    ha="right",
                    va="center",
                    family="DejaVu Sans Mono",
                )
            for j, change in enumerate(changes):
                ax.text(j + 0.46, 1.05, change, ha="center", va="bottom", fontsize=10)
            caption = (
                "7 edits in shared code"
                if procedural
                else "0 edits: add one new class each"
            )
            ax.text(
                1.4, -3.85, caption, ha="center", color=ACCENT if procedural else GREEN
            )
            ax.set_xlim(-1.9, 3.2)
            ax.set_ylim(-4.2, 1.6)
            ax.set_title(title, color=NAVY)
            ax.axis("off")
        self.save(fig, "fig_spaghetti_vs_patterns.png")

    def strategy_uml(self) -> None:
        """Context -> strategy interface <- concrete strategies (+ adapter)."""
        fig, ax = plt.subplots(figsize=(11, 5.4))
        box(
            ax,
            (0.2, 3.4),
            (3.2, 1.5),
            "RegressionExperiment\n(context)\nstrategy: Regressor\nrun(train, test)",
            GREY_FILL,
            fontsize=9,
        )
        box(
            ax,
            (4.6, 3.3),
            (3.6, 1.7),
            "<<abstract>> Regressor\n(strategy interface)\nfit(X, y) -> self\n"
            "predict(X)\nscore(X, y) = RMSE",
            BASE_FILL,
            fontsize=9,
        )
        arrow(ax, (3.4, 4.15), (4.6, 4.15))
        ax.text(4.0, 4.3, "uses", ha="center", fontsize=8.5)
        concretes = [
            "MeanBaseline\nmean of y",
            "LinearLeast\nSquares\nnormal eqs",
            "KNearest\nNeighbors\nmean of k",
            "GradientDescent\nRegressor\ntrainable",
            "SklearnRegressor\n(adapter)",
        ]
        for k, label in enumerate(concretes):
            x = 0.2 + 2.2 * k
            box(ax, (x, 0.6), (2.0, 1.4), label, SUB_FILL, dashed=k == 4, fontsize=8.5)
            arrow(ax, (x + 1.0, 2.0), (6.4, 3.3))
        box(
            ax,
            (9.0, 3.6),
            (2.0, 1.0),
            "any sklearn\nestimator",
            GREY_FILL,
            fontsize=8.5,
        )
        arrow(ax, (9.0 + 1.0, 2.0), (10.0, 3.6), style="-|>", dashed=True)
        ax.text(10.6, 2.75, "wraps", fontsize=8.5)
        ax.set_xlim(0, 11.3)
        ax.set_ylim(0.3, 5.2)
        ax.axis("off")
        self.save(fig, "fig_strategy_uml.png")

    def observer_sequence(self) -> None:
        """Sequence diagram: Trainer notifying three callbacks over epochs."""
        lanes = ["Trainer", "History", "EarlyStopping", "ProgressLogger"]
        events = [
            ("on_train_begin", 0),
            ("on_epoch_end (0)", 1),
            ("on_epoch_end (1)", 2),
            ("...", 3),
            ("on_epoch_end (k): stop!", 4),
            ("on_train_end", 5),
        ]
        fig, ax = plt.subplots(figsize=(11, 5.0))
        xs = [1.0 + 3.0 * i for i in range(len(lanes))]
        for x, lane in zip(xs, lanes):
            box(
                ax,
                (x - 1.1, 6.3),
                (2.2, 0.6),
                lane,
                BASE_FILL if x == xs[0] else SUB_FILL,
            )
            ax.plot([x, x], [0.2, 6.3], color="0.6", ls=":", lw=1)
        for label, row in events:
            y = 5.6 - row
            if label == "...":
                ax.text(xs[0] + 0.2, y, "...", fontsize=12)
                continue
            for x in xs[1:]:
                arrow(ax, (xs[0], y), (x, y), color=NAVY)
            ax.text(xs[0] + 0.1, y + 0.12, label, fontsize=9, family="DejaVu Sans Mono")
        ax.annotate(
            "state.request_stop()",
            xy=(xs[0] + 0.1, 1.45),
            xytext=(xs[2] + 0.2, 1.05),
            color=ACCENT,
            fontsize=9,
            arrowprops={"arrowstyle": "-|>", "color": ACCENT},
        )
        ax.set_xlim(-0.3, xs[-1] + 1.4)
        ax.set_ylim(0.0, 7.1)
        ax.axis("off")
        self.save(fig, "fig_observer_sequence.png")

    def factory_registry(self) -> None:
        """Name -> registry -> class -> configured loader -> Dataset."""
        fig, ax = plt.subplots(figsize=(11, 3.8))
        box(ax, (0.1, 1.5), (2.3, 1.0), '"fuel_flow_\nsynthetic"', GREY_FILL)
        box(
            ax,
            (3.0, 0.4),
            (3.0, 3.2),
            "DatasetFactory._registry\n\n"
            '"csv" -> CsvTableLoader\n'
            '"flight_log_csv"\n   -> FlightLogCsvLoader\n'
            '"fuel_flow_synthetic"\n   -> SyntheticFuel...\n'
            '"your_loader" -> ...',
            BASE_FILL,
            fontsize=8.5,
        )
        box(ax, (6.6, 1.5), (2.0, 1.0), "loader =\ncls(**kwargs)", SUB_FILL, fontsize=9)
        box(
            ax, (9.2, 1.5), (1.9, 1.0), "loader.load()\n-> Dataset", OK_FILL, fontsize=9
        )
        arrow(ax, (2.4, 2.0), (3.0, 2.0))
        arrow(ax, (6.0, 2.0), (6.6, 2.0))
        arrow(ax, (8.6, 2.0), (9.2, 2.0))
        ax.text(
            4.5,
            0.05,
            '@DatasetFactory.register("your_loader") adds a row; nothing else changes',
            ha="center",
            color=ACCENT,
            fontsize=9,
        )
        ax.set_xlim(0, 11.3)
        ax.set_ylim(-0.2, 3.8)
        ax.axis("off")
        self.save(fig, "fig_factory_registry.png")

    def power_curve(self) -> None:
        """Fuel flow vs airspeed: model components and the synthetic data."""
        aircraft = FuelFlowAircraft()
        speed = np.linspace(14.0, 38.0, 200)
        rho = isa_density(np.zeros_like(speed))
        weight = REFERENCE_MASS_KG * STANDARD_GRAVITY_MPS2
        to_gph = aircraft.bsfc_kg_per_kwh / aircraft.propeller_efficiency
        parasite = (
            0.5 * rho * speed**3 * aircraft.wing_area_m2 * aircraft.zero_lift_drag
        )
        induced = (
            2
            * weight**2
            / (
                rho
                * speed
                * aircraft.wing_area_m2
                * np.pi
                * aircraft.oswald_efficiency
                * aircraft.aspect_ratio
            )
        )
        total = aircraft.fuel_flow_gph(speed, 0.0, 0.0, REFERENCE_MASS_KG)
        dataset = DatasetFactory.load("fuel_flow_synthetic")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        ax = axes[0]
        ax.plot(speed, parasite * to_gph, "--", color=NAVY, label="parasite part")
        ax.plot(speed, induced * to_gph, ":", color=NAVY, label="induced part")
        ax.plot(speed, total, color=ACCENT, lw=2, label="total (level, sea level)")
        ax.plot(25.0, 270.9, "o", color="k")
        ax.annotate(
            "Example 1: 270.9 g/h",
            (25.0, 270.9),
            (26.5, 120.0),
            fontsize=9,
            arrowprops={"arrowstyle": "-|>"},
        )
        ax.set_xlabel("Airspeed V [m/s]")
        ax.set_ylabel("Fuel flow [g/h]")
        ax.set_title(f"Model, {REFERENCE_MASS_KG:g} kg, no climb")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
        ax = axes[1]
        points = ax.scatter(
            dataset.X[:, 0], dataset.y, c=dataset.X[:, 2], cmap="coolwarm", s=10
        )
        fig.colorbar(points, ax=ax, label="climb rate [m/s]")
        ax.set_xlabel("Airspeed V [m/s]")
        ax.set_title(f"Synthetic data: {len(dataset)} rows, 3 % noise")
        ax.grid(alpha=0.3)
        self.save(fig, "fig_power_curve.png")

    def model_comparison(self) -> None:
        """Test RMSE of each strategy on the chronological fuel-flow split."""
        train, test = DatasetFactory.load("fuel_flow_synthetic").split()
        strategies = [
            MeanBaseline(),
            KNearestNeighbors(k=10),
            KNearestNeighbors(k=3),
            KNearestNeighbors(k=5),
            LinearLeastSquares(),
            GradientDescentRegressor(),
        ]
        results = RegressionExperiment(MeanBaseline()).compare(strategies, train, test)
        labels = [r.model.replace("(", "\n(").replace(", ", ",\n") for r in results]
        fig, ax = plt.subplots(figsize=(11, 4.0))
        bars = ax.bar(
            range(len(results)),
            [r.test_rmse for r in results],
            color=BASE_FILL,
            edgecolor=NAVY,
        )
        bars[-1].set_color(GREY_FILL)
        for i, r in enumerate(results):
            ax.text(i, r.test_rmse + 4, f"{r.test_rmse:.1f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(results)), labels, fontsize=7.5)
        ax.set_ylabel("Test RMSE [g/h]")
        ax.set_title("Same experiment, six strategies (one line changes between runs)")
        ax.grid(axis="y", alpha=0.3)
        self.save(fig, "fig_model_comparison.png")

    def early_stopping(self) -> None:
        """In-class Problem 2: the patience counter on a validation curve."""
        losses = np.array(EARLY_STOP_LOSSES)
        epochs = np.arange(len(losses))
        best, wait, best_epoch, stop = np.inf, 0, -1, None
        waits = []
        for e, loss in enumerate(losses):
            if loss < best - EARLY_STOP_MIN_DELTA:
                best, wait, best_epoch = loss, 0, e
            else:
                wait += 1
            waits.append(wait)
            if wait >= EARLY_STOP_PATIENCE:
                stop = e
                break
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(epochs, losses, "o-", color="0.6", label="val_loss (full run)")
        ax.plot(
            epochs[: stop + 1],
            losses[: stop + 1],
            "o-",
            color=NAVY,
            label="seen by trainer",
        )
        ax.axhline(
            losses[best_epoch] - EARLY_STOP_MIN_DELTA,
            color=ACCENT,
            ls="--",
            lw=1,
            label="best - min_delta",
        )
        for e in range(stop + 1):
            ax.text(e, losses[e] + 0.03, f"wait={waits[e]}", ha="center", fontsize=8.5)
        ax.plot(
            best_epoch, losses[best_epoch], "*", ms=16, color=GREEN, label="restored"
        )
        ax.axvline(stop, color=ACCENT, lw=1)
        ax.text(stop + 0.05, 0.95, f"stop after epoch {stop}", color=ACCENT, fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation loss")
        ax.set_title(
            f"EarlyStopping(patience={EARLY_STOP_PATIENCE}, "
            f"min_delta={EARLY_STOP_MIN_DELTA})"
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
        self.save(fig, "fig_early_stopping.png")

    def gradient_step(self) -> None:
        """In-class Problem 1: three points and the line before/after one step."""
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 5.0])
        model = GradientDescentRegressor(learning_rate=0.1, epochs=1, standardize=False)
        model.initialize(X, y)
        lines = [(model.weights_.copy(), "w = (0, 0), MSE = 15.0")]
        loss = model.train_step(X, y)
        lines.append((model.weights_.copy(), f"after 1 step, MSE = {loss:.3f}"))
        ols = LinearLeastSquares().fit(X, y)
        x_plot = np.linspace(0.0, 3.5, 20)
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(X, y, "o", ms=9, color="k", label="data")
        for (w, label), style in zip(lines, ("--", "-")):
            ax.plot(x_plot, w[0] + w[1] * x_plot, style, color=NAVY, label=label)
        ax.plot(
            x_plot,
            ols.intercept_ + ols.coef_[0] * x_plot,
            ":",
            color=ACCENT,
            label="least squares (y = 0.667 + 1.5 x)",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
        ax.set_title("One gradient-descent step, eta = 0.1")
        self.save(fig, "fig_gradient_step.png")

    def contract_tests(self) -> None:
        """Matrix: every strategy passes every contract test."""
        strategies = ["mean", "ols", "ridge", "knn", "gd", "your model"]
        tests = [
            "fit returns self",
            "predict shape",
            "unfitted raises",
            "bad features raise",
            "bad data raise",
            "beats baseline",
        ]
        fig, ax = plt.subplots(figsize=(9, 3.6))
        for i, test in enumerate(tests):
            for j, name in enumerate(strategies):
                todo = name == "your model"
                ax.add_patch(
                    plt.Rectangle(
                        (j, -i),
                        0.92,
                        0.85,
                        fc=GREY_FILL if todo else OK_FILL,
                        ec="white",
                    )
                )
                ax.text(
                    j + 0.46,
                    -i + 0.42,
                    "?" if todo else "pass",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="0.4" if todo else GREEN,
                )
            ax.text(-0.1, -i + 0.42, test, ha="right", va="center", fontsize=9)
        for j, name in enumerate(strategies):
            ax.text(
                j + 0.46, 1.0, name, ha="center", fontsize=9, family="DejaVu Sans Mono"
            )
        ax.set_xlim(-2.6, len(strategies))
        ax.set_ylim(-len(tests) + 0.6, 1.5)
        ax.set_title(
            "@pytest.fixture(params=STRATEGIES): 6 tests x N models", color=NAVY
        )
        ax.axis("off")
        self.save(fig, "fig_contract_tests.png")

    def build_all(self) -> None:
        """Render every figure."""
        self.spaghetti_vs_patterns()
        self.strategy_uml()
        self.observer_sequence()
        self.factory_registry()
        self.power_curve()
        self.model_comparison()
        self.early_stopping()
        self.gradient_step()
        self.contract_tests()


if __name__ == "__main__":
    Week04Figures(OUT_DIR).build_all()
