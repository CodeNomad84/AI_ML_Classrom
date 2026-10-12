"""Render the figures used by the Week 3 slides and lecture notes.

Provenance: new content (Week 3). Output: slides/assets/week03/*.png

Usage:
    python tools/make_week03_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerospace_ai.pipeline import (  # noqa: E402
    MAD_TO_SIGMA,
    FillGaps,
    HampelFilter,
    RollingMean,
    TimeDerivative,
)
from aerospace_ai.utils import ACCEL_COLUMNS, STANDARD_GRAVITY_MPS2  # noqa: E402

DATA_DIR: Final[Path] = REPO_ROOT / "data" / "sample"
OUT_DIR: Final[Path] = REPO_ROOT / "slides" / "assets" / "week03"
FLIGHT_LOG: Final[Path] = DATA_DIR / "week01_uav_flight_log.csv"
DPI: Final[int] = 150
SMOOTH_WINDOW: Final[int] = 5
DESPIKE_WINDOW: Final[int] = 7
CLIMB_WINDOW_ROWS: Final[int] = 51
RESAMPLE_RATE_HZ: Final[float] = 10.0
HAND_WINDOW: Final[int] = 5
N_SIGMAS: Final[float] = 3.0
HAND_ALTITUDE: Final[tuple[float, ...]] = (
    100.0,
    100.4,
    99.8,
    106.0,
    100.3,
    100.1,
    99.9,
)
STEP_SIGNAL: Final[tuple[float, ...]] = (1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0)
GAP_START_S: Final[float] = 99.98
GAP_END_S: Final[float] = 101.2

NAVY: Final[str] = "#1f3a5f"
ACCENT: Final[str] = "#e07a1f"
GOOD: Final[str] = "#2e7d4f"
BAD: Final[str] = "#c0392b"
BASE_FILL: Final[str] = "#dbe6f3"
SUB_FILL: Final[str] = "#fbe3cc"
GREY_FILL: Final[str] = "#eeeeee"
LEARN_FILL: Final[str] = "#dff0e3"


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    text: str,
    fill: str,
    dashed: bool = False,
    fontsize: float = 10.0,
    edge: str = NAVY,
) -> None:
    """Draw a rounded box with centred, multi-line text."""
    ax.add_patch(
        FancyBboxPatch(
            xy,
            *size,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            fc=fill,
            ec=edge,
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
    color: str = NAVY,
    style: str = "-|>",
) -> None:
    """Draw an arrow from ``start`` to ``end``."""
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=16, color=color, lw=1.4
        )
    )


def load_factor(frame: pd.DataFrame) -> np.ndarray:
    """Return n = |f| / g0 from the accelerometer columns."""
    accel = frame[list(ACCEL_COLUMNS)].to_numpy(dtype=float)
    return np.linalg.norm(accel, axis=1) / STANDARD_GRAVITY_MPS2


class Week03Figures:
    """Build every Week 3 figure."""

    def __init__(self, out_dir: Path) -> None:
        """Make sure the output folder exists and load the flight log once."""
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = pd.read_csv(FLIGHT_LOG)

    def save(self, fig: plt.Figure, name: str) -> None:
        """Save ``fig`` as ``name`` in the output folder and close it."""
        fig.savefig(self.out_dir / name, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("wrote", (self.out_dir / name).relative_to(REPO_ROOT))

    def two_analysts(self) -> None:
        """Smooth-only versus despike-then-smooth on the Week 1 log (two zooms)."""
        accel = list(ACCEL_COLUMNS)
        smooth = RollingMean(accel, window=SMOOTH_WINDOW)
        recipe_a = smooth(self.log)
        recipe_b = (HampelFilter(accel, window=DESPIKE_WINDOW) >> smooth)(self.log)
        t = self.log["time_s"].to_numpy()
        n_raw, n_a, n_b = (load_factor(f) for f in (self.log, recipe_a, recipe_b))
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
        windows = (((64.5, 69.0), "glitch in level cruise"), ((186, 198), "pull-up"))
        for ax, ((t0, t1), title) in zip(axes, windows):
            keep = (t >= t0) & (t <= t1)
            ax.plot(t[keep], n_raw[keep], color="0.75", lw=0.8, label="raw")
            ax.plot(t[keep], n_a[keep], color=ACCENT, lw=1.6, label="A: smooth")
            ax.plot(t[keep], n_b[keep], color=NAVY, lw=1.6, label="B: despike, smooth")
            ax.set_title(title)
            ax.set_xlabel("time [s]")
            ax.grid(alpha=0.3)
        for n_series, color, label in ((n_a, ACCENT, "A"), (n_b, NAVY, "B")):
            k = int(np.argmax(n_series))
            ax_peak = axes[0] if t[k] < windows[0][0][1] else axes[1]
            ax_peak.annotate(
                f"{label} peak {n_series[k]:.2f} g\nt = {t[k]:.1f} s",
                (t[k], n_series[k]),
                xytext=(10, -40 if label == "B" else 5),
                textcoords="offset points",
                color=color,
                fontsize=9,
                arrowprops={"arrowstyle": "->", "color": color},
            )
        axes[0].set_ylim(0.6, 2.3)
        axes[0].set_ylabel("load factor n [g]")
        axes[0].legend(loc="upper left", fontsize=8)
        fig.suptitle("Same log, same two operations, different order", y=1.02)
        self.save(fig, "fig_two_analysts.png")

    def hampel_by_hand(self) -> None:
        """The seven-sample altitude example with window, median and threshold."""
        x = np.asarray(HAND_ALTITUDE)
        k = np.arange(len(x))
        stage = HampelFilter(["h"], window=HAND_WINDOW, n_sigmas=N_SIGMAS)
        is_outlier, median = stage.outlier_mask(pd.Series(x))
        centre = 3
        window = x[centre - 2 : centre + 3]
        mad = np.median(np.abs(window - median[centre]))
        threshold = N_SIGMAS * MAD_TO_SIGMA * mad
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.axvspan(centre - 2.4, centre + 2.4, color=BASE_FILL, label="window k = 1..5")
        ax.fill_between(
            [centre - 2.4, centre + 2.4],
            median[centre] - threshold,
            median[centre] + threshold,
            color=LEARN_FILL,
            label=f"m ± 3·1.4826·MAD = {median[centre]:.1f} ± {threshold:.2f}",
        )
        ax.axhline(median[centre], color=GOOD, lw=1.2, ls="--")
        ax.plot(k, x, "o-", color=NAVY, label="h_k")
        ax.plot(k[is_outlier], x[is_outlier], "o", ms=12, mfc="none", mec=BAD, mew=2)
        ax.annotate(
            "106.0 → 100.3\n|106.0 − 100.3| = 5.7 > 0.89",
            (centre, x[centre]),
            xytext=(centre + 0.6, 104.5),
            color=BAD,
            arrowprops={"arrowstyle": "->", "color": BAD},
        )
        ax.text(
            centre + 0.05, median[centre] - 0.7, "median 100.3, MAD 0.2", color=GOOD
        )
        ax.set(xlabel="sample k", ylabel="altitude [m]", ylim=(98.5, 107))
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("Hampel filter by hand (window 5, n_sigma = 3)")
        self.save(fig, "fig_hampel_by_hand.png")

    def order_matters(self) -> None:
        """H then S versus S then H on a one-sample glitch."""
        frame = pd.DataFrame({"x": STEP_SIGNAL})
        hampel = HampelFilter(["x"], window=3)
        mean = RollingMean(["x"], window=3)
        paths = (
            ("Despike, then smooth", [frame, hampel(frame), (hampel >> mean)(frame)]),
            ("Smooth, then despike", [frame, mean(frame), (mean >> hampel)(frame)]),
        )
        k = np.arange(len(STEP_SIGNAL))
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
        for ax, (title, frames) in zip(axes, paths):
            styles = (
                ("input", ":", "0.55", 2.0, 6),
                ("after step 1", "--", ACCENT, 3.5, 10),
                ("output", "-", NAVY, 1.5, 5),
            )
            for frame_i, (label, style, color, width, size) in zip(frames, styles):
                ax.plot(
                    k,
                    frame_i["x"],
                    style,
                    color=color,
                    lw=width,
                    marker="o",
                    ms=size,
                    label=label,
                )
            peak = frames[-1]["x"].max()
            color = GOOD if peak < 2 else BAD
            ax.set_title(f"{title}: peak = {peak:g}", color=color)
            ax.set_xlabel("sample k")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("x")
        axes[0].legend(fontsize=8)
        self.save(fig, "fig_order_matters.png")

    def pipeline_chain(self) -> None:
        """Typed stages joined by their data types; a mismatch caught at build time."""
        fig, ax = plt.subplots(figsize=(12, 3.8))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 4)
        ax.axis("off")
        stages = ("SensorToFrame", "HampelFilter", "RollingMean", "ToArray")
        types = ("BaseSensor", "DataFrame", "DataFrame", "DataFrame", "ndarray")
        width, x0 = 2.1, 0.2
        for i, name in enumerate(stages):
            x = x0 + 0.9 + i * 2.75
            box(ax, (x, 2.3), (width, 0.9), name, BASE_FILL, fontsize=9.5)
            ax.text(x - 0.42, 2.95, types[i], ha="center", fontsize=8, color=ACCENT)
            arrow(ax, (x - 0.85, 2.75), (x, 2.75))
        ax.text(
            x0 + 0.9 + 3 * 2.75 + width + 0.45,
            2.95,
            types[-1],
            ha="center",
            fontsize=8,
            color=ACCENT,
        )
        arrow(
            ax,
            (x0 + 0.9 + 3 * 2.75 + width, 2.75),
            (x0 + 0.9 + 3 * 2.75 + width + 0.9, 2.75),
        )
        ax.text(
            0.2,
            1.75,
            "SensorToFrame() >> HampelFilter(...) >> RollingMean(...) >> ToArray(...)"
            "   ✓ built: each output type is accepted by the next input type",
            fontsize=9,
            family="DejaVu Sans Mono",
            color=GOOD,
        )
        box(ax, (0.9, 0.2), (width, 0.9), "ToArray", BASE_FILL, fontsize=9.5)
        box(
            ax,
            (4.2, 0.2),
            (width, 0.9),
            "RollingMean",
            SUB_FILL,
            fontsize=9.5,
            edge=BAD,
        )
        arrow(ax, (0.9 + width, 0.65), (4.2, 0.65), color=BAD)
        ax.text(3.55, 0.95, "ndarray ≠ DataFrame", ha="center", fontsize=8, color=BAD)
        ax.text(
            6.6,
            0.55,
            "TypeError when the pipeline is BUILT:\n"
            "step 0 ToArray outputs ndarray,\nbut step 1 RollingMean expects DataFrame",
            fontsize=9,
            family="DejaVu Sans Mono",
            color=BAD,
            va="center",
        )
        self.save(fig, "fig_pipeline_chain.png")

    def stage_hierarchy(self) -> None:
        """PipelineStage ABC, the concrete dataclass stages and DataPipeline."""
        fig, ax = plt.subplots(figsize=(11, 5.4))
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 6.2)
        ax.axis("off")
        box(
            ax,
            (3.2, 4.5),
            (4.6, 1.5),
            "PipelineStage(ABC, Generic[InT, OutT])\n"
            "INPUT_TYPE, OUTPUT_TYPE, registry\n"
            "@abstractmethod transform(data)\n"
            "fit(data) · is_fitted · __call__ · >>",
            BASE_FILL,
            fontsize=9,
        )
        frozen = (
            "SensorToFrame",
            "SelectColumns",
            "DropNonFinite",
            "FillGaps",
            "HampelFilter",
            "RollingMean",
            "TimeDerivative",
            "ToArray",
        )
        for i, name in enumerate(frozen):
            col, row = i % 4, i // 4
            box(
                ax,
                (0.2 + col * 2.05, 2.3 - row * 0.85),
                (1.9, 0.62),
                name,
                SUB_FILL,
                fontsize=8.5,
            )
        ax.text(
            1.3,
            3.1,
            "@dataclass(frozen=True): configuration only",
            fontsize=9,
            color=ACCENT,
        )
        box(
            ax,
            (8.55, 1.45),
            (2.25, 1.47),
            "Standardize\n@dataclass\nmean_, std_:\ninit/repr/compare\n= False",
            LEARN_FILL,
            fontsize=8.5,
        )
        ax.text(9.75, 3.1, "learns in fit()", fontsize=9, color=GOOD)
        arrow(ax, (5.5, 3.55), (5.5, 4.45))
        ax.plot([1.15, 9.65], [3.55, 3.55], color=NAVY, lw=1.4)
        for x in (1.15, 9.65):
            ax.plot([x, x], [2.95, 3.55], color=NAVY, lw=1.4)
        box(
            ax,
            (1.0, 0.05),
            (9.0, 0.75),
            "DataPipeline: tuple of stages · __repr__ __eq__ __len__ __iter__ __getitem__\n"
            "run() → StageReport per stage · to_config() / from_config()",
            GREY_FILL,
            fontsize=8.5,
        )
        ax.text(
            5.5,
            0.95,
            "◆ composes (has-a, not is-a)",
            ha="center",
            fontsize=9,
            color=NAVY,
        )
        self.save(fig, "fig_stage_hierarchy.png")

    def dataclass_generated(self) -> None:
        """What @dataclass writes for a frozen stage."""
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 4)
        ax.axis("off")
        box(ax, (0.1, 0.6), (3.9, 2.8), "", BASE_FILL)
        ax.text(
            0.35,
            2.0,
            "@dataclass(frozen=True)\nclass RollingMean(...):\n"
            "    columns: tuple[str, ...]\n    window: int = 5\n\n"
            "    def __post_init__(self):\n        # validate, normalise",
            ha="left",
            va="center",
            multialignment="left",
            fontsize=9,
            family="DejaVu Sans Mono",
        )
        generated = (
            "__init__(self, columns, window=5)",
            "__repr__ → RollingMean(columns=('x',), window=5)",
            "__eq__ → same class and (columns, window)",
            "__hash__ → hash((columns, window))   [frozen]",
            "__setattr__ → FrozenInstanceError   [frozen]",
        )
        for i, text in enumerate(generated):
            box(ax, (5.0, 3.15 - i * 0.68), (5.9, 0.52), text, SUB_FILL, fontsize=8.5)
        arrow(ax, (4.1, 2.0), (4.9, 2.0))
        ax.text(4.5, 2.2, "writes", ha="center", fontsize=9, color=NAVY)
        self.save(fig, "fig_dataclass_generated.png")

    def leakage(self) -> None:
        """Standardising a test sample with statistics from train versus train + test."""
        train = np.array([2.0, 4.0, 6.0])
        test = 8.0
        fig, axes = plt.subplots(2, 1, figsize=(9, 3.4), sharex=True)
        cases = (
            ("fit on train only", train, GOOD),
            ("fit on train + test (leakage)", np.append(train, test), BAD),
        )
        for ax, (title, fitted, color) in zip(axes, cases):
            mean, std = fitted.mean(), fitted.std()
            ax.axvspan(mean - std, mean + std, color=GREY_FILL)
            ax.axvline(mean, color="0.4", ls="--")
            ax.plot(train, np.zeros_like(train), "o", color=NAVY, ms=9, label="train")
            ax.plot([test], [0], "s", color=color, ms=10, label="test")
            z = (test - mean) / std
            ax.text(
                test + 0.15,
                0.25,
                f"z = (8 − {mean:g}) / {std:.3f} = {z:.3f}",
                color=color,
            )
            ax.set_title(
                f"{title}: μ = {mean:g}, σ = {std:.3f}", fontsize=10, loc="left"
            )
            ax.set_yticks([])
            ax.set_ylim(-0.6, 0.8)
        axes[0].legend(loc="upper left", fontsize=8, ncol=2)
        axes[1].set_xlabel("altitude [m]")
        axes[1].set_xlim(0, 12)
        self.save(fig, "fig_leakage.png")

    def resample_gap(self) -> None:
        """Climb rate near the logger gap: row-counting versus a uniform time grid."""
        alt = "baro_alt_m"
        row_based = (
            FillGaps([alt], max_gap=5, time_column="time_s")
            >> HampelFilter([alt], window=DESPIKE_WINDOW)
            >> RollingMean([alt], window=CLIMB_WINDOW_ROWS)
            >> TimeDerivative(alt, "climb_mps", time_column="time_s")
        )(self.log)
        # Uniform 10 Hz grid by linear interpolation (what Practice Assignment 3 builds).
        t = self.log["time_s"].to_numpy()
        values = self.log[alt].to_numpy()
        known = ~np.isnan(values)
        n_points = int(np.floor((t[-1] - t[0]) * RESAMPLE_RATE_HZ + 1e-9)) + 1
        grid = t[0] + np.arange(n_points) / RESAMPLE_RATE_HZ
        uniform = pd.DataFrame(
            {"time_s": grid, alt: np.interp(grid, t[known], values[known])}
        )
        gridded = (
            HampelFilter([alt], window=DESPIKE_WINDOW)
            >> RollingMean([alt], window=SMOOTH_WINDOW)
            >> TimeDerivative(alt, "climb_mps", time_column="time_s")
        )(uniform)
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.axvspan(GAP_START_S, GAP_END_S, color=GREY_FILL, label="1.2 s logger gap")
        ax.plot(
            row_based["time_s"],
            row_based["climb_mps"],
            color=ACCENT,
            lw=1.2,
            label="row window (51 rows)",
        )
        ax.plot(
            gridded["time_s"],
            gridded["climb_mps"],
            color=NAVY,
            lw=1.2,
            label="10 Hz grid, 5-sample window",
        )
        ax.set(
            xlim=(90, 112), ylim=(-2, 24), xlabel="time [s]", ylabel="climb rate [m/s]"
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_title("A window of rows is not a window of seconds")
        self.save(fig, "fig_resample_gap.png")

    def build_all(self) -> None:
        """Render every figure."""
        self.two_analysts()
        self.hampel_by_hand()
        self.order_matters()
        self.pipeline_chain()
        self.stage_hierarchy()
        self.dataclass_generated()
        self.leakage()
        self.resample_gap()


if __name__ == "__main__":
    Week03Figures(OUT_DIR).build_all()
