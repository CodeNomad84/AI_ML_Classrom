"""Render the figures used by the Week 1 slides and lecture notes.

Provenance: new content (Week 1). Output: slides/assets/week01/*.png

Usage:
    python tools/make_week01_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerospace_ai.utils import GYRO_COLUMNS, FlightLog  # noqa: E402

DATA_PATH: Final[Path] = REPO_ROOT / "data" / "sample" / "week01_uav_flight_log.csv"
OUT_DIR: Final[Path] = REPO_ROOT / "slides" / "assets" / "week01"
DPI: Final[int] = 150
LOAD_LIMIT_G: Final[float] = 2.0
GROUND_END_S: Final[float] = 25.0
SPIKE_ZOOM_HALF_WIDTH_S: Final[float] = 0.3


class Week01Figures:
    """Build every Week 1 figure from the sample flight log."""

    def __init__(self, log: FlightLog, out_dir: Path) -> None:
        """Store the log and make sure the output folder exists.

        Args:
            log: Loaded Week 1 flight log.
            out_dir: Folder that receives the PNG files.
        """
        self.log = log
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig: plt.Figure, name: str) -> None:
        """Save ``fig`` as ``name`` in the output folder and close it."""
        fig.savefig(self.out_dir / name, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("wrote", (self.out_dir / name).relative_to(REPO_ROOT))

    def dashboard(self) -> None:
        """Four-panel overview: altitude, airspeed, load factor, climb rate."""
        t = self.log.time
        data = self.log.data
        panels = [
            (data["baro_alt_m"], "Altitude [m]", "tab:blue"),
            (data["airspeed_mps"], "Airspeed [m/s]", "tab:orange"),
            (self.log.robust_load_factor(), "Load factor [g]", "tab:red"),
            (self.log.climb_rate(window_s=2.0), "Climb rate [m/s]", "tab:green"),
        ]
        fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
        for ax, (series, label, color) in zip(axes, panels):
            ax.plot(t, series, lw=0.8, color=color)
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        for _, gap in self.log.time_gaps().iterrows():
            for ax in axes:
                ax.axvspan(gap["gap_start_s"], gap["gap_end_s"], color="k", alpha=0.2)
        axes[2].axhline(LOAD_LIMIT_G, ls="--", color="k", lw=0.8)
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle("UAV test flight (grey band = logger gap)")
        fig.tight_layout()
        self.save(fig, "fig_dashboard.png")

    def spike_vs_maneuver(self) -> None:
        """Zoom on one spike (left) and on the real pull-up (right)."""
        t = self.log.time
        raw = self.log.load_factor()
        med = self.log.robust_load_factor()
        spike_t = float(t[raw.idxmax()])
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
        windows = [
            (
                spike_t - SPIKE_ZOOM_HALF_WIDTH_S,
                spike_t + SPIKE_ZOOM_HALF_WIDTH_S,
                "Spike: 1 sample (sensor glitch)",
            ),
            (186.0, 198.0, "Pull-up: ~200 samples (real)"),
        ]
        for ax, (start, end, title) in zip(axes, windows):
            mask = t.between(start, end)
            ax.plot(t[mask], raw[mask], ".-", ms=3, lw=0.6, color="0.5", label="raw")
            ax.plot(
                t[mask], med[mask], lw=1.6, color="tab:red", label="rolling median (5)"
            )
            ax.axhline(LOAD_LIMIT_G, ls="--", color="k", lw=0.8)
            ax.set(title=title, xlabel="Time [s]")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Load factor [g]")
        axes[0].legend(loc="upper left")
        fig.tight_layout()
        self.save(fig, "fig_spike_vs_maneuver.png")

    def gyro_noise(self) -> None:
        """Histograms of the three gyros while parked, with Gaussian fits."""
        rest = self.log.segment(0.0, GROUND_END_S).data
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
        for ax, column in zip(axes, GYRO_COLUMNS):
            values = rest[column]
            mu, sigma = values.mean(), values.std(ddof=1)
            grid = np.linspace(values.min(), values.max(), 200)
            pdf = np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi)
            )
            ax.hist(values, bins=40, density=True, alpha=0.6)
            ax.plot(grid, pdf, color="k", lw=1)
            ax.set(title=f"{column}: s = {sigma:.2f}", xlabel="deg/s")
        axes[0].set_ylabel("density")
        fig.tight_layout()
        self.save(fig, "fig_gyro_noise.png")

    def climb_rate(self) -> None:
        """Raw versus smoothed derivative of barometric altitude."""
        t = self.log.time.to_numpy()
        altitude = (
            self.log.data["baro_alt_m"].interpolate(limit_direction="both").to_numpy()
        )
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(
            t, np.gradient(altitude, t), lw=0.3, color="0.7", label="raw derivative"
        )
        ax.plot(
            t,
            self.log.climb_rate(window_s=2.0),
            lw=1.4,
            color="tab:blue",
            label="smooth (2 s) then differentiate",
        )
        ax.set(xlabel="Time [s]", ylabel="Climb rate [m/s]", ylim=(-40, 40))
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self.save(fig, "fig_climb_rate.png")

    @staticmethod
    def environment_stack() -> plt.Figure:
        """Layer diagram: OS, Python, isolated environment, packages, notebook."""
        layers = [
            ("Operating system (Windows / Linux / macOS)", "#d9d9d9"),
            ("Python 3.11 interpreter", "#c6dbef"),
            ("Isolated environment: conda 'aero-ai' or .venv", "#9ecae1"),
            ("Pinned packages: numpy 1.26, pandas 2.1, matplotlib 3.8 ...", "#6baed6"),
            ("Your work: Jupyter notebooks + src/aerospace_ai + Git", "#3182bd"),
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        for level, (label, color) in enumerate(layers):
            ax.add_patch(
                FancyBboxPatch(
                    (0.05, 0.05 + level * 0.18),
                    0.9,
                    0.15,
                    boxstyle="round,pad=0.01",
                    fc=color,
                    ec="k",
                    lw=0.8,
                )
            )
            ax.text(
                0.5,
                0.125 + level * 0.18,
                label,
                ha="center",
                va="center",
                fontsize=11,
                color="white" if level >= 3 else "black",
            )
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        return fig

    @staticmethod
    def axis_and_broadcasting() -> plt.Figure:
        """Grid of an (N, 3) array with axis arrows, plus the broadcasting picture."""
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 4))
        rows, cols = 5, 3
        for r in range(rows):
            for c in range(cols):
                ax_a.add_patch(
                    plt.Rectangle((c, rows - 1 - r), 0.95, 0.95, fc="#c6dbef", ec="k")
                )
        for c, label in enumerate("xyz"):
            ax_a.text(c + 0.47, rows + 0.2, label, ha="center", fontsize=12)
        ax_a.annotate(
            "",
            xy=(3.6, 0.2),
            xytext=(3.6, 4.8),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "tab:red"},
        )
        ax_a.text(
            3.75, 2.5, "axis=0\nper-axis mean\nshape (3,)", va="center", color="tab:red"
        )
        ax_a.annotate(
            "",
            xy=(2.9, -0.6),
            xytext=(0.0, -0.6),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "tab:blue"},
        )
        ax_a.text(
            1.45,
            -1.3,
            "axis=1: per-sample norm, shape (N,)",
            ha="center",
            color="tab:blue",
        )
        ax_a.set(xlim=(-0.3, 6), ylim=(-1.8, 6), title="accel.shape = (N, 3)")
        ax_a.axis("off")
        for r in range(4):
            for c in range(3):
                ax_b.add_patch(
                    plt.Rectangle((c, 3 - r), 0.95, 0.95, fc="#9ecae1", ec="k")
                )
                ax_b.add_patch(
                    plt.Rectangle(
                        (c + 4.2, 3 - r),
                        0.95,
                        0.95,
                        fc="#fdd0a2" if r == 0 else "#fff5eb",
                        ec="k",
                        ls="-" if r == 0 else "--",
                    )
                )
        ax_b.text(3.55, 2.0, "−", fontsize=26, ha="center", va="center")
        ax_b.text(1.45, 4.3, "a: (4, 3)", ha="center")
        ax_b.text(5.65, 4.3, "a.mean(axis=0): (3,)\nstretched to (4, 3)", ha="center")
        ax_b.set(
            xlim=(-0.3, 7.6),
            ylim=(-0.5, 5.2),
            title="Broadcasting: (4, 3) − (3,) → (4, 3)",
        )
        ax_b.axis("off")
        fig.tight_layout()
        return fig

    @staticmethod
    def banked_turn() -> plt.Figure:
        """Free-body diagram of a level coordinated turn: L cos(phi) = W."""
        phi = np.radians(30.0)
        lift = 1.0 / np.cos(phi)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        wing = np.array([[-1.2, 1.2], [0.0, 0.0]])
        rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        span = rot @ wing
        ax.plot(span[0], span[1], lw=6, color="0.3")
        arrow = {"width": 0.02, "length_includes_head": True, "head_width": 0.08}
        ax.arrow(
            0, 0, -lift * np.sin(phi), lift * np.cos(phi), color="tab:blue", **arrow
        )
        ax.arrow(0, 0, 0, -1.0, color="tab:red", **arrow)
        ax.arrow(0, 0, -lift * np.sin(phi), 0, color="tab:green", **arrow)
        ax.plot([-lift * np.sin(phi)] * 2, [0, lift * np.cos(phi)], "k:", lw=1)
        ax.text(
            -lift * np.sin(phi) - 0.15,
            lift * np.cos(phi) + 0.05,
            "L",
            color="tab:blue",
            fontsize=14,
        )
        ax.text(0.08, -1.0, "W", color="tab:red", fontsize=14)
        ax.text(
            -0.3,
            0.06,
            "L sin φ (centripetal)",
            color="tab:green",
            fontsize=9,
            ha="center",
            va="bottom",
        )
        ax.text(0.55, 0.12, "φ", fontsize=14)
        ax.text(
            0.0,
            -1.45,
            "L cos φ = W  ⇒  n = L / W = 1 / cos φ",
            ha="center",
            fontsize=12,
        )
        ax.set(
            xlim=(-1.6, 1.6),
            ylim=(-1.7, 1.4),
            aspect="equal",
            title="Level coordinated turn",
        )
        ax.axis("off")
        return fig

    @staticmethod
    def gap_timeline() -> plt.Figure:
        """Timestamps every 0.02 s with one 1.22 s hole: 60 missing samples."""
        nominal = 0.02
        before = 99.9 + nominal * np.arange(5)
        after = 101.2 + nominal * np.arange(5)
        fig, ax = plt.subplots(figsize=(9, 2.2))
        ax.plot(before, np.zeros_like(before), "o", color="tab:blue")
        ax.plot(after, np.zeros_like(after), "o", color="tab:blue")
        ax.axvspan(before[-1], after[0], color="tab:red", alpha=0.15)
        ax.text(
            (before[-1] + after[0]) / 2,
            0.25,
            "dt = 1.22 s = 61 steps → 60 samples missing",
            ha="center",
            color="tab:red",
            fontsize=12,
        )
        ax.text(before[2], -0.35, "dt = 0.02 s (50 Hz)", ha="center")
        ax.set(ylim=(-0.6, 0.6), xlabel="Time [s]")
        ax.get_yaxis().set_visible(False)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        return fig

    @staticmethod
    def mean_vs_median() -> plt.Figure:
        """Number line of a five-sample window: the spike drags the mean, not the median."""
        window = np.array([1.00, 1.01, 5.10, 0.99, 1.00])
        fig, ax = plt.subplots(figsize=(9, 2.2))
        ax.plot(window, np.zeros_like(window), "o", ms=9, color="0.4")
        ax.axvline(window.mean(), color="tab:orange", lw=2)
        ax.axvline(np.median(window), color="tab:red", lw=2)
        ax.text(
            window.mean() + 0.05,
            0.3,
            f"mean = {window.mean():.2f} g",
            color="tab:orange",
            fontsize=12,
        )
        ax.text(
            np.median(window) + 0.05,
            -0.4,
            f"median = {np.median(window):.2f} g",
            color="tab:red",
            fontsize=12,
        )
        ax.text(5.1, 0.15, "spike", ha="center")
        ax.set(ylim=(-0.6, 0.6), xlim=(0.6, 5.5), xlabel="Load factor [g]")
        ax.get_yaxis().set_visible(False)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        return fig

    @staticmethod
    def central_difference() -> plt.Figure:
        """Three altitude samples and the central-difference secant."""
        t = np.array([80.00, 80.02, 80.04])
        h = np.array([250.0, 250.3, 250.5])
        slope = (h[2] - h[0]) / (t[2] - t[0])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, h, "o", ms=9, color="tab:blue")
        grid = np.linspace(79.995, 80.045, 10)
        ax.plot(
            grid,
            h[0] + slope * (grid - t[0]),
            "--",
            color="tab:red",
            label=f"secant slope = {slope:.1f} m/s",
        )
        for ti, hi in zip(t, h):
            ax.annotate(
                f"({ti:.2f}, {hi:.1f})",
                (ti, hi),
                textcoords="offset points",
                xytext=(5, -15),
            )
        ax.set(
            xlabel="Time [s]",
            ylabel="Altitude [m]",
            title="ḣ(80.02) ≈ (250.5 − 250.0) / 0.04 = 12.5 m/s",
        )
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def render_all(self) -> None:
        """Render every figure."""
        self.dashboard()
        self.spike_vs_maneuver()
        self.gyro_noise()
        self.climb_rate()
        self.save(self.environment_stack(), "fig_environment_stack.png")
        self.save(self.axis_and_broadcasting(), "fig_axis_broadcasting.png")
        self.save(self.banked_turn(), "fig_banked_turn.png")
        self.save(self.gap_timeline(), "fig_gap_timeline.png")
        self.save(self.mean_vs_median(), "fig_mean_vs_median.png")
        self.save(self.central_difference(), "fig_central_difference.png")


if __name__ == "__main__":
    Week01Figures(FlightLog.from_csv(DATA_PATH), OUT_DIR).render_all()
