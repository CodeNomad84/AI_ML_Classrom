"""Render the figures used by the Week 5 slides and lecture notes.

Provenance: new content (Week 5). Output: slides/assets/week05/*.png

Usage:
    python tools/make_week05_figures.py
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

from aerospace_ai.datasets import (  # noqa: E402
    ROVER_FEATURES,
    ROVER_TARGET,
    DatasetFactory,
    rover_cleaning_pipeline,
)
from aerospace_ai.models import KNearestNeighbors, LinearLeastSquares  # noqa: E402
from aerospace_ai.pipeline import DataPipeline, HampelFilter  # noqa: E402
from aerospace_ai.preprocessing import (
    ImputeMedian,
    RobustScale,
    audit_frame,
)  # noqa: E402
from aerospace_ai.rover import (  # noqa: E402
    LIDAR_MAX_RANGE_M,
    RoverLogConfig,
    RoverLogGenerator,
)
from aerospace_ai.splits import (  # noqa: E402
    ChronologicalSplit,
    GroupHoldoutSplit,
    RandomRowSplit,
    split_dataset,
    split_frame,
)

OUT_DIR: Final[Path] = REPO_ROOT / "slides" / "assets" / "week05"
ROVER_CSV: Final[Path] = REPO_ROOT / "data" / "sample" / "week05_rover_log.csv"
SEED: Final[int] = 5
DEMO_RUN: Final[int] = 3
DPI: Final[int] = 150
TUKEY_K: Final[float] = 1.5

NAVY: Final[str] = "#1f3a5f"
ACCENT: Final[str] = "#e07a1f"
GREEN: Final[str] = "#3a8f4b"
RED: Final[str] = "#c0392b"
GREY: Final[str] = "#9a9a9a"
BASE_FILL: Final[str] = "#dbe6f3"
SUB_FILL: Final[str] = "#fbe3cc"
OK_FILL: Final[str] = "#d6efd9"


def save(fig: plt.Figure, name: str) -> None:
    """Write ``fig`` to the output folder and close it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    text: str,
    fill: str,
    fontsize: float = 10.0,
) -> None:
    """Draw a rounded box with centred text."""
    ax.add_patch(
        FancyBboxPatch(
            xy,
            *size,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            fc=fill,
            ec=NAVY,
            lw=1.4,
        )
    )
    ax.text(
        xy[0] + size[0] / 2,
        xy[1] + size[1] / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=NAVY,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Draw a navy arrow."""
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14, color=NAVY, lw=1.4
        )
    )


def fig_raw_vs_truth(raw: pd.DataFrame, truth: pd.DataFrame) -> None:
    """Four sensor channels of one run: logged values over the truth."""
    run_raw, run_true = (
        raw[raw["run_id"] == DEMO_RUN],
        truth[truth["run_id"] == DEMO_RUN],
    )
    units = ["m", "m/s", "m/s²", "°/s"]
    fig, axes = plt.subplots(4, 1, figsize=(9, 6.5), sharex=True)
    for ax, col, unit in zip(axes, ROVER_FEATURES, units):
        ax.plot(run_true["time_s"], run_true[col], color=GREY, lw=2.5, label="truth")
        ax.plot(run_raw["time_s"], run_raw[col], ".", ms=3.5, color=RED, label="logged")
        ax.set_ylabel(f"{col}\n[{unit}]", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].axhline(LIDAR_MAX_RANGE_M, ls="--", color="k", lw=0.8)
    axes[0].annotate("beyond 12 m: logged as 0.0", (1.0, 0.3), fontsize=8, color=RED)
    axes[2].set_ylim(-4, 4)
    axes[2].annotate(
        "spikes up to ±25 m/s² (clipped in this view)",
        (0.3, 2.8),
        fontsize=8,
        color=RED,
    )
    axes[3].annotate("frozen for 3 s", (8.0, 3.2), fontsize=8, color=RED)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Run {DEMO_RUN}: what the logger wrote vs what happened", color=NAVY)
    fig.tight_layout()
    save(fig, "fig_raw_vs_truth.png")


def fig_audit(raw: pd.DataFrame) -> None:
    """Bar charts of unusable cells and longest repeats per sensor column."""
    table = audit_frame(raw, {"lidar_range_m": 0.0}).loc[list(ROVER_FEATURES)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    ax1.barh(table.index, table["pct_unusable"], color=[RED, ACCENT, NAVY, NAVY])
    for y, v in enumerate(table["pct_unusable"]):
        ax1.text(v + 0.3, y, f"{v:.1f} %", va="center", fontsize=9)
    ax1.set_xlabel("unusable cells (NaN, inf, sentinel 0.0) [%]")
    ax1.invert_yaxis()
    ax2.barh(table.index, table["longest_repeat"], color=[GREY, GREY, NAVY, RED])
    for y, v in enumerate(table["longest_repeat"]):
        ax2.text(v + 0.5, y, f"{int(v)}", va="center", fontsize=9)
    ax2.set_xlabel("longest run of identical values [samples]")
    ax2.invert_yaxis()
    ax2.set_yticklabels([])
    ax2.text(36, 1.35, "zeros at rest:\ngenuine", fontsize=8, color=GREY)
    ax2.text(36, 3.15, "frozen gyro", fontsize=8, color=RED)
    fig.suptitle(
        "audit_frame(raw): look before you clean (2533 rows, 16 runs)", color=NAVY
    )
    fig.tight_layout()
    save(fig, "fig_audit.png")


def fig_mnar(truth: pd.DataFrame) -> None:
    """Histogram of true ranges, split into seen and unseen by the 12 m limit."""
    r = truth["lidar_range_m"].to_numpy()
    seen, unseen = r[r <= LIDAR_MAX_RANGE_M], r[r > LIDAR_MAX_RANGE_M]
    bins = np.arange(0, 19, 0.5)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(seen, bins=bins, color=NAVY, alpha=0.8, label=f"logged ({seen.size})")
    ax.hist(
        unseen,
        bins=bins,
        color=RED,
        alpha=0.8,
        label=f"missing: wall beyond 12 m ({unseen.size})",
    )
    for value, color, text, side in (
        (r.mean(), "k", "true mean", 0.15),
        (seen.mean(), NAVY, "observed mean", -2.6),
    ):
        ax.axvline(value, color=color, ls="--", lw=1.2)
        ax.text(
            value + side,
            ax.get_ylim()[1] * 0.85,
            f"{text}\n{value:.2f} m",
            fontsize=8,
            color=color,
        )
    ax.set_xlabel("true range to the wall [m]")
    ax.set_ylabel("samples")
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("MNAR: the LiDAR is silent because the value is large", color=NAVY)
    fig.tight_layout()
    save(fig, "fig_mnar.png")


def fig_masking() -> None:
    """Worked example: z-score vs robust score, and the sqrt(n - 1) bound."""
    glitch = np.array([0.2, -0.1, 0.0, 0.3, -0.2, 0.1, 0.0, 25.0])
    z = (glitch - glitch.mean()) / glitch.std()
    median = np.median(glitch)
    robust = np.abs(glitch - median) / (1.4826 * np.median(np.abs(glitch - median)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.bar(["|z|", "robust score"], [abs(z[-1]), robust[-1]], color=[GREY, GREEN])
    ax1.axhline(3, color=RED, ls="--")
    ax1.set_yscale("log")
    ax1.text(0.55, 3.4, "threshold 3", color=RED, fontsize=9)
    for x, v in enumerate([abs(z[-1]), robust[-1]]):
        ax1.text(
            x, v * 1.15, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center", fontsize=10
        )
    ax1.set_title("glitch of 25 m/s² among 7 samples at rest", fontsize=10, color=NAVY)
    n = np.arange(2, 21)
    ax2.plot(
        n, np.sqrt(n - 1), "o-", color=NAVY, ms=4, label="largest possible |z| = √(n−1)"
    )
    ax2.axhline(3, color=RED, ls="--", label="threshold 3")
    ax2.axvspan(1.5, 9.5, color=RED, alpha=0.08)
    ax2.text(
        2.2, 3.6, "n < 10: one outlier\ncan never be flagged", fontsize=9, color=RED
    )
    ax2.set_xlabel("sample size n")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.3)
    ax2.set_title("Samuelson's bound: masking", fontsize=10, color=NAVY)
    fig.tight_layout()
    save(fig, "fig_masking.png")


def fig_outlier_rules(raw: pd.DataFrame) -> None:
    """Acceleration of one run with Tukey (global) and Hampel (local) flags."""
    log = raw.drop_duplicates(["run_id", "time_s"]).reset_index(drop=True)
    accel = log["accel_x_mps2"]
    q1, q3 = accel.quantile([0.25, 0.75])
    low, high = q1 - TUKEY_K * (q3 - q1), q3 + TUKEY_K * (q3 - q1)
    run = log[log["run_id"] == 7]
    hampel, _ = HampelFilter(("accel_x_mps2",)).outlier_mask(run["accel_x_mps2"])
    tukey = (run["accel_x_mps2"] < low) | (run["accel_x_mps2"] > high)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(
        run["time_s"], run["accel_x_mps2"], color=NAVY, lw=1, label="accel_x (run 7)"
    )
    ax.axhspan(
        low,
        high,
        color=GREY,
        alpha=0.2,
        label=f"Tukey fences k=1.5 [{low:.2f}, {high:.2f}]",
    )
    ax.plot(
        run["time_s"][tukey],
        run["accel_x_mps2"][tukey],
        "s",
        mfc="none",
        color=ACCENT,
        ms=7,
        label=f"outside the fences ({int(tukey.sum())})",
    )
    ax.plot(
        run["time_s"][hampel],
        run["accel_x_mps2"][hampel],
        "x",
        color=RED,
        ms=8,
        label=f"Hampel flags ({int(hampel.sum())})",
    )
    ax.set_ylim(-3, 3)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("m/s²")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left", ncol=2)
    ax.set_title(
        "Global fences flag the braking manoeuvre; the local filter flags spikes",
        color=NAVY,
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "fig_outlier_rules.png")


def fig_per_group(raw: pd.DataFrame) -> None:
    """Target shift across a run boundary: plain vs per run."""
    log = raw.drop_duplicates(["run_id", "time_s"]).reset_index(drop=True)
    seq = log[log["run_id"].isin([1, 2])].reset_index(drop=True)
    r = seq["lidar_range_m"].replace(0.0, np.nan)
    end = int(seq.index[seq["run_id"] == 1][-1])
    window = slice(end - 25, end + 15)
    idx = np.arange(len(seq))[window]
    plain = r.shift(-10).to_numpy()[window]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(
        idx,
        r.to_numpy()[window],
        "o-",
        ms=3,
        color=NAVY,
        label="LiDAR range (0.0 shown as gap)",
    )
    ax.plot(idx, plain, "s", ms=4, color=RED, label="target by plain shift(-10)")
    ax.axvline(end + 0.5, color="k", lw=1)
    ax.text(end - 24, 9, "run 1 ends at rest, 0.8 m from the wall", fontsize=8)
    ax.text(end + 1.5, 8.5, "run 2: first reading\n11 m away", fontsize=8)
    ax.annotate(
        "last 10 targets of run 1\ncome from run 2",
        (end - 4, 10.7),
        (end - 22, 5.5),
        fontsize=8,
        color=RED,
        arrowprops={"arrowstyle": "->", "color": RED},
    )
    ax.set_xlabel("row number")
    ax.set_ylabel("m")
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    ax.set_title(
        "PerGroup(ForecastTarget(...)) stops the shift at the run boundary",
        color=NAVY,
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "fig_per_group.png")


def fig_scaling() -> None:
    """Worked example: nearest neighbour in raw and in scaled units."""
    q, a, b = (5.0, 0.0), (5.1, -1.0), (6.2, 0.0)
    scales = (3.0, 0.4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (sx, sy), title in zip(
        axes, [(1.0, 1.0), scales], ["raw units", "scaled by training σ"]
    ):
        pts = {
            name: (p[0] / sx, p[1] / sy) for name, p in {"q": q, "A": a, "B": b}.items()
        }
        for name, (x, y) in pts.items():
            ax.plot(x, y, "o", ms=10, color={"q": NAVY, "A": RED, "B": GREEN}[name])
            ax.annotate(name, (x, y), (6, 6), textcoords="offset points", fontsize=11)
        for name in ("A", "B"):
            d = np.hypot(pts[name][0] - pts["q"][0], pts[name][1] - pts["q"][1])
            ax.plot(*zip(pts["q"], pts[name]), "--", color=GREY)
            mid = ((pts["q"][0] + pts[name][0]) / 2, (pts["q"][1] + pts[name][1]) / 2)
            ax.text(*mid, f" d={d:.3f}", fontsize=9)
        ax.margins(0.25)
        ax.set_title(title, color=NAVY, fontsize=10)
        ax.set_xlabel("range" + (" [m]" if sx == 1 else " / 3.0 m"))
        ax.set_ylabel("accel" + (" [m/s²]" if sy == 1 else " / 0.4 m/s²"))
        ax.grid(alpha=0.3)
    fig.suptitle(
        "q cruising at 5 m: A braking at 5.1 m, B cruising at 6.2 m", color=NAVY
    )
    fig.tight_layout()
    save(fig, "fig_scaling.png")


def fig_split_leakage() -> None:
    """Which rows go to test for each splitter, and what k-NN scores."""
    data = DatasetFactory.load("rover_range_forecast", path=ROVER_CSV)
    splitters = {
        "random rows": RandomRowSplit(0.25, seed=0),
        "chronological": ChronologicalSplit(0.25),
        "chronological, gap 10": ChronologicalSplit(0.25, gap=10),
        "by run (group)": GroupHoldoutSplit(0.25),
    }
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2, 1]}
    )
    scores = {}
    for y, (name, splitter) in enumerate(splitters.items()):
        (fold,) = splitter.split(len(data), data.groups)
        lane = np.full(len(data), np.nan)
        lane[fold.train], lane[fold.test] = 0.0, 1.0
        colors = np.where(
            np.isnan(lane), "white", np.where(lane == 1.0, RED, BASE_FILL)
        )
        ax1.scatter(
            np.arange(len(data)), np.full(len(data), y), c=colors, marker="|", s=120
        )
        ((train, test),) = split_dataset(data, splitter)
        scores[name] = (
            KNearestNeighbors(k=5).fit(train.X, train.y).score(test.X, test.y)
        )
    ax1.set_yticks(range(len(splitters)), list(splitters))
    ax1.invert_yaxis()
    ax1.set_xlabel(
        "row (runs 1 to 16 in time order); red = test, blue = train, white = purged"
    )
    ax2.barh(list(scores), list(scores.values()), color=[RED, ACCENT, GREEN, GREEN])
    for y, v in enumerate(scores.values()):
        ax2.text(v + 0.01, y, f"{v:.3f} m", va="center", fontsize=9)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("k-NN test RMSE [m]")
    ax2.set_xlim(0, 0.85)
    fig.suptitle("A random split hides 10 Hz twins in the training set", color=NAVY)
    fig.tight_layout()
    save(fig, "fig_split_leakage.png")


def fig_workflow() -> None:
    """The leak-free order: stateless cleaning, split, fit on train, transform both."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    box(ax, (0.1, 1.3), (1.6, 0.8), "raw log\n(all runs)", "#eeeeee")
    box(
        ax,
        (2.1, 1.3),
        (2.2, 0.8),
        "stateless cleaning\nsort, mark, Hampel,\nfill, target (per run)",
        BASE_FILL,
        9,
    )
    box(ax, (4.7, 1.3), (1.4, 0.8), "split\nby run", SUB_FILL)
    box(ax, (6.6, 2.3), (1.9, 0.8), "train runs", OK_FILL)
    box(ax, (6.6, 0.3), (1.9, 0.8), "test runs", "#f6d5d1")
    box(
        ax,
        (8.9, 2.3),
        (2.0, 0.8),
        "fit learned stages\nimpute, clip, scale",
        OK_FILL,
        9,
    )
    box(ax, (8.9, 0.3), (2.0, 0.8), "transform only\n(same numbers)", "#f6d5d1", 9)
    arrow(ax, (1.7, 1.7), (2.1, 1.7))
    arrow(ax, (4.3, 1.7), (4.7, 1.7))
    arrow(ax, (6.1, 1.9), (6.6, 2.6))
    arrow(ax, (6.1, 1.5), (6.6, 0.8))
    arrow(ax, (8.5, 2.7), (8.9, 2.7))
    arrow(ax, (8.5, 0.7), (8.9, 0.7))
    arrow(ax, (9.9, 2.3), (9.9, 1.1))
    ax.text(10.0, 1.65, "statistics", fontsize=8, color=NAVY)
    ax.text(
        2.1,
        0.9,
        "may see every row:\nlooks at one run at a time",
        fontsize=8,
        color=GREY,
    )
    fig.suptitle(
        "Leak-free order: nothing learned from the test runs reaches the model",
        color=NAVY,
    )
    save(fig, "fig_workflow.png")


def fig_forecast() -> None:
    """Linear forecast vs measured range for two unseen runs, with the naive model."""
    raw = pd.read_csv(ROVER_CSV)
    clean = rover_cleaning_pipeline().fit_transform(raw)
    ((train, test),) = split_frame(clean, GroupHoldoutSplit(0.25), "run_id")
    features = list(ROVER_FEATURES)
    learned = DataPipeline(
        [ImputeMedian(("gyro_z_dps",)), RobustScale(ROVER_FEATURES)]
    ).fit(train)
    model = LinearLeastSquares().fit(
        learned.transform(train)[features].to_numpy(), train[ROVER_TARGET]
    )
    predicted = model.predict(learned.transform(test)[features].to_numpy())
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for k, run_id in enumerate(test["run_id"].unique()[:2]):
        rows = (test["run_id"] == run_id).to_numpy()
        ax.plot(
            test.loc[rows, "time_s"],
            test.loc[rows, ROVER_TARGET],
            color=GREY,
            lw=3,
            label="measured range 1 s later" if k == 0 else None,
        )
        ax.plot(
            test.loc[rows, "time_s"],
            predicted[rows],
            "--",
            color=[NAVY, ACCENT][k],
            label=f"linear forecast, run {run_id}",
        )
    rmse = float(np.sqrt(np.mean((predicted - test[ROVER_TARGET].to_numpy()) ** 2)))
    ax.set_title(
        f"Unseen runs 13 and 14: test RMSE over runs 13-16 = {rmse:.3f} m (naive: 2.73 m)",
        color=NAVY,
        fontsize=10,
    )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_forecast.png")


def main() -> None:
    """Render every Week 5 figure."""
    raw = pd.read_csv(ROVER_CSV)
    truth = RoverLogGenerator(RoverLogConfig(seed=SEED)).generate_truth()
    fig_raw_vs_truth(raw, truth)
    fig_audit(raw)
    fig_mnar(truth)
    fig_masking()
    fig_outlier_rules(raw)
    fig_per_group(raw)
    fig_scaling()
    fig_split_leakage()
    fig_workflow()
    fig_forecast()


if __name__ == "__main__":
    main()
