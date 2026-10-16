"""Render the figures of the bonus-week slides and notes ("Vibe Coding in Engineering").

Provenance: new content (bonus week). Every plotted number is computed here,
except the study results in ``fig_evidence``, which are copied from the cited
sources (see ``EVIDENCE``). Output: slides/assets/week_extra_vibe_coding/*.png

Usage:
    python tools/make_week_extra_vibe_coding_figures.py
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerospace_ai.sensors import (  # noqa: E402
    BARO_MAX_PRESSURE_PA,
    ISA_SEA_LEVEL_PRESSURE_PA,
    pressure_to_altitude,
)
from aerospace_ai.verification import observed_order  # noqa: E402

OUT_DIR: Final[Path] = REPO_ROOT / "slides" / "assets" / "week_extra_vibe_coding"
DPI: Final[int] = 150
NAVY: Final[str] = "#1f3a5f"
ACCENT: Final[str] = "#e07a1f"
GREEN: Final[str] = "#3a8f4b"
RED: Final[str] = "#c0392b"
GREY: Final[str] = "#9a9a9a"
PASS_FILL: Final[str] = "#d6efd9"
FAIL_FILL: Final[str] = "#f6d5d1"
BOX_FILL: Final[str] = "#dbe6f3"

UNIVERSAL_GAS_CONSTANT: Final[float] = 8.314
SPECIFIC_GAS_CONSTANT: Final[float] = 287.05287
LAPSE_K_PER_M: Final[float] = 0.0065
T0_K: Final[float] = 288.15
G0: Final[float] = 9.80665

OMEGA_N: Final[float] = 3.0
ZETA: Final[float] = 0.5
THETA0: Final[float] = 0.1
T_END: Final[float] = 2.0
STEPS: Final[tuple[float, ...]] = (0.1, 0.05, 0.025, 0.0125)

# Change in completion time with AI (%), negative = faster, with the reported CI.
# Peng et al., arXiv:2302.06590 (55.8 % faster, CI 21-89 %); METR, arXiv:2507.09089
# (+19 %, CI +2 to +39 %); METR blog 2026-02-24 (-18 %, CI -38 to +9 %;
# -4 %, CI -15 to +9 %).
EVIDENCE: Final[tuple[tuple[str, float, float, float], ...]] = (
    ("Peng et al. 2023\n1 greenfield task, Copilot", -55.8, -89.0, -21.0),
    ("METR 2025 (arXiv:2507.09089)\nexperienced devs, own repos", 19.0, 2.0, 39.0),
    ("METR 2026 update\nreturning devs", -18.0, -38.0, 9.0),
    ("METR 2026 update\nnew devs", -4.0, -15.0, 9.0),
)
MUTATION_SCORES: Final[dict[str, tuple[int, int]]] = {
    "weak\n(runs, 0 m at sea level)": (1, 6),
    "typical\n(+ one hand value)": (4, 6),
    "strong\n(+ table, < 0 m, array,\nmonotonic)": (6, 6),
    "strong minus the\nassertions D fails": (4, 6),
}


def save(fig: plt.Figure, name: str) -> None:
    """Write ``fig`` to the output folder and close it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def candidate_b(p: float) -> float:
    """Universal instead of specific gas constant (teaching example)."""
    exponent = UNIVERSAL_GAS_CONSTANT * LAPSE_K_PER_M / G0
    return T0_K / LAPSE_K_PER_M * (1 - (p / ISA_SEA_LEVEL_PRESSURE_PA) ** exponent)


def candidate_d(p: float) -> float:
    """Correct formula, clamped to 0 m at and above 101 325 Pa (teaching example)."""
    if p >= ISA_SEA_LEVEL_PRESSURE_PA:
        return 0.0
    return float(pressure_to_altitude(p))


def fig_check_matrix() -> None:
    """Which candidate fails which check (outcomes as computed in the lab notebook)."""
    checks = [
        "table\nvalues",
        "monotonic",
        "round\ntrip",
        "below sea\nlevel",
        "vs Week 2\nlibrary",
    ]
    outcomes = {
        "A  rounded constants": [1, 1, 1, 1, 1],
        "B  R = 8.314": [0, 1, 0, 0, 0],
        "C  hPa in, feet out": [0, 1, 0, 0, 0],
        "D  clamp 'for safety'": [1, 0, 0, 0, 0],
    }
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    for row, (label, values) in enumerate(outcomes.items()):
        for col, ok in enumerate(values):
            ax.add_patch(
                plt.Rectangle(
                    (col, -row - 1), 0.95, 0.9, color=PASS_FILL if ok else FAIL_FILL
                )
            )
            ax.text(
                col + 0.475,
                -row - 0.55,
                "pass" if ok else "FAIL",
                ha="center",
                va="center",
                color=GREEN if ok else RED,
                fontsize=10,
                fontweight="bold",
            )
        ax.text(-0.1, -row - 0.55, label, ha="right", va="center", fontsize=10)
    for col, name in enumerate(checks):
        ax.text(
            col + 0.475, 0.15, name, ha="center", va="bottom", fontsize=9, color=NAVY
        )
    ax.set_xlim(-3.2, len(checks))
    ax.set_ylim(-len(outcomes) - 0.1, 1.2)
    ax.axis("off")
    ax.set_title("Four plausible answers, five checks: only A survives", color=NAVY)
    save(fig, "fig_check_matrix.png")


def fig_candidates() -> None:
    """Altitude of candidates B and D against the ISA formula (barometer range)."""
    pressure = np.linspace(30_000.0, BARO_MAX_PRESSURE_PA, 400)
    truth = pressure_to_altitude(pressure)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    left.plot(pressure / 1000, truth / 1000, color=NAVY, lw=2, label="ISA formula")
    left.plot(
        pressure / 1000,
        [candidate_b(p) / 1000 for p in pressure],
        color=RED,
        ls="--",
        label="B: R = 8.314",
    )
    left.set_xlabel("static pressure (kPa)")
    left.set_ylabel("pressure altitude (km)")
    left.legend()
    left.set_title("B: a plausible number, 5.4 km wrong at 50 kPa", fontsize=10)
    zoom = np.linspace(95_000.0, BARO_MAX_PRESSURE_PA, 200)
    right.plot(
        zoom / 1000, pressure_to_altitude(zoom), color=NAVY, lw=2, label="ISA formula"
    )
    right.plot(
        zoom / 1000,
        [candidate_d(p) for p in zoom],
        color=ACCENT,
        ls="--",
        lw=2,
        label="D: clamped",
    )
    right.axvline(ISA_SEA_LEVEL_PRESSURE_PA / 1000, color=GREY, lw=1)
    right.set_xlabel("static pressure (kPa)")
    right.set_ylabel("pressure altitude (m)")
    right.legend()
    right.set_title("D: hides every high-pressure day", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_candidates.png")


def pitch_rhs(t: float, x: np.ndarray) -> np.ndarray:
    """Damped pitch oscillator, state (theta, theta_dot)."""
    return np.array([x[1], -2.0 * ZETA * OMEGA_N * x[1] - OMEGA_N**2 * x[0]])


def pitch_exact(t: float) -> float:
    """Exact theta(t) for theta(0) = THETA0, theta_dot(0) = 0."""
    omega_d = OMEGA_N * math.sqrt(1.0 - ZETA**2)
    sigma = ZETA * OMEGA_N
    return (
        THETA0
        * math.exp(-sigma * t)
        * (math.cos(omega_d * t) + sigma / omega_d * math.sin(omega_d * t))
    )


Step = Callable[[Callable, float, np.ndarray, float], np.ndarray]


def rk4_step(
    f: Callable, t: float, x: np.ndarray, h: float, typo: bool = False
) -> np.ndarray:
    """Classical RK4; with ``typo`` the third stage uses k1 instead of k2."""
    k1 = f(t, x)
    k2 = f(t + h / 2, x + h / 2 * k1)
    k3 = f(t + h / 2, x + h / 2 * (k1 if typo else k2))
    k4 = f(t + h, x + h * k3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def pitch_error(step: Step, h: float) -> float:
    """Error at T_END of a fixed-step integration."""
    x = np.array([THETA0, 0.0])
    for i in range(round(T_END / h)):
        x = step(pitch_rhs, i * h, x, h)
    return abs(x[0] - pitch_exact(T_END))


def fig_convergence() -> None:
    """Log-log error versus step: Euler, RK4 with a typo, RK4."""
    methods: dict[str, tuple[Step, str]] = {
        "explicit Euler": (lambda f, t, x, h: x + h * f(t, x), GREY),
        "RK4 with k3 typo": (
            lambda f, t, x, h: rk4_step(f, t, x, h, typo=True),
            ACCENT,
        ),
        "RK4": (rk4_step, NAVY),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, (step, color) in methods.items():
        errors = [pitch_error(step, h) for h in STEPS]
        order = observed_order(errors, STEPS)[-1]
        ax.loglog(
            STEPS,
            errors,
            "o-",
            color=color,
            label=f"{label} (observed order {order:.2f})",
        )
    ax.axhline(1e-5, color=RED, ls=":", lw=1)
    ax.text(STEPS[-1], 1.4e-5, "one-off test tolerance 1e-5", color=RED, fontsize=8)
    ax.set_xlabel("step size h (s)")
    ax.set_ylabel("|theta(2 s) error| (rad)")
    ax.set_title("Halving h must divide the RK4 error by 16", color=NAVY)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    save(fig, "fig_convergence.png")


def fig_mutation() -> None:
    """Mutation score of four test suites on six seeded bugs (computed in the lab)."""
    labels = list(MUTATION_SCORES)
    killed = [k for k, _ in MUTATION_SCORES.values()]
    total = [n for _, n in MUTATION_SCORES.values()]
    colors = [RED, ACCENT, GREEN, ACCENT]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.bar(labels, killed, color=colors)
    for i, (k, n) in enumerate(zip(killed, total, strict=True)):
        ax.text(i, k + 0.1, f"{k}/{n}", ha="center", color=NAVY, fontweight="bold")
    ax.set_ylim(0, max(total) + 1)
    ax.set_ylabel("mutants killed")
    ax.set_title(
        "All four suites pass on correct code; only one catches every bug", color=NAVY
    )
    ax.tick_params(axis="x", labelsize=8)
    save(fig, "fig_mutation.png")


def fig_evidence() -> None:
    """Forest plot of the verified productivity results (numbers from the sources)."""
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    y = np.arange(len(EVIDENCE))[::-1]
    for yi, (_label, mid, low, high) in zip(y, EVIDENCE, strict=True):
        ax.errorbar(
            mid, yi, xerr=[[mid - low], [high - mid]], fmt="o", capsize=4, color=NAVY
        )
        ax.text(high + 2, yi, f"{mid:+.0f} %", va="center", fontsize=9, color=NAVY)
    ax.axvline(0.0, color=GREY, lw=1)
    ax.set_yticks(y, [row[0] for row in EVIDENCE], fontsize=8)
    ax.set_xlabel("change in task completion time with AI (%), negative = faster")
    ax.set_xlim(-100, 60)
    ax.set_title("Randomised trials disagree: the setting decides", color=NAVY)
    save(fig, "fig_evidence.png")


def box(ax: plt.Axes, xy: tuple[float, float], text: str, fill: str = BOX_FILL) -> None:
    """Rounded box of width 1.9 and height 0.9 with centred text."""
    ax.add_patch(
        FancyBboxPatch(
            xy, 1.9, 0.9, boxstyle="round,pad=0.05", fc=fill, ec=NAVY, lw=1.2
        )
    )
    ax.text(xy[0] + 0.95, xy[1] + 0.45, text, ha="center", va="center", fontsize=9)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Straight arrow between two points."""
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=NAVY)
    )


def fig_loop() -> None:
    """The verification loop: specify, generate small, check, review, commit."""
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    steps = [
        ("1 specify\nunits, ranges,\nreference values", BOX_FILL),
        ("2 generate\none small piece", "#eeeeee"),
        ("3 static checks\ndeliverables,\nimports", BOX_FILL),
        ("4 run checks\nreference, property,\norder, mutants", PASS_FILL),
        ("5 review + commit\nread every line;\nGit checkpoint", BOX_FILL),
    ]
    for i, (text, fill) in enumerate(steps):
        box(ax, (0.1 + 2.2 * i, 1.2), text, fill)
        if i < len(steps) - 1:
            arrow(ax, (2.0 + 2.2 * i, 1.65), (2.3 + 2.2 * i, 1.65))
    ax.add_patch(
        FancyArrowPatch(
            (9.9, 1.1),
            (1.05, 1.1),
            connectionstyle="arc3,rad=-0.12",
            arrowstyle="-|>",
            mutation_scale=14,
            color=ACCENT,
        )
    )
    ax.text(
        5.5,
        0.15,
        "a check fails: fix the specification or regenerate",
        ha="center",
        color=ACCENT,
        fontsize=9,
    )
    fig.suptitle(
        "AI writes, the engineer verifies: one small loop per piece", color=NAVY
    )
    save(fig, "fig_loop.png")


def main() -> None:
    """Render every figure."""
    fig_check_matrix()
    fig_candidates()
    fig_convergence()
    fig_mutation()
    fig_evidence()
    fig_loop()


if __name__ == "__main__":
    main()
