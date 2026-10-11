"""Render the figures used by the Week 2 slides and lecture notes.

Provenance: new content (Week 2). Output: slides/assets/week02/*.png

Usage:
    python tools/make_week02_figures.py
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
sys.path.insert(0, str(REPO_ROOT / "tools"))

from aerospace_ai.sensors import IMUSensor, pressure_to_altitude  # noqa: E402
from aerospace_ai.utils import STANDARD_GRAVITY_MPS2  # noqa: E402
from make_week02_data import LidarLogConfig, ray_range  # noqa: E402

DATA_DIR: Final[Path] = REPO_ROOT / "data" / "sample"
OUT_DIR: Final[Path] = REPO_ROOT / "slides" / "assets" / "week02"
DPI: Final[int] = 150
IMU_RANGE_G: Final[float] = 4.0
GROUND_END_S: Final[float] = 25.0
LOAD_LIMIT_G: Final[float] = 2.0
EARTH_MEAN_RADIUS_M: Final[float] = 6_371_000.0
GPS_ORIGIN: Final[tuple[float, float]] = (35.6892, 51.3134)
MAX_HDOP: Final[float] = 5.0

NAVY: Final[str] = "#1f3a5f"
ACCENT: Final[str] = "#e07a1f"
BASE_FILL: Final[str] = "#dbe6f3"
SUB_FILL: Final[str] = "#fbe3cc"
GREY_FILL: Final[str] = "#eeeeee"


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
    patch = FancyBboxPatch(
        xy,
        *size,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        fc=fill,
        ec=NAVY,
        lw=1.4,
        ls="--" if dashed else "-",
    )
    ax.add_patch(patch)
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
) -> None:
    """Draw an arrow from ``start`` to ``end``."""
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=16, color=color, lw=1.4
        )
    )


class Week02Figures:
    """Build every Week 2 figure."""

    def __init__(self, out_dir: Path) -> None:
        """Make sure the output folder exists."""
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig: plt.Figure, name: str) -> None:
        """Save ``fig`` as ``name`` in the output folder and close it."""
        fig.savefig(self.out_dir / name, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("wrote", (self.out_dir / name).relative_to(REPO_ROOT))

    def procedural_vs_oop(self) -> None:
        """Grid of functions x sensor types: what must change to add a barometer."""
        functions = ["validate", "preprocess", "to_frame", "plot"]
        sensors = ["IMU", "LiDAR", "GPS", "Baro (new)"]
        inherited = {"to_frame", "plot"}
        new_column = len(sensors) - 1
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        panels = (
            ("Procedural: if/elif in every function", False, "edit 4 functions"),
            ("OOP: one new subclass", True, "2 methods; rest inherited"),
        )
        for ax, (title, oop, caption) in zip(axes, panels):
            for i, fn in enumerate(functions):
                for j in range(len(sensors)):
                    color = BASE_FILL
                    if j == new_column:
                        color = GREY_FILL if (oop and fn in inherited) else ACCENT
                    ax.add_patch(
                        plt.Rectangle((j, -i), 0.92, 0.85, fc=color, ec="white")
                    )
                ax.text(
                    -0.1,
                    -i + 0.42,
                    fn,
                    ha="right",
                    va="center",
                    family="DejaVu Sans Mono",
                )
            for j, sensor in enumerate(sensors):
                ax.text(j + 0.46, 1.05, sensor, ha="center", va="bottom", fontsize=10)
            ax.text(
                new_column + 0.46,
                -3.85,
                caption,
                ha="center",
                color=ACCENT,
                fontsize=10,
            )
            ax.set_xlim(-1.6, 4.6)
            ax.set_ylim(-4.2, 1.6)
            ax.set_title(title, color=NAVY)
            ax.axis("off")
        self.save(fig, "fig_procedural_vs_oop.png")

    def class_hierarchy(self) -> None:
        """UML-style diagram of SensorReading, BaseSensor and its subclasses."""
        fig, ax = plt.subplots(figsize=(11, 5.6))
        box(
            ax,
            (3.3, 3.2),
            (4.4, 1.9),
            "<<abstract>> BaseSensor(ABC)\n"
            "sensor_id, rate_hz, period_s\n"
            "channel_names, n_rejected\n"
            "add_reading()  to_array()  to_frame()\n"
            "@abstractmethod validate(raw)\n"
            "@abstractmethod preprocess(raw)",
            BASE_FILL,
            fontsize=9,
        )
        box(
            ax,
            (8.4, 3.55),
            (2.5, 1.2),
            "@dataclass(frozen)\nSensorReading\ntimestamp, values,\nsensor_id, is_valid",
            GREY_FILL,
            fontsize=8.5,
        )
        arrow(ax, (7.7, 4.15), (8.4, 4.15), style="-")
        ax.text(8.05, 4.3, "stores *", ha="center", fontsize=8)
        subclasses = [
            ("IMUSensor\nbias removal\nsaturation check", False),
            ("BarometerSensor\nISA altitude\nrange check", False),
            ("LidarSensor\nNaN for no echo\nnearest_obstacle()", True),
            ("GPSSensor\nlat/lon -> ENU m\nHDOP check", True),
        ]
        for k, (label, assignment) in enumerate(subclasses):
            x = 0.2 + 2.75 * k
            box(
                ax,
                (x, 0.3),
                (2.4, 1.4),
                label,
                SUB_FILL,
                dashed=assignment,
                fontsize=8.8,
            )
            arrow(ax, (x + 1.2, 1.7), (5.5, 3.2), style="-|>")
        ax.text(
            9.6,
            0.05,
            "dashed = Assignment 1",
            ha="center",
            fontsize=9,
            color=ACCENT,
        )
        ax.set_xlim(0, 11.2)
        ax.set_ylim(-0.1, 5.3)
        ax.axis("off")
        self.save(fig, "fig_class_hierarchy.png")

    def template_method(self) -> None:
        """Flow of BaseSensor.add_reading: fixed steps vs subclass steps."""
        steps = [
            ("1. check shape", BASE_FILL),
            ("2. check time order", BASE_FILL),
            ("3. validate(raw)", SUB_FILL),
            ("4. preprocess(raw)", SUB_FILL),
            ("5. store SensorReading", BASE_FILL),
        ]
        fig, ax = plt.subplots(figsize=(11, 2.6))
        for k, (label, fill) in enumerate(steps):
            box(
                ax,
                (0.1 + 2.2 * k, 0.6),
                (1.9, 0.9),
                label.replace(" ", "\n", 1),
                fill,
                fontsize=9,
            )
            if k:
                arrow(ax, (2.2 * k, 1.05), (0.1 + 2.2 * k, 1.05))
        ax.text(
            5.6,
            0.2,
            "blue: BaseSensor (written once)   orange: each subclass",
            ha="center",
        )
        ax.annotate(
            "invalid: skip 4, store raw\nwith is_valid=False",
            xy=(5.5, 1.5),
            xytext=(1.0, 1.75),
            fontsize=8.5,
            arrowprops={"arrowstyle": "->", "color": ACCENT},
            color=ACCENT,
        )
        ax.set_xlim(0, 11.2)
        ax.set_ylim(0, 2.3)
        ax.axis("off")
        self.save(fig, "fig_template_method.png")

    def shared_list(self) -> None:
        """Memory picture of a class attribute list vs per-instance lists."""
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        for ax, shared in zip(axes, (True, False)):
            box(ax, (0.2, 2.0), (1.6, 0.8), "g1", BASE_FILL)
            box(ax, (0.2, 0.4), (1.6, 0.8), "g2", BASE_FILL)
            if shared:
                box(ax, (3.2, 1.2), (2.4, 0.8), "[1.5, 1.0]", ACCENT)
                ax.text(4.4, 2.25, "Gyro.samples (class)", ha="center", fontsize=9)
                arrow(ax, (1.8, 2.4), (3.2, 1.7))
                arrow(ax, (1.8, 0.8), (3.2, 1.5))
                ax.set_title("class attribute: one shared list", color=ACCENT)
            else:
                box(ax, (3.2, 2.0), (2.4, 0.8), "[1.5]", SUB_FILL)
                box(ax, (3.2, 0.4), (2.4, 0.8), "[1.0]", SUB_FILL)
                arrow(ax, (1.8, 2.4), (3.2, 2.4))
                arrow(ax, (1.8, 0.8), (3.2, 0.8))
                ax.set_title("self.samples = [] in __init__: one list each", color=NAVY)
            ax.set_xlim(0, 6)
            ax.set_ylim(0.1, 3.1)
            ax.axis("off")
        self.save(fig, "fig_shared_list.png")

    def imu_validation(self) -> None:
        """Week 1 log: raw load factor vs IMUSensor with saturated samples rejected."""
        log = pd.read_csv(DATA_DIR / "week01_uav_flight_log.csv")
        channels = list(IMUSensor.CHANNELS)
        imu = IMUSensor(
            "imu", 50.0, accel_range_mps2=IMU_RANGE_G * STANDARD_GRAVITY_MPS2
        )
        imu.calibrate(log.loc[log["time_s"] < GROUND_END_S, channels].to_numpy())
        imu.add_readings(log["time_s"].to_numpy(), log[channels].to_numpy())
        raw_n = (
            np.linalg.norm(log[channels[:3]].to_numpy(), axis=1) / STANDARD_GRAVITY_MPS2
        )
        fig, ax = plt.subplots(figsize=(10, 3.4))
        ax.plot(log["time_s"], raw_n, lw=0.5, color="0.6", label="raw log")
        ax.plot(
            imu.timestamps(),
            imu.load_factor(),
            lw=0.8,
            color="tab:red",
            label=f"IMUSensor, ±{IMU_RANGE_G:g} g: {imu.n_rejected} saturated samples rejected",
        )
        ax.axhline(LOAD_LIMIT_G, ls="--", color="k", lw=0.8, label="2.0 g limit")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Load factor n [-]")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        self.save(fig, "fig_imu_validation.png")

    def isa_altitude(self) -> None:
        """ISA pressure-altitude curve with the worked example marked."""
        pressure = np.linspace(60_000.0, 101_325.0, 200)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(pressure / 1000.0, pressure_to_altitude(pressure), color=NAVY)
        example_p = 95_000.0
        example_h = pressure_to_altitude(example_p)
        ax.plot(example_p / 1000.0, example_h, "o", color=ACCENT)
        ax.annotate(
            f"95.0 kPa -> {example_h:.1f} m",
            (example_p / 1000.0, example_h),
            xytext=(78, 1600),
            arrowprops={"arrowstyle": "->"},
        )
        ax.set_xlabel("Static pressure [kPa]")
        ax.set_ylabel("ISA pressure altitude [m]")
        ax.grid(alpha=0.3)
        self.save(fig, "fig_isa_altitude.png")

    def lidar_scan(self) -> None:
        """Top view of the hall with the 16 beams of one scan, unusable beams marked."""
        cfg = LidarLogConfig()
        scans = pd.read_csv(DATA_DIR / "assignment01_lidar_scans.csv")
        k = len(scans) - 1
        ranges = scans.iloc[k, 1:].to_numpy(dtype=float)
        position = np.array([cfg.start_x_m + cfg.speed_mps * scans.iloc[k, 0], cfg.y_m])
        angles = np.radians(-180.0 + 22.5 * np.arange(cfg.n_beams))
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.add_patch(
            plt.Rectangle((0, 0), cfg.hall_length_m, cfg.hall_width_m, fill=False, lw=2)
        )
        ax.add_patch(plt.Circle(cfg.pillar_center, cfg.pillar_radius_m, color="0.4"))
        for r, a in zip(ranges, angles):
            true_r = ray_range(position, a, cfg)
            end = position + true_r * np.array([np.cos(a), np.sin(a)])
            usable = np.isfinite(r) and 0.15 <= r <= 12.0
            ax.plot(
                *zip(position, end),
                color="tab:green" if usable else ACCENT,
                lw=1.0 if usable else 1.6,
                ls="-" if usable else "--",
            )
            if usable:
                ax.plot(*end, "o", color="tab:green", ms=4)
        ax.plot(*position, "k^", ms=11)
        ax.plot([], [], color="tab:green", label="echo in range")
        ax.plot([], [], color=ACCENT, ls="--", label="0.0 / inf / dust: becomes NaN")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)
        ax.set_title("16-beam LiDAR scan in a 10 m x 6 m hall with a pillar")
        self.save(fig, "fig_lidar_scan.png")

    def gps_enu(self) -> None:
        """GPS track converted to local metres, rejected fixes marked."""
        track = pd.read_csv(DATA_DIR / "assignment01_gps_track.csv")
        lat0, lon0 = GPS_ORIGIN
        north = EARTH_MEAN_RADIUS_M * np.radians(track["lat_deg"] - lat0)
        east = (
            EARTH_MEAN_RADIUS_M
            * np.cos(np.radians(lat0))
            * np.radians(track["lon_deg"] - lon0)
        )
        bad_hdop = track["hdop"] > MAX_HDOP
        plausible = north.abs() < 1_000
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        ok = ~bad_hdop & plausible
        ax.plot(east[ok], north[ok], ".", ms=4, color=NAVY, label="valid fix")
        ax.plot(
            east[bad_hdop],
            north[bad_hdop],
            "x",
            ms=8,
            color=ACCENT,
            label=f"HDOP > {MAX_HDOP:g} (rejected)",
        )
        ax.plot(0, 0, "k*", ms=12, label="origin: Mehrabad")
        ax.set_aspect("equal")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.grid(alpha=0.3)
        ax.legend(loc="center", fontsize=8.5)
        ax.set_title("GPS track in the local ENU frame")
        self.save(fig, "fig_gps_enu.png")

    def build_all(self) -> None:
        """Render every figure."""
        self.procedural_vs_oop()
        self.class_hierarchy()
        self.template_method()
        self.shared_list()
        self.imu_validation()
        self.isa_altitude()
        self.lidar_scan()
        self.gps_enu()


if __name__ == "__main__":
    Week02Figures(OUT_DIR).build_all()
