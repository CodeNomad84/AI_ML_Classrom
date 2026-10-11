"""Assignment 1: LiDAR and GPS sensors on top of the course BaseSensor.

Provenance: new content (Week 2). OOP concepts follow (rewritten, not copied)
W3Schools "Python Inheritance" and GeeksforGeeks "Abstract Classes in Python";
flat-Earth local coordinates adapted from Groves, *Principles of GNSS, Inertial,
and Multisensor Integrated Navigation Systems*, Ch. 2.

Fill in every ``# TODO`` block. Do not rename classes, methods or constants:
the notebook and the instructor's tests import them by name.
Write all comments in English.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar, Final

import numpy as np
import pandas as pd

from aerospace_ai.sensors import DEFAULT_BUFFER_SIZE, BaseSensor

# ---------------------------------------------------------------------------
# Constants (use these; do not type the numbers again in the classes).
# ---------------------------------------------------------------------------
EARTH_MEAN_RADIUS_M: Final[float] = 6_371_000.0
MAX_LATITUDE_DEG: Final[float] = 90.0
MAX_LONGITUDE_DEG: Final[float] = 180.0
DEFAULT_MAX_HDOP: Final[float] = 5.0

DEFAULT_N_BEAMS: Final[int] = 16
DEFAULT_FOV_DEG: Final[float] = 360.0
DEFAULT_MIN_RANGE_M: Final[float] = 0.15
DEFAULT_MAX_RANGE_M: Final[float] = 12.0
DEFAULT_MIN_VALID_FRACTION: Final[float] = 0.5

REPORT_COLUMNS: Final[list[str]] = [
    "sensor_type",
    "n_readings",
    "n_valid",
    "n_rejected",
    "pct_rejected",
]
PERCENT: Final[float] = 100.0


class LidarSensor(BaseSensor):
    """Planar scanning LiDAR with ``n_beams`` equally spaced beams.

    Raw sample: one range per beam in metres. A beam with no echo reports
    0.0 or ``inf``. Beam ``i`` points at ``angles_deg[i]`` measured from the
    vehicle's x-axis (forward), counter-clockwise positive.
    """

    __slots__ = (
        "_n_beams",
        "fov_deg",
        "min_range_m",
        "max_range_m",
        "min_valid_fraction",
    )

    def __init__(
        self,
        sensor_id: str,
        rate_hz: float,
        n_beams: int = DEFAULT_N_BEAMS,
        fov_deg: float = DEFAULT_FOV_DEG,
        min_range_m: float = DEFAULT_MIN_RANGE_M,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
        min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        """Create a LiDAR (given: read it as a model for GPSSensor.__init__).

        Args:
            sensor_id: Unique name, for example ``"lidar0"``.
            rate_hz: Scan rate in Hz.
            n_beams: Number of beams per scan (positive).
            fov_deg: Angular field of view in degrees, in (0, 360].
            min_range_m: Shortest range the sensor can measure.
            max_range_m: Longest range the sensor can measure.
            min_valid_fraction: Fraction of beams that must be in range for
                the whole scan to be valid.
            buffer_size: Maximum number of stored scans.

        Raises:
            ValueError: If any argument is outside its allowed range.
        """
        if n_beams <= 0:
            raise ValueError("n_beams must be positive")
        if not 0.0 < fov_deg <= DEFAULT_FOV_DEG:
            raise ValueError("fov_deg must be in (0, 360]")
        if not 0.0 <= min_range_m < max_range_m:
            raise ValueError("need 0 <= min_range_m < max_range_m")
        if not 0.0 < min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        # n_beams must exist before super().__init__ in case the base class
        # ever reads channel_names during initialisation.
        self._n_beams = n_beams
        super().__init__(sensor_id, rate_hz, buffer_size)
        self.fov_deg = fov_deg
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m
        self.min_valid_fraction = min_valid_fraction

    @property
    def n_beams(self) -> int:
        """Return the number of beams per scan."""
        return self._n_beams

    # ----------------------------------------------------------------- Task 4
    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return one channel name per beam: range_00_m, range_01_m, ...

        The number has two digits (format spec ``:02d``). This property
        *overrides* the base-class version, which returns ``CHANNELS``.
        """
        # TODO (Task 4): build the tuple with a generator expression.
        raise NotImplementedError

    @property
    def angles_deg(self) -> np.ndarray:
        """Return the beam angles in degrees, starting at -fov/2.

        Beam i points at -fov/2 + i * fov / n_beams. For the default 16 beams
        over 360 deg: [-180, -157.5, ..., 157.5].
        """
        # TODO (Task 4): one NumPy expression, no loop.
        raise NotImplementedError

    # ----------------------------------------------------------------- Task 5
    def _in_range(self, ranges: np.ndarray) -> np.ndarray:
        """Return a boolean mask of beams with a usable echo.

        A beam is usable if its range is finite and
        ``min_range_m <= range <= max_range_m``.
        """
        # TODO (Task 5): careful, comparisons with inf/NaN must not warn or
        # give a wrong answer.
        raise NotImplementedError

    def validate(self, raw: np.ndarray) -> bool:
        """Accept a scan if at least ``min_valid_fraction`` of beams are usable."""
        # TODO (Task 5)
        raise NotImplementedError

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Replace beams without a usable echo by NaN (keep the others)."""
        # TODO (Task 5)
        raise NotImplementedError

    # ----------------------------------------------------------------- Task 6
    def nearest_obstacle(self) -> tuple[float, float]:
        """Return (distance_m, angle_deg) of the closest echo in the latest valid scan.

        Use ``self.readings(valid_only=True)`` (inherited) and ignore NaN beams.

        Raises:
            ValueError: If there is no valid scan yet.
        """
        # TODO (Task 6)
        raise NotImplementedError


class GPSSensor(BaseSensor):
    """GNSS receiver that reports positions in a local East-North-Up frame.

    Raw sample: ``(latitude_deg, longitude_deg, altitude_m, hdop)``.
    Stored values: ``(east_m, north_m, up_m, hdop)`` relative to the origin,
    using the flat-Earth approximation (valid within a few kilometres)::

        north = R * radians(lat - lat0)
        east  = R * cos(radians(lat0)) * radians(lon - lon0)
        up    = alt - alt0
    """

    CHANNELS: ClassVar[tuple[str, ...]] = ("east_m", "north_m", "up_m", "hdop")
    UNITS: ClassVar[tuple[str, ...]] = ("m", "m", "m", "-")

    __slots__ = ("_origin", "max_hdop")

    # ----------------------------------------------------------------- Task 1
    def __init__(
        self,
        sensor_id: str,
        rate_hz: float,
        origin_lat_deg: float,
        origin_lon_deg: float,
        origin_alt_m: float = 0.0,
        max_hdop: float = DEFAULT_MAX_HDOP,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        """Create a GPS receiver.

        Args:
            sensor_id: Unique name, for example ``"gps0"``.
            rate_hz: Fix rate in Hz.
            origin_lat_deg: Latitude of the local frame origin.
            origin_lon_deg: Longitude of the local frame origin.
            origin_alt_m: Altitude of the local frame origin.
            max_hdop: Fixes with a larger HDOP are rejected.
            buffer_size: Maximum number of stored fixes.

        Raises:
            ValueError: If the origin is not a valid coordinate or
                ``max_hdop`` is not positive.
        """
        # TODO (Task 1):
        #   1. let the base class initialise id, rate and buffer (super()),
        #   2. check the origin with _valid_coordinate and max_hdop > 0,
        #   3. store the origin as a tuple of three floats in self._origin,
        #   4. store max_hdop.
        raise NotImplementedError

    @property
    def origin(self) -> tuple[float, float, float]:
        """Return the local frame origin (lat_deg, lon_deg, alt_m), read-only."""
        # TODO (Task 1)
        raise NotImplementedError

    @staticmethod
    def _valid_coordinate(lat_deg: float, lon_deg: float) -> bool:
        """Return True for a finite latitude/longitude inside its range (given)."""
        return (
            math.isfinite(lat_deg)
            and math.isfinite(lon_deg)
            and abs(lat_deg) <= MAX_LATITUDE_DEG
            and abs(lon_deg) <= MAX_LONGITUDE_DEG
        )

    # ----------------------------------------------------------------- Task 2
    def validate(self, raw: np.ndarray) -> bool:
        """Reject impossible coordinates and fixes with poor geometry.

        Valid when: the coordinate is valid, the altitude and HDOP are finite,
        and 0 < hdop <= max_hdop.
        """
        # TODO (Task 2)
        raise NotImplementedError

    # ----------------------------------------------------------------- Task 3
    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Convert (lat, lon, alt, hdop) to (east, north, up, hdop) in metres."""
        # TODO (Task 3): use the formulas in the class docstring.
        raise NotImplementedError

    def distance_travelled_m(self) -> float:
        """Return the horizontal path length through all valid fixes.

        Sum of the straight-line (east, north) distances between consecutive
        valid fixes; 0.0 if there are fewer than two.
        """
        # TODO (Task 3): np.diff + np.linalg.norm, no Python loop.
        raise NotImplementedError


# --------------------------------------------------------------------- Task 7
def sensor_report(sensors: Sequence[BaseSensor]) -> pd.DataFrame:
    """Summarise any mix of sensors in one table (polymorphism in action).

    Args:
        sensors: Any objects derived from :class:`BaseSensor`.

    Returns:
        DataFrame indexed by ``sensor_id`` (index name ``"sensor_id"``) with
        columns ``REPORT_COLUMNS``: the class name, number of stored readings,
        valid readings, rejected readings and the rejected percentage
        (0.0 for a sensor without readings).

    Raises:
        ValueError: If two sensors share the same ``sensor_id``.
    """
    # TODO (Task 7): use only the BaseSensor interface (len, readings, ...),
    # never isinstance checks for specific sensor types.
    raise NotImplementedError
