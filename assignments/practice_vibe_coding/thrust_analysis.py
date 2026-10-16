"""Practice Assignment VC: static-fire thrust analysis (student file).

Provenance: new content (bonus week "Vibe Coding in Engineering").

This file is the *specification*: the signatures, docstrings and constants
below are what your AI assistant must implement. Give it this file and
``SPEC.md``, keep its first answer unedited in ``ai_raw_output.py``, then put
your reviewed and corrected version here. Every ``# TODO`` must be gone when
you submit. Do not change names, signatures or the constants: the hidden
tests import them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: F401  (you will need it for from_csv)

from aerospace_ai.utils import STANDARD_GRAVITY_MPS2  # noqa: F401  (for Isp)

BURN_THRESHOLD_FRACTION: Final[float] = 0.05
MIN_SAMPLES: Final[int] = 2


@dataclass(frozen=True, slots=True)
class ThrustCurve:
    """Thrust samples of one static-fire test.

    The constructor stores read-only float copies of both arrays and raises
    ``ValueError`` when: the arrays are not 1-D or differ in length; there are
    fewer than ``MIN_SAMPLES`` samples; any value is NaN or infinite; or the
    times are not strictly increasing (a repeated timestamp is an error).

    Attributes:
        time_s: Sample times in seconds, strictly increasing, NOT necessarily
            evenly spaced (the logger jitters and drops packets).
        thrust_n: Measured thrust in newtons, same length as ``time_s``.
    """

    time_s: np.ndarray
    thrust_n: np.ndarray

    def __post_init__(self) -> None:
        # TODO: validate and store read-only float copies (see the class docstring).
        raise NotImplementedError

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        time_column: str = "time_s",
        thrust_column: str = "thrust_n",
    ) -> ThrustCurve:
        """Read a curve from a CSV file with a time and a thrust column."""
        # TODO
        raise NotImplementedError

    def clipped_thrust(self) -> np.ndarray:
        """Thrust with negative samples (load-cell noise) set to zero."""
        # TODO
        raise NotImplementedError

    def total_impulse(self) -> float:
        """Total impulse in N s: trapezoidal rule on the clipped thrust.

        Use the actual timestamps (uneven spacing). Must run with NumPy 1.26,
        the course's pinned version.
        """
        # TODO
        raise NotImplementedError

    def peak_thrust(self) -> float:
        """Largest measured (not clipped) thrust in N."""
        # TODO
        raise NotImplementedError

    def burn_time(self, threshold_fraction: float = BURN_THRESHOLD_FRACTION) -> float:
        """Burn time in s, by the course convention (not a standard).

        The time from the first to the last sample whose measured thrust is
        greater than or equal to ``threshold_fraction * peak_thrust()``.

        Raises:
            ValueError: If ``threshold_fraction`` is not strictly between 0 and 1.
        """
        # TODO
        raise NotImplementedError

    def average_thrust(
        self, threshold_fraction: float = BURN_THRESHOLD_FRACTION
    ) -> float:
        """Total impulse divided by the burn time, in N.

        Raises:
            ValueError: If the burn time is zero.
        """
        # TODO
        raise NotImplementedError


def specific_impulse(total_impulse_ns: float, propellant_mass_kg: float) -> float:
    """Specific impulse in s: I / (m_p g0), with g0 = ``STANDARD_GRAVITY_MPS2``.

    Raises:
        ValueError: If the propellant mass is not positive.
    """
    # TODO
    raise NotImplementedError
