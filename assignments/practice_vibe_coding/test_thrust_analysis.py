"""Your tests for thrust_analysis.py (Practice Assignment VC).

Provenance: new content (bonus week). Two examples are given; add your own
below them. A good suite fails on every plausible wrong implementation: think
about uneven time steps, negative noise samples, the >= in the burn-time
rule, units of the specific impulse, and invalid input. At least one expected
value must come from a calculation you did by hand (write it in a comment).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from thrust_analysis import ThrustCurve, specific_impulse  # noqa: F401


def test_triangle_pulse_impulse_by_hand() -> None:
    # Hand calculation: area of a triangle, base 2 s, height 10 N -> 10 N s.
    # The trapezoidal rule is exact for piecewise-linear thrust.
    curve = ThrustCurve(np.array([0.0, 1.0, 2.0]), np.array([0.0, 10.0, 0.0]))
    assert math.isclose(curve.total_impulse(), 10.0, rel_tol=1e-12)


def test_decreasing_time_is_rejected() -> None:
    with pytest.raises(ValueError):
        ThrustCurve(np.array([0.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0]))


# TODO: add your tests here (at least 8 in total).
