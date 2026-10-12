"""Your pytest suite for Assignment 2 (Task 6). Provenance: new content (Week 5).

Two examples are given. Add at least 5 more tests: use a fixture, at least one
``@pytest.mark.parametrize`` and at least one ``pytest.raises``. Good tests
check numbers you worked out by hand (a five-value column whose quartiles you
can compute on paper), not just types.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # noqa: F401 (for your tests)
import pytest

from sensor_cleaning import (  # noqa: F401 (for your tests)
    FlagStuck,
    IQRClipper,
    audit_report,
    build_cleaning_pipeline,
    build_learned_pipeline,
)


def test_flag_stuck_hand_example() -> None:
    values = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
    mask = FlagStuck(("x",), min_run=3).stuck_mask(values)
    assert mask.tolist() == [False, True, True, True, False, False]


def test_iqr_clipper_needs_positive_k() -> None:
    with pytest.raises(ValueError):
        IQRClipper(("x",), k=0.0)


# TODO: your tests below (at least 5 more)
