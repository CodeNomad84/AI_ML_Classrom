"""Your pytest tests for Practice Assignment 3 (Task 5).

Provenance: new content (Week 3). Pytest usage follows (rewritten, not copied)
GeeksforGeeks "Getting Started with Pytest".

Run from the repository root:

    python -m pytest assignments/practice03 -q

Rules for Task 5 (see the README):
* at least 6 test functions in total (the two examples below count),
* use at least one ``pytest.raises``, one ``@pytest.mark.parametrize`` and one
  fixture,
* tests must be self-contained: small hand-made frames, no CSV files,
* every test must pass on a correct implementation. The instructor also runs
  your tests against three subtly broken implementations; each one your tests
  catch earns points.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from airdata_stages import (  # noqa: F401  (you will need the others)
    MinMaxScale,
    PlausibleRange,
    Resample,
    build_airdata_pipeline,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Return a five-row frame sampled at 2 Hz with one impossible value."""
    return pd.DataFrame(
        {"time_s": [0.0, 0.5, 1.0, 1.5, 2.0], "x": [1.0, 2.0, 100.0, 4.0, 5.0]}
    )


def test_plausible_range_turns_impossible_values_into_nan(frame: pd.DataFrame) -> None:
    out = PlausibleRange(["x"], 0.0, 10.0)(frame)
    assert math.isnan(out["x"].iloc[2])
    assert out["x"].drop(index=2).tolist() == [1.0, 2.0, 4.0, 5.0]
    assert frame["x"].iloc[2] == 100.0, "the input must not be modified"


@pytest.mark.parametrize("rate_hz", [0.0, -10.0, math.inf, math.nan])
def test_resample_rejects_bad_rates(rate_hz: float) -> None:
    with pytest.raises(ValueError):
        Resample(rate_hz)


# TODO: at least four more tests. Ideas: Resample on an uneven time axis (check
# the grid and one interpolated value by hand), MinMaxScale before fit
# (NotFittedError), after fit on a tiny frame (check exact numbers, including a
# custom feature_range), inverse_transform, the pipeline's stage order.
