"""Your pytest suite for Practice 4 (Task 4). Provenance: new content (Week 4).

Two examples are given. Add at least 6 more tests: use a fixture, at least one
``@pytest.mark.parametrize`` and at least one ``pytest.raises``. Good tests
check numbers you worked out by hand, not just types.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerospace_ai.datasets import DatasetFactory  # noqa: F401 (for your tests)
from aerospace_ai.models import LinearLeastSquares
from patterns_practice import (  # noqa: F401 (for your tests)
    FlightRegimeLoader,
    PolynomialLeastSquares,
    ReduceLROnPlateau,
)


def test_polynomial_degree_must_be_positive() -> None:
    with pytest.raises(ValueError):
        PolynomialLeastSquares(degree=0)


def test_polynomial_degree_one_without_interactions_equals_linear() -> None:
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(20, 3)), rng.normal(size=20)
    poly = PolynomialLeastSquares(degree=1, interactions=False).fit(X, y)
    np.testing.assert_allclose(
        poly.predict(X), LinearLeastSquares().fit(X, y).predict(X)
    )


# TODO: your tests below (at least 6 more)
