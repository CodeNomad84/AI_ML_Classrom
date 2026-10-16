"""Unit tests for aerospace_ai.verification (bonus week). Provenance: new content."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aerospace_ai.verification import (
    ConvergenceCheck,
    DifferentialCheck,
    Mutant,
    MutationRunner,
    Outcome,
    PropertyCheck,
    ReferenceCase,
    ReferenceCheck,
    Verifier,
    audit_deliverables,
    defined_names,
    imported_modules,
    observed_order,
    unresolvable_imports,
)

SQUARE_SOURCE = '''
def square(x):
    """Return x squared."""
    return x * x
'''


def euler_error(_func: object, h: float) -> float:
    """Global error of explicit Euler on dx/dt = -x, x(0) = 1, at t = 1."""
    x = 1.0
    for _ in range(round(1.0 / h)):
        x += h * (-x)
    return abs(x - math.exp(-1.0))


def test_reference_case_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError):
        ReferenceCase("bad", (1.0,), 1.0, rel_tol=-1.0)


def test_reference_check_pass_fail_and_error() -> None:
    check = ReferenceCheck(
        [ReferenceCase("two", (2.0,), 4.0), ReferenceCase("three", (3.0,), 9.0)]
    )
    assert check(lambda x: x * x).passed
    failed = check(lambda x: 2 * x)
    assert (
        failed.outcome is Outcome.FAIL
        and "three" in failed.detail
        and "two" not in failed.detail
    )
    crashed = check(lambda x: 1 / 0)
    assert crashed.outcome is Outcome.ERROR and "ZeroDivisionError" in crashed.detail
    with pytest.raises(ValueError):
        ReferenceCheck([])


def test_property_check_finds_counterexample_reproducibly() -> None:
    def sample(rng: np.random.Generator) -> tuple[float, float]:
        a, b = sorted(rng.uniform(-5.0, 5.0, size=2))
        return (a, b)

    def monotonic(func, args) -> bool:  # type: ignore[no-untyped-def]
        return func(args[0]) <= func(args[1])

    check = PropertyCheck("monotonic", sample, monotonic, trials=50, seed=3)
    assert check(lambda x: x**3).passed
    first, second = check(lambda x: x * x), check(lambda x: x * x)
    assert first.outcome is Outcome.FAIL and first.detail == second.detail
    with pytest.raises(ValueError):
        PropertyCheck("p", sample, monotonic, trials=0)


def test_differential_check_against_oracle() -> None:
    check = DifferentialCheck(np.sin, lambda rng: (rng.uniform(-1.0, 1.0),), trials=100)
    assert check(lambda x: math.sin(x)).passed
    taylor = check(lambda x: x - x**3 / 6.0)
    assert taylor.outcome is Outcome.FAIL


def test_observed_order_exact_and_validation() -> None:
    steps = [0.1, 0.05, 0.025]
    errors = [3.0 * h**2 for h in steps]
    assert np.allclose(observed_order(errors, steps), [2.0, 2.0])
    with pytest.raises(ValueError):
        observed_order([1.0], [0.1])
    with pytest.raises(ValueError):
        observed_order([1.0, 0.0], [0.1, 0.05])


def test_convergence_check_on_euler() -> None:
    steps = [0.1, 0.05, 0.025, 0.0125]
    assert ConvergenceCheck(euler_error, steps, expected_order=1.0)(None).passed  # type: ignore[arg-type]
    wrong = ConvergenceCheck(euler_error, steps, expected_order=2.0)(None)  # type: ignore[arg-type]
    assert wrong.outcome is Outcome.FAIL and "observed orders" in wrong.detail


def test_verifier_report_and_compare() -> None:
    verifier = Verifier().add(ReferenceCheck([ReferenceCase("one", (1.0,), 1.0)]))
    verifier.add(ReferenceCheck([ReferenceCase("two", (2.0,), 4.0)], name="second"))
    report = verifier.verify(lambda x: x * x, "square")
    assert report.passed and report.summary() == "square: 2/2 checks passed"
    assert list(report.to_frame().columns) == ["check", "outcome", "detail"]
    table = verifier.compare({"good": lambda x: x * x, "bad": lambda x: x})
    assert table.loc["good"].tolist() == ["pass", "pass"]
    assert table.loc["bad"].tolist() == ["pass", "fail"]
    with pytest.raises(ValueError):
        Verifier().verify(abs)


def test_mutation_runner_scores_weak_and_strong_suites() -> None:
    mutants = [
        Mutant("plus", "x * x", "x + x", "adds instead of multiplying"),
        Mutant("cube", "x * x", "x * x * x", "wrong power"),
    ]
    runner = MutationRunner(SQUARE_SOURCE, "square", mutants)
    assert runner.build()(3) == 9

    def weak(func) -> bool:  # type: ignore[no-untyped-def]
        return func(2) == 4  # 2 + 2 == 2 * 2: the "plus" mutant survives

    def strong(func) -> bool:  # type: ignore[no-untyped-def]
        return func(2) == 4 and func(3) == 9

    weak_report = runner.run(weak)
    assert weak_report.killed == ("cube",) and weak_report.survived == ("plus",)
    assert str(weak_report) == "mutation score 1/2 = 50%"
    assert runner.run(strong).score == 1.0
    with pytest.raises(ValueError, match="original"):
        runner.run(lambda func: func(2) == 5)


def test_mutant_pattern_must_be_unique() -> None:
    with pytest.raises(ValueError, match="2 times"):
        Mutant("x", "x", "y").apply("x + x")
    with pytest.raises(ValueError):
        MutationRunner(SQUARE_SOURCE, "square", [Mutant("missing", "x ** 2", "x")])
    with pytest.raises(ValueError):
        MutationRunner(SQUARE_SOURCE, "square", [])


def test_static_checks_do_not_execute_code() -> None:
    source = (
        "import numpy as np\nimport os.path\nfrom math import pi\n"
        "from . import sibling\nimport surely_not_a_real_module_xyz\n"
        "raise SystemExit('this must never run')\n"
        "def altitude(p):\n    return 0\nclass Report:\n    pass\n"
    )
    assert defined_names(source) == {"altitude", "Report"}
    assert imported_modules(source) == {
        "numpy",
        "os",
        "math",
        "surely_not_a_real_module_xyz",
    }
    assert unresolvable_imports(source) == {"surely_not_a_real_module_xyz"}
    audit = audit_deliverables(source, ["altitude", "inverse"])
    assert audit.missing == {"inverse"} and audit.unrequested == {"Report"}
    assert not audit.complete
