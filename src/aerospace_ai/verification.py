"""Verification harness for engineering code, whoever (or whatever) wrote it.

Provenance: new content (bonus week "Vibe Coding in Engineering"). The ideas
are rewritten, not copied, from these sources:

* Code verification against exact solutions and the *observed order of
  accuracy*, p = ln(e_coarse / e_fine) / ln(h_coarse / h_fine), follow
  S. P. Veluri, C. J. Roy and E. A. Luke, "Comprehensive Code Verification
  Techniques for Finite Volume CFD Codes", *Computers & Fluids* (2012), Eq. 4.
* Mutation testing (seed a known bug, check that the tests fail) follows
  R. A. DeMillo, R. J. Lipton and F. G. Sayward, "Hints on Test Data
  Selection: Help for the Practicing Programmer", *IEEE Computer* 11(4), 1978.
* "Count what you asked for" (an AI assistant can silently omit or delete
  parts of a task) paraphrases G. Kim and S. Yegge, *Vibe Coding* (IT
  Revolution, 2025), Chs. 4 and 7.
* The unresolvable-import check is motivated by J. Spracklen et al., "We Have
  a Package for You! A Comprehensive Analysis of Package Hallucinations by Code
  Generating LLMs", USENIX Security 2025 (arXiv:2406.10279).

Design:

* :class:`Check` is an abstract base class with a template method
  (:meth:`Check.__call__`) that turns any exception raised by the code under
  test into an ``ERROR`` result instead of crashing the harness. Concrete
  checks: :class:`ReferenceCheck`, :class:`PropertyCheck`,
  :class:`DifferentialCheck` and :class:`ConvergenceCheck`.
* :class:`Verifier` is a composite: it runs a list of checks on one callable
  and returns an immutable :class:`VerificationReport`.
* :class:`MutationRunner` measures how good a *test suite* is by running it
  against deliberately broken copies (:class:`Mutant`) of a trusted source.
* :func:`audit_deliverables` and :func:`unresolvable_imports` are static
  checks on source text: nothing is executed.

Safety note: :class:`MutationRunner` executes source code with ``exec``. Only
give it code you have read. Run unknown AI-generated code in a throw-away
environment (a container or a fresh virtual environment), never with access to
your real files, credentials or lab hardware.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final

import numpy as np
import pandas as pd

DEFAULT_REL_TOL: Final[float] = 1e-9
DEFAULT_TRIALS: Final[int] = 200
DEFAULT_SEED: Final[int] = 0
MIN_STEPS_FOR_ORDER: Final[int] = 2


class Outcome(Enum):
    """Result of one check: the code passed, failed, or raised an exception."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Immutable record of one check run on one callable."""

    check: str
    outcome: Outcome
    detail: str

    @property
    def passed(self) -> bool:
        """True only for ``Outcome.PASS``."""
        return self.outcome is Outcome.PASS


@dataclass(frozen=True, slots=True)
class ReferenceCase:
    """One input with a trusted expected output and the source of that value.

    Attributes:
        label: Short human-readable name, for example ``"tropopause"``.
        args: Positional arguments passed to the function under test.
        expected: Trusted output (a table value, a hand calculation, an exact
            solution).
        rel_tol: Relative tolerance for :func:`math.isclose`.
        abs_tol: Absolute tolerance for :func:`math.isclose`.
        source: Where the expected value comes from. Never leave it empty in
            real work: an expected value without a source is just another guess.
    """

    label: str
    args: tuple[Any, ...]
    expected: float
    rel_tol: float = DEFAULT_REL_TOL
    abs_tol: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        if self.rel_tol < 0 or self.abs_tol < 0:
            raise ValueError("tolerances must be non-negative")


class Check(ABC):
    """Abstract verification check applied to one callable.

    Subclasses implement :meth:`run`. Calling the check object runs it inside
    an exception guard, so a crash in the code under test is reported as
    ``Outcome.ERROR`` with the exception text.
    """

    kind: ClassVar[str] = "check"

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.kind

    @abstractmethod
    def run(self, func: Callable[..., Any]) -> CheckResult:
        """Run the check on ``func`` and return the result."""

    def __call__(self, func: Callable[..., Any]) -> CheckResult:
        """Template method: run the check and convert exceptions into ERROR."""
        try:
            return self.run(func)
        except Exception as error:  # noqa: BLE001 - any failure of func is data
            return CheckResult(
                self.name, Outcome.ERROR, f"{type(error).__name__}: {error}"
            )

    def _result(self, ok: bool, detail: str) -> CheckResult:
        return CheckResult(self.name, Outcome.PASS if ok else Outcome.FAIL, detail)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class ReferenceCheck(Check):
    """Compare a function with trusted reference values (tables, exact solutions)."""

    kind = "reference values"

    def __init__(self, cases: Sequence[ReferenceCase], name: str | None = None) -> None:
        super().__init__(name)
        if not cases:
            raise ValueError("ReferenceCheck needs at least one case")
        self.cases: tuple[ReferenceCase, ...] = tuple(cases)

    def run(self, func: Callable[..., Any]) -> CheckResult:
        failures = []
        for case in self.cases:
            got = float(func(*case.args))
            if not math.isclose(
                got, case.expected, rel_tol=case.rel_tol, abs_tol=case.abs_tol
            ):
                failures.append(
                    f"{case.label}: got {got:.6g}, expected {case.expected:.6g}"
                )
        if failures:
            return self._result(False, "; ".join(failures))
        return self._result(True, f"{len(self.cases)} of {len(self.cases)} cases match")


class PropertyCheck(Check):
    """Test an invariant on many seeded random inputs.

    Args:
        name: What the property says, for example ``"monotonic"``.
        sample: Draws one argument tuple from a NumPy generator.
        holds: ``holds(func, args)`` is True when the property is satisfied.
        trials: Number of random inputs.
        seed: Seed of the generator, so a failure can be reproduced.
    """

    kind = "property"

    def __init__(
        self,
        name: str,
        sample: Callable[[np.random.Generator], tuple[Any, ...]],
        holds: Callable[[Callable[..., Any], tuple[Any, ...]], bool],
        trials: int = DEFAULT_TRIALS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        super().__init__(name)
        if trials < 1:
            raise ValueError("trials must be at least 1")
        self.sample = sample
        self.holds = holds
        self.trials = trials
        self.seed = seed

    def run(self, func: Callable[..., Any]) -> CheckResult:
        rng = np.random.default_rng(self.seed)
        for trial in range(self.trials):
            args = self.sample(rng)
            if not self.holds(func, args):
                return self._result(
                    False, f"counterexample at trial {trial}: args={args}"
                )
        return self._result(
            True, f"held on {self.trials} random inputs (seed {self.seed})"
        )


class DifferentialCheck(Check):
    """Compare a candidate with an independent, already-trusted implementation.

    Args:
        oracle: The trusted implementation (for example a library function that
            has its own tests).
        sample: Draws one argument tuple from a NumPy generator.
        rel_tol: Relative tolerance; ``abs_tol`` is the absolute tolerance.
    """

    kind = "differential"

    def __init__(
        self,
        oracle: Callable[..., Any],
        sample: Callable[[np.random.Generator], tuple[Any, ...]],
        trials: int = DEFAULT_TRIALS,
        rel_tol: float = 1e-6,
        abs_tol: float = 0.0,
        seed: int = DEFAULT_SEED,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.oracle = oracle
        self.sample = sample
        self.trials = trials
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.seed = seed

    def run(self, func: Callable[..., Any]) -> CheckResult:
        rng = np.random.default_rng(self.seed)
        worst_gap = 0.0
        for trial in range(self.trials):
            args = self.sample(rng)
            got, want = float(func(*args)), float(self.oracle(*args))
            worst_gap = max(worst_gap, abs(got - want))
            if not math.isclose(got, want, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                return self._result(
                    False,
                    f"trial {trial}: args={args}, got {got:.6g}, oracle {want:.6g}",
                )
        return self._result(
            True, f"agrees on {self.trials} inputs, max gap {worst_gap:.3g}"
        )


def observed_order(errors: Sequence[float], steps: Sequence[float]) -> np.ndarray:
    """Observed order of accuracy between successive step sizes.

    p_i = ln(e_i / e_{i+1}) / ln(h_i / h_{i+1})   (Veluri, Roy and Luke, Eq. 4)

    Args:
        errors: Error norms against an exact solution, one per step size.
        steps: Step sizes (any order, but paired with ``errors``).

    Returns:
        Array of ``len(steps) - 1`` observed orders.

    Raises:
        ValueError: If the inputs differ in length, have fewer than two
            entries, or contain non-positive values.
    """
    e = np.asarray(errors, dtype=float)
    h = np.asarray(steps, dtype=float)
    if e.shape != h.shape or e.size < MIN_STEPS_FOR_ORDER:
        raise ValueError("need matching errors and steps, at least two of each")
    if np.any(e <= 0) or np.any(h <= 0):
        raise ValueError("errors and steps must be positive")
    return np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])


class ConvergenceCheck(Check):
    """Check that a numerical method converges at its theoretical order.

    The callable under test is handed to ``error_at(func, h)``, which must run
    the method with step ``h`` and return the error against an exact solution.

    Args:
        error_at: Computes the error of ``func`` at one step size.
        steps: Step sizes, typically halved each time.
        expected_order: Theoretical order (1 for Euler, 4 for classical RK4).
        tolerance: Allowed deviation of the *finest* observed order.
    """

    kind = "convergence order"

    def __init__(
        self,
        error_at: Callable[[Callable[..., Any], float], float],
        steps: Sequence[float],
        expected_order: float,
        tolerance: float = 0.2,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        if len(steps) < MIN_STEPS_FOR_ORDER:
            raise ValueError("need at least two step sizes")
        self.error_at = error_at
        self.steps = tuple(float(h) for h in steps)
        self.expected_order = expected_order
        self.tolerance = tolerance

    def errors(self, func: Callable[..., Any]) -> np.ndarray:
        """Return the error at every step size."""
        return np.array([self.error_at(func, h) for h in self.steps])

    def run(self, func: Callable[..., Any]) -> CheckResult:
        orders = observed_order(self.errors(func), self.steps)
        finest = float(orders[-1])
        ok = abs(finest - self.expected_order) <= self.tolerance
        shown = ", ".join(f"{p:.2f}" for p in orders)
        return self._result(
            ok, f"observed orders [{shown}], expected {self.expected_order}"
        )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """All check results for one callable."""

    subject: str
    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        """True when every check passed."""
        return all(result.passed for result in self.results)

    @property
    def n_passed(self) -> int:
        """Number of checks that passed."""
        return sum(result.passed for result in self.results)

    def to_frame(self) -> pd.DataFrame:
        """One row per check: check name, outcome, detail."""
        return pd.DataFrame(
            {
                "check": [r.check for r in self.results],
                "outcome": [r.outcome.value for r in self.results],
                "detail": [r.detail for r in self.results],
            }
        )

    def summary(self) -> str:
        """One line, for example ``'candidate_b: 1/3 checks passed'``."""
        return f"{self.subject}: {self.n_passed}/{len(self.results)} checks passed"


@dataclass
class Verifier:
    """Composite of checks: run every check on one callable.

    Example:
        >>> verifier = Verifier([ReferenceCheck([ReferenceCase("one", (1.0,), 1.0)])])
        >>> verifier.verify(lambda x: x).passed
        True
    """

    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> Verifier:
        """Append a check and return ``self`` so calls can be chained."""
        self.checks.append(check)
        return self

    def verify(
        self, func: Callable[..., Any], subject: str | None = None
    ) -> VerificationReport:
        """Run all checks on ``func``."""
        if not self.checks:
            raise ValueError("Verifier has no checks")
        name = subject or getattr(func, "__name__", repr(func))
        return VerificationReport(name, tuple(check(func) for check in self.checks))

    def compare(self, candidates: dict[str, Callable[..., Any]]) -> pd.DataFrame:
        """Outcome table: one row per candidate, one column per check."""
        rows = {}
        for label, func in candidates.items():
            report = self.verify(func, label)
            rows[label] = {r.check: r.outcome.value for r in report.results}
        return pd.DataFrame.from_dict(rows, orient="index")


# ----------------------------------------------------------------- mutation


@dataclass(frozen=True, slots=True)
class Mutant:
    """A deliberately broken copy of a trusted source: replace ``find`` by ``replace``.

    ``find`` must occur exactly once in the source, so the mutation is
    unambiguous.
    """

    name: str
    find: str
    replace: str
    description: str = ""

    def apply(self, source: str) -> str:
        """Return the mutated source text."""
        count = source.count(self.find)
        if count != 1:
            raise ValueError(
                f"mutant {self.name!r}: pattern found {count} times, need 1"
            )
        return source.replace(self.find, self.replace)


@dataclass(frozen=True, slots=True)
class MutationReport:
    """Which mutants a test suite killed (detected) and which survived."""

    killed: tuple[str, ...]
    survived: tuple[str, ...]

    @property
    def score(self) -> float:
        """Fraction of mutants killed (1.0 is the target)."""
        total = len(self.killed) + len(self.survived)
        return len(self.killed) / total if total else math.nan

    def __str__(self) -> str:
        total = len(self.killed) + len(self.survived)
        return f"mutation score {len(self.killed)}/{total} = {self.score:.0%}"


TestSuite = Callable[[Callable[..., Any]], bool]


class MutationRunner:
    """Measure a test suite by running it against mutants of a trusted source.

    Args:
        source: Source text that defines ``entry_point`` (read it first!).
        entry_point: Name of the function the test suite receives.
        mutants: Broken variants of ``source``.
        namespace: Extra globals (for example ``{"np": np}``) for ``exec``.
    """

    def __init__(
        self,
        source: str,
        entry_point: str,
        mutants: Sequence[Mutant],
        namespace: dict[str, Any] | None = None,
    ) -> None:
        if not mutants:
            raise ValueError("MutationRunner needs at least one mutant")
        names = [m.name for m in mutants]
        if len(set(names)) != len(names):
            raise ValueError("mutant names must be unique")
        self.source = source
        self.entry_point = entry_point
        self.mutants: tuple[Mutant, ...] = tuple(mutants)
        self.namespace = dict(namespace or {})
        for mutant in self.mutants:
            mutant.apply(source)  # fail early if a pattern is missing or ambiguous

    def build(self, mutant: Mutant | None = None) -> Callable[..., Any]:
        """Compile the original (``mutant=None``) or a mutant; return the function."""
        text = self.source if mutant is None else mutant.apply(self.source)
        scope: dict[str, Any] = {"__name__": "mutant", **self.namespace}
        exec(compile(text, f"<{mutant.name if mutant else 'original'}>", "exec"), scope)
        return scope[self.entry_point]

    @staticmethod
    def _passes(suite: TestSuite, func: Callable[..., Any]) -> bool:
        try:
            return bool(suite(func))
        except Exception:  # noqa: BLE001 - a crashing test counts as a failing test
            return False

    def run(self, suite: TestSuite) -> MutationReport:
        """Run ``suite`` on every mutant.

        Raises:
            ValueError: If the suite fails on the original source: a suite that
                rejects correct code cannot be used to judge anything.
        """
        if not self._passes(suite, self.build()):
            raise ValueError("the test suite fails on the original source")
        killed, survived = [], []
        for mutant in self.mutants:
            (survived if self._passes(suite, self.build(mutant)) else killed).append(
                mutant.name
            )
        return MutationReport(tuple(killed), tuple(survived))


# ----------------------------------------------------------- static checks


def defined_names(source: str) -> set[str]:
    """Top-level function and class names defined in ``source`` (not executed)."""
    tree = ast.parse(source)
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    return {node.name for node in tree.body if isinstance(node, kinds)}


@dataclass(frozen=True, slots=True)
class DeliverableAudit:
    """What was requested versus what the source actually defines."""

    requested: frozenset[str]
    found: frozenset[str]

    @property
    def missing(self) -> frozenset[str]:
        """Requested but not defined."""
        return self.requested - self.found

    @property
    def unrequested(self) -> frozenset[str]:
        """Defined but never asked for (review these too)."""
        return self.found - self.requested

    @property
    def complete(self) -> bool:
        """True when nothing requested is missing."""
        return not self.missing


def audit_deliverables(source: str, requested: Iterable[str]) -> DeliverableAudit:
    """Count what you asked for: compare requested names with defined ones."""
    return DeliverableAudit(frozenset(requested), frozenset(defined_names(source)))


def imported_modules(source: str) -> set[str]:
    """Top-level module names imported by ``source`` (relative imports skipped)."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def unresolvable_imports(source: str) -> set[str]:
    """Imported top-level modules that cannot be found in *this* environment.

    A name in the result is not proof that the package does not exist: it may
    simply not be installed. It is a prompt to check the package registry, the
    publisher and the download history *before* installing anything, because
    attackers register package names that code assistants tend to invent.
    """
    builtin = set(sys.builtin_module_names)
    return {
        name
        for name in imported_modules(source)
        if name not in builtin and importlib.util.find_spec(name) is None
    }
