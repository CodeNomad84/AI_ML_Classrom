"""Dataset loaders behind a factory: ask for data by name, not by class.

Provenance: new content (Week 4). The Factory Method / registry idea is
rewritten from GeeksforGeeks ("Factory Method - Python Design Patterns") and
Gamma et al., *Design Patterns* (1994), Ch. 3; dictionaries and decorators
follow the W3Schools Python tutorial. The fuel-flow model is standard
performance analysis (power required = parasite + induced + climb power),
adapted from Anderson, *Aircraft Performance and Design*, Ch. 5, with the
ISA density from Anderson, *Introduction to Flight*, Ch. 3.

Design:

* :class:`Dataset` is an immutable container (features, target, names) with a
  leak-free chronological :meth:`Dataset.split`.
* :class:`DatasetLoader` is the abstract *product*: every loader has one job,
  :meth:`DatasetLoader.load`.
* :class:`DatasetFactory` is the *creator*. Loaders register themselves under
  a string key with the :meth:`DatasetFactory.register` class decorator, and
  client code calls ``DatasetFactory.create("fuel_flow_synthetic", seed=1)``.
  Adding a data source never edits the factory or the experiment code: it is
  one new decorated class (the Open/Closed Principle).
"""

from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Final, TypeVar

import numpy as np
import pandas as pd

from aerospace_ai.pipeline import DataPipeline
from aerospace_ai.sensors import (
    ISA_EXPONENT,
    ISA_LAPSE_RATE_K_PER_M,
    ISA_SEA_LEVEL_TEMPERATURE_K,
)
from aerospace_ai.utils import STANDARD_GRAVITY_MPS2, FlightLog

DEFAULT_TEST_FRACTION: Final[float] = 0.25
ISA_SEA_LEVEL_DENSITY_KGPM3: Final[float] = 1.225
WATTS_PER_KILOWATT: Final[float] = 1000.0
GRAMS_PER_KILOGRAM: Final[float] = 1000.0

FUEL_FEATURES: Final[tuple[str, ...]] = (
    "airspeed_mps",
    "altitude_m",
    "climb_rate_mps",
    "mass_kg",
)
FUEL_TARGET: Final[str] = "fuel_flow_gph"

LoaderT = TypeVar("LoaderT", bound="type[DatasetLoader]")


class UnknownDatasetError(KeyError):
    """Raised when the factory is asked for a name that is not registered."""


# --------------------------------------------------------------------- Dataset
@dataclass(frozen=True)
class Dataset:
    """A supervised-learning table: feature matrix, target vector and names.

    Attributes:
        X: Feature matrix, shape (n_samples, n_features).
        y: Target vector, shape (n_samples,).
        feature_names: One name per column of ``X``.
        target_name: Name of the target quantity (with units).
        name: Short identifier of the data source.
        metadata: Free-form information about how the data was made.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    target_name: str
    name: str = "dataset"
    metadata: dict[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        """Check that shapes and names agree.

        Raises:
            ValueError: If ``X`` is not 2-D, ``y`` is not 1-D, the row counts
                differ, or the number of names does not match the columns.
        """
        X = np.asarray(self.X, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        if len(self.feature_names) != X.shape[1]:
            raise ValueError(
                f"{len(self.feature_names)} feature names for {X.shape[1]} columns"
            )
        # The dataclass is frozen, so bypass __setattr__ to store float copies.
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    def __len__(self) -> int:
        """Return the number of samples."""
        return int(self.y.shape[0])

    def __repr__(self) -> str:
        """Return a one-line summary (arrays are not printed)."""
        return (
            f"Dataset(name={self.name!r}, n_samples={len(self)}, "
            f"features={list(self.feature_names)}, target={self.target_name!r})"
        )

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return int(self.X.shape[1])

    def to_frame(self) -> pd.DataFrame:
        """Return features and target as one DataFrame."""
        frame = pd.DataFrame(self.X, columns=list(self.feature_names))
        frame[self.target_name] = self.y
        return frame

    def subset(self, rows: np.ndarray | slice, suffix: str) -> Dataset:
        """Return a new dataset with the selected rows.

        Args:
            rows: Row indices, boolean mask or slice.
            suffix: Appended to the name, for example ``"train"``.

        Returns:
            A new :class:`Dataset` sharing names and metadata.
        """
        return Dataset(
            self.X[rows],
            self.y[rows],
            self.feature_names,
            self.target_name,
            f"{self.name}/{suffix}",
            dict(self.metadata),
        )

    def split(
        self,
        test_fraction: float = DEFAULT_TEST_FRACTION,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> tuple[Dataset, Dataset]:
        """Split into a training and a test set.

        The default is a *chronological* split (first part train, last part
        test). For time-ordered sensor data this is the leak-free choice: a
        random split would let the model see the future of every test point.

        Args:
            test_fraction: Fraction of samples in the test set, in (0, 1).
            shuffle: Shuffle rows before splitting (only for i.i.d. data).
            seed: Random seed used when ``shuffle`` is True.

        Returns:
            ``(train, test)`` datasets.

        Raises:
            ValueError: If ``test_fraction`` is outside (0, 1) or either part
                would be empty.
        """
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
        n_test = int(round(test_fraction * len(self)))
        if n_test == 0 or n_test == len(self):
            raise ValueError("split would leave an empty train or test set")
        order = np.arange(len(self))
        if shuffle:
            order = np.random.default_rng(seed).permutation(order)
        n_train = len(self) - n_test
        return (
            self.subset(order[:n_train], "train"),
            self.subset(order[n_train:], "test"),
        )


# ------------------------------------------------------------- Loader product
class DatasetLoader(ABC):
    """Abstract product of the factory: something that can build a Dataset.

    Subclasses implement :meth:`load`. ``registry_name`` is filled in by
    :meth:`DatasetFactory.register`.
    """

    registry_name: ClassVar[str] = ""

    @abstractmethod
    def load(self) -> Dataset:
        """Build and return the dataset."""

    def __repr__(self) -> str:
        """Return the class name and registry key."""
        return f"{type(self).__name__}(registry_name={self.registry_name!r})"


# -------------------------------------------------------------------- Factory
class DatasetFactory:
    """Registry-based factory for dataset loaders.

    The registry is a class-level dictionary from name to loader class, so it
    is shared by the whole program. The class is never instantiated; all
    methods are class methods.

    Example:
        >>> loader = DatasetFactory.create("fuel_flow_synthetic", n_samples=50)
        >>> len(loader.load())
        50
    """

    _registry: ClassVar[dict[str, type[DatasetLoader]]] = {}

    def __init__(self) -> None:
        """Forbid instances: the factory is used through its class methods.

        Raises:
            TypeError: Always.
        """
        raise TypeError("DatasetFactory is used through its class methods")

    @classmethod
    def register(cls, name: str) -> Callable[[LoaderT], LoaderT]:
        """Return a class decorator that registers a loader under ``name``.

        Args:
            name: Registry key, for example ``"flight_log_csv"``.

        Returns:
            A decorator that stores the class and returns it unchanged.

        Raises:
            ValueError: If ``name`` is empty or already taken by another class.
        """
        key = name.strip().lower()
        if not key:
            raise ValueError("registry name must not be empty")

        def decorator(loader_cls: LoaderT) -> LoaderT:
            if not (
                isinstance(loader_cls, type) and issubclass(loader_cls, DatasetLoader)
            ):
                raise TypeError(f"{loader_cls!r} is not a DatasetLoader subclass")
            existing = cls._registry.get(key)
            if existing is not None and existing is not loader_cls:
                raise ValueError(
                    f"{key!r} is already registered to {existing.__name__}"
                )
            cls._registry[key] = loader_cls
            loader_cls.registry_name = key
            return loader_cls

        return decorator

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove ``name`` from the registry (used by tests and notebooks).

        Raises:
            UnknownDatasetError: If the name is not registered.
        """
        key = name.strip().lower()
        if key not in cls._registry:
            raise UnknownDatasetError(key)
        del cls._registry[key]

    @classmethod
    def available(cls) -> list[str]:
        """Return the registered names in alphabetical order."""
        return sorted(cls._registry)

    @classmethod
    def loader_class(cls, name: str) -> type[DatasetLoader]:
        """Return the loader class registered under ``name``.

        Raises:
            UnknownDatasetError: If the name is unknown; the message suggests
                the closest registered names (typos are common).
        """
        key = name.strip().lower()
        try:
            return cls._registry[key]
        except KeyError:
            close = difflib.get_close_matches(key, cls._registry, n=3)
            hint = f" Did you mean {close}?" if close else ""
            raise UnknownDatasetError(
                f"no dataset loader named {key!r}. Available: {cls.available()}.{hint}"
            ) from None

    @classmethod
    def create(cls, name: str, **kwargs: object) -> DatasetLoader:
        """Instantiate the loader registered under ``name``.

        Args:
            name: Registry key.
            **kwargs: Passed to the loader's constructor.

        Returns:
            A configured loader (call ``.load()`` on it).
        """
        return cls.loader_class(name)(**kwargs)

    @classmethod
    def load(cls, name: str, **kwargs: object) -> Dataset:
        """Shortcut for ``create(name, **kwargs).load()``."""
        return cls.create(name, **kwargs).load()


# --------------------------------------------------------- Concrete products
def isa_density(altitude_m: np.ndarray) -> np.ndarray:
    """Return ISA air density in the troposphere.

    rho = rho0 * (T / T0) ** (g0 / (R L) - 1), with T = T0 - L h.

    Args:
        altitude_m: Geopotential altitude in metres (below 11 km).

    Returns:
        Density in kg/m^3, same shape as the input.
    """
    temperature_ratio = (
        ISA_SEA_LEVEL_TEMPERATURE_K - ISA_LAPSE_RATE_K_PER_M * np.asarray(altitude_m)
    ) / ISA_SEA_LEVEL_TEMPERATURE_K
    return ISA_SEA_LEVEL_DENSITY_KGPM3 * temperature_ratio ** (1.0 / ISA_EXPONENT - 1.0)


@dataclass(frozen=True, slots=True)
class FuelFlowAircraft:
    """Airframe and engine constants of the small piston UAV used in Week 4.

    Attributes:
        wing_area_m2: Reference wing area S.
        aspect_ratio: Wing aspect ratio AR.
        oswald_efficiency: Span efficiency factor e.
        zero_lift_drag: Parasite drag coefficient C_D0.
        propeller_efficiency: Shaft-to-thrust power efficiency eta_p.
        bsfc_kg_per_kwh: Brake-specific fuel consumption of the engine.
        idle_power_w: Minimum shaft power (the engine never stops in flight).
    """

    wing_area_m2: float = 1.5
    aspect_ratio: float = 10.0
    oswald_efficiency: float = 0.8
    zero_lift_drag: float = 0.03
    propeller_efficiency: float = 0.75
    bsfc_kg_per_kwh: float = 0.35
    idle_power_w: float = 150.0

    def fuel_flow_gph(
        self,
        airspeed_mps: np.ndarray,
        altitude_m: np.ndarray,
        climb_rate_mps: np.ndarray,
        mass_kg: np.ndarray,
    ) -> np.ndarray:
        """Return the noise-free fuel flow in grams per hour.

        P_req = 0.5 rho V^3 S C_D0 + 2 W^2 / (rho V S pi e AR) + W * climb_rate
        P_shaft = max(P_req / eta_p, P_idle), fuel = BSFC * P_shaft.

        Args:
            airspeed_mps: True airspeed V.
            altitude_m: Altitude (for the ISA density).
            climb_rate_mps: Rate of climb (negative in a descent).
            mass_kg: Aircraft mass.

        Returns:
            Fuel flow in g/h.
        """
        rho = isa_density(altitude_m)
        weight_n = np.asarray(mass_kg) * STANDARD_GRAVITY_MPS2
        speed = np.asarray(airspeed_mps)
        parasite = 0.5 * rho * speed**3 * self.wing_area_m2 * self.zero_lift_drag
        induced = (2.0 * weight_n**2) / (
            rho
            * speed
            * self.wing_area_m2
            * np.pi
            * self.oswald_efficiency
            * self.aspect_ratio
        )
        climb = weight_n * np.asarray(climb_rate_mps)
        shaft_w = np.maximum(
            (parasite + induced + climb) / self.propeller_efficiency, self.idle_power_w
        )
        return self.bsfc_kg_per_kwh * shaft_w / WATTS_PER_KILOWATT * GRAMS_PER_KILOGRAM


@DatasetFactory.register("fuel_flow_synthetic")
class SyntheticFuelFlowLoader(DatasetLoader):
    """Seeded synthetic fuel-flow measurements of a small piston UAV.

    Features: airspeed, altitude, climb rate, mass. Target: fuel flow in g/h
    with multiplicative measurement noise. Fuel flow is convex in airspeed
    (parasite power grows with V^3) and contains a climb-rate x mass product
    (climb power = W * climb rate), so a straight-line model cannot fit it
    perfectly: a good test case for comparing model strategies.
    """

    AIRSPEED_RANGE_MPS: ClassVar[tuple[float, float]] = (16.0, 36.0)
    ALTITUDE_RANGE_M: ClassVar[tuple[float, float]] = (0.0, 3000.0)
    CLIMB_RANGE_MPS: ClassVar[tuple[float, float]] = (-2.0, 3.0)
    MASS_RANGE_KG: ClassVar[tuple[float, float]] = (25.0, 40.0)

    def __init__(
        self,
        n_samples: int = 400,
        noise_fraction: float = 0.03,
        seed: int = 4,
        aircraft: FuelFlowAircraft | None = None,
    ) -> None:
        """Configure the generator.

        Args:
            n_samples: Number of rows.
            noise_fraction: Standard deviation of the relative noise.
            seed: Random seed (same seed, same data).
            aircraft: Airframe constants; the default UAV if None.

        Raises:
            ValueError: If ``n_samples`` < 2 or ``noise_fraction`` < 0.
        """
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        if noise_fraction < 0.0:
            raise ValueError("noise_fraction must be non-negative")
        self.n_samples = n_samples
        self.noise_fraction = noise_fraction
        self.seed = seed
        self.aircraft = aircraft or FuelFlowAircraft()

    def load(self) -> Dataset:
        """Draw flight conditions uniformly and evaluate the fuel model."""
        rng = np.random.default_rng(self.seed)
        ranges = (
            self.AIRSPEED_RANGE_MPS,
            self.ALTITUDE_RANGE_M,
            self.CLIMB_RANGE_MPS,
            self.MASS_RANGE_KG,
        )
        X = np.column_stack([rng.uniform(lo, hi, self.n_samples) for lo, hi in ranges])
        clean = self.aircraft.fuel_flow_gph(*X.T)
        y = clean * (1.0 + self.noise_fraction * rng.standard_normal(self.n_samples))
        return Dataset(
            X,
            y,
            FUEL_FEATURES,
            FUEL_TARGET,
            self.registry_name,
            {"seed": self.seed, "noise_fraction": self.noise_fraction},
        )


@DatasetFactory.register("csv")
class CsvTableLoader(DatasetLoader):
    """Generic loader: pick feature and target columns from any CSV file.

    An optional Week 3 :class:`~aerospace_ai.pipeline.DataPipeline` cleans the
    raw table first (for example ``DropNonFinite() >> HampelFilter(...)``).
    Use only cleaning stages here: anything that learns statistics, such as
    ``Standardize``, must be fitted on the training split, after
    :meth:`Dataset.split`. Rows still missing a value in a selected column are
    dropped.
    """

    def __init__(
        self,
        path: str | Path,
        features: Sequence[str],
        target: str,
        cleaning: DataPipeline[pd.DataFrame, pd.DataFrame] | None = None,
    ) -> None:
        """Store the file path and column choice.

        Args:
            path: CSV file.
            features: Columns used as features, in order.
            target: Column used as the target.
            cleaning: Optional DataFrame-to-DataFrame pipeline run on the raw table.

        Raises:
            ValueError: If no feature is given or the target is also a feature.
        """
        if not features:
            raise ValueError("at least one feature column is required")
        if target in features:
            raise ValueError(f"target {target!r} is also listed as a feature")
        self.path = Path(path)
        self.features = tuple(features)
        self.target = target
        self.cleaning = cleaning

    def read_frame(self) -> pd.DataFrame:
        """Read the raw table (subclasses may override, e.g. to add columns)."""
        return pd.read_csv(self.path)

    def load(self) -> Dataset:
        """Read the file, check the columns and drop incomplete rows.

        Raises:
            KeyError: If a requested column is not in the file.
        """
        frame = self.read_frame()
        if self.cleaning is not None:
            frame = self.cleaning.fit_transform(frame)
        wanted = [*self.features, self.target]
        missing = [c for c in wanted if c not in frame.columns]
        if missing:
            raise KeyError(f"columns {missing} not in {self.path.name}")
        clean = frame[wanted].dropna()
        return Dataset(
            clean[list(self.features)].to_numpy(),
            clean[self.target].to_numpy(),
            self.features,
            self.target,
            self.registry_name,
            {"path": str(self.path), "rows_dropped": len(frame) - len(clean)},
        )


@DatasetFactory.register("flight_log_csv")
class FlightLogCsvLoader(CsvTableLoader):
    """Week 1 UAV flight log, with the derived load factor available as a column.

    Reuses :class:`aerospace_ai.utils.FlightLog`, so ``"load_factor"`` can be
    requested as a feature or target even though it is not in the file.
    """

    DERIVED_LOAD_FACTOR: ClassVar[str] = "load_factor"

    def read_frame(self) -> pd.DataFrame:
        """Load the log and append the load-factor column."""
        log = FlightLog.from_csv(self.path)
        frame = log.data.copy()
        frame[self.DERIVED_LOAD_FACTOR] = log.load_factor()
        return frame
