---
title: "Week 3: ABCs, Dataclasses, Typing and a Composable DataPipeline"
subtitle: "One reproducible recipe for cleaning sensor data"
course: "AI for Aerospace & Mechanical Engineering (PhD)"
footer: "Week 03 | OOP: typed, composable pipelines"
language: en
provenance: "New content. Python concepts rewritten from W3Schools (abc Module, dataclasses Module, typing Module, __repr__(), __eq__(), Iterators) and GeeksforGeeks (Abstract Classes in Python, Understanding Python Dataclasses, Python Generics, str() vs repr()); Hampel filter adapted from Pearson, IEEE TCST 10(1), 2002; MAD constant from Rousseeuw & Croux, JASA 88, 1993."
---

<!-- Build: python tools/build_slides.py slides/en/week03_data_pipeline_EN.md -->

## SLIDE 1 | Same log, two analysts, two answers
**Content:**
- Structures asks: peak load factor of the Week 1 test flight
- Analyst A: 5-sample moving average → 1.87 g at t = 66.7 s
- Analyst B: rolling median, then average → 1.80 g at t = 192 s
- A's "peak" is a smeared glitch in level cruise
- Q: which recipe produced the number in the report?
**Visual:** IMAGE ../assets/week03/fig_two_analysts.png
**Notes:** Run Section 0 of the lab live. Ask the room which number goes in the report and how a reader of the report could tell which script made it. Write the final question on the board.

## SLIDE 2 | Why scripts are not recipes
**Content:**
- Order of steps hidden in a function body
- Parameters (5, 7, center=True) not saved with the result
- Two scripts cannot be compared: analyst_a == analyst_b is meaningless
- Wrong data type found only when something crashes, or never
**Visual:** Two script boxes "analyst_a.py" and "analyst_b.py" feeding one report figure, with four red question marks: order? parameters? same? types?
**Notes:** Every bullet maps to a tool this week: order → a stage tuple; parameters → dataclass fields; comparison → __eq__; types → INPUT_TYPE/OUTPUT_TYPE checks.

## SLIDE 3 | Example 1: a spike through a moving average
**Content:**
- Cruise a_z ≈ 10.35 m/s² (n ≈ 1.05); one sample reads 50.33 (+40)
- 5-sample mean around it: (4 × 10.35 + 50.33) / 5 = 18.35 m/s²
- n = 18.35 / 9.807 = 1.87 g on five samples in a row
- Despike first (median of 7 ignores 1 bad sample) → 1.80 g, the pull-up
- Lesson: despike before you smooth
**Visual:** Two rows of 7 boxes: raw a_z samples 10.27 … 10.38 with one orange 50.33; after the 5-mean, five boxes at ≈ 18.3 shaded orange.
**Notes:** Do the arithmetic on the board before running code (log rows 3331–3337, t = 66.62–66.74 s). The point: a filter that averages cannot remove what it spreads.

## SLIDE 4 | Abstract base classes: the stage contract
**Content:**
- class Stage(ABC) with @abstractmethod transform(data)
- Missing method → TypeError at construction, not mid-run
- Abstract property: @property + @abstractmethod
- Concrete shared methods live in the ABC (__call__ checks the type)
**Visual:** Code: `class Stage(ABC): @property @abstractmethod def description(...)`, `@abstractmethod def transform(...)`, `def __call__(...)`; below, `Forgetful()` → red "TypeError: Can't instantiate abstract class Forgetful with abstract method description".
**Notes:** Rewritten from GeeksforGeeks "Abstract Classes in Python" and W3Schools "Python abc Module". Lab Section 1, Exercise 1.1 (PlausibleRange).

## SLIDE 5 | Dataclasses: the configuration is the class
**Content:**
- Annotated fields → __init__, __repr__, __eq__ written for you
- frozen=True: immutable and hashable recipes
- __post_init__: validate (window odd ≥ 3), normalise (list → tuple)
- Frozen: object.__setattr__(self, "columns", tuple(...)) once
**Visual:** IMAGE ../assets/week03/fig_dataclass_generated.png
**Notes:** GeeksforGeeks "Understanding Python Dataclasses" and W3Schools "Python dataclasses Module". Show ManualMovingAverage == ManualMovingAverage → False first.

## SLIDE 6 | Example 2: what did @dataclass write?
**Content:**
- Stage("smooth") prints Stage(name='smooth', window=5, tags=[])
- default_factory=list: every object gets its own list
- tags: list = [] → ValueError when the class is defined
- field(compare=False): learned mean_ ignored by ==
- eq without frozen → __hash__ = None → unhashable
**Visual:** Code of In-class Problem 1 with the four printed lines on the right; the ValueError and TypeError lines in red.
**Notes:** In-class Problem 1, 5 minutes. Connect default_factory to Week 2's shared-list bug.

## SLIDE 7 | Learned state is not configuration
**Content:**
- Standardize learns mean_ and std_ in fit()
- field(init=False, repr=False, compare=False)
- Fitted and unfitted Standardize(["x"]) are the same recipe
- transform before fit → NotFittedError
- get_params() returns only init=True fields
**Visual:** IMAGE ../assets/week03/fig_stage_hierarchy.png
**Notes:** The same convention as scikit-learn (trailing underscore = learned). We meet sklearn in Week 6; this is why its API looks the way it does.

## SLIDE 8 | Writing __repr__ and __eq__ yourself
**Content:**
- __repr__: for developers; aim for eval(repr(x)) == x
- __str__: for reports; print() prefers it
- __eq__ returns NotImplemented for foreign types
- Define __eq__ → __hash__ becomes None unless you define it
- Hash exactly what __eq__ compares, and only if it never changes
**Visual:** Code: class Route with __repr__ → "Route(['OIII', 'OIFM', 'OISS'])", __str__ → "OIII -> OIFM -> OISS", __eq__, __hash__; below, the round-trip line printing True.
**Notes:** W3Schools "__repr__()" and "__eq__()" pages; GeeksforGeeks "str() vs repr()". Exercise 3.1 (Waypoint).

## SLIDE 9 | Example 3: Hampel filter by hand
**Content:**
- h = 100.0, 100.4, 99.8, 106.0, 100.3, 100.1, 99.9 m; window 5
- k = 3: median 100.3, deviations 0.1 0.5 5.7 0.0 0.2 → MAD 0.2
- Threshold 3 × 1.4826 × 0.2 = 0.89 m
- |106.0 − 100.3| = 5.7 > 0.89 → replace by 100.3
- k = 4: |100.3 − 100.1| = 0.2 → keep
**Visual:** IMAGE ../assets/week03/fig_hampel_by_hand.png
**Notes:** 1.4826 = 1/Φ⁻¹(0.75) makes MAD estimate σ for Gaussian noise (Rousseeuw & Croux 1993). Lab Section 5 reproduces these numbers with HampelFilter.outlier_mask.

## SLIDE 10 | Type hints and generics
**Content:**
- Hints document intent; Python does not enforce them
- TypeVar: a placeholder type; Generic[InT, OutT]: a typed class
- PipelineStage[DataFrame, ndarray] reads like list[int]
- Parameters are erased at run time → store INPUT_TYPE, OUTPUT_TYPE
- ClassVar, Final, X | None: the vocabulary of the course code
**Visual:** Code: `InT = TypeVar("InT")`, `class TypedStage(ABC, Generic[InT, OutT])`, `class FrameToArray(TypedStage[pd.DataFrame, np.ndarray]): INPUT_TYPE = pd.DataFrame`; below, `peak_g("9.8")` failing only inside the body.
**Notes:** GeeksforGeeks "Python Generics", W3Schools "Python typing Module". mypy is not in the pinned environment; mention it, do not install it.

## SLIDE 11 | Composition: stages joined by types
**Content:**
- a >> b builds DataPipeline((a, b))
- __init__ checks issubclass(left.OUTPUT_TYPE, right.INPUT_TYPE)
- Mismatch → TypeError when BUILT, before any data
- Immutable: >> and slicing return new pipelines
- Has-a (composition), not is-a (inheritance)
**Visual:** IMAGE ../assets/week03/fig_pipeline_chain.png
**Notes:** Exercise 4.1 writes can_follow(). In-class Problem 4 asks which errors are build-time and which are run-time.

## SLIDE 12 | Example 4: does the order matter?
**Content:**
- x = 1, 1, 1, 10, 1, 1, 1; Hampel window 3, mean window 3
- Despike then smooth: MAD 0 at k = 3 → 10 → 1; peak 1
- Smooth then despike: 1, 1, 4, 4, 4, 1, 1; nothing flagged; peak 4
- H∘S ≠ S∘H: composition is not commutative
- So H >> S and S >> H must compare unequal
**Visual:** IMAGE ../assets/week03/fig_order_matters.png
**Notes:** In-class Problem 3. Same mechanism as the 1.87 g vs 1.80 g opener.

## SLIDE 13 | The course library: aerospace_ai.pipeline
**Content:**
- PipelineStage(ABC, Generic): transform, fit, __call__, registry
- Frozen stages: SensorToFrame, SelectColumns, DropNonFinite, FillGaps
- HampelFilter, RollingMean, TimeDerivative, ToArray
- Learning stage: Standardize; container: DataPipeline
- 36 pytest tests in tests/test_pipeline.py
**Visual:** Code: `recipe = SelectColumns([...]) >> HampelFilter(ACCEL, window=7) >> RollingMean(ACCEL, window=5)` and its multi-line repr.
**Notes:** Instructor: read pipeline.py before class. It also uses __init_subclass__ for the registry, typing.overload on __getitem__, and __slots__ on DataPipeline.

## SLIDE 14 | A pipeline that reports and saves itself
**Content:**
- run() → output + one StageReport per stage
- rows in/out, NaN out, cells changed, seconds
- to_config() → plain dict → json.dumps
- DataPipeline.from_config(json) == original → True
- Save the JSON next to every number you report
**Visual:** Table of the report for pipeline B on the flight log (HampelFilter changed 2140 cells, RollingMean 44 815), and a JSON snippet with "stage": "HampelFilter", "window": 7.
**Notes:** The registry filled by __init_subclass__ maps the "stage" names back to classes; Week 4 calls this the Factory pattern.

## SLIDE 15 | Example 5: both analysts as pipelines
**Content:**
- A = RollingMean(5) >> SelectColumns → 1.87 g at 66.7 s
- B = HampelFilter(7) >> RollingMean(5) >> SelectColumns → 1.80 g at 192.0 s
- A == B → False, and repr shows why
- B's config JSON rebuilds B exactly
**Visual:** IMAGE ../assets/week03/fig_two_analysts.png
**Notes:** Lab Section 8. Hampel replaces about 4.8 % of accelerometer samples with a 7-sample window: a small MAD from 7 noisy samples flags some good samples too. Harmless here (they become their local median); ask what window would flag fewer.

## SLIDE 16 | Fit on training data only
**Content:**
- Train {2, 4, 6} m: μ = 4, σ = 1.633 → test x = 8: z = 2.449
- Fit on train + test: μ = 5, σ = 2.236 → z = 1.342
- The test sample looks ordinary because it defined "ordinary"
- Pipeline.fit fits each stage on the output of the previous one
**Visual:** IMAGE ../assets/week03/fig_leakage.png
**Notes:** Leakage is Week 5's main topic; here it motivates why learned state lives in fit and why Practice Assignment 3 splits by time.

## SLIDE 17 | Any sensor in, a model-ready array out
**Content:**
- SensorToFrame accepts any BaseSensor (polymorphism)
- IMU (±4 g) → Hampel → mean → ToArray: (14 937, 3), peak 1.80 g
- Your Assignment 1 LiDAR and GPS plug in unchanged
- The pipeline never imports a concrete sensor class
**Visual:** Diagram: IMUSensor, BarometerSensor, "your GPSSensor", "your LidarSensor" all with arrows into one SensorToFrame box, then the chain.
**Notes:** The reference LiDAR/GPS sensors join src only after Assignment 1 is graded; the pipeline is written against BaseSensor so it does not need them. Exercise 10.1.

## SLIDE 18 | Example 6: rows are not seconds
**Content:**
- Climb-rate pipeline with a 51-row window: max 20.99 m/s at 99.76 s
- That is next to a 1.2 s logger gap (t = 99.98 → 101.20 s)
- The window spans more time on one side → false slope
- Resample to a uniform 10 Hz grid first → 19.91 m/s at 84.0 s
**Visual:** IMAGE ../assets/week03/fig_resample_gap.png
**Notes:** Exercise 8.1 finds the artefact; Practice Assignment 3 Task 2 builds the Resample stage that removes it.

## SLIDE 19 | Testing stages with pytest
**Content:**
- Test numbers from hand examples (Hampel at k = 3 → 100.3)
- Test refusals: bad window, wrong type, unfitted, missing column
- Test invariants: input untouched, repr round-trips, JSON round-trips
- parametrize the bad inputs; fixtures for small frames
**Visual:** Code: `@pytest.mark.parametrize("window", [0, 1, 2, 4, -3])` → `pytest.raises(ValueError, match="odd")`; terminal "77 passed".
**Notes:** tests/test_pipeline.py is the model for Practice Assignment 3 Task 5, which is graded against three broken implementations.

## SLIDE 20 | This week's work
**Content:**
- Lab: notebooks/week03/week03_data_pipeline.ipynb (6 TODO exercises)
- In class: 4 warm-ups (dataclass output, Hampel, order, build vs run errors)
- Practice Assignment 3: air-data pipeline, about 3.5 h, due Week 4
- Assignment 1 is due at the start of this lecture
**Visual:** Three icons: notebook, pencil, Git branch.
**Notes:** Practice assignments fill the weeks without a syllabus milestone; their numbers follow the week, so they never clash with A1–A6.

## SLIDE 21 | What We Now Know How To Do
**Content:**
- Write contracts as ABCs; configurations as frozen dataclasses
- Separate learned state with field(init=False, repr=False, compare=False)
- Write __repr__, __eq__ and a consistent __hash__
- Use TypeVar/Generic and check types where Python does not
- Compose typed stages into a pipeline that reports and saves itself
**Visual:** The opener's two scripts replaced by one printed DataPipeline repr and a JSON file icon next to the report figure, green check marks on order, parameters, equality and types.
**Notes:** Return to the opener question: the JSON next to the figure answers "which recipe produced this number". Next week: Strategy, Observer and Factory patterns.
