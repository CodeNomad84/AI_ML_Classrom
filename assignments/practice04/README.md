# Practice Assignment 4 (Week 4): Extend the three patterns

> Provenance: new content. Patterns rewritten (not copied) from GeeksforGeeks ("Strategy Design
> Pattern", "Observer Design Pattern", "Factory Method - Python Design Patterns", "Pytest
> Fixtures") and Gamma et al., *Design Patterns* (1994); polynomial features adapted from Bishop,
> *Pattern Recognition and Machine Learning*, Sec. 1.1 and 3.1; fuel-flow model adapted from
> Anderson, *Aircraft Performance and Design*, Ch. 5.

| | |
|---|---|
| Released | Week 4 |
| Due | Start of Week 5 lecture |
| Effort | About 3 hours |
| Weight | Formative: feedback only, not part of the course grade (the syllabus milestones are A1 in Week 2 and A2 in Week 5) |
| Notebook | [`practice04_design_patterns.ipynb`](practice04_design_patterns.ipynb) (Persian problem statement, checks, expected output) |
| Your code | [`patterns_practice.py`](patterns_practice.py) (skeleton with `# TODO`) |
| Your tests | [`test_patterns_practice.py`](test_patterns_practice.py) (two examples given) |
| Data | `DatasetFactory.load("fuel_flow_synthetic", seed=4)` (same rows as [`week04_fuel_flow.csv`](../../data/sample/week04_fuel_flow.csv)) |

## The task in one paragraph

The straight-line model predicts the UAV's fuel flow with a test RMSE of about 51 g/h although
the measurement noise is only about 12 g/h, because parasite power grows with $V^3$ and climb
power is mass × climb rate. Without editing a single line of `src/aerospace_ai`, you add a new
**strategy** (`PolynomialLeastSquares`, built by composition on `LinearLeastSquares`), a new
**observer** (`ReduceLROnPlateau`), and a new **factory product** (`FlightRegimeLoader`, which
builds any registered dataset through the factory and keeps one flight regime), and you test all
three with pytest.

```bash
python -m pytest assignments/practice04 -q          # your tests
jupyter lab assignments/practice04/                  # the notebook
```

## Time budget

| Part | Suggested time |
|------|----------------|
| Re-read `models.py`, `training.py`, `datasets.py` | 30 min |
| Task 1: polynomial strategy | 50 min |
| Task 2: plateau callback | 40 min |
| Task 3: regime loader | 30 min |
| Task 4: your test suite | 30 min |

## Rubric (100 points, for feedback)

| Task | Points | Full marks when | Typical deductions |
|------|-------:|-----------------|--------------------|
| 1. `PolynomialLeastSquares` | 30 | Training-set standardisation; column order as documented; `linear` is a `LinearLeastSquares` (composition); `repr` shows degree and interactions; test RMSE 22.59 g/h with interactions | Standardising with test statistics (−8); inheriting from `LinearLeastSquares` instead of composing (−5); no `MIN_STD` floor, constant column gives NaN (−4) |
| 2. `ReduceLROnPlateau` | 30 | Same improvement rule as `EarlyStopping`; floor at `min_lr`; counter reset after a reduction; `reductions` only when the rate changed; state reset in `on_train_begin`; arguments validated | Resetting `best` after a reduction (−5); recording a reduction at the floor (−3) |
| 3. `FlightRegimeLoader` | 25 | Registered as `"fuel_flow_regime"`; base built *through the factory* with `**base_kwargs`; regimes partition the data (80 / 199 / 121); clear `ValueError`s | Hard-coding `SyntheticFuelFlowLoader` instead of the factory (−8); `<=` vs `<` at ±0.5 (−3) |
| 4. Your pytest suite | 15 | ≥ 8 tests pass on a correct implementation; fixture + `parametrize` + `raises` used; at least one value computed by hand | Tests that only check types |

Tasks 1 to 3 are checked by hidden instructor tests; Task 4 by running your suite against the
reference solution. Code quality (type hints, Google-style docstrings, Black formatting, no magic
numbers) is commented on in the feedback.

## Expected output

| Quantity | Expected value |
|----------|----------------|
| `expand([[1, 10], [3, 30]])`, degree 2, interactions | `[[-1, -1, 1, 1, 1], [1, 1, 1, 1, 1]]` |
| Test RMSE: linear / poly-2 / poly-2 + interactions | 51.46 / 41.02 / 22.59 g/h |
| Plateau run (patience 3, factor 0.5, min_delta 1) | first reduction after epoch 32, 17 reductions, final rate 1e-06 |
| Regime sizes (seed 4) | cruise 80, climb 199, descent 121 |
