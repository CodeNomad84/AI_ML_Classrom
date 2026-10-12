# Assignment 2 (Week 5): Leak-free cleaning of a noisy rover log

> Provenance: new content. Missing-data mechanisms after Little and Rubin, *Statistical Analysis
> with Missing Data* (2019), Ch. 1; Tukey's fences from Tukey, *Exploratory Data Analysis*
> (1977), Ch. 2; cleaning and leakage rules rewritten (not copied) from W3Schools ("Pandas -
> Cleaning Data") and GeeksforGeeks ("Handling Missing Values in Machine Learning", "Detect and
> Remove the Outliers using Python", "Data Leakage").

| | |
|---|---|
| Released | Week 5 |
| Due | Start of Week 6 lecture |
| Effort | About 4 hours |
| Weight | Graded milestone **A2** (syllabus) |
| Notebook | [`assignment02_sensor_cleaning.ipynb`](assignment02_sensor_cleaning.ipynb) (Persian problem statement, checks, expected output) |
| Your code | [`sensor_cleaning.py`](sensor_cleaning.py) (skeleton with `# TODO`) |
| Your tests | [`test_sensor_cleaning.py`](test_sensor_cleaning.py) (two examples given) |
| Data | [`assignment02_rover_log.csv`](../../data/sample/assignment02_rover_log.csv) (16 test drives, seed 52; the lecture log uses seed 5) |

## The task in one paragraph

The project rover's braking controller needs the LiDAR range to the wall **one second ahead**.
The raw log has every defect from the lecture: LiDAR zeros (no echo), NaN bursts in the wheel
speed, accelerometer spikes, dust echoes, a frozen gyroscope and re-sent rows. Starting from the
lecture recipe (`aerospace_ai.datasets.rover_cleaning_pipeline`) and the Week 5 library, you
count the defects and separate the two LiDAR missing-data mechanisms (Task 1), write a stateless
frozen-sensor detector (Task 2) and a learned Tukey-fence clipper (Task 3), assemble the
stateless and the learned recipes (Task 4), and deliver model-ready arrays from a split by
**run** with every learned statistic taken from the training runs only (Task 5). Nothing in
`src/` may be edited.

```bash
python -m pytest assignments/assignment02 -q          # your tests
jupyter lab assignments/assignment02/                  # the notebook
```

## Time budget

| Part | Suggested time |
|------|----------------|
| Re-read the Week 5 lab, `preprocessing.py`, `splits.py` | 40 min |
| Task 1: audit | 35 min |
| Task 2: frozen sensor | 40 min |
| Task 3: Tukey clipper | 30 min |
| Task 4: recipes | 25 min |
| Task 5: leak-free preparation | 45 min |
| Task 6: tests and answers | 35 min |

## Rubric (100 points)

| Task | Points | Full marks when | Typical deductions |
|------|-------:|-----------------|--------------------|
| 1. `audit_report` | 15 | Duplicates counted on the raw table; zeros split per run into leading (MNAR) and isolated (MCAR) after de-duplication | Counting a re-sent row twice (−4); "leading" decided over the whole log instead of per run (−6) |
| 2. `FlagStuck` | 20 | Every sample of a run of ≥ `min_run` equal values (including the first) becomes NaN; NaN never forms a run; `min_run` validated; input untouched | Missing the first sample of the run (−5); a Python loop over rows instead of NumPy (code quality, −3) |
| 3. `IQRClipper` | 15 | Subclasses `ColumnStatStage`; fences from training quartiles; clipping (not dropping); NaN preserved; `k` validated | Fences from test data (−8); dropping rows instead of clipping (−5) |
| 4. Recipes | 15 | `FlagStuck` per run directly after de-duplication; no learned stage in the cleaning recipe; learned recipe in the documented order with `k = 3`; config round-trips | Applying `FlagStuck` across run boundaries (−5) |
| 5. `prepare` | 25 | Group split by run (runs 13–16 test); learned pipeline fitted on training rows only; shapes and targets as expected; leakage report OK; linear test RMSE 0.109 m | Fitting the scaler before the split (−10); splitting rows instead of runs (−10) |
| 6. Tests and answers | 10 | ≥ 7 tests pass on a correct implementation, with a fixture, `parametrize` and `raises`; three short, correct answers | Tests that only check types; answers without the mechanism or the leak named |

Tasks 1 to 5 are checked by hidden instructor tests; Task 6 by running your suite against the
reference solution and reading your answers. Code quality (type hints, Google-style docstrings,
Black formatting, no magic numbers) is part of every task's mark.

## Expected output

| Quantity | Expected value |
|----------|----------------|
| `DefectCounts` | duplicate_rows 32, lidar_leading_zeros 482, lidar_isolated_zeros 50, speed_nan 25 |
| Frozen gyro samples | 62 (two segments of 31, runs 3 and 15) |
| Hand example, Task 3 | fences (−1, 7) for k = 1.5 and (−4, 10) for k = 3 on `[1, 2, 3, 4, 100]` |
| Rows after cleaning | 1919 (62 with a NaN gyro value) |
| `prepare` | X_train (1426, 4), X_test (493, 4); runs 1–12 train, 13–16 test; leakage OK |
| Linear test RMSE | 0.109 m (0.112 m with k = 1.5) |
