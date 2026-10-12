# Practice Assignment 3 (Week 3): A reproducible air-data pipeline

> Provenance: new content. Dataclass and ABC usage follows (rewritten, not copied) W3Schools
> "Python dataclasses Module" and "Python abc Module" and GeeksforGeeks "Understanding Python
> Dataclasses", "Abstract Classes in Python" and "Getting Started with Pytest"; min-max scaling
> adapted from Géron, *Hands-On Machine Learning*, 3rd ed., Ch. 2.

| | |
|---|---|
| Released | Week 3 |
| Due | Start of Week 4 lecture |
| Effort | About 3.5 hours |
| Notebook | [`practice03_airdata_pipeline.ipynb`](practice03_airdata_pipeline.ipynb) (Persian problem statement, checks, expected output, rubric) |
| Your code | [`airdata_stages.py`](airdata_stages.py) (skeleton with `# TODO`) |
| Your tests | [`test_airdata_stages.py`](test_airdata_stages.py) (two examples given) |
| Data | [`week01_uav_flight_log.csv`](../../data/sample/week01_uav_flight_log.csv) (the Week 1 log) |
| Submit | Your branch on the course repository with the executed notebook and `airdata_pipeline.json` |

**Why "practice"?** The syllabus milestones keep their numbers (A1 = Week 2, A2 = Week 5,
A3 = Week 7, ...). Weeks without a milestone get a graded *practice* assignment named after the
week, so Practice Assignment 3 belongs to Week 3 and does not collide with Assignment 3.

## The task in one paragraph

The Week 6 fuel-consumption model will be trained on barometric altitude, airspeed, vertical
acceleration and climb rate from the Week 1 flight log. The log has three impossible
accelerometer spikes, 75 NaN dropouts per air-data channel and a 1.2 s logger gap at t ≈ 100 s,
where a row-counting moving average invents a 21 m/s climb rate. You write three dataclass
stages on top of the course `PipelineStage` ABC (`PlausibleRange`, `Resample` and the learning
stage `MinMaxScale`), assemble the seven-stage pipeline in the notebook's table, fit it on the
first 70 % of the flight only, save and reload its configuration as JSON, and write your own
pytest suite.

```bash
python -m pytest assignments/practice03 -q            # your tests
jupyter lab assignments/practice03/                    # the notebook
```

## Time budget

| Part | Suggested time |
|------|----------------|
| Read `src/aerospace_ai/pipeline.py` (`HampelFilter`, `Standardize`, `DataPipeline`) | 30 min |
| Task 1: `PlausibleRange` | 20 min |
| Task 2: `Resample` | 40 min |
| Task 3: `MinMaxScale` | 40 min |
| Task 4: pipeline, time split, save/load | 40 min |
| Task 5: your test suite | 40 min |

## Rubric (100 points)

| Task | Points | Full marks when | Typical deductions |
|------|-------:|-----------------|--------------------|
| 1. `PlausibleRange` | 15 | Frozen; tuple columns; bounds validated; closed interval; NaN, not clip; input untouched | Clipping (−8); `<` instead of `<=` (−3) |
| 2. `Resample` | 20 | Validated rate; `period_s` property; round-off-safe grid; NaN bridged; non-numeric columns dropped | `np.arange(t0, t1, dt)` length errors (−5) |
| 3. `MinMaxScale` | 25 | Learned fields with `init/repr/compare=False`; `NotFittedError`; exact formulas both ways; constant column rejected | Learned state compared (−5); `feature_range` ignored (−8) |
| 4. Pipeline, split, save/load | 25 | Exact stage order and settings; time split; fit on the training part only; indented JSON round trip | Fitting on all rows (−10) |
| 5. Your pytest suite | 15 | ≥ 6 tests passing on a correct implementation (3); `raises` + `parametrize` + fixture used (3); 3 pts per hidden broken implementation caught (9) | Tests that only check types catch nothing |

Tasks 1 to 5 are auto-graded by hidden instructor tests (Task 5 runs your tests against the
reference solution and three subtly broken versions of it).

**Code-quality gate** (applies to all tasks, up to −10): type hints on every signature,
Google-style docstrings, English-only comments, no magic numbers (use the module constants),
Black formatting (88 characters), and *Restart & Run All* completes without errors.

## Academic integrity

Discuss ideas freely; write your own code. Cite any source you adapt (for example a W3Schools
or GeeksforGeeks page) in a comment.
