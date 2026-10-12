# Week 5 package: data engineering for noisy sensor data

**Problem:** the project rover's braking controller needs the LiDAR range one second ahead. The
naive route (read the CSV, shift, drop NaN, split rows at random, fit) gives a linear model that
misses by 2.73 m, because the log reads "no echo" as 0 m, has NaN bursts, accelerometer spikes,
dust echoes, a frozen gyro, re-sent rows, and 16 drives glued together. **Tools:** a data audit,
missing-data mechanisms (MCAR, MAR, MNAR), local robust outlier rules (Hampel vs z-score and
Tukey), per-run processing, scalers fitted on training rows only, and leakage-free splits by
time and by group, with a leakage audit.

| # | Deliverable | Files |
|---|-------------|-------|
| 1 | Slides (outline + PPTX) | [`slides/en/week05_data_engineering_EN.md`](../../slides/en/week05_data_engineering_EN.md) / [`.pptx`](../../slides/en/week05_data_engineering_EN.pptx), [`slides/fa/week05_data_engineering_FA.md`](../../slides/fa/week05_data_engineering_FA.md) / [`.pptx`](../../slides/fa/week05_data_engineering_FA.pptx) |
| 2 | Lecture notes (PDF + Markdown source) | [`notes/en/week05_notes_EN.pdf`](../../notes/en/week05_notes_EN.pdf), [`notes/fa/week05_notes_FA.pdf`](../../notes/fa/week05_notes_FA.pdf) |
| 3 | Lab notebook (TODO + solved) | [`week05_data_engineering.ipynb`](week05_data_engineering.ipynb), [`week05_data_engineering_solved.ipynb`](week05_data_engineering_solved.ipynb) |
| 4 | **Assignment 2** (graded milestone A2) + rubric | [`assignments/assignment02/`](../../assignments/assignment02/) |
| 5 | In-class problems | [`in_class_problems/week05_problems.ipynb`](../../in_class_problems/week05_problems.ipynb) (solutions: instructor copy, not in the public repo) |
| + | Reusable code and tests | [`src/aerospace_ai/rover.py`](../../src/aerospace_ai/rover.py), [`preprocessing.py`](../../src/aerospace_ai/preprocessing.py), [`splits.py`](../../src/aerospace_ai/splits.py), [`datasets.py`](../../src/aerospace_ai/datasets.py) (`rover_range_forecast`); [`tests/test_rover.py`](../../tests/test_rover.py), [`test_preprocessing.py`](../../tests/test_preprocessing.py), [`test_splits.py`](../../tests/test_splits.py) |
| + | Data | [`data/sample/week05_rover_log.csv`](../../data/sample/week05_rover_log.csv) (lecture, seed 5), [`assignment02_rover_log.csv`](../../data/sample/assignment02_rover_log.csv) (A2, seed 52); see the data README |

## Worked numerical examples (hand first, then code)

1. Mean imputation under MNAR: true ranges 2 … 17 m, LiDAR limit 12 m; true mean 9.375 m,
   observed and imputed mean 6.0 m (bias −3.4 m).
2. Interpolating a gap: 8.0 → ? → ? → 7.4 m gives 7.8 and 7.6 m.
3. Masking: one 25 m/s² spike in eight samples has z = 2.645 ≤ √7 (Samuelson bound), never
   flagged; robust score 112. Exercise: z = 3 needs n ≥ 10.
4. Scaling changes the nearest neighbour: raw d_A = 1.005, d_B = 1.200; scaled 2.500 and 0.400.
5. Split comparison, k-NN RMSE: random rows 0.319 (leaky), chronological 0.634 (leaky),
   chronological with a 10-row gap 0.634, by run 0.641 m. Linear: 0.098 to 0.121 m.
6. Leak-free workflow: mean 3.833, linear **0.121** (from 2.73 naive), k-NN 0.641 m.
7. In class: a dust echo in seven readings (z = −√6, robust score 229), a saturating temperature
   sensor (MNAR), a five-line leakage hunt and scaling by hand.

## What the library adds

* **Stateless stages** (`preprocessing.py`): `SortAndDeduplicate`, `MarkInvalid` (sentinels and
  valid range to NaN), `CausalRollingMean`, `ForecastTarget`, and the Composite `PerGroup`,
  which applies any stateless stage to each run separately and round-trips through the Week 3
  config. Audit helpers `audit_frame`, `count_duplicate_keys`, `longest_true_run`.
* **Learned stages**: template-method base `ColumnStatStage` (`fit` → `learn`, `transform` →
  `apply`) with `ImputeMedian` (optional `_was_missing` indicators), `MinMaxScale`,
  `RobustScale`.
* **Splitters** (`splits.py`, Strategy): `RandomRowSplit`, `ChronologicalSplit(gap)`,
  `GroupHoldoutSplit`, `ExpandingWindowCV`; `check_split` returns a `LeakageReport`.
* **Factory loader** `rover_range_forecast` runs `rover_cleaning_pipeline()` and returns a
  `Dataset` with the new `groups` field (run ids).

## Run and rebuild

```bash
pytest                                              # library tests (tests/)
python -m pytest assignments/assignment02 -q        # Assignment 2 student tests
jupyter lab notebooks/week05/                       # the lab
python tools/make_week05_data.py                    # regenerate both rover logs
python tools/make_week05_figures.py                 # regenerate slide figures
python tools/build_slides.py slides/en/week05_data_engineering_EN.md
python tools/build_slides.py slides/fa/week05_data_engineering_FA.md
python tools/build_notes.py notes/en/week05_notes_EN.md   # needs pandoc + Chrome/Chromium
python tools/build_notes.py notes/fa/week05_notes_FA.md
```

All notebooks were executed top to bottom with Python 3.11, numpy 1.26, pandas 2.1,
matplotlib 3.8, scikit-learn 1.4 and pytest 8.1 (the course pins).

## Sources

Partly expanded from instructor notes: slides 2 and 3 and lab Section 0 expand slides 2 to 8 of
the instructor's deck "Machine Learning and Decision Trees" (AI, ML and DL; feature extraction;
classical methods need preprocessing; the task taxonomy). Its decision-tree slides (9 to 11) are
kept for Week 7. Everything else is new: missing-data mechanisms from Little and Rubin,
*Statistical Analysis with Missing Data* (2019), Ch. 1; cleaning, imputation, outliers, scaling
and leakage rewritten (not copied) from W3Schools ("Pandas - Cleaning Data", "ML Scaling",
"ML Train/Test") and GeeksforGeeks ("Handling Missing Values in Machine Learning", "Detect and
Remove the Outliers using Python", "Data Leakage"); LiDAR error types from Thrun, Burgard and
Fox, *Probabilistic Robotics* (2005), Sec. 6.3; the z-score bound from Samuelson, JASA 63 (1968)
and Shiffler, *Am. Stat.* 42 (1988); the Hampel filter from Pearson, IEEE TCST 10 (2002);
blocked and grouped validation from Roberts et al., *Ecography* 40 (2017).
