# Assignment 0 (Week 1): Flight-log quality check

> Provenance: new content. Numbering: this Week 1 warm-up is **Assignment 0** so the graded
> milestones keep their syllabus numbers (Assignment 1 = Week 2 Sensor OOP hierarchy).

| | |
|---|---|
| Released | Week 1 |
| Due | Start of Week 2 lecture |
| Effort | About 3 to 4 hours |
| Notebook | [`assignment00_flight_log_qc.ipynb`](assignment00_flight_log_qc.ipynb) |
| Data | [`data/sample/assignment00_uav_flight_log.csv`](../../data/sample/assignment00_uav_flight_log.csv) |
| Submit | Your branch on the course repository plus the executed notebook |

## The task in one paragraph

A small UAV flew a five-minute test (take-off, climb, steep turn, pull-up, descent, landing).
Its 50 Hz log has spikes, NaN dropouts and a logger outage. You write seven small, typed,
documented functions that load and validate the log, find the sample rate and gaps, report
missing values, compute and filter the load factor, list exceedances of the 2.0 g limit,
estimate gyro noise at rest, and draw a dashboard. Then you write a short engineering verdict:
should the aircraft be grounded for inspection?

The problem statement inside the notebook is in Persian, with an English summary.

## Time budget

| Part | Suggested time |
|------|----------------|
| Environment, clone, branch (Task 1) | 20 min |
| Loading, gaps, missing values (Tasks 2 to 4) | 60 min |
| Load factor and exceedances (Task 5) | 60 min |
| Gyro noise and dashboard (Tasks 6 and 7) | 45 min |
| Verdict and Git history (Task 8) | 20 min |

## Rubric (100 points)

| Task | Points | Full marks when | Typical deductions |
|------|-------:|-----------------|--------------------|
| 1. Environment report | 5 | Correct keys and version strings | Hard-coded versions (−3) |
| 2. Load and validate | 10 | Sorted, fresh index, `ValueError` naming missing columns | Silent failure on missing column (−5) |
| 3. Sample rate and gaps | 15 | Median-based rate; correct gap rows; empty frame when no gap | Mean-based rate (−5); off-by-one in missing samples (−4) |
| 4. Missing-value report | 10 | Counts and percentages, sorted | Fractions instead of percent (−3) |
| 5. Load factor and exceedances | 20 | Vectorised norm; centred median; contiguous runs with duration and peak | Python loop over rows (−5); spikes reported as exceedances (−8) |
| 6. Gyro noise | 10 | Bias and `ddof=1` std over the parked segment | `ddof=0` (−3); wrong time window (−4) |
| 7. Dashboard | 15 | Four labelled panels, limit line, shaded gap, returns `Figure` | Missing units (−1 each); `plt.show()` inside (−2) |
| 8. Verdict and Git | 15 | Quotes filtered peak, exceedance time and duration, explains spike rejection; ≥3 own commits | No numbers (−6); fewer than 3 commits (−5) |

Tasks 1 to 7 (85 points) are auto-graded by hidden instructor tests. Task 8 (15 points) is
graded by hand.

**Code-quality gate** (applies to all tasks, up to −10): type hints on every signature,
Google-style docstrings, English-only comments, no magic numbers (use the constants in the
setup cell), and *Restart & Run All* completes without errors.

## Academic integrity

Discuss ideas freely; write your own code. Cite any source you adapt (for example a W3Schools
or GeeksforGeeks page) in a comment.
