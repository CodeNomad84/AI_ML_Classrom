# Assignment 1 (Week 2): Sensor OOP hierarchy

> Provenance: new content. OOP concepts follow (rewritten, not copied) W3Schools "Python
> Inheritance" and "Python Polymorphism" and GeeksforGeeks "Abstract Classes in Python" and
> "Getting Started with Pytest"; flat-Earth local coordinates adapted from Groves, *Principles
> of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, Ch. 2.

| | |
|---|---|
| Released | Week 2 |
| Due | Start of Week 3 lecture |
| Effort | About 4 hours |
| Notebook | [`assignment01_sensor_hierarchy.ipynb`](assignment01_sensor_hierarchy.ipynb) (Persian problem statement, checks, expected output) |
| Your code | [`sensor_hierarchy.py`](sensor_hierarchy.py) (skeleton with `# TODO`) |
| Your tests | [`test_sensor_hierarchy.py`](test_sensor_hierarchy.py) (two examples given) |
| Data | [`assignment01_gps_track.csv`](../../data/sample/assignment01_gps_track.csv), [`assignment01_lidar_scans.csv`](../../data/sample/assignment01_lidar_scans.csv) |
| Submit | Your branch on the course repository with the executed notebook |

## The task in one paragraph

A ground robot drives one 40 m circle at Tehran Mehrabad airport (GPS, 5 Hz) and then moves
along a 10 m × 6 m hall with a pillar (16-beam LiDAR, 10 Hz). Both logs contain realistic
defects. You implement `GPSSensor` and `LidarSensor` as subclasses of the course's abstract
`BaseSensor` (`src/aerospace_ai/sensors.py`), each supplying only its own `validate` and
`preprocess` plus a few sensor-specific properties and methods; you write a polymorphic
`sensor_report` that works for any sensor; and you write your own pytest suite.

Code goes in the `.py` files (pytest needs importable modules); the notebook loads your module,
runs public checks, plots the results and holds the instructor's hidden test cells.

```bash
python -m pytest assignments/assignment01 -q          # your tests
jupyter lab assignments/assignment01/                  # the notebook
```

## Time budget

| Part | Suggested time |
|------|----------------|
| Read `BaseSensor`, `IMUSensor`, `BarometerSensor` | 30 min |
| GPS: Tasks 1 to 3 | 75 min |
| LiDAR: Tasks 4 to 6 | 60 min |
| `sensor_report` (Task 7) | 20 min |
| Your test suite (Task 8) | 45 min |

## Rubric (100 points)

| Task | Points | Full marks when | Typical deductions |
|------|-------:|-----------------|--------------------|
| 1. `GPSSensor.__init__`, `origin` | 10 | `super().__init__` called; origin and `max_hdop` validated; origin read-only tuple of floats | Base class not initialised (−10); origin settable (−3) |
| 2. `GPSSensor.validate` | 15 | Rejects high/zero/NaN HDOP, impossible or NaN coordinates, non-finite altitude; 288 valid fixes | `hdop < max_hdop` instead of `<=` (−3) |
| 3. `preprocess`, `distance_travelled_m` | 15 | ENU within 1e-6 of the formulas; ≈ 284.0 m path, vectorised | Missing `cos φ0` (−8); Python loop (−3) |
| 4. `channel_names`, `angles_deg` | 10 | `range_00_m` … tuple; angles from −fov/2 in one NumPy expression | Wrong start angle (−5) |
| 5. `_in_range`, `validate`, `preprocess` | 15 | NaN for 0.0 / inf / out-of-range; no NumPy warnings; 46 valid scans | Zeros kept as ranges (−8) |
| 6. `nearest_obstacle` | 10 | Latest **valid** scan, NaN ignored, `ValueError` without a scan; 1.875 m at −22.5° | Uses an invalid scan (−5) |
| 7. `sensor_report` | 10 | Works for any `BaseSensor` (including `BarometerSensor`); correct columns, index and percentages; duplicate IDs raise | `isinstance` checks per type (−5) |
| 8. Your pytest suite | 15 | ≥ 8 tests passing on a correct implementation (3); `raises` + `parametrize` + fixture used (3); 3 pts per hidden broken implementation caught (9) | Tests that only check types catch nothing |

Tasks 1 to 8 are auto-graded by hidden instructor tests (Task 8 runs your tests against the
reference solution and three subtly broken versions of it).

**Code-quality gate** (applies to all tasks, up to −10): type hints on every signature,
Google-style docstrings, English-only comments, no magic numbers (use the module constants),
Black formatting (88 characters), and *Restart & Run All* completes without errors.

## Academic integrity

Discuss ideas freely; write your own code. Cite any source you adapt (for example a W3Schools
or GeeksforGeeks page) in a comment. The reference `LidarSensor` and `GPSSensor` join
`src/aerospace_ai/sensors.py` after the deadline, because Week 3's pipeline uses them.
