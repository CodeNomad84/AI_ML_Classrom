# Week 2 package: classes, `__init__` and inheritance

**Problem:** a test vehicle carries an IMU, a LiDAR and a GPS, each with its own raw format and
defects. Procedural `if/elif` code breaks silently and must be edited everywhere to add a sensor.
**Tools:** classes, `__init__`/`self`, class vs instance attributes, properties, inheritance and
`super()`, abstract base classes, dataclasses, polymorphism, pytest.

| # | Deliverable | Files |
|---|-------------|-------|
| 1 | Slides (outline + PPTX) | [`slides/en/week02_oop_sensor_hierarchy_EN.md`](../../slides/en/week02_oop_sensor_hierarchy_EN.md) / [`.pptx`](../../slides/en/week02_oop_sensor_hierarchy_EN.pptx), [`slides/fa/week02_oop_sensor_hierarchy_FA.md`](../../slides/fa/week02_oop_sensor_hierarchy_FA.md) / [`.pptx`](../../slides/fa/week02_oop_sensor_hierarchy_FA.pptx) |
| 2 | Lecture notes (PDF + Markdown source) | [`notes/en/week02_notes_EN.pdf`](../../notes/en/week02_notes_EN.pdf), [`notes/fa/week02_notes_FA.pdf`](../../notes/fa/week02_notes_FA.pdf) |
| 3 | Lab notebook (TODO + solved) | [`week02_oop_sensors.ipynb`](week02_oop_sensors.ipynb), [`week02_oop_sensors_solved.ipynb`](week02_oop_sensors_solved.ipynb) |
| 4 | Assignment 1 + rubric | [`assignments/assignment01/`](../../assignments/assignment01/) |
| 5 | In-class problems | [`in_class_problems/week02_problems.ipynb`](../../in_class_problems/week02_problems.ipynb) (solutions: instructor copy, not in the public repo) |
| + | Reusable code and tests | [`src/aerospace_ai/sensors.py`](../../src/aerospace_ai/sensors.py), [`tests/test_sensors.py`](../../tests/test_sensors.py) |
| + | Data | [`data/sample/`](../../data/sample/) (synthetic GPS and LiDAR logs, see its README) |

## Worked numerical examples (hand first, then code)

1. ISA pressure altitude: 95 000 Pa → 540.3 m (472.2 m with QNH 100 500 Pa).
2. IMU bias removal and saturation: four samples, two rejected; load factors 1.005 and 2.003.
3. Week 1 flight log through `IMUSensor` (±4 g): 3 saturated spikes rejected, peak 1.81 g.
4. GPS to local metres: 0.001° N and E of Mehrabad → (90.3 m E, 111.2 m N).
5. In class: shared-list bug, a three-level `super()` chain, IMU calibration by hand
   (n = 1.734), barometric climb rate two ways (10.55 m/s).

## The sensor hierarchy

`BaseSensor(ABC)` fixes the algorithm in `add_reading` (shape check → time check →
`validate` → `preprocess` → store a frozen `SensorReading`) and leaves `validate` and
`preprocess` to subclasses (template method). Week 2 ships `IMUSensor` and `BarometerSensor`;
`LidarSensor` and `GPSSensor` are Assignment 1 and join the library after its deadline.

## Run and rebuild

```bash
pytest                                              # library tests (tests/)
python -m pytest assignments/assignment01 -q        # Assignment 1 student tests
jupyter lab notebooks/week02/                       # the lab
python tools/make_week02_data.py                    # regenerate the GPS/LiDAR CSVs
python tools/make_week02_figures.py                 # regenerate slide/notes figures
python tools/build_slides.py slides/en/week02_oop_sensor_hierarchy_EN.md
python tools/build_slides.py slides/fa/week02_oop_sensor_hierarchy_FA.md
python tools/build_notes.py notes/en/week02_notes_EN.md   # needs pandoc + Chrome/Chromium
python tools/build_notes.py notes/fa/week02_notes_FA.md
```

All notebooks were executed top to bottom with Python 3.11, numpy 1.26, pandas 2.1,
matplotlib 3.8 and pytest 8.1 (the course pins).

## Sources

New content. OOP material rewritten from W3Schools (Python Classes, `__init__()` Method, Class
Properties, Inheritance, Polymorphism) and GeeksforGeeks (class vs instance variables, `super()`,
`property()`, Abstract Classes in Python, Understanding Python Dataclasses, Getting Started with
Pytest); ISA model adapted from Anderson, *Introduction to Flight*, Ch. 3; flat-Earth frame
adapted from Groves, *Principles of GNSS, Inertial, and Multisensor Integrated Navigation
Systems*, Ch. 2. The instructor's PDFs do not cover Python OOP.
