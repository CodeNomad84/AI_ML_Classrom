---
title: "Week 2: Classes, __init__ and Inheritance"
subtitle: "One interface for every sensor on the vehicle"
course: "AI for Aerospace & Mechanical Engineering (PhD)"
footer: "Week 02 | OOP: the sensor hierarchy"
language: en
provenance: "New content. OOP concepts rewritten from W3Schools (Classes, __init__, Class Properties, Inheritance, Polymorphism) and GeeksforGeeks (class vs instance variables, super(), property(), Abstract Classes, dataclasses, Getting Started with Pytest); ISA model adapted from Anderson, Introduction to Flight, Ch. 3; flat-Earth frame adapted from Groves, Principles of GNSS, Ch. 2."
---

<!-- Build: python tools/build_slides.py slides/en/week02_oop_sensor_hierarchy_EN.md -->

## SLIDE 1 | Three sensors, one robot
**Content:**
- IMU: 6 numbers at 50 Hz, biased, can saturate
- LiDAR: 16 ranges at 10 Hz, 0.0 or inf when no echo
- GPS: lat, lon, alt, HDOP at 5 Hz, some fixes are garbage
- Every later module (lanes, obstacles, control) consumes them
- Q: how do we add a 4th sensor without editing everything?
**Visual:** Top view of a small ground robot with three labelled sensors (IMU in the body, LiDAR on the roof with a ring of beams, GPS antenna), each with an arrow to a different raw-data format.
**Notes:** Open with the vehicle, not with Python. Ask the room how they would organise code for three different sensors. Write the final question on the board and keep it there.

## SLIDE 2 | Why the procedural version breaks
**Content:**
- One function per job, an if/elif chain on the sensor type inside each
- Adding a barometer = edit validate, preprocess, to_frame, plot
- Configuration (biases) ends up in globals
- Typos fall through the chain silently
**Visual:** IMAGE ../assets/week02/fig_procedural_vs_oop.png
**Notes:** Left grid: the new column touches every function. Right grid: the new sensor only writes the two methods that are really different; the rest is inherited.

## SLIDE 3 | Example 1: the silent typo
**Content:**
- preprocess("imu", raw) removes the bias
- preprocess("IMU", raw) returns raw data: no error
- preprocess("imu", [0.35, -0.05]) accepts 2 values: zip stops early
- Biased data flows into the model unnoticed
**Visual:** Code block: `def preprocess(sensor_type, data): if sensor_type == "imu": ... elif "lidar": ... return data` with the three calls and their outputs below; the two wrong outputs highlighted in orange.
**Notes:** Run Section 0 of the lab live. The point: errors that do not raise are the expensive ones.

## SLIDE 4 | Class, object, __init__, self
**Content:**
- Class = blueprint; object = one thing built from it
- __init__ runs at creation and stores attributes on self
- Method = function in the class; Python passes the object as self
- Two objects, one class, different state (two QNH settings)
**Visual:** Code: `class SimpleBarometer: def __init__(self, sensor_id, reference_pressure_pa=101325.0): ... def altitude_m(self, p): ...` next to two object boxes "baro0 (101 325 Pa)" and "baro_qnh (100 500 Pa)".
**Notes:** Rewritten from W3Schools "Python Classes" and "__init__() Method". Stress that self is just the first parameter; the name is a convention.

## SLIDE 5 | Class attributes vs instance attributes
**Content:**
- Class body: shared by all objects (good for constants)
- self.x in __init__: one per object (state)
- Mutating a shared list leaks data between sensors
- Assignment on an instance shadows; mutation does not
**Visual:** IMAGE ../assets/week02/fig_shared_list.png
**Notes:** GeeksforGeeks "Class (static) and instance variables". In-class Problem 1 drills this. Rule: mutable state goes in __init__.

## SLIDE 6 | Example 2: ISA altitude by hand
**Content:**
- h = (T0/L) [1 − (p/p_ref)^(RL/g0)], T0/L = 44 330.8 m, RL/g0 = 0.190263
- p = 95 000 Pa: ratio 0.937577, ln = −0.064457
- power = e^(−0.012264) = 0.987811
- h = 44 330.8 × 0.012189 = 540.3 m
- QNH 100 500 Pa instead: 472.2 m (same class, other object)
**Visual:** IMAGE ../assets/week02/fig_isa_altitude.png
**Notes:** Do the arithmetic on the board first, then run SimpleBarometer. Exercise 1.1 is the QNH case.

## SLIDE 7 | Properties: attributes that check themselves
**Content:**
- @property: looks like an attribute, runs a method
- Setter validates: rate_hz = 0 raises ValueError
- No setter = read-only (sensor_id)
- Derived values stay consistent: period_s = 1 / rate_hz
- _name: "internal, do not touch" convention
**Visual:** Code: `@property def rate_hz(self)` / `@rate_hz.setter def rate_hz(self, value): if not value > 0: raise ValueError`; arrow from `imu.rate_hz = 0` to a red "ValueError".
**Notes:** GeeksforGeeks "Python property() function". Ask: why not store period_s in __init__? Because it would go stale when the rate changes.

## SLIDE 8 | Inheritance and super()
**Content:**
- class Barometer(Sensor): gets every method of Sensor
- Override: same method name, sensor-specific body
- super().__init__(...) lets the parent set up its part
- isinstance(baro, Sensor) is True
- MRO: the order Python searches for a method
**Visual:** Two stacked boxes Sensor (sensor_id, rate_hz, readings, add_reading, preprocess) and Barometer (reference_pressure_pa, preprocess override) joined by an inheritance arrow.
**Notes:** W3Schools "Python Inheritance" and GeeksforGeeks "super()". Live: print Barometer.__mro__.

## SLIDE 9 | Example 3: follow super() through three classes
**Content:**
- IMU → InertialSensor → Sensor → object
- Each __init__ prints, then calls super().__init__
- Printed: IMU init, Inertial init, Sensor init
- describe() returns "IMU/inertial imu0 @ 200 Hz"
- Forget one super().__init__: AttributeError much later
**Visual:** Vertical chain of four boxes (IMU, InertialSensor, Sensor, object); down arrows labelled "super().__init__", up arrows labelled "return value of describe()".
**Notes:** In-class Problem 2. Students predict before running. The late AttributeError is the key lesson.

## SLIDE 10 | Abstract base classes: the interface as a contract
**Content:**
- class BaseSensor(ABC) with @abstractmethod validate, preprocess
- BaseSensor(...) raises TypeError: cannot instantiate
- Subclass missing a method (or with a typo) also raises TypeError
- Error at construction, not mid-flight
**Visual:** Code: `class ForgetfulBarometer(AbstractSensor): def pre_process(...)` and the resulting `TypeError: Can't instantiate abstract class ... with abstract method preprocess`.
**Notes:** GeeksforGeeks "Abstract Classes in Python". Week 3 goes deeper (abstract properties, typing).

## SLIDE 11 | Dataclasses: SensorReading
**Content:**
- @dataclass writes __init__, __repr__, __eq__ from the fields
- frozen=True: a logged measurement cannot change (and is hashable)
- slots=True: no per-object __dict__, less memory
- values is a tuple: arrays break == and hash
**Visual:** Side by side: 15-line ManualReading class vs 6-line `@dataclass(frozen=True, slots=True) class SensorReading: timestamp: float; values: tuple[float, ...]; sensor_id: str; is_valid: bool = True`.
**Notes:** GeeksforGeeks "Understanding Python Dataclasses". Show FrozenInstanceError live.

## SLIDE 12 | Example 4: IMU bias and saturation by hand
**Content:**
- ±4 g IMU: full scale 39.23 m/s²; bias a = (0.10, −0.05, 0.20), g = (0.8, −0.3, 0.1)
- (0.35, −0.05, 10.05 | 0.8, −0.3, 2.1) → (0.25, 0, 9.85 | 0, 0, 2.0), n = 1.005
- NaN in a_y → invalid, stored raw
- a_z = 39.50 ≥ 39.23 → saturated, invalid: true value unknown
- (1.30, −0.45, 19.80 | ...) → (1.2, −0.4, 19.6), n = 2.003
**Visual:** Four-row table: raw sample, valid?, preprocessed values, load factor; invalid rows shaded orange.
**Notes:** Same numbers as Lab Section 7 and tests. The last sample is Week 1's worked example; continuity matters for weak programmers.

## SLIDE 13 | The course sensor hierarchy
**Content:**
- BaseSensor owns buffering, time checks, arrays, DataFrames
- Subclasses own only validate() and preprocess()
- IMUSensor, BarometerSensor: in src/aerospace_ai/sensors.py
- LidarSensor, GPSSensor: Assignment 1
**Visual:** IMAGE ../assets/week02/fig_class_hierarchy.png
**Notes:** Instructor: read sensors.py before class; it also uses __slots__, __init_subclass__ and a deque buffer, which we discuss in Weeks 3 and 4.

## SLIDE 14 | Template method: add_reading
**Content:**
- Algorithm fixed once in the base class
- Steps 3 and 4 delegated to the subclass
- Invalid samples kept raw with is_valid=False
- to_array() returns valid samples only
**Visual:** IMAGE ../assets/week02/fig_template_method.png
**Notes:** Name the pattern now; Week 4 covers Strategy, Observer and Factory.

## SLIDE 15 | Example 5: the Week 1 log through IMUSensor
**Content:**
- Calibrate biases from t < 25 s (parked, level)
- Feed 14 940 samples; ±4 g full scale
- 3 spikes (≈ 50 m/s²) rejected as saturated
- Peak load factor 1.81 g: below the 2.0 g limit
- Week 1 needed a rolling median; here the object knows its limits
**Visual:** IMAGE ../assets/week02/fig_imu_validation.png
**Notes:** Lab Section 9. Caveat: rejection works here because the spikes exceed full scale; spikes inside the range still need filtering.

## SLIDE 16 | Polymorphism: one loop, many sensors
**Content:**
- for sensor in suite: sensor.add_reading(t, raw)
- No if/elif on types anywhere
- A new ThermometerSensor joins with zero changes to the loop
- Same call, type-specific behaviour
**Visual:** Code loop over `[IMUSensor, BarometerSensor, ThermometerSensor]` printing id, class, channels, readings, rejected.
**Notes:** W3Schools "Python Polymorphism". Exercise 8.1 builds the thermometer (kelvin in, Celsius out).

## SLIDE 17 | LiDAR: beams, angles and missing echoes
**Content:**
- Beam i at −180° + i × 22.5° (16 beams, 360°)
- No echo: 0.0 or inf; dust: 0.05 m < min range
- Unusable beams become NaN, not zero
- Scan valid if ≥ 50 % of beams usable (fog rejects 4 of 50)
- nearest_obstacle(): np.nanargmin on the latest valid scan
**Visual:** IMAGE ../assets/week02/fig_lidar_scan.png
**Notes:** Why NaN and not 0: a zero range would be read as an obstacle touching the robot. Assignment 1, Tasks 4 to 6.

## SLIDE 18 | Example 6: GPS to local metres by hand
**Content:**
- 1° latitude = Rπ/180 = 111 195 m (R = 6 371 km)
- 1° longitude = 111 195 × cos φ0; at Mehrabad (35.69°): 90 314 m
- 0.001° N and E of origin → (east, north) = (90.3, 111.2) m
- Reject: HDOP > 5, NaN, |lat| > 90°
**Visual:** IMAGE ../assets/week02/fig_gps_enu.png
**Notes:** Flat-Earth approximation adapted from Groves, Ch. 2; error below 1 cm over 80 m. Forgetting cos φ0 is one of the grader's mutants.

## SLIDE 19 | Testing classes with pytest
**Content:**
- test_*.py files, test_* functions, plain assert
- pytest.raises: the error must happen
- @pytest.mark.parametrize: one test, many inputs
- @pytest.fixture: a fresh object per test
- Good tests check numbers, not just types
**Visual:** Code: fixture `imu()`, `test_imu_removes_bias`, parametrized `test_rate_must_be_positive`; terminal line "41 passed".
**Notes:** GeeksforGeeks "Getting Started with Pytest". Assignment 1 grades students' tests by running them against three subtly broken implementations.

## SLIDE 20 | This week's work
**Content:**
- Lab: notebooks/week02/week02_oop_sensors.ipynb (5 TODO exercises)
- In class: 4 warm-ups (shared list, super chain, IMU by hand, climb rate)
- Assignment 1: LiDAR + GPS sensors + your pytest suite, about 4 h, due Week 3
- python -m pytest assignments/assignment01 before you push
**Visual:** Three icons: notebook, pencil, Git branch.
**Notes:** Assignment 1 hidden tests check numbers from the provided CSVs; the expected-output table in the notebook lets students self-check.

## SLIDE 21 | What We Now Know How To Do
**Content:**
- Bundle data and behaviour in classes; use __init__ and self correctly
- Guard state with properties; keep mutable state per object
- Reuse with inheritance and super(); enforce interfaces with ABCs
- Write records with frozen dataclasses; process sensors polymorphically
- Test a class hierarchy with pytest
**Visual:** The opener's robot with its three sensors now all pointing into one box "BaseSensor interface", green check marks on each.
**Notes:** Return to the opener question: a fourth sensor is one new subclass with two methods.
