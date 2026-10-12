---
title: "Week 1: Python and Environment Review"
subtitle: "From a raw flight log to engineering answers"
course: "AI for Aerospace & Mechanical Engineering (PhD)"
footer: "Week 01 | Python & environment review"
language: en
provenance: "New content. Python tooling rewritten from W3Schools (Python, NumPy, Pandas, Matplotlib, Git tutorials) and GeeksforGeeks (virtual environments, Anaconda, Jupyter, describe()); load factor adapted from Anderson, Introduction to Flight, Ch. 6."
---

<!-- Build: python tools/build_slides.py slides/en/week01_python_environment_EN.md -->

## SLIDE 1 | A flight log lands on your desk
**Content:**
- 5-minute UAV test flight, IMU at 50 Hz + baro, pitot, temperature
- Q1: What was the sample rate? Did the logger lose data?
- Q2: Peak load factor? Was the 2.0 g limit exceeded?
- Q3: How noisy are the gyros while parked?
- Q4: What was the peak climb rate?
**Visual:** IMAGE ../assets/week01/fig_dashboard.png
**Notes:** Open with the file, not with Python. Ask the room how they would answer Q2 with Excel. Keep the four questions on the board all session; every tool we introduce answers one of them.

## SLIDE 2 | Today: problem → tools → answers
**Content:**
- Reproducible environment: conda / venv, Jupyter, Git
- NumPy: vectorised maths on arrays
- Pandas: labelled tables, read_csv, describe
- Matplotlib: figures that answer questions
- Preview: the same logic as a reusable class
**Visual:** Left-to-right chain of five boxes: "CSV file" → "Environment" → "NumPy" → "Pandas" → "Matplotlib" → "Answers Q1–Q4".
**Notes:** Students with weak programming backgrounds need to see that each tool exists for a reason.

## SLIDE 3 | Example 0: what the raw file looks like
**Content:**
- time_s, accel_x/y/z [m/s²], gyro_x/y/z [deg/s]
- baro_alt_m, airspeed_mps, temp_c
- 14 940 rows × 10 columns, ~1 MB
- Units live in the column names: a convention, not luck
**Visual:** Code block of the first three CSV lines:
`time_s,accel_x_mps2,...,temp_c` / `0.0,-0.052,0.0375,9.8537,...` / `0.02,0.0389,0.0033,9.863,...`
**Notes:** Point out accel_z ≈ 9.81 on the ground: the accelerometer measures specific force, so it reads +1 g when parked.

## SLIDE 4 | Why an isolated environment?
**Content:**
- "Works on my machine" is not a result
- One environment per project, pinned versions
- conda: Python + compiled libraries (PyTorch later)
- venv + pip: built-in, lightweight
- environment.yml / requirements.txt = the recipe
**Visual:** IMAGE ../assets/week01/fig_environment_stack.png
**Notes:** Source: GeeksforGeeks "Python Virtual Environment" and "Set up virtual environment for Python using Anaconda" (rewritten).

## SLIDE 5 | The commands you need this week
**Content:**
- conda create -n aero-ai python=3.11 → conda activate aero-ai
- python -m venv .venv → activate → pip install -r requirements.txt
- jupyter lab (run cells: Shift+Enter; restart & run all)
- git clone / switch -c / add / commit / push
- git status before every commit
**Visual:** Two-column cheat sheet: left "conda / venv / jupyter", right "git". Highlight `git status`.
**Notes:** Demo live; have students type along. Git basics follow W3Schools "Git Introduction" (repository, staging, commit, branch, push/pull).

## SLIDE 6 | Example: first environment and first commit
**Content:**
- conda env create -f environment.yml
- jupyter lab → open notebooks/week01/week01_python_env.ipynb
- Run the version-check cell: all "ok"?
- git switch -c week01-yourname
- git add, commit -m "Week 1: environment ok", push
**Visual:** Screenshot placeholder of the version-check cell output with four "ok" lines.
**Notes:** 10-minute hands-on. Anyone who fails here gets help now, not in Week 9.

## SLIDE 7 | NumPy: arrays, shape, axis
**Content:**
- ndarray: one block of numbers, one dtype
- accel.shape = (N, 3): rows = samples, columns = x, y, z
- axis=0 collapses samples; axis=1 collapses components
- Vectorised: np.sqrt(np.sum(a**2, axis=1)) → ~10× faster than a loop
**Visual:** IMAGE ../assets/week01/fig_axis_broadcasting.png
**Notes:** Rewritten from W3Schools "Introduction to NumPy". The timing cell in the notebook shows the speed-up on 200 000 vectors.

## SLIDE 8 | Broadcasting and boolean masks
**Content:**
- (4, 3) − (3,) → (4, 3): trailing dimensions align
- Remove per-axis IMU bias: a − a.mean(axis=0)
- Mask: on_ground = t < 25 → df.loc[on_ground]
- Masks replace if-statements inside loops
**Visual:** IMAGE ../assets/week01/fig_axis_broadcasting.png
**Notes:** In-class Problem 4 drills this.

## SLIDE 9 | Example 1: noise statistics by hand (ddof)
**Content:**
- x = [0.2, −0.4, 0.1, 0.5, −0.4] deg/s → mean = 0
- Σ(x − x̄)² = 0.62
- Population σ = √(0.62/5) = 0.352 (np.std default)
- Sample s = √(0.62/4) = 0.394 (pandas .std default)
- Always write ddof explicitly
**Visual:** Two-row table: "divide by N → 0.352 (NumPy)", "divide by N−1 → 0.394 (Pandas)", with the 12 % difference highlighted.
**Notes:** Do it on the board first, then run the cell. This is a classic silent bug.

## SLIDE 10 | Load factor from an accelerometer
**Content:**
- Accelerometer measures specific force f (not gravity)
- n = ‖f‖ / g₀, with g₀ = 9.80665 m/s²
- Level coordinated turn: n = 1 / cos φ
- 30° bank → 1.155 g; 60° bank → 2.0 g
**Visual:** IMAGE ../assets/week01/fig_banked_turn.png
**Notes:** Adapted from Anderson, Introduction to Flight, Ch. 6 (turning flight).

## SLIDE 11 | Pandas: first look at any dataset
**Content:**
- df = pd.read_csv(path)
- df.head(): does it look right?
- df.info(): dtypes and non-null counts
- df.describe().T: count, mean, std, min, quartiles, max
- Read count and max like an engineer
**Visual:** describe() table with two cells circled: count = 14 865 for baro_alt_m (missing data) and max = 50.3 m/s² for accel_z (≈ 5 g?).
**Notes:** Rewritten from W3Schools "Pandas Read CSV" / "Analyzing DataFrames" and GeeksforGeeks "DataFrame describe()".

## SLIDE 12 | Example 2: load factor by hand
**Content:**
- f = (1.2, −0.4, 19.6) m/s²
- ‖f‖² = 1.44 + 0.16 + 384.16 = 385.76
- ‖f‖ = 19.641 m/s² → n = 19.641 / 9.80665 = 2.003
- One sample above 2.0 g. Real exceedance or not?
**Visual:** 3-D vector arrow with its three components and the norm labelled.
**Notes:** Then the vectorised version: np.linalg.norm(samples, axis=1) / G0.

## SLIDE 13 | Q1: sample rate and gaps
**Content:**
- dt = t.diff(); fs = 1 / median(dt) = 50 Hz
- Median, not mean: one gap ruins the mean
- Nyquist = fs / 2 = 25 Hz
- Gap: dt > 1.5 × nominal → 1.22 s step = 60 missing samples
- 0.5 % NaN in baro and airspeed
**Visual:** IMAGE ../assets/week01/fig_gap_timeline.png
**Notes:** Answer to Q1 goes on the board.

## SLIDE 14 | Q2: spike or manoeuvre?
**Content:**
- Raw peak: 5.13 g at t = 66.68 s
- Neighbours: 1.06, 1.06, 5.13, 1.06, 1.06 → one sample
- Real manoeuvres last seconds (hundreds of samples)
- Rolling median (5 samples) removes spikes, keeps manoeuvres
- Filtered peak 1.80 g at 192 s → limit not exceeded
**Visual:** IMAGE ../assets/week01/fig_spike_vs_maneuver.png
**Notes:** This is the first data-cleaning decision of the course; Week 5 generalises it.

## SLIDE 15 | Example 3: mean vs median, and a gap by hand
**Content:**
- Window [1.00, 1.01, 5.10, 0.99, 1.00]
- Mean = 9.10 / 5 = 1.82 g (spike leaks in)
- Median = sort → [0.99, 1.00, 1.00, 1.01, 5.10] → 1.00 g
- Timestamps 99.94, 99.96, 99.98, 101.20 → 1.22 / 0.02 − 1 = 60 lost
**Visual:** IMAGE ../assets/week01/fig_mean_vs_median.png
**Notes:** Students compute both on paper before you reveal them.

## SLIDE 16 | Q3 and Q4: noise and derivatives
**Content:**
- Parked (t < 25 s): gyro s ≈ 0.30 deg/s, bias ≈ 0
- Histogram ≈ Gaussian → white noise
- Differentiating noise amplifies it: ±28 m/s raw
- Smooth (2 s) first, then differentiate
- Peak climb ≈ 16 m/s at 84 s (design 15 m/s)
**Visual:** IMAGE ../assets/week01/fig_climb_rate.png
**Notes:** Show fig_gyro_noise.png as a backup if time allows.

## SLIDE 17 | Matplotlib: figures that answer questions
**Content:**
- fig, axes = plt.subplots(4, 1, sharex=True)
- One question per panel; units on every axis
- Reference lines: ax.axhline(2.0, ls="--")
- Shade gaps: ax.axvspan(start, end)
- Return the Figure from functions; show it outside
**Visual:** IMAGE ../assets/week01/fig_gyro_noise.png
**Notes:** Rewritten from W3Schools "Matplotlib Pyplot".

## SLIDE 18 | Example 4: a derivative by hand
**Content:**
- h = [250.0, 250.3, 250.5] m at t = [80.00, 80.02, 80.04] s
- Central difference: (250.5 − 250.0) / 0.04 = 12.5 m/s
- Baro noise σ = 0.8 m → difference noise √2 · 0.8 = 1.13 m
- 1.13 m / 0.04 s ≈ ±28 m/s noise on a 15 m/s signal
**Visual:** IMAGE ../assets/week01/fig_central_difference.png
**Notes:** Motivates smoothing; the same reasoning returns in Week 12 (trajectory velocity).

## SLIDE 19 | From notebook to class: FlightLog (preview)
**Content:**
- FlightLog.from_csv(path): @classmethod constructor
- log.sample_rate_hz: @property, computed on access
- __slots__, __repr__, type hints, docstrings
- Same answers as the notebook in five lines
- Week 2: Sensor hierarchy (IMU / LiDAR / GPS)
**Visual:** UML box "FlightLog" with attributes (source) and methods (time_gaps, load_factor, robust_load_factor, climb_rate, segment, plot_overview).
**Notes:** Code: src/aerospace_ai/utils.py. Instructor: read it before Week 2; it previews several OOP features.

## SLIDE 20 | This week's work
**Content:**
- Lab: notebooks/week01/week01_python_env.ipynb (7 TODO exercises)
- In class: 4 warm-ups (ddof, bank angle, gaps, NumPy shapes)
- Assignment 0: flight-log quality check, 3–4 h, due Week 2
- Push your branch; Restart & Run All before submitting
**Visual:** Three icons: notebook, pencil, Git branch.
**Notes:** Assignment 0 has hidden tests; the public tests show the expected format.

## SLIDE 21 | What We Now Know How To Do
**Content:**
- Build and activate a pinned environment; version-control a notebook
- Load a sensor CSV and read describe() like an engineer
- Find the true sample rate, gaps and missing values
- Separate sensor spikes from real manoeuvres
- Estimate noise and derivatives from noisy data, and plot the answers
**Visual:** The four questions Q1–Q4 from slide 1, each with a green check mark and its answer.
**Notes:** Close by returning to the opener: every question now has a number.
