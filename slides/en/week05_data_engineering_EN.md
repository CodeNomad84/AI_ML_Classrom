---
title: "Week 5: Data Engineering for Noisy Sensor Data"
subtitle: "Missing data, outliers, normalisation and splits without leakage"
course: "AI for Aerospace & Mechanical Engineering (PhD)"
footer: "Week 05 | Data engineering"
language: en
provenance: "Partly expanded from instructor notes: slides 2 and 3 expand the instructor's deck 'Machine Learning & Decision Trees' (slides 2 to 8: AI, ML and DL; feature extraction; classical models need preprocessing; classification, clustering, regression). The rest is new content: missing-data mechanisms from Little and Rubin, Statistical Analysis with Missing Data (2019), Ch. 1; cleaning, imputation, outliers, scaling and leakage rewritten from W3Schools (Pandas - Cleaning Data, ML Scaling, ML Train/Test) and GeeksforGeeks (Handling Missing Values in Machine Learning, Detect and Remove the Outliers using Python, Data Leakage); LiDAR error types from Thrun, Burgard and Fox, Probabilistic Robotics (2005), Sec. 6.3; z-score bound from Samuelson, JASA 63 (1968); Hampel filter from Pearson (2002); blocked validation from Roberts et al., Ecography 40 (2017)."
---

<!-- Build: python tools/build_slides.py slides/en/week05_data_engineering_EN.md -->

## SLIDE 1 | A braking controller that trusts a raw log
**Content:**
- Task: forecast the distance to the wall 1 s ahead (project module M3)
- 16 test drives, 10 Hz: LiDAR range, wheel speed, acceleration, yaw rate
- Naive pipeline: read CSV, shift, drop NaN, random split, fit
- Linear model: 2.73 m error on a 1 m stopping distance
- Q: what must happen between the CSV and the model?
**Visual:** IMAGE ../assets/week05/fig_raw_vs_truth.png
**Notes:** Run Section 0 of the lab live. Point at run 3: zeros where the wall is beyond 12 m, a dust echo, accelerometer spikes, a frozen gyroscope. Write the closing question on the board and keep it there until the last slide.

## SLIDE 2 | Where data engineering sits in machine learning
**Content:**
- AI ⊃ ML ⊃ DL (instructor's deck, slide 2)
- Classical ML (trees, SVM, naive Bayes, logistic regression) needs features, not raw files
- Deep learning learns features, but still needs clean, honest data
- This week: regression (continuous target), supervised (labels from the log)
- Garbage in, garbage out applies to both
**Visual:** Nested circles AI ⊃ ML ⊃ DL on the left; on the right the chain raw log → cleaning → features → model → decision, with "this week" bracketing the first two arrows.
**Notes:** Expanded from the instructor's deck (slides 2 to 8, adapted by the instructor from the Faradars blog): its slide 3 says classical models "cannot be applied directly to raw data such as CSV files" and need a preprocessing step. Decision trees (deck slides 9 to 11) come back in Week 7; note for later that trees do not need feature scaling.

## SLIDE 3 | Example 1: what the naive pipeline got wrong
**Content:**
- Random split: mean 3.89 m, linear 2.73 m, k-NN 2.29 m
- 533 LiDAR zeros read as "wall touching the sensor"
- Targets of each run's last 10 rows come from the next run
- 25 duplicated rows, 11 spikes up to ±25 m/s², a gyro frozen for 3 s
- After this lecture: linear 0.121 m on unseen drives
**Visual:** Two-column table "naive" vs "clean, split by run": mean 3.891 / 3.833, linear 2.734 / 0.121, k-NN 2.285 / 0.641 (m), with the 0.121 cell highlighted.
**Notes:** Ask the room which number they would have put in a paper. Then reveal that two of the naive numbers are not even honest: the random split leaks (slide 18). Keep this table; we return to it on slide 21.

## SLIDE 4 | Look before you clean: the audit
**Content:**
- audit_frame counts NaN, ±inf and sentinel cells per column
- LiDAR: 21 % unusable, 66 zeros in a row at a run start
- Wheel speed: 19 NaN in bursts of up to 4 samples
- Longest repeat: 32 identical gyro values (noise 0.3 °/s!)
- Repeats are hints, not verdicts: 51 zeros = rover at rest
**Visual:** IMAGE ../assets/week05/fig_audit.png
**Notes:** W3Schools "Pandas - Cleaning Data" lists four kinds of bad data: empty cells, wrong format, wrong data, duplicates. Sensor logs add two: sentinels (a number that means "no number") and frozen values. Exercise 1.1 counts the 25 duplicated rows with count_duplicate_keys.

## SLIDE 5 | A taxonomy of sensor defects
**Content:**
- Sentinel: LiDAR 0.0 = "no echo" (not "0 m")
- Gap: NaN bursts from lost CAN packets
- Outlier: dust echo, connector glitch, a 2.5 g spike
- Frozen value: the sensor repeats its last reading
- Duplicate: the logger re-sends a packet
**Visual:** Five small time-series sketches in a row, one per defect, each with the tool that fixes it underneath: MarkInvalid, FillGaps, HampelFilter, FlagStuck (Assignment 2), SortAndDeduplicate.
**Notes:** The LiDAR error classes follow the beam model of Thrun, Burgard and Fox, Probabilistic Robotics, Sec. 6.3: max-range readings, random measurements and short readings. The synthetic log injects each defect with a known count, so every cleaning step can be checked exactly.

## SLIDE 6 | Example 2: mean imputation under MNAR, by hand
**Content:**
- True ranges: 2, 4, 6, 8, 10, 13, 15, 17 m; LiDAR limit 12 m
- True mean 75/8 = 9.375 m; observed mean 30/5 = 6.0 m
- Fill the three gaps with 6.0: errors −7, −9, −11 m
- Imputed mean stays 6.0: biased by −3.4 m
- Rover log: true mean 6.86 m, observed about 5.1 m
**Visual:** IMAGE ../assets/week05/fig_mnar.png
**Notes:** The imputer invents a wall 6 m away where there is none. The real information is "farther than 12 m": keep it as an indicator column or restrict the model to the sensor's range. In-class Problem 2 repeats the idea with a saturating temperature sensor.

## SLIDE 7 | Why is it missing? MCAR, MAR, MNAR
**Content:**
- MCAR: unrelated to anything (lost CAN packet)
- MAR: depends on other observed columns (GPS lost, "in_hangar" logged)
- MNAR: depends on the missing value itself (LiDAR beyond range)
- The mechanism, not the missing fraction, decides the remedy
- MNAR has no neutral fix: flag it, model it, or restrict scope
**Visual:** Three panels with the same scatter of true values; dots removed at random (MCAR), removed where a second variable is high (MAR), removed where the value itself is high (MNAR).
**Notes:** Definitions from Little and Rubin, Ch. 1, as summarised by GeeksforGeeks "Handling Missing Values in Machine Learning". In the rover log, leading LiDAR zeros are MNAR and isolated zeros are MCAR; Assignment 2 Task 1 counts them separately (482 and 50).

## SLIDE 8 | Remedies, and when each is safe
**Content:**
- Drop rows: MCAR, and when few rows are lost
- Interpolate in time: short interior gaps (≤ 5 samples here)
- Impute (median) + indicator: weak features, MAR or MNAR
- Restrict the scope: "model valid while the wall is within 12 m"
- Never: interpolate across long gaps or across runs
**Visual:** Decision tree: "Why missing?" → MCAR (short gap? interpolate : drop) / MAR (impute from the related column) / MNAR (indicator or restrict scope), with the rover example on each leaf.
**Notes:** GeeksforGeeks lists deletion, mean/median/mode imputation, forward/backward fill and interpolation. Imputers learn a number, so they are fitted on the training split (slide 16). MarkInvalid turns the sentinel into NaN first, so every remedy sees "unknown" rather than "0 m".

## SLIDE 9 | Example 3: interpolating a gap by hand
**Content:**
- r(0) = 8.0 m, r(0.3) = 7.4 m; samples at 0.1 and 0.2 s missing
- r(t) = 8.0 + (7.4 − 8.0)·t/0.3
- r(0.1) = 7.8 m, r(0.2) = 7.6 m (implied speed 2 m/s)
- FillGaps(max_gap=5) reproduces both values
- Longer gaps stay NaN: interpolation would invent data
**Visual:** Four points on a time axis, two hollow (missing) on the straight line joining 8.0 and 7.4, labelled 7.8 and 7.6.
**Notes:** At 10 Hz and below 2.2 m/s, a 0.5 s gap stays within a few centimetres of the truth unless the rover starts braking inside the gap. FillGaps uses the time column as abscissa, so it also handles irregular sampling. Lab Worked Example 2.

## SLIDE 10 | Outliers: the z-score cannot see one outlier in a small sample
**Content:**
- Seven samples at rest and one glitch of 25 m/s²
- Mean 3.16, σ 8.26: z = 2.65 < 3, not flagged
- Samuelson: |z| ≤ √(n − 1): 2.65 for n = 8, 3 only from n = 10
- The outlier inflates the σ that should expose it (masking)
- Median 0.05, MAD 0.15: robust score 112, flagged
**Visual:** IMAGE ../assets/week05/fig_masking.png
**Notes:** Samuelson, JASA 63 (1968), for the population standard deviation; with the sample standard deviation the bound is (n − 1)/√n (Shiffler, 1988). GeeksforGeeks "Detect and Remove the Outliers using Python" teaches the z-score and IQR rules; this slide shows when the first one cannot work. Lab Exercise 3.1.

## SLIDE 11 | Spike or manoeuvre? Global vs local rules
**Content:**
- Tukey fences [Q1 − 1.5 IQR, Q3 + 1.5 IQR]: robust but global
- On the rover log they flag 239 plausible samples (braking!)
- Hampel: median and MAD in a moving window of 7 samples
- Compares a sample with its neighbours, not with the whole log
- Use Tukey (k = 3) only as a final safety net, fitted on train
**Visual:** IMAGE ../assets/week05/fig_outlier_rules.png
**Notes:** A dust echo of 1.2 m while the wall is 8 m away is a valid range in absolute terms; only its neighbours show it is wrong. The Hampel step also nudges noisy samples by about 4 cm (range) and 0.3 m/s² (acceleration): small next to what it removes, but report it.

## SLIDE 12 | Example 4: a dust echo in seven LiDAR readings
**Content:**
- 8.02, 7.98, 8.01, 1.20, 7.97, 8.03, 7.99 m
- Mean 7.03, σ 2.38: z(1.20) = −2.449 = −√6, the largest possible
- Median 7.99, MAD 0.02: robust score 229
- Tukey fences [7.915, 8.075]: flagged, but only valid for this window
- Hampel, window 7: flags exactly one sample
**Visual:** Seven dots on a number line from 0 to 9 m with the mean, the median and the band median ± 3·1.4826·MAD (±0.089 m) drawn; the echo far to the left.
**Notes:** In-class Problem 1. On the vehicle a centred window costs 0.3 s of delay; a trailing window has none but reacts later. Ask: which would the braking controller prefer?

## SLIDE 13 | Respect the boundaries: per-run processing
**Content:**
- The log is 16 separate drives glued together
- Interpolation, Hampel windows, rolling means, shifts must stop at run ends
- Plain shift(−10): the last 10 targets of a run come from the next run
- PerGroup(stage, "run_id") applies any stateless stage per run
- A composite: it is a stage and contains a stage
**Visual:** IMAGE ../assets/week05/fig_per_group.png
**Notes:** PerGroup refuses learned stages (a scaler per run would be a different model) and round-trips through the Week 3 JSON config. It is the Week 4 idea again: extend behaviour by wrapping, not by editing HampelFilter or FillGaps.

## SLIDE 14 | The lecture recipe: stateless cleaning
**Content:**
- SortAndDeduplicate: 2533 → 2508 rows
- MarkInvalid: 530 LiDAR cells become NaN (0.0, outside 0.3 to 12 m)
- PerGroup(Hampel), PerGroup(FillGaps), PerGroup(ForecastTarget)
- DropNonFinite: 1859 rows with all features and the target
- Nothing learned: safe to run before the split
**Visual:** The run report table: stage, rows_in, rows_out, nan_out, cells_changed (2533/2508/19; 2508/2508/549/530; 167; 58; 0; 1859).
**Notes:** aerospace_ai.datasets.rover_cleaning_pipeline; the rover_range_forecast factory loader (Week 4 Factory) runs it and returns a Dataset whose groups are the run ids. The 649 dropped rows are mostly the MNAR start of each run: the model's scope is "wall within range".

## SLIDE 15 | Example 5: which neighbour is nearest?
**Content:**
- q = (5.0 m, 0.0 m/s²) cruising; A = (5.1, −1.0) braking; B = (6.2, 0.0)
- Raw: d_A = 1.005, d_B = 1.200, so A, the braking sample
- Scaled by σ = 3.0 m and 0.4 m/s²: d_A = 2.500, d_B = 0.400
- Now B, the cruising sample, is nearest
- Units must not decide who is similar
**Visual:** IMAGE ../assets/week05/fig_scaling.png
**Notes:** Week 4 showed unscaled k-NN losing to the mean baseline on fuel flow. Distance-based models (k-NN, SVM with RBF kernel, K-Means in Week 8) and gradient descent need scaling; trees do not (Week 7).

## SLIDE 16 | Three scalers and one rule
**Content:**
- z-score (x − μ)/σ; min-max (x − min)/(max − min); robust (x − median)/IQR
- Every scaler, imputer and clipper learns numbers from data
- Learn them from training rows only; apply the same numbers to test
- ColumnStatStage: template method fit → learn(), transform → apply()
- Test values outside [0, 1] after min-max are a finding, not a bug
**Visual:** UML: ColumnStatStage (abstract learn, apply; fit, transform) with subclasses ImputeMedian, MinMaxScale, RobustScale and a dashed IQRClipper (Assignment 2).
**Notes:** W3Schools "ML Scaling" introduces the z-score; GeeksforGeeks "Data Leakage" states the rule. Lab Exercise 5.1: training median 5.076 m vs 5.188 m from all rows. In-class Problem 4 shows the mildest form of leakage: a test value of 12 m looks like z = 1.61 instead of 2.12.

## SLIDE 17 | Leakage: when the test set is not really unseen
**Content:**
- Preprocessing statistics computed before the split
- Features that will not exist at prediction time (future samples)
- Targets that cross into the test period or another run
- Random splits of time series: 10 Hz twins on both sides
- P(a test row has a training neighbour) = 1 − 0.25² = 0.94
**Visual:** Timeline of 20 samples at 10 Hz, test samples in red scattered among blue training samples, each red one bracketed by blue neighbours 0.1 s away.
**Notes:** Causes rewritten from GeeksforGeeks "Data Leakage" (splitting incorrectly, preprocessing before the split, features unavailable at prediction time, pipeline errors). In-class Problem 3 hunts five leaks in five lines of code.

## SLIDE 18 | Example 6: four splitters, one model
**Content:**
- Random rows: k-NN 0.319 m (LEAKY)
- Chronological: 0.634 m (LEAKY: one run cut in two)
- Chronological with a 10-row gap: 0.634 m (OK)
- By run (GroupHoldoutSplit): 0.641 m (OK): the real question
- Linear model: 0.098 to 0.121 m whatever the split
**Visual:** IMAGE ../assets/week05/fig_split_leakage.png
**Notes:** Had we tuned k on the random split, we would have reported 0.32 m and delivered 0.64 m. The linear model has four parameters and cannot memorise neighbours, so it barely notices. check_split reports shared rows, shared runs and the closest train/test distance against the horizon. Lab Exercise 6.1.

## SLIDE 19 | Non-causal features and time-series CV
**Content:**
- Week 3's RollingMean is centred: it uses future samples
- CausalRollingMean uses only the past: it can run on the rover
- ChronologicalSplit(gap = horizon) purges overlapping targets
- ExpandingWindowCV: train on blocks 1..k, test on block k + 1
- Week 7: pass such a splitter to GridSearchCV
**Visual:** Step input 0, 0, 0, 0, 9, 9: the centred mean (window 3) rises at index 3, before the step; the trailing mean rises at index 4. Below, four expanding-window folds as coloured bars.
**Notes:** Blocked and grouped validation for autocorrelated data: Roberts et al., Ecography 40 (2017). Purging rows whose targets overlap the test period: Lopez de Prado (2018), Ch. 7.

## SLIDE 20 | The leak-free workflow
**Content:**
- 1. Stateless cleaning on the whole log, one run at a time
- 2. Split by run (or by time, with a gap)
- 3. Fit learned stages (impute, clip, scale) on training rows
- 4. Transform training and test rows with the same fitted pipeline
- 5. Fit and score the model; report the split
**Visual:** IMAGE ../assets/week05/fig_workflow.png
**Notes:** The Week 3 DataPipeline already separates fit and transform; Week 5 adds the discipline of where each call happens. The same order holds for every model in Blocks 1 and 2.

## SLIDE 21 | Example 7: the same model, before and after
**Content:**
- Naive, random split: linear 2.734 m, k-NN 2.285 m
- Clean, split by run: linear 0.121 m, k-NN 0.641 m
- Mean baseline: 3.83 m, beaten by a factor of 30
- Weights: future range ≈ range − speed × 1 s, plus a little acceleration
- Every number on this slide can be reproduced from the repo
**Visual:** IMAGE ../assets/week05/fig_forecast.png
**Notes:** Return to the opener: nothing in the model changed; only the data engineering did. Lab Section 7. The yaw rate has a weight close to zero: a weak feature, which is why median imputation is acceptable for its frozen segments.

## SLIDE 22 | This week's work
**Content:**
- Lab: notebooks/week05/week05_data_engineering.ipynb (6 exercises)
- In class: 4 warm-ups (dust echo, MNAR, leakage hunt, scaling)
- Assignment 2 (graded, about 4 h): leak-free cleaning of a new rover log
- Due at the start of Week 6
**Visual:** Three icons: notebook, pencil, Git branch.
**Notes:** Assignment 2 adds a frozen-sensor detector, a learned Tukey clipper and a leak-free prepare() on a log with seed 52. Expected numbers are in its README: 1919 clean rows, runs 13 to 16 as test, linear RMSE 0.109 m.

## SLIDE 23 | What We Now Know How To Do
**Content:**
- Audit a raw sensor log before touching it
- Name the missing-data mechanism and pick a remedy that fits it
- Tell spikes from manoeuvres with local, robust outlier rules
- Keep neighbour-based steps inside one run; scale with training statistics
- Split time-ordered, grouped data without leakage, and prove it
**Visual:** The opener's naive pipeline crossed out, replaced by the five-step leak-free workflow with green check marks and the RMSE 2.73 m → 0.12 m.
**Notes:** Return to the opener question: between the CSV and the model sit an audit, stateless cleaning per run, a split by run, and learned stages fitted on training rows only.
