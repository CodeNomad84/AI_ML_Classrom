# Week 4 package: design patterns for ML (Strategy, Observer, Factory) and pytest

**Problem:** one `run_experiment(model_type, data_source, verbose, early_stop, ...)` function
predicts a UAV's fuel flow. A misspelled `"KNN"` silently returns the mean baseline, its k-NN
branch has an unscaled-distance bug nobody sees, and every new model, data source or training
feature edits the same function. **Tools:** the Strategy pattern (with an Adapter for
scikit-learn), the Observer pattern (a `Trainer` with callbacks), the Factory pattern (a
registry with a class decorator), `typing.Protocol`, and pytest contract tests, test doubles and
fixtures.

| # | Deliverable | Files |
|---|-------------|-------|
| 1 | Slides (outline + PPTX) | [`slides/en/week04_design_patterns_EN.md`](../../slides/en/week04_design_patterns_EN.md) / [`.pptx`](../../slides/en/week04_design_patterns_EN.pptx), [`slides/fa/week04_design_patterns_FA.md`](../../slides/fa/week04_design_patterns_FA.md) / [`.pptx`](../../slides/fa/week04_design_patterns_FA.pptx) |
| 2 | Lecture notes (PDF + Markdown source) | [`notes/en/week04_notes_EN.pdf`](../../notes/en/week04_notes_EN.pdf), [`notes/fa/week04_notes_FA.pdf`](../../notes/fa/week04_notes_FA.pdf) |
| 3 | Lab notebook (TODO + solved) | [`week04_design_patterns.ipynb`](week04_design_patterns.ipynb), [`week04_design_patterns_solved.ipynb`](week04_design_patterns_solved.ipynb) |
| 4 | Practice Assignment 4 + rubric | [`assignments/practice04/`](../../assignments/practice04/) |
| 5 | In-class problems | [`in_class_problems/week04_problems.ipynb`](../../in_class_problems/week04_problems.ipynb) (solutions: instructor copy, not in the public repo) |
| + | Reusable code and tests | [`src/aerospace_ai/models.py`](../../src/aerospace_ai/models.py), [`training.py`](../../src/aerospace_ai/training.py), [`datasets.py`](../../src/aerospace_ai/datasets.py); [`tests/test_models.py`](../../tests/test_models.py), [`test_training.py`](../../tests/test_training.py), [`test_datasets.py`](../../tests/test_datasets.py) |
| + | Data | [`data/sample/week04_fuel_flow.csv`](../../data/sample/week04_fuel_flow.csv) (400 synthetic UAV fuel-flow measurements, see its README) |

**Assignment numbering.** Week 4 has no syllabus milestone (A2 is Week 5), so, as in Week 3, its
weekly assignment is the formative *Practice Assignment 4*.

## Worked numerical examples (hand first, then code)

1. Fuel flow at 25 m/s, sea level, 30 kg, level flight: 430.66 + 149.94 W → 774.13 W shaft →
   270.95 g/h; with a 1 m/s climb 408.24 g/h.
2. Three strategies on four flights: mean RMSE 79.73, k-NN (k = 2) 32.60, least squares
   (slope 19.25, intercept −199.5) 14.18 g/h; at 16 m/s the line extrapolates to 108.5 vs ~162.
3. One gradient-descent step: L0 = 15, ∇L = (−7.333, −16.667), w = (0.7333, 1.6667),
   L1 = 0.2341; η = 0.5 diverges (L1 = 309; λmax = 11.09 so η < 0.18).
4. Early stopping (patience 2, min_delta 0.05): stops after epoch 5, restores epoch 3 (0.70);
   patience 3 reaches 0.50.
5. Full data, chronological 300/100 split: linear 51.5, scaled k-NN 51.6, random forest via the
   adapter 37.65, mean 225.4, **unscaled k-NN 235.3 g/h (worse than the mean)**.
6. In class: strategy choice for payload extrapolation (linear RMSE 7.07 vs 1-NN 41.7 vs mean
   92.1 g/h) and the observer-order puzzle (same training, different printout).

## The three patterns in the library

* **Strategy** (`models.py`): `Regressor(ABC)` with template-method `fit`/`predict`;
  `MeanBaseline`, `LinearLeastSquares`, `KNearestNeighbors`, `GradientDescentRegressor`;
  `SklearnRegressor` adapter (structural `SupportsFitPredict` protocol); context
  `RegressionExperiment`. `NotFittedError` is shared with the Week 3 pipeline.
* **Observer** (`training.py`): subject `Trainer`, observers `Callback` (`History`,
  `ProgressLogger`, `EarlyStopping`, `DivergenceGuard`, `StepDecay`), shared `TrainingState`,
  models typed by the `TrainableModel` protocol.
* **Factory** (`datasets.py`): `DatasetFactory` registry with `@DatasetFactory.register(name)`,
  products `DatasetLoader` → frozen `Dataset` with a chronological `split`; loaders `csv`
  (optionally cleaned by a Week 3 `DataPipeline`), `flight_log_csv`, `fuel_flow_synthetic`.

## Run and rebuild

```bash
pytest                                              # library tests (tests/)
python -m pytest assignments/practice04 -q          # Practice Assignment 4 student tests
jupyter lab notebooks/week04/                       # the lab
python tools/make_week04_data.py                    # regenerate the fuel-flow CSV
python tools/make_week04_figures.py                 # regenerate slide/notes figures
python tools/build_slides.py slides/en/week04_design_patterns_EN.md
python tools/build_slides.py slides/fa/week04_design_patterns_FA.md
python tools/build_notes.py notes/en/week04_notes_EN.md   # needs pandoc + Chrome/Chromium
python tools/build_notes.py notes/fa/week04_notes_FA.md
```

All notebooks were executed top to bottom with Python 3.11, numpy 1.26, pandas 2.1,
matplotlib 3.8, scikit-learn 1.4 and pytest 8.1 (the course pins).

## Sources

New content; the instructor's PDFs do not cover design patterns. Patterns rewritten from
GeeksforGeeks ("Strategy Design Pattern", "Observer Design Pattern", "Factory Method - Python
Design Patterns", "Getting Started with Pytest", "Pytest Fixtures") and Gamma, Helm, Johnson
and Vlissides, *Design Patterns* (1994); Python details from W3Schools (Classes, Inheritance,
Polymorphism, Dictionaries, Lambda). The KodAmoz Python bootcamp
(kodamoz.com/fa/bootcamps/python) was used for Persian terminology; it covers OOP but not
design patterns or testing. Least squares and k-NN adapted from Bishop, *PRML*, Ch. 3 and
Sec. 2.5.2; early stopping from Goodfellow et al., *Deep Learning*, Sec. 7.8; the fuel-flow model
from Anderson, *Aircraft Performance and Design*, Ch. 5, with ISA density from Anderson,
*Introduction to Flight*, Ch. 3.
