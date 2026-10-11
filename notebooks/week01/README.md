# Week 1 package: Python and environment review

**Problem:** a 5-minute UAV flight log arrives; answer four engineering questions (sample rate
and data loss, peak load factor vs the 2.0 g limit, gyro noise at rest, peak climb rate).
**Tools:** conda/venv, Jupyter, Git, NumPy, Pandas, Matplotlib.

| # | Deliverable | Files |
|---|-------------|-------|
| 1 | Slides (outline + PPTX) | [`slides/en/week01_python_environment_EN.md`](../../slides/en/week01_python_environment_EN.md) / [`.pptx`](../../slides/en/week01_python_environment_EN.pptx), [`slides/fa/week01_python_environment_FA.md`](../../slides/fa/week01_python_environment_FA.md) / [`.pptx`](../../slides/fa/week01_python_environment_FA.pptx) |
| 2 | Lecture notes (PDF + Markdown source) | [`notes/en/week01_notes_EN.pdf`](../../notes/en/week01_notes_EN.pdf), [`notes/fa/week01_notes_FA.pdf`](../../notes/fa/week01_notes_FA.pdf) |
| 3 | Lab notebook (TODO + solved) | [`week01_python_env.ipynb`](week01_python_env.ipynb), [`week01_python_env_solved.ipynb`](week01_python_env_solved.ipynb) |
| 4 | Assignment 0 + rubric | [`assignments/assignment00/`](../../assignments/assignment00/) |
| 5 | In-class problems | [`in_class_problems/week01_problems.ipynb`](../../in_class_problems/week01_problems.ipynb) (solutions: instructor copy, not in the public repo) |
| + | Reusable code and tests | [`src/aerospace_ai/utils.py`](../../src/aerospace_ai/utils.py), [`tests/test_utils.py`](../../tests/test_utils.py) |
| + | Data | [`data/sample/`](../../data/sample/) (synthetic, see its README) |

## Worked numerical examples (hand first, then code)

1. Population vs sample standard deviation of five gyro samples: 0.352 vs 0.394 deg/s (`ddof`).
2. Load factor of f = (1.2, −0.4, 19.6) m/s²: n = 2.003.
3. Sample rate and gap from timestamps: 50 Hz, 60 missing samples; mean vs median of a spiky window: 1.82 vs 1.00 g.
4. Central-difference climb rate: 12.5 m/s, with ±28 m/s noise if not smoothed.

## Run and rebuild

```bash
pytest                                              # unit tests for src/aerospace_ai
jupyter lab notebooks/week01/                       # the lab
python tools/make_sample_data.py                    # regenerate the CSV logs
python tools/make_week01_figures.py                 # regenerate slide/notes figures
python tools/build_slides.py slides/en/week01_python_environment_EN.md
python tools/build_slides.py slides/fa/week01_python_environment_FA.md
python tools/build_notes.py notes/en/week01_notes_EN.md   # needs pandoc + Chrome/Chromium
python tools/build_notes.py notes/fa/week01_notes_FA.md
```

All notebooks were executed top to bottom with Python 3.11, numpy 1.26, pandas 2.1 and
matplotlib 3.8 (the course pins). The environment files (`environment.yml`,
`requirements.txt`) are delivery item 3 and are not part of this package yet.

## Sources

New content. Python tooling rewritten from W3Schools (NumPy intro, Pandas Read CSV, Pandas
Analyzing DataFrames, Matplotlib Pyplot, Git Introduction) and GeeksforGeeks (Python Virtual
Environment, Anaconda virtual environments, Getting started with Jupyter, DataFrame
describe()); turning-flight relations adapted from Anderson, *Introduction to Flight*, Ch. 6.
The instructor's "environment setup + ML intro" PDF was not available; reconcile when provided.
