# Week 3 package: ABCs, dataclasses, typing and a composable `DataPipeline`

**Problem:** two analysts clean the same flight log with two reasonable scripts and report two
different peak load factors (1.87 g in level cruise vs. the real 1.80 g pull-up). Nothing in
either result records the order of the steps, their parameters or whether the two recipes are
the same. **Tools:** abstract base classes (abstract methods and properties), dataclasses
(`field`, `frozen`, `__post_init__`, learned state), hand-written `__repr__`/`__eq__`/`__hash__`,
type hints with `TypeVar`/`Generic`, composition, a Hampel filter, fit/transform without
leakage.

| # | Deliverable | Files |
|---|-------------|-------|
| 1 | Slides (outline + PPTX) | [`slides/en/week03_data_pipeline_EN.md`](../../slides/en/week03_data_pipeline_EN.md) / [`.pptx`](../../slides/en/week03_data_pipeline_EN.pptx), [`slides/fa/week03_data_pipeline_FA.md`](../../slides/fa/week03_data_pipeline_FA.md) / [`.pptx`](../../slides/fa/week03_data_pipeline_FA.pptx) |
| 2 | Lecture notes (PDF + Markdown source) | [`notes/en/week03_notes_EN.pdf`](../../notes/en/week03_notes_EN.pdf), [`notes/fa/week03_notes_FA.pdf`](../../notes/fa/week03_notes_FA.pdf) |
| 3 | Lab notebook (TODO + solved) | [`week03_data_pipeline.ipynb`](week03_data_pipeline.ipynb), [`week03_data_pipeline_solved.ipynb`](week03_data_pipeline_solved.ipynb) |
| 4 | Practice Assignment 3 + rubric | [`assignments/practice03/`](../../assignments/practice03/) |
| 5 | In-class problems | [`in_class_problems/week03_problems.ipynb`](../../in_class_problems/week03_problems.ipynb) (solutions: instructor copy, not in the public repo) |
| + | Reusable code and tests | [`src/aerospace_ai/pipeline.py`](../../src/aerospace_ai/pipeline.py), [`tests/test_pipeline.py`](../../tests/test_pipeline.py) |

**Assignment numbering.** The syllabus milestones keep their numbers (A1 = Week 2,
A2 = Week 5, ...). Week 3 has no milestone, so its weekly assignment is *Practice Assignment 3*
(named after the week), which never collides with Assignment 3 in Week 7.

## Worked numerical examples (hand first, then code)

1. One +40 m/s² spike through a 5-sample mean: (4 × 10.35 + 50.33) / 5 = 18.35 m/s², n = 1.87 g.
2. Hampel filter by hand: window 5 at k = 3, median 100.3, MAD 0.2, threshold 0.89 m, 106.0 → 100.3.
3. Order matters: despike-then-smooth peaks at 1, smooth-then-despike at 4.
4. Standardisation without leakage: z = 2.449 (fit on train) vs 1.342 (fit on train + test).
5. Flight log: pipeline A 1.87 g at 66.7 s, pipeline B 1.80 g at 192.0 s; row-window climb rate
   20.99 m/s at the logger gap vs 19.91 m/s on a 10 Hz grid.
6. In class: dataclass output prediction, Hampel with a two-sample spike (window ≥ 2L + 1),
   order of stages, build-time vs run-time errors.

## The pipeline

`PipelineStage(ABC, Generic[InT, OutT])` declares `INPUT_TYPE`/`OUTPUT_TYPE`, an abstract
`transform`, an optional `fit`, and a registry filled by `__init_subclass__`. Concrete stages are
dataclasses (`SensorToFrame`, `SelectColumns`, `DropNonFinite`, `FillGaps`, `HampelFilter`,
`RollingMean`, `TimeDerivative`, `ToArray`; `Standardize` learns). `a >> b` builds an immutable
`DataPipeline` that type-checks the chain when it is built, reports what each stage did, and
round-trips through JSON (`to_config`/`from_config`).

The pipeline is written against `BaseSensor`, so it does **not** depend on the LiDAR and GPS
sensors of Assignment 1 (held back from `src/` until that assignment is graded); students'
own sensors plug into `SensorToFrame` unchanged.

## Run and rebuild

```bash
pytest                                              # library tests (tests/), 77 tests
python -m pytest assignments/practice03 -q          # Practice Assignment 3 student tests
jupyter lab notebooks/week03/                       # the lab
python tools/make_week03_figures.py                 # regenerate slide/notes figures
python tools/build_slides.py slides/en/week03_data_pipeline_EN.md
python tools/build_slides.py slides/fa/week03_data_pipeline_FA.md
python tools/build_notes.py notes/en/week03_notes_EN.md   # needs pandoc + Chrome/Chromium
python tools/build_notes.py notes/fa/week03_notes_FA.md
```

All notebooks were executed top to bottom with Python 3.11, numpy 1.26, pandas 2.1,
matplotlib 3.8 and pytest 8.1 (the course pins).

## Sources

New content. Python material rewritten from W3Schools ([abc Module](https://www.w3schools.com/python/ref_module_abc.asp),
[dataclasses Module](https://www.w3schools.com/python/ref_module_dataclasses.asp),
[typing Module](https://www.w3schools.com/python/ref_module_typing.asp),
[`__repr__()`](https://www.w3schools.com/python/python_magic_repr.asp),
[`__eq__()`](https://www.w3schools.com/python/python_magic_eq.asp),
[Iterators](https://www.w3schools.com/python/python_iterators.asp)) and GeeksforGeeks
([Abstract Classes in Python](https://www.geeksforgeeks.org/python/abstract-classes-in-python/),
[Understanding Python Dataclasses](https://www.geeksforgeeks.org/python/understanding-python-dataclasses/),
[Python Generics](https://www.geeksforgeeks.org/python/python-generics/),
[str() vs repr()](https://www.geeksforgeeks.org/str-vs-repr-in-python/)). Hampel filter adapted
from Pearson, *IEEE TCST* 10(1), 2002; MAD constant from Rousseeuw & Croux, *JASA* 88(424), 1993;
min-max scaling (Practice Assignment 3) from Géron, *Hands-On Machine Learning*, 3rd ed., Ch. 2.
The instructor's PDFs do not cover Python OOP.
