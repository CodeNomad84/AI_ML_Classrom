# Bonus week package: Vibe Coding in Engineering

**Problem:** one prompt to an AI assistant returns four altimeter functions for the project UAV.
All four run, have docstrings and return a number; at 50 kPa they give 5574 m, 172 m, −159 889
and 5574 m. Which one would you fly, and how do you prove it? **Tools:** reference values with a
source, property checks, differential testing against a trusted implementation, the observed
order of accuracy (code verification), mutation testing of the tests themselves, static checks
(missing deliverables, imports that do not resolve), pinned environments, and a critical reading
of the productivity and security studies on AI coding assistants.

The bonus week sits outside the 16-week numbering and the grading plan; it can be taught any
time after Week 4.

| # | Deliverable | Files |
|---|-------------|-------|
| 1 | Slides (outline + PPTX) | [`slides/en/week_extra_vibe_coding_EN.md`](../../slides/en/week_extra_vibe_coding_EN.md) / [`.pptx`](../../slides/en/week_extra_vibe_coding_EN.pptx), [`slides/fa/week_extra_vibe_coding_FA.md`](../../slides/fa/week_extra_vibe_coding_FA.md) / [`.pptx`](../../slides/fa/week_extra_vibe_coding_FA.pptx) |
| 2 | Lecture notes (PDF + Markdown source) | [`notes/en/week_extra_vibe_coding_notes_EN.pdf`](../../notes/en/week_extra_vibe_coding_notes_EN.pdf), [`notes/fa/week_extra_vibe_coding_notes_FA.pdf`](../../notes/fa/week_extra_vibe_coding_notes_FA.pdf) |
| 3 | Lab notebook (TODO + solved) | [`week_extra_vibe_coding.ipynb`](week_extra_vibe_coding.ipynb), [`week_extra_vibe_coding_solved.ipynb`](week_extra_vibe_coding_solved.ipynb) |
| 4 | **Practice Assignment VC** (formative) + rubric | [`assignments/practice_vibe_coding/`](../../assignments/practice_vibe_coding/) |
| 5 | In-class problems | [`in_class_problems/week_extra_vibe_coding_problems.ipynb`](../../in_class_problems/week_extra_vibe_coding_problems.ipynb) (solutions: instructor copy, not in the public repo) |
| + | Reusable code and tests | [`src/aerospace_ai/verification.py`](../../src/aerospace_ai/verification.py) (`ReferenceCheck`, `PropertyCheck`, `DifferentialCheck`, `ConvergenceCheck`, `Verifier`, `MutationRunner`, `audit_deliverables`, `unresolvable_imports`); [`tests/test_verification.py`](../../tests/test_verification.py) |
| + | Data | [`data/sample/practice_vc_static_fire.csv`](../../data/sample/practice_vc_static_fire.csv) (synthetic, seed 2026); see the data README |

## Worked numerical examples (hand first, then code)

1. ISA pressure altitude at 50 kPa: exponent 0.19026, ratio^exponent 0.87425, T0/L = 44 331 m,
   h ≈ 5575 m (5574.4 m). Candidate B with R = 8.314: exponent 0.005511 (34.5 times too small),
   h = 172.2 m.
2. Check matrix (lab Section 5): A passes all five checks (max deviation 0.002 m); B and C fail
   table, round trip, below sea level and differential; D passes the table and fails the rest.
   C with a Pa → hPa and ft → m adapter agrees with the library to 3.88 m.
3. One RK4 step of dx/dt = −x, h = 0.1: 0.9048375 (error 8.2 × 10⁻⁸). Pitch oscillator
   (ωn = 3 rad/s, ζ = 0.5, T = 2 s): observed orders RK4 3.95/3.99/4.00, RK4 with a k3 typo
   2.69/2.55/2.38, Euler 1.38/1.18/1.09; at h = 0.05 both RK4 versions pass a 10⁻⁵ test.
4. Mutation scores on six ISA mutants: weak suite 1/6, typical 4/6, strong 6/6, strong minus the
   assertions D fails 4/6.
5. Static-fire log: I = 122.30 N s, peak 120.51 N, burn 1.541 s, average 79.37 N, Isp 191.86 s
   (m = 65 g); even-spacing bug 97.85 N s; unclipped 122.11 N s.
6. In class: B's exponent and C's unit contract (5572.1 m), pass@k with n = 10, c = 3 (0.3,
   0.7083, 0.9167), observed order 2.547 from two errors, and a test-weakening diff (a 20.0 %
   error passes rel_tol = 0.25).

## Run and rebuild

```bash
pytest                                                  # library tests (tests/)
python -m pytest assignments/practice_vibe_coding -q    # Practice VC student tests
jupyter lab notebooks/week_extra_vibe_coding/           # the lab
python tools/make_week_extra_vibe_coding_data.py        # regenerate the static-fire log
python tools/make_week_extra_vibe_coding_figures.py     # regenerate slide figures
python tools/build_slides.py slides/en/week_extra_vibe_coding_EN.md
python tools/build_slides.py slides/fa/week_extra_vibe_coding_FA.md
python tools/build_notes.py notes/en/week_extra_vibe_coding_notes_EN.md   # pandoc + Chromium
python tools/build_notes.py notes/fa/week_extra_vibe_coding_notes_FA.md
```

All notebooks were executed top to bottom with Python 3.11, numpy 1.26, pandas 2.1,
matplotlib 3.8, scikit-learn 1.4 and pytest 8.1 (the course pins).

## Sources

New content. The uploaded books were used as follows: G. Kim and S. Yegge, *Vibe Coding* (IT
Revolution, 2025), Chs. 1, 4 and 7 (definitions, head-chef idea, cautionary tales, delegation
rules, "count your babies"), paraphrased; N. AlNasser, *The Vibe Coder's Handbook* (2026),
Chs. 14 and 17, cross-checked for the safety angle; its statistics have no sources and are not
used (lab Section 9 lists them). Karpathy's coinage and the review rule: S. Willison,
simonwillison.net, 19 Mar 2025. Code verification: Veluri, Roy and Luke, *Computers & Fluids*
(2012). Mutation testing: DeMillo, Lipton and Sayward, *IEEE Computer* 11(4) (1978). pass@k:
Chen et al., arXiv:2107.03374. Studies: Peng et al. (arXiv:2302.06590), METR (arXiv:2507.09089;
blog, 24 Feb 2026), DORA 2024 (Google Cloud blog), Pearce et al. (arXiv:2108.09293), Perry et al.
(ACM CCS 2023), Spracklen et al. (USENIX Security 2025), Veracode press release (30 Jul 2025).
Mars Climate Orbiter: NASA MIB Phase I report (10 Nov 1999). Holzmann, "The Power of 10"
(NASA/JPL). ISA: ISO 2533:1975. Each number was read in the source or computed in the course
environment on 2026-10-16.
