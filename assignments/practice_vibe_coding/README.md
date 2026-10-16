# Practice Assignment VC (bonus week): Vibe-code it, then prove it

> Provenance: new content. Code verification with manufactured solutions and the observed
> order of accuracy after S. P. Veluri, C. J. Roy and E. A. Luke, *Computers & Fluids* (2012);
> mutation testing after R. A. DeMillo, R. J. Lipton and F. G. Sayward, *IEEE Computer* 11(4),
> 34–41 (1978); "count what you asked for" and the deleted-tests story paraphrased from G. Kim
> and S. Yegge, *Vibe Coding* (IT Revolution, 2025), Chs. 4 and 7; package hallucination after
> J. Spracklen et al., USENIX Security 2025 (arXiv:2406.10279). The static-fire log is synthetic.

| | |
|---|---|
| Released | Bonus week "Vibe Coding in Engineering" (any time after Week 4) |
| Due | One week after release |
| Effort | About 4 hours |
| Weight | Formative: feedback only (the bonus week is outside the 16-week grading plan; the syllabus milestones A1 to A6 are unchanged) |
| Notebook | [`practice_vibe_coding.ipynb`](practice_vibe_coding.ipynb) (Persian problem statement, checks, expected output) |
| Specification for the AI | [`SPEC.md`](SPEC.md) and the skeleton [`thrust_analysis.py`](thrust_analysis.py) |
| Evidence files | [`prompt_log.md`](prompt_log.md), [`ai_raw_output.py`](ai_raw_output.py), [`REVIEW.md`](REVIEW.md) |
| Your tests | [`test_thrust_analysis.py`](test_thrust_analysis.py) (two examples given) |
| Data | [`data/sample/practice_vc_static_fire.csv`](../../data/sample/practice_vc_static_fire.csv) (synthetic, 498 rows) |

## The task in one paragraph

A load cell logged a small solid motor on a test stand. You need its total impulse, peak
thrust, burn time, average thrust and specific impulse. This time an AI assistant writes the
code from a specification you give it; your job is to decide, with evidence, whether that code
can be trusted. You record the prompts and the unedited answer, run static checks, verify the
answer against hand-computed values, a property and the convergence order on an exact
(manufactured) pulse, fix what is wrong, and write a pytest suite that you prove effective by
running it against deliberately broken copies (mutants) of your own code.

```bash
python -m pytest assignments/practice_vibe_coding -q     # your tests on your final file
jupyter lab assignments/practice_vibe_coding/             # the notebook
```

No AI tool? Ask the instructor for a supplied answer to review; everything else is the same.
Use any assistant you have legitimate access to, and never paste data or code that your lab or
sponsor has not cleared for sharing with an external service.

## Time budget

| Part | Suggested time |
|------|----------------|
| Read `SPEC.md`, prompt the assistant, record the answer | 30 min |
| Static checks and the hand calculations (Exercise 3.1) | 30 min |
| Verification, reproducing and fixing defects | 90 min |
| Your test suite and your mutants | 60 min |
| `REVIEW.md` and reflection | 30 min |

## Rubric (100 points, for feedback)

| Task | Points | Full marks when | Typical deductions |
|------|-------:|-----------------|--------------------|
| 1. Prompt log and raw answer | 10 | Every prompt recorded verbatim with tool, model and date; first answer kept unedited | Raw answer "tidied up" before review (−5) |
| 3. Final `thrust_analysis.py` | 35 | Hidden tests pass (validation, uneven spacing, clipping, `>=` threshold, units, sample-log values below); type hints, docstrings, no magic numbers | Works only on even spacing (−10); `np.trapezoid` (NumPy 2 only) in the pinned NumPy 1.26 env (−5) |
| 4. Your test suite | 25 | ≥ 8 tests, pass on the reference; fixture + `parametrize` + `raises`; one hand-computed value; kills most of the six hidden mutants; at least three own mutants, all killed | Tests that only check types or call the code without asserting (−10); expected value copied from the code's own output (−5) |
| 5. `REVIEW.md` | 20 | Each defect reproduced with an input, the check that found it, and the fix; static-check results; honest "none found" with evidence if the answer was correct | A defect claimed but not reproduced (−5 each) |
| Reflection | 10 | Three answers that use this assignment's evidence | Generic statements about AI |

Task 3 is checked by hidden instructor tests; Task 4 by running your suite against the
reference solution and six hidden mutants. A suite that fails on the reference earns no
mutation credit.

## Expected output

| Quantity | Expected value |
|----------|----------------|
| Exercise 3.1 a) / b) / c) | 10 / 12 / 10 N s |
| Samples in the log | 498 |
| Total impulse (clipped, trapezoid on actual times) | 122.296 N s (noise-free pulse: 2 F t_b / π = 122.231 N s) |
| Peak thrust / burn time (≥ 5 % of peak) / average thrust | 120.508 N / 1.5408 s / 79.37 N |
| Specific impulse, m_p = 0.065 kg | 191.86 s |
| Observed order on the half-sine (h = t_b/20, t_b/40, t_b/80) | 2.00 |
| What the classic defects give | even spacing assumed: 97.85 N s; negative noise integrated: 122.114 N s |

Note that a convergence check on an evenly spaced grid cannot see an "even spacing" defect: you
need a reference case on an uneven grid as well. That is the point of combining several checks.
