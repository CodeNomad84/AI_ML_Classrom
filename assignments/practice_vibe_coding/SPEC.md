# Specification: static-fire thrust analysis

> Provenance: new content (bonus week "Vibe Coding in Engineering"). This is the text you give
> your AI assistant, together with `thrust_analysis.py`. You may rephrase it, but keep every
> requirement, and record what you actually sent in `prompt_log.md`.

## Context

A load cell measured the thrust of a small solid motor on a test stand. The logger wrote
`data/sample/practice_vc_static_fire.csv` with two columns: `time_s` (s) and `thrust_n` (N).
The file is **synthetic** (a half-sine pulse with noise, see
`tools/make_week_extra_vibe_coding_data.py`), so its true answer is known.

Facts about the log that the code must respect:

1. Timestamps are strictly increasing but **unevenly spaced**: the nominal rate is 200 Hz,
   every interval jitters by up to ±20 %, and dropped packets leave gaps of about 40 ms.
2. Thrust is noisy: before ignition and after burn-out the load cell reads small **negative**
   values. Negative samples are noise and count as zero thrust.
3. The propellant mass of this motor is 0.065 kg.

## Requirements

Implement every `# TODO` in `thrust_analysis.py` without changing names, signatures or
constants. The docstrings in that file are part of this specification.

* Python 3.11, NumPy 1.26, pandas 2.1 (the course's pinned environment). No other packages.
* Type hints on all signatures, Google-style docstrings, English comments, no magic numbers.
* `total_impulse` uses the trapezoidal rule on the **actual** timestamps of the clipped thrust.
* `burn_time` follows the course convention in its docstring (≥ 5 % of peak, first to last
  sample). This is a course convention, not an industry standard.
* `specific_impulse` returns seconds: $I_{sp} = I / (m_p\,g_0)$, $g_0 = 9.80665$ m/s².
* Invalid input raises `ValueError` with a clear message.

## Prompting advice (not part of the specification)

* Ask for a plan first ("list the steps and the edge cases before writing code"), then for the
  code, then for tests: three small requests are easier to check than one big one.
* Ask the assistant to state its assumptions (units, spacing, sign conventions). Compare them
  with the facts above.
* Never paste data, code or documents that your lab or sponsor has not cleared for sharing
  with an external service.
* Do not install any package an answer imports without checking that it exists, who publishes
  it, and that you need it. This assignment needs none.
