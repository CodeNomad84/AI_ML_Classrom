---
title: "Bonus Week: Vibe Coding in Engineering"
subtitle: "AI writes the code, the engineer verifies it: checks, tests, review, limits"
course: "AI for Aerospace & Mechanical Engineering (PhD)"
footer: "Bonus week | Vibe coding in engineering"
language: en
provenance: "New content (bonus week). Definitions and cautionary tales paraphrased from G. Kim and S. Yegge, Vibe Coding (IT Revolution, 2025), Chs. 1, 4 and 7; Karpathy's coinage and the review rule from S. Willison, 'Not all AI-assisted programming is vibe coding' (simonwillison.net, 19 Mar 2025); safety and honest-limits angle cross-checked with N. AlNasser, The Vibe Coder's Handbook (2026), Chs. 14 and 17, whose unsourced statistics are not used. Code verification and the observed order after Veluri, Roy and Luke, Computers & Fluids (2012); mutation testing after DeMillo, Lipton and Sayward, IEEE Computer 11(4) (1978); package hallucination from Spracklen et al., USENIX Security 2025; productivity results from Peng et al. (arXiv:2302.06590), METR (arXiv:2507.09089 and blog 24 Feb 2026) and the DORA 2024 report (Google Cloud blog); security results from Pearce et al. (arXiv:2108.09293), Perry et al. (ACM CCS 2023) and the Veracode press release of 30 Jul 2025; Mars Climate Orbiter from the NASA Mishap Investigation Board Phase I report (10 Nov 1999); rules 5 and 10 from G. J. Holzmann, 'The Power of 10' (NASA/JPL); ISA values from ISO 2533:1975. Every number was computed in the course environment or read in the cited source on 2026-10-16."
---

<!-- Build: python tools/build_slides.py slides/en/week_extra_vibe_coding_EN.md -->

## SLIDE 1 | An altimeter written in five minutes
**Content:**
- Project M3 needs pressure altitude from the UAV barometer
- One prompt to an AI assistant: "write pressure_altitude(p), ISA"
- Four answers: all run, all have docstrings, all return a number
- At 50 kPa they give 5574 m, 172 m, −159 889 and 5574 m
- Q: which one would you fly, and how would you prove it?
**Visual:** Four code cards side by side (candidates A to D from the lab, Section 1), each with a green "runs" badge and its output at 50 kPa underneath.
**Notes:** The four candidates are teaching examples written for this lab, not recorded outputs of a particular tool: say so. Ask for a show of hands before anything is run. Keep the question on the board; the last slide answers it.

## SLIDE 2 | What "vibe coding" means, and what this week is about
**Content:**
- Karpathy (Feb 2025): give in to the vibes, "forget that the code even exists"
- He called it fine for "throwaway weekend projects"
- Willison: if you review, test and can explain it, it is not vibe coding
- Kim and Yegge: you are the head chef; you delegate cooking, not responsibility
- This week: AI writes, the engineer verifies
**Visual:** A horizontal spectrum from "accept without reading" (left) to "specify, review, test, explain" (right), with a flight-software badge on the far right and "weekend prototype" on the far left.
**Notes:** Karpathy's wording is quoted from Simon Willison's post of 19 March 2025, which reproduces the tweet. Collins Dictionary named "vibe coding" its Word of the Year 2025. Kim and Yegge use "vibe coding" broadly for all AI-assisted programming with an engineer in charge; Willison keeps the narrow meaning. Both agree on the rule that matters for us: you sign the result.

## SLIDE 3 | Example 1: the right answer, by hand
**Content:**
- ISA troposphere: h = (T0/L) [1 − (p/p0)^(R L/g0)]
- Exponent 287.05 × 0.0065 / 9.80665 = 0.19026
- (50 000/101 325)^0.19026 = 0.87425; T0/L = 44 331 m
- h = 44 331 × (1 − 0.87425) ≈ 5575 m (5574.4 m exact)
- B used R = 8.314 J/(mol K): exponent 0.005511, h ≈ 172 m
**Visual:** IMAGE ../assets/week_extra_vibe_coding/fig_candidates.png
**Notes:** Do the arithmetic on the board before showing any code output. The universal gas constant 8.314 J/(mol K) and the specific constant of dry air 287.05 J/(kg K) differ by the molar mass; a dimension check of R L / g0 catches it in one line. Right panel: candidate D, which we meet again on slide 6.

## SLIDE 4 | Plausible code fails in engineering ways
**Content:**
- Wrong constant, wrong unit contract, a "safety" clamp that changes physics
- Kim and Yegge, Ch. 4: an agent silently deleted 80 % of one test suite
- Their other tales: a lost repository, a near-wiped CNC machine, ignored instructions
- Mars Climate Orbiter (1999): ground software in lbf·s, navigation expected N·s
- None of these crash: the program runs and returns a number
**Visual:** Four icons in a row (constant, units, clamp, deleted tests), each with one line: "runs, wrong".
**Notes:** The MCO root cause, as worded by the Mishap Investigation Board Phase I report (10 Nov 1999): "failure to use metric units in the coding of a ground software file, 'Small Forces', used in trajectory models". No AI was involved: the point is that these defects are ordinary, which is why AI reproduces them. The CNC story: the assistant proposed wiping the machine's storage device and the engineer almost pressed Enter.

## SLIDE 5 | Check 1: reference values you did not compute with the code
**Content:**
- ISA table (ISO 2533:1975): 101 325 Pa at 0 m, 22 632 Pa at 11 000 m
- Table rounded to 1 Pa; |dh/dp| ≈ 0.28 m/Pa near 11 km, so ±1 m is fair
- Write each case with its source: ReferenceCase(..., source=...)
- B and C fail at once; D passes
- Two table values are necessary, not sufficient
**Visual:** A two-row table (sea level, tropopause) with source, expected altitude, tolerance, and a pass/fail column for A to D.
**Notes:** Lab Section 3 and Exercise 3.1. The tolerance is justified from the rounding of the table, not chosen to make the test pass. Ask: what is the smallest test that would expose D?

## SLIDE 6 | Example 2: four candidates, five checks
**Content:**
- Properties: monotonic, round trip, negative below sea level
- Differential: within 1 m of the Week 2 library over 30 to 110 kPa
- A passes all five (rounded constants, max deviation 0.002 m)
- D passes the table but fails the other four
- A portfolio of checks, not one test, separates them
**Visual:** IMAGE ../assets/week_extra_vibe_coding/fig_check_matrix.png
**Notes:** Lab Sections 4 and 5. D's clamp returns 0 m for every pressure at or above 101 325 Pa: on a high-pressure day the UAV on the ground reads 0 m, and climbing the first metres shows no change. The check matrix is computed by Verifier.compare in the lab.

## SLIDE 7 | Properties and differential testing
**Content:**
- Property: a statement true for every valid input (sampled 200 times)
- Monotonic: higher pressure means lower altitude
- Round trip: altitude_to_pressure(pressure_altitude(p)) = p
- Differential: compare with an independent, trusted implementation
- Seeded random samples: a failure can be reproduced exactly
**Visual:** Left: a scatter of random pressures with a monotone curve through them; right: two curves (new, trusted) and their difference with a ±1 m band.
**Notes:** PropertyCheck and DifferentialCheck in src/aerospace_ai/verification.py take a sampler and a seed. Exercise 4.1 writes the round-trip property. Differential testing needs an oracle you trust for independent reasons (here: the Week 2 function, its tests and its review).

## SLIDE 8 | Unit contracts: candidate C is not wrong, it is foreign
**Content:**
- C takes hPa and returns feet (145 366.45 ft scale, 1013.25 hPa)
- Fed Pa, it gives −159 889: nonsense, but no error
- With an adapter (Pa → hPa, ft → m) it is within 3.88 m of the library
- The defect is the missing contract, not the formula
- Write units into names, docstrings and tests: p_pa, altitude_m
**Visual:** A block diagram: Pa → [÷100] → C → [× 0.3048] → m, with the comparison against pressure_to_altitude.
**Notes:** The 3.88 m difference comes from C's own rounded constants (exponent 0.190284 and the feet scale). This is the Mars Climate Orbiter pattern in miniature: two correct pieces, one undocumented unit interface. Ask the AI for units explicitly and test the contract.

## SLIDE 9 | Example 3: an RK4 that passes the usual test
**Content:**
- By hand: one RK4 step of dx/dt = −x, h = 0.1 gives 0.9048375
- Exact e^−0.1 = 0.904837418; error 8.2 × 10⁻⁸
- Pitch oscillator: ωn = 3 rad/s, ζ = 0.5, θ0 = 0.1 rad, T = 2 s
- Typo: k3 computed from k1 instead of k2
- At h = 0.05 both versions pass a 10⁻⁵ tolerance test
**Visual:** Two code listings side by side, identical except one highlighted character (k1 vs k2) in the k3 line, each with a green "PASS (error < 1e-5)" stamp.
**Notes:** Lab Section 6. Errors at h = 0.05: 1.21 × 10⁻⁷ for RK4, 6.93 × 10⁻⁶ for the typo version. A single-run tolerance test cannot see a lowered order of accuracy when the step is small enough.

## SLIDE 10 | Code verification: does the error fall at the promised rate?
**Content:**
- Observed order p̂ = ln(e1/e2) / ln(h1/h2) (Veluri, Roy and Luke, 2012)
- Halving h must divide the RK4 error by 2⁴ = 16
- RK4: 3.95, 3.99, 4.00; Euler: 1.38, 1.18, 1.09
- Typo version: 2.69, 2.55, 2.38: the defect is visible at once
- Needs an exact or manufactured solution
**Visual:** IMAGE ../assets/week_extra_vibe_coding/fig_convergence.png
**Notes:** Exercise 6.1 computes 2.547 by hand from the two typo errors at h = 0.05 and 0.025. The method of manufactured solutions extends this to equations without a known solution: choose the solution, derive the source term, check the order. ConvergenceCheck in the library automates it.

## SLIDE 11 | Are the tests any good? Mutation testing
**Content:**
- Passing tests prove nothing if they also pass on wrong code
- Mutation testing (DeMillo, Lipton, Sayward, 1978): seed known bugs
- A test suite "kills" a mutant when at least one test fails
- Mutation score = killed / total
- Track it: a silent test deletion becomes a visible drop
**Visual:** A trusted function in the centre with six arrows to mutated copies (universal R, hPa reference, clamp, inverted exponent, sign, scalars only), each marked killed or survived.
**Notes:** Mutant and MutationRunner in the library apply a one-place text patch to trusted source and rebuild the function. Kim and Yegge's "vanishing tests" story is exactly the case where the score drops and nobody looks.

## SLIDE 12 | Example 4: four suites, same green CI
**Content:**
- Weak (runs, 0 m at sea level): 1 of 6 mutants killed
- Typical (+ one hand-computed value): 4 of 6
- Strong (+ table, below sea level, array, monotonic): 6 of 6
- Strong minus the assertions D fails: back to 4 of 6
- Survivors: the sea-level clamp and the scalars-only version
**Visual:** IMAGE ../assets/week_extra_vibe_coding/fig_mutation.png
**Notes:** Lab Section 7 and Exercise 7.2. All four suites pass on the correct function, so a green CI badge alone cannot tell them apart. In the practice assignment the starter suite (two tests) kills 1 of 6 hidden mutants; a good suite kills all six.

## SLIDE 13 | Look before you run: static checks
**Content:**
- Count what you asked for: missing or unrequested functions
- Imports that do not resolve may be hallucinated packages
- Spracklen et al. (USENIX Sec. 2025): 576 000 samples, 16 models
- Non-existent packages: ≥ 5.2 % (commercial), 21.7 % (open-source)
- An attacker can publish malware under an invented name
**Visual:** An AI answer with the missing altitude_to_pressure greyed out and "import hallucinated_isa_toolkit" underlined in red.
**Notes:** "Count your babies" is Kim and Yegge's Ch. 7 advice. audit_deliverables and unresolvable_imports (library) need no execution, so they are safe on code you have not read. A name that does not resolve may simply be uninstalled: check the registry and the publisher before you install anything.

## SLIDE 14 | Pinned environments and old engineering rules
**Content:**
- NumPy 1.26 (course env): np.trapz exists, np.trapezoid does not
- NumPy 2.x: the reverse; assistants mix both
- Pin versions and run tests in the pinned environment
- Holzmann, Power of 10 (rule 5): at least two assertions per function
- Rule 10: zero compiler warnings, static analysis every day
**Visual:** Two terminal boxes: "NumPy 1.26.4 → trapz ✓ trapezoid ✗" and "NumPy 2.x → trapz ✗ trapezoid ✓".
**Notes:** Both facts were checked in this course's containers (NumPy 1.26.4 and 2.5.3). The Power of 10 rules were written for safety-critical C at NASA/JPL; the two quoted here transfer directly to AI-written Python: assertions as executable specification, and warnings as defects.

## SLIDE 15 | Example 5: a static-fire log (practice assignment)
**Content:**
- 498 samples, jittered 200 Hz, three dropped packets, load-cell noise
- Correct: clip negatives, trapezoid on actual times: I = 122.30 N·s
- "dt = t[1] − t[0]" (even spacing assumed): 97.85 N·s, 20 % low
- Integrating negative noise: 122.11 N·s
- A convergence check on an even grid cannot see the spacing bug
**Visual:** A thrust trace with the three gaps shaded and two shaded areas: the true impulse and the uniform-dt estimate.
**Notes:** The log is synthetic (a half-sine of 120 N over 1.6 s, exact impulse 2 F t_b / π = 122.23 N·s), generated by tools/make_week_extra_vibe_coding_data.py. Peak 120.51 N, burn time 1.541 s, average 79.37 N, Isp 191.86 s with 65 g propellant. Point: each check sees some bugs; combine a reference case on an uneven grid with the order check.

## SLIDE 16 | What the productivity studies say
**Content:**
- Peng et al. (2023, RCT, 95 people, one JS task): 55.8 % faster
- METR (2025, RCT, 16 experienced devs, own repos): 19 % slower
- Those developers expected −24 % and afterwards believed −20 %
- METR update (Feb 2026): −18 % and −4 %, CIs include zero
- The setting decides; self-reported speed-ups are not evidence
**Visual:** IMAGE ../assets/week_extra_vibe_coding/fig_evidence.png
**Notes:** Peng: 95 % CI 21 % to 89 %; success rate +7 points, not significant. METR 2025: CI +2 % to +39 %, Cursor Pro with Claude 3.5/3.7 Sonnet, 246 issues. METR 2026 calls its own update very weak evidence because of selection effects. DORA 2024 (survey): as AI adoption increased, throughput fell 1.5 % and delivery stability 7.2 % (estimates).

## SLIDE 17 | What the security studies say
**Content:**
- Pearce et al. (2021): 89 scenarios, 1689 Copilot programs, ~40 % vulnerable
- Perry et al. (ACM CCS 2023): assisted users wrote less secure code
- ...and were more likely to believe it was secure
- Veracode (vendor, Jul 2025): flaws in 45 % of tests (80 tasks, 100+ LLMs)
- DORA 2024: 39 % of respondents have little or no trust in AI code
**Visual:** Three cards (lab study, user study, vendor test) with the sample, the result and the type of evidence; the vendor card marked "not peer reviewed (not stated)".
**Notes:** Note the hierarchy of evidence: a controlled study with stated methods outranks a vendor press release, which outranks an unsourced number. Perry et al. used codex-davinci-002. Veracode: Java worst at over 70 %; XSS defences failed in 86 % and log injection in 88 % of relevant samples (press release figures).

## SLIDE 18 | Example 6: checking a statistic like an AI answer
**Content:**
- Handbook: "a 2025 study found that 45% of AI-generated code contained security vulnerabilities"
- No citation; the number matches the Veracode press release (vendor test)
- "170 critical vulnerabilities in 1,645 vibe-coded apps": only third-party posts
- The CVE-2025-48757 record itself does not give these counts
- "63 % non-developers", "$4.7 billion": no source found, not used
**Visual:** A three-step funnel: claim → original source? → type of evidence, with three claims dropping out at different levels.
**Notes:** The Vibe Coder's Handbook (AlNasser, 2026) is useful on practice but cites no sources for its statistics. We checked each one on 2026-10-16. The same habit applies to AI answers: an unsourced number stays unverified, however confident the wording.

## SLIDE 19 | A working method: one small loop per piece
**Content:**
- Delegate small, well-defined tasks with checkable criteria
- Supervise more when the task is novel, complex or safety-relevant
- Guardrails: no test edits, no data, no hardware commands
- Check after every step; commit to Git before each agent change
- Keep a project file: units, conventions, pinned versions
**Visual:** IMAGE ../assets/week_extra_vibe_coding/fig_loop.png
**Notes:** The five bullets paraphrase Kim and Yegge's delegation rules (Ch. 4) for a research lab. Willison's test is the exit criterion of the loop: do not commit code you could not explain to someone else.

## SLIDE 20 | A prompt that makes the answer checkable, and the safety rules
**Content:**
- State inputs and outputs with units, ranges, conventions, environment
- Give two or three reference values with their source
- Ask for assumptions and edge cases first, then code, then tests
- Never paste data or code not cleared for external services
- Install nothing until the package and its publisher are checked
**Visual:** The prompt template from lab Section 10 as a text block, with the units, reference-value and "do not modify tests" lines highlighted.
**Notes:** Read the template aloud. Remind students of sponsor and export-control rules for project data. Run unknown code in a throw-away environment. "The AI wrote it" is not an engineering justification.

## SLIDE 21 | This week's work
**Content:**
- Lab: notebooks/week_extra_vibe_coding (5 exercises, about 2 h)
- In class: 4 problems (B's exponent, pass@k, observed order, a test-weakening diff)
- Practice Assignment VC (formative, about 4 h): vibe-code it, then prove it
- Evidence files: prompt log, raw AI answer, REVIEW.md, your mutants
**Visual:** Three icons: notebook, pencil, Git branch.
**Notes:** The bonus week is outside the 16-week grading plan; A1 to A6 are unchanged. Students without an AI tool get a supplied answer to review. Expected numbers are in assignments/practice_vibe_coding/README.md.

## SLIDE 22 | What We Now Know How To Do
**Content:**
- Tell vibe coding from AI-assisted engineering, and own the result
- Verify AI code with reference values, properties and differential tests
- Measure the observed order of accuracy of numerical code
- Judge a test suite by its mutation score; notice weakened tests
- Run static checks and read AI studies and statistics critically
**Visual:** The opener's four candidates with only A ticked, and the five-check matrix beside it.
**Notes:** Answer the opener: fly A, and only because it passed table values, three properties and a differential test; you can show the evidence. Without the evidence you could not have told A from D.
