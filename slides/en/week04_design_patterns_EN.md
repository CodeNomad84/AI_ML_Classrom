---
title: "Week 4: Design Patterns for ML"
subtitle: "Strategy for models, Observer for training, Factory for data, pytest for all three"
course: "AI for Aerospace & Mechanical Engineering (PhD)"
footer: "Week 04 | Design patterns for ML"
language: en
provenance: "New content. Patterns rewritten from GeeksforGeeks (Strategy, Observer and Factory Method design patterns, Getting Started with Pytest, Pytest Fixtures) and Gamma, Helm, Johnson and Vlissides, Design Patterns (1994); Python details from W3Schools (Classes, Inheritance, Polymorphism, Dictionaries, Lambda); Persian terms checked against the KodAmoz Python bootcamp; least squares and k-NN from Bishop, PRML, Ch. 3 and Sec. 2.5.2; early stopping from Goodfellow et al., Deep Learning, Sec. 7.8; fuel-flow model from Anderson, Aircraft Performance and Design, Ch. 5."
---

<!-- Build: python tools/build_slides.py slides/en/week04_design_patterns_EN.md -->

## SLIDE 1 | The experiment script that keeps growing
**Content:**
- Task: predict a UAV's fuel flow [g/h] from V, altitude, climb rate, mass
- First attempt: run_experiment(model_type, data_source, verbose, early_stop, ...)
- "KNN" and "lienar" silently fall back to the mean baseline
- Every new model, data source or training feature edits the same function
- Q: how do we add things without touching what already works?
**Visual:** Code block of run_experiment with three if/elif chains highlighted in orange; below it the printed table where "KNN" and "lienar" both show 225.41 g/h (the baseline).
**Notes:** Run Section 0 of the lab live. Ask: which result would you have put in the paper? Write the closing question on the board and keep it there until the last slide.

## SLIDE 2 | Three axes of change, three patterns
**Content:**
- The model varies: Strategy
- What happens during training varies: Observer
- Where the data comes from varies: Factory
- Goal: add a class, edit nothing (Open/Closed Principle)
**Visual:** IMAGE ../assets/week04/fig_spaghetti_vs_patterns.png
**Notes:** Left grid: adding a model, a dataset or a callback edits seven places in shared code. Right grid: each addition is one new class. That is the whole lecture in one picture.

## SLIDE 3 | Example 1: fuel flow at 25 m/s by hand
**Content:**
- P_par = ½ρV³S·C_D0 = 0.5·1.225·15 625·1.5·0.03 = 430.66 W
- P_ind = 2W²/(ρVSπe·AR) = 173 106 / 1154.5 = 149.94 W
- P_shaft = (430.66 + 149.94) / 0.75 = 774.13 W
- Fuel = 0.35 kg/kWh · 0.77413 kW = 270.95 g/h
- With a 1 m/s climb: + W·ḣ/η = +392.3 W → 408.24 g/h
**Visual:** IMAGE ../assets/week04/fig_power_curve.png
**Notes:** m = 30 kg, sea level, S = 1.5 m², AR = 10, e = 0.8. The curve is convex in V and contains a mass × climb-rate product: a straight line cannot capture either. That is why model choice matters. Adapted from Anderson, Aircraft Performance and Design, Ch. 5.

## SLIDE 4 | Design patterns: named answers to recurring problems
**Content:**
- Pattern = a proven structure for a problem that keeps coming back
- Catalogued by Gamma, Helm, Johnson, Vlissides (1994): 23 patterns
- Built from what we already know: ABCs, inheritance, composition
- A shared vocabulary: "make the loss a Strategy" says a lot in 4 words
- Not a goal in itself: use one only when that axis really changes
**Visual:** Three cards labelled Creational (Factory), Behavioural (Strategy, Observer), Structural (Adapter), each with a one-line definition.
**Notes:** GeeksforGeeks lists the pros and the cost of each pattern: more classes. Week 2 already used one: the template method in BaseSensor.add_reading.

## SLIDE 5 | Strategy: make the algorithm a replaceable part
**Content:**
- Strategy interface: what every algorithm offers (fit, predict)
- Concrete strategies: one class per algorithm
- Context: holds "some strategy", calls only the interface
- Swap at run time: experiment.strategy = KNearestNeighbors(k=5)
**Visual:** IMAGE ../assets/week04/fig_strategy_uml.png
**Notes:** Rewritten from GeeksforGeeks "Strategy Design Pattern". The context never contains an if on the model type; that is the test of whether you applied the pattern.

## SLIDE 6 | Example 2: three strategies, four flights
**Content:**
- Train: (18, 170), (22, 200), (26, 280), (30, 400); test: (25, 270), (29, 375)
- Mean: 262.5 for both → RMSE 79.73 g/h
- k-NN, k = 2: 240 and 340 → RMSE 32.60 g/h
- Least squares: slope 1540/80 = 19.25, intercept −199.5 → 281.75, 358.75 → RMSE 14.18
- At V = 16 (true 162): line 108.5, k-NN 185: the best strategy depends on the question
**Visual:** Scatter of the four training points and two test points with the three fitted predictors (flat line, k-NN steps, straight line); extrapolation region shaded.
**Notes:** Do the arithmetic on the board, then run the lab cell where the same two calls (fit, predict) serve all three models. Polymorphism from Week 2 is what makes Strategy work.

## SLIDE 7 | Strategy in Python: function or class?
**Content:**
- Functions are objects: a dict of lambdas is a strategy table
- Enough while the strategy has no state and no settings
- Real models have both: learned coef_, chosen k and alpha
- Regressor(ABC): fit/predict check inputs once, call _fit/_predict
- New model = one subclass with two methods
**Visual:** Two code panels: PREDICTORS = {"mean": lambda ..., "median": lambda ...} and class MeanBaseline(Regressor) with _fit and _predict; arrow "add settings and state".
**Notes:** W3Schools "Python Lambda" for the first panel. The public fit/predict are template methods, exactly like BaseSensor.add_reading. Exercise 1.1 writes MedianBaseline.

## SLIDE 8 | Adapter: bringing scikit-learn in
**Content:**
- sklearn estimators have fit/predict but are not Regressor subclasses
- SklearnRegressor(estimator) wraps them: an Adapter
- typing.Protocol checks the shape of an object, not its ancestry
- isinstance(obj, SupportsFitPredict) works with @runtime_checkable
- Result: RandomForest joins the comparison with zero edits
**Visual:** Plug diagram: sklearn RandomForestRegressor (round plug) → SklearnRegressor adapter → Regressor socket (square); code `SklearnRegressor(RandomForestRegressor(n_estimators=200))`.
**Notes:** Protocol is "duck typing with type hints" (Week 3 typing). The adapter also flattens sklearn's occasional 2-D predictions. Exercise 1.2: the forest reaches 37.65 g/h.

## SLIDE 9 | Example 3: six strategies on 400 flights
**Content:**
- Chronological split: first 300 train, last 100 test (no leakage)
- Linear, ridge, gradient descent: 51.1 to 51.5 g/h
- k-NN (k = 5, scaled): 51.6 g/h; mean baseline: 225.4 g/h
- k-NN unscaled: 235.3 g/h, worse than the mean!
- Only experiment.strategy changed between runs
**Visual:** IMAGE ../assets/week04/fig_model_comparison.png
**Notes:** Unscaled, altitude (0 to 3000 m) swamps airspeed (16 to 36 m/s) in the distance, so "nearest" means "similar altitude". The comparison exposed a bug that was invisible inside run_experiment. Gradient descent and the normal equations agree: a free self-check.

## SLIDE 10 | Observer: events without touching the loop
**Content:**
- Subject keeps a list of observers and notifies them
- Observers react; the subject does not know what they do
- Trainer = subject; Callback = observer
- Events: on_train_begin, on_epoch_end, on_train_end
- Observers talk back only through TrainingState
**Visual:** IMAGE ../assets/week04/fig_observer_sequence.png
**Notes:** Rewritten from GeeksforGeeks "Observer Design Pattern" (weather station and displays). Keras and PyTorch Lightning callbacks are exactly this.

## SLIDE 11 | The Trainer loop and TrainingState
**Content:**
- model.initialize → notify("on_train_begin")
- each epoch: train_step, val loss, notify("on_epoch_end")
- if state.stop_training: break → notify("on_train_end")
- TrainingState: epoch, losses, learning_rate, stop_training, reason
- About 15 lines that never change again
**Visual:** Code of Trainer.fit with the three notify lines highlighted; side box with the TrainingState dataclass fields.
**Notes:** Callback hooks are not abstract on purpose: an observer overrides only the events it cares about. TrainableModel is a Protocol, so any model with initialize/train_step/loss/get_state/set_state can be trained, including PyTorch wrappers in Block 2.

## SLIDE 12 | Example 4: one gradient-descent step by hand
**Content:**
- Data (1, 2), (2, 4), (3, 5); ŷ = w0 + w1·x; start w = (0, 0)
- L0 = (4 + 16 + 25)/3 = 15
- ∇L = (2/n)·Aᵀ(Aw − y) = (−7.333, −16.667)
- η = 0.1: w = (0.7333, 1.6667), L1 = 0.2341
- η = 0.5: w = (3.667, 8.333), L1 = 309: divergence
**Visual:** IMAGE ../assets/week04/fig_gradient_step.png
**Notes:** In-class Problem 1. Stability needs η < 2/λmax of the Hessian (2/n)AᵀA; here λmax = 11.09 so η < 0.18. Week 6 derives it; today a DivergenceGuard callback catches it.

## SLIDE 13 | The callbacks we ship
**Content:**
- History: losses and learning rate per epoch (for plots)
- ProgressLogger: a line every n epochs, a summary at the end
- EarlyStopping: patience, min_delta, restore best weights
- DivergenceGuard: stop if loss is non-finite or 10× the first epoch
- StepDecay: multiply the learning rate by γ every k epochs
**Visual:** Five plug-in cards attached to a Trainer box, each with its hook names (on_epoch_end, on_train_end).
**Notes:** DivergenceGuard is the training equivalent of an engine over-temperature cut-off. Practice 4 adds ReduceLROnPlateau as a sixth card without editing the Trainer.

## SLIDE 14 | Observer pitfalls: order and coupling
**Content:**
- Observers run in attachment order
- Logger before StepDecay prints the old rate; after it, the new one
- Training is identical either way: the Trainer reads the rate next epoch
- Never call one observer from another; share state instead
- Reset counters in on_train_begin so objects can be reused
**Visual:** Two timelines of the same epoch: [Logger → StepDecay] printing lr 0.4 and [StepDecay → Logger] printing lr 0.2, with identical loss values 0.6533 and 0.2708.
**Notes:** In-class Problem 4e. The loose coupling that makes Observer flexible also makes it harder to debug: the GeeksforGeeks article lists that as the main cost.

## SLIDE 15 | Example 5: early stopping by hand
**Content:**
- val_loss: 1.00, 0.80, 0.78, 0.70, 0.69, 0.68, 0.50; patience 2, min_delta 0.05
- Improve only if loss < best − 0.05: epochs 0, 1, 3
- wait: 0, 0, 1, 0, 1, 2 → stop after epoch 5
- Restore epoch 3 weights (0.70)
- Patience 3 would reach epoch 6 (0.50): patience is a trade-off
**Visual:** IMAGE ../assets/week04/fig_early_stopping.png
**Notes:** In-class Problem 2 and test_early_stopping_hand_trace. On the fuel data min_delta = 1 stops after 39 epochs instead of 170 with the same test RMSE (51.19 vs 51.21 g/h).

## SLIDE 16 | Factory: ask for data by name
**Content:**
- Client says what it wants: "fuel_flow_synthetic"
- Factory decides how to build it and returns a DatasetLoader
- Product interface: one method, load() → Dataset
- Typos fail loudly, with a suggestion
- Settings travel as keyword arguments, not new flags
**Visual:** IMAGE ../assets/week04/fig_factory_registry.png
**Notes:** Rewritten from GeeksforGeeks "Factory Method - Python Design Patterns", whose if/else vehicle client is the anti-pattern we started from. Week 5 adds a noisy LiDAR loader, Week 11 a camera-image loader, both as new rows in this registry.

## SLIDE 17 | Registry and class decorator in Python
**Content:**
- _registry: ClassVar[dict[str, type[DatasetLoader]]]
- @DatasetFactory.register("name") stores the class and returns it unchanged
- create(name, **kwargs) → loader; load(name, **kwargs) → Dataset
- Duplicate names and non-loaders are rejected
- difflib suggests the closest name on a typo
**Visual:** Code: the register classmethod returning an inner decorator; below, `@DatasetFactory.register("csv") class CsvTableLoader(DatasetLoader): ...`.
**Notes:** A decorator is just a function that takes a class and returns it (W3Schools Dictionaries for the registry itself). Week 2 showed the alternative, __init_subclass__; the decorator keeps registration explicit and lets one module register several names.

## SLIDE 18 | Example 6: a new loader in six lines
**Content:**
- Subclass CsvTableLoader (it already reads and checks columns)
- Defaults: path = week04_fuel_flow.csv, the four features, the target
- Decorate with @DatasetFactory.register("fuel_flow_csv")
- DatasetFactory.load("fuel_flow_csv") → 400 rows
- Matches the synthetic loader to 0.0005 g/h (3-decimal CSV)
**Visual:** Code of FuelFlowCsvLoader (decorator, class line, __init__ calling super().__init__) and the printed Dataset repr.
**Notes:** Lab Exercise 3.1. Re-running a notebook cell defines a new class object, so the cell unregisters the old one first: the registry refuses to hand an existing name to a different class.

## SLIDE 19 | Testing patterns: contract tests
**Content:**
- One fixture with params = every strategy
- Same 6 tests for all: fit returns self, shapes, NotFittedError, bad data
- "Beats or equals the mean baseline" is a test, not a hope
- New model added to the list = tested for free
- Liskov: any strategy must work where a Regressor is expected
**Visual:** IMAGE ../assets/week04/fig_contract_tests.png
**Notes:** @pytest.fixture(params=sorted(STRATEGIES)) in tests/test_models.py. GeeksforGeeks "Pytest Fixtures" for the fixture syntax.

## SLIDE 20 | Test doubles and isolation
**Content:**
- SpyCallback records events: assert the exact notification order
- ScriptedModel returns fixed losses: early stopping becomes a test
- FakeEstimator: test the adapter without scikit-learn
- Fixture registers a throw-away loader and unregisters it after
- tmp_path for files, capsys for printed output, pytest.approx for floats
**Visual:** Three small code snippets: class SpyCallback(Callback) with events.append, a ScriptedModel with next(self._val), a fixture with yield and unregister.
**Notes:** The patterns make this easy: each piece talks only to an interface, so a fake on the other side of the interface is enough.

## SLIDE 21 | Example 7: testing the notification order
**Content:**
- Two spies share one log list, epochs = 2
- Expected: a:begin, b:begin, a:epoch0, b:epoch0, a:epoch1, b:epoch1, a:end, b:end
- One assert checks subject behaviour and order together
- 96 Week 4 tests run in about 2 s
**Visual:** Code of test_events_arrive_in_order_for_every_observer with the expected list; terminal line "96 passed".
**Notes:** Ask the room to predict the list before showing it. Then ask what breaks if notify iterated over a set instead of a list.

## SLIDE 22 | This week's work
**Content:**
- Lab: notebooks/week04/week04_design_patterns.ipynb (6 exercises)
- In class: 4 warm-ups (GD step, early stopping, strategy choice, name the pattern)
- Practice 4 (formative, about 3 h): polynomial strategy, plateau callback, regime loader, tests
- Assignment 2 (graded) is released next week
**Visual:** Three icons: notebook, pencil, Git branch.
**Notes:** Practice 4 expected numbers are in its README. Polynomial features with interactions drop the test RMSE from 51.5 to 22.6 g/h: the climb × mass product is the physics.

## SLIDE 23 | What We Now Know How To Do
**Content:**
- Spot the axes of change in ML code: model, training behaviour, data source
- Swap models with Strategy; wrap foreign libraries with an Adapter
- Add training behaviour as Observer callbacks without editing the loop
- Create loaders by name with a registry Factory and a class decorator
- Test it all with contract tests, test doubles and fixtures
**Visual:** The opener's run_experiment function crossed out, replaced by three boxes (Strategy, Observer, Factory) feeding one small experiment loop; green check marks.
**Notes:** Return to the opener question: a new model, callback or loader is now one new class and zero edits to code that works.
