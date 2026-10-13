# TrustLap: resume and interview notes

Everything here is meant to be cut down. The prose is hand-written and stays as you edit it. The block at the end between the `resume:start` / `resume:end` markers is regenerated from `metrics.json` by `python -m trustlap.export`, so every measured number in it matches the website and README. When the new training runs land, that block updates itself; the prose doesn't need to.

Live site: https://trustlap.vercel.app · Code: https://github.com/hitenjain22/trustlap

---

## The 30-second version

I built a simulator for race cars in changing weather, trained AI drivers in it with reinforcement learning, and gave one of them a sense of grip: it works out how slippery the road is from how the car has been moving, the way a human driver feels the car slide. I then tested it honestly against an identical driver without that sense, on hundreds of tracks neither had seen, and published every result, including where it loses. All of it runs live in the browser: every car on the website is the actual trained neural network.

## The 2-minute version

Self-driving and robotics systems are trained in simulation and then meet conditions they never practised in. The question behind TrustLap is whether a learned driver can *adapt* to conditions it can't observe directly, like a wet road, and whether it can tell when it's out of its depth.

1. **Simulator.** I wrote a vehicle-dynamics simulator in JAX (tire slip, grip that changes with rain, puddles, wind, tire wear, heavier cars, steering lag). Because it's pure array math, one GPU runs tens of thousands of cars at once, which is what makes training on billions of driving decisions practical.
2. **Training.** Drivers learn by reinforcement learning (PPO). A population trains side by side and periodically copies the best performer while trying new settings (population-based training), and an automatic curriculum makes the weather harder as they improve.
3. **Feeling the road.** Using the Rapid Motor Adaptation recipe, a "teacher" driver is first trained while being *told* the true conditions; then a "student" network learns to infer those conditions from the last second of motion, so the same driving skill works without being told anything. I call these the Teacher AI and the Adaptive AI on the site; the comparison point is a Basic AI trained the same way but with no sense of grip.
4. **Knowing when it's unsure.** Five copies of the student vote; their disagreement plus speed, the curve ahead and the felt grip feed a calibrated crash-risk estimate, and a safety governor can ease off the throttle when risk is high.
5. **Honest evaluation.** Every driver races the same held-out tracks in the same conditions from the same start (paired comparisons), with confidence intervals from a cluster bootstrap over tracks. Conditions it never trained on are tested separately. Every number on the website and README is generated from the evaluation output; nothing is typed by hand.
6. **The website.** An exact TypeScript port of the physics and a hand-written neural-network runtime run the real trained drivers in a Web Worker in your browser, with a Playground (change the weather, draw your own track, run a puddle test), a training timeline, and a plain-language results page.

**The result, honestly:** the Adaptive AI wins big where grip changes (rain, puddles, worn tires), loses on dry roads and in wind, and both fail in conditions far outside training. I traced the losses to how the student imitates the teacher and built two fixes (fine-tuning the driving on its own estimates, and a dry-road fallback); those, plus a three-seed replication with equal training budgets, are the current runs.

---

## What I built, piece by piece

### 1. A GPU vehicle simulator (Python, JAX)

- **Physics:** a dynamic bicycle model (front and rear axle), Pacejka "Magic Formula" tire forces with a friction ellipse (braking and cornering share the same grip), aerodynamic drag, rolling resistance, wind and random gusts, a low-speed blend so the model stays stable when nearly stopped, steering-actuator lag and input delay. Four physics substeps per 50 ms control step.
- **Conditions:** rain lowers grip everywhere; puddles are Gaussian patches of low grip under each axle; tires wear over the lap; mass, power, brakes, drag, steering lag, input delay and sensor noise all vary. One "difficulty" dial widens every range together.
- **Tracks:** procedurally generated closed circuits (random star-shaped control points, centripetal Catmull–Rom splines), validated for length, minimum corner radius, self-intersection and pinch points, with an automatic Laplacian-smoothing fix for near-misses. Separate train, validation and test track banks.
- **Design choices:** pure functions over arrays, written once for both NumPy and JAX and vectorized with `vmap`; all randomness from explicit PRNG keys and a PCG32 generator; every constant in one config file that is exported to the browser, so nothing is copied by hand between languages.
- **Benchmark:** measured throughput on a T4 GPU against the same code on CPU (numbers below).

### 2. An exact browser port (TypeScript)

- The simulator, tracks, conditions and a dependency-free MLP runtime are re-implemented in TypeScript and **parity-tested against Python with golden files**: Python writes thousands of random cases (physics steps, observations, network outputs, closed-loop laps, the risk model) and the TypeScript tests must reproduce them within tight tolerances. Any physics change regenerates the fixtures and must pass both suites.
- Physics runs in a Web Worker at a fixed 20 Hz; the page interpolates between snapshots and draws with Canvas 2D at the screen's refresh rate (layered static background, puddles, skid marks, rain, cars, overlays).

### 3. Reinforcement learning at scale

- **PPO in the PureJaxRL style:** rollouts, generalized advantage estimation and update epochs compiled into one JAX function. Details that matter: GAE distinguishes *terminated* (crash: no future value) from *truncated* (time limit: bootstrap from the critic); running observation normalization frozen at deployment; reward scaling by the running return std; an asymmetric critic that sees privileged information during training only.
- **Population-based training:** several drivers train in parallel; every 25 updates the weakest copies the strongest and perturbs its learning rate and exploration; the full lineage is recorded (the Training lab draws the family tree).
- **Automatic curriculum:** difficulty rises when drivers finish over 80% of laps twice in a row and backs off below 50%, with a share of easy and "stretch" episodes so skills aren't forgotten.

### 4. Teacher–student adaptation (Rapid Motor Adaptation)

- **Teacher AI:** its actor gets the normal observation (33 numbers: speed, slip, steering, the road ahead) plus an 8-number summary of the true conditions produced by an encoder over 14 privileged values.
- **Adaptive AI (student):** an adaptation network reads the last second of motion (20 steps × 10 numbers) and predicts that 8-number summary plus the grip under each axle; the teacher's driving network then acts on the estimate. Trained DAgger-style (the student drives, the teacher labels), with a warm start.
- **Basic AI:** the same PPO training without any condition information, as the controlled comparison.

### 5. Uncertainty and safety

- **Ensemble:** five students with different seeds; disagreement between their estimates and resulting actions signals unfamiliar conditions.
- **Crash-risk model:** disagreement, speed, the sharpest curve in the next ~40 m and *grip demand* (the cornering grip the curve needs at this speed divided by the grip the car feels it has), combined by logistic regression and calibrated with isotonic regression on validation runs, so "20% risk" means about 20% of those moments end in a crash. The browser runs the same model (parity-tested).
- **Safety governor:** above a risk threshold it caps the throttle; the threshold is chosen on validation tracks only.

### 6. Evaluation built to avoid fooling myself

- Held-out test tracks never used for training or tuning; every driver races the same track, conditions and start (paired), so differences come from the driver.
- 95% confidence intervals from a paired cluster bootstrap over tracks (laps on one track are correlated).
- Twelve condition suites, four of them never trained on (to measure the cliff outside the training range).
- Lap times compared only on laps every driver finished, so slow-but-safe drivers aren't flattered by survivor bias.
- Model selection (calibration, governor threshold, fallback threshold) on a separate validation bank; the test bank is touched once.
- A controlled puddle test with a no-puddle control run, after I caught my own metric over-crediting the model (story below).
- In progress: three independent training seeds with the published one fixed in advance, so each verdict says whether it held in every run.

### 7. The product

- A React + TypeScript site (Vite, Tailwind, Zustand, Comlink, zod-validated artifacts) on Vercel with CI on GitHub Actions (Python tests, web unit and parity tests, Playwright end-to-end tests, bundle-size budgets).
- **Home:** 48 live cars in the rain with a rain slider; a live side-by-side of the Adaptive and Basic AI on the same track and rain.
- **Playground:** weather and car sliders with the training range shaded, scenario presets, draw-your-own-track (simplification, scaling, validation with specific error messages, auto-fix), a puddle test scored like the evaluation, chase camera, tire-force arrows, a grip trail coloured by estimate error, crash explanations in plain words, a safety-governor switch, shareable URLs, keyboard shortcuts, a phone layout with a bottom sheet.
- **Training lab:** the real saved checkpoints driving live on a scrubbable timeline, the population curves and family tree.
- **Results:** a plain-language summary first (one chart, generated takeaways), the full evaluation underneath.
- Lighthouse desktop performance 99–100 and accessibility 100 on every page; keyboard and screen-reader support; no layout shift; works on phones.

---

## Debugging stories (good interview material)

- **The evaluation that quietly used the wrong tracks.** Generalization looked terrible: the drivers seemed to fail on their own *training* tracks. The cause was a JAX compilation cache that captured the first track bank it saw, so later runs silently reused the test tracks. I made the bank an explicit argument to the compiled function, added a regression test and reran the whole evaluation.
- **A metric that flattered my model, twice.** My first "time to notice a puddle" metric fired when the car *left* the puddle. The fix reported zero detections. On a later run it reported dozens of detections, but the timing was suspicious, so I added a control run with no puddle: every "detection" came seconds after the car had left the puddle, a knock-on wobble, not detection. The metric now only counts responses while grip is actually low and reports the control's false alarms. The honest answer is that the student doesn't detect single puddles, and the site says so.
- **NaN geometry that passed validation.** Hand-drawn tracks that ended exactly on their start point collapsed to NaN in the simplification step, and NaN comparisons are always false, so the broken track passed every validity check. Fixed the degenerate case and added a finiteness check.
- **A "slow device" fallback that undid the user's choice.** A performance fallback re-initialized the simulation with the original track, silently reverting a track the user had just picked, and made the field jump to a new layout. It now drops cars in place.
- **A scripted reference driver that crashed every lap,** fixed one cause at a time from per-step traces: power oversteer, pure-pursuit oscillation, late braking, pedal chatter.
- **A flaky browser test** that turned out to measure wall-clock time on a CPU-starved machine; it now waits on the simulator's own clock.

## Honest limitations (say these before an interviewer does)

- Simulation only: a flat 2D world, one car type, two axles standing in for four tires.
- The Adaptive AI loses to the Basic AI on dry roads and in wind; both fail far outside the training range.
- The student doesn't detect individual puddles; why it does better in the wet isn't fully pinned down.
- The five students share one driving policy, so their disagreement measures uncertainty about the road, not about the driving.
- What I'm doing about it: equal training budgets for teacher and baseline, longer student training, three-seed replication, fine-tuning the driving network on the student's own estimates, and a validated dry-road fallback.

## Skills and keywords

Reinforcement learning (PPO, GAE, population-based training, curriculum learning, teacher–student / privileged learning, Rapid Motor Adaptation, DAgger), uncertainty estimation (deep ensembles, isotonic calibration, logistic regression, ECE, Brier, AUROC), experimental design (held-out evaluation, paired comparisons, cluster bootstrap, controls, replication), JAX (jit, vmap, scan), Flax, Optax, NumPy, vehicle dynamics (bicycle model, Pacejka tires), TypeScript, React, Web Workers, Canvas 2D, golden-file parity testing, Vitest, Playwright, pytest, CI (GitHub Actions), Vercel, GPU training on Kaggle, accessibility (WCAG, Lighthouse), performance profiling.

---

## Measured numbers and candidate bullets (generated)

<!-- resume:start -->
**Simulator and compute**

- 65,536 cars simulated in parallel on one T4 GPU, 926× the same JAX code on its CPU
- Training: 3.0 billion environment steps in 83 minutes on NVIDIA T4 GPU (Kaggle)

**Evaluation protocol**

- 200 held-out test tracks × 3 condition draws = 600 laps per driver per condition; 12 conditions (4 never trained on); 2000 paired cluster-bootstrap resamples over tracks

**Laps finished out of 100, unseen test tracks**

| Condition | Adaptive AI | Basic AI | Teacher AI | Rule-based |
|---|---|---|---|---|
| dry | 64% | 97% | 96% | 100% |
| wet | 39% | 4% | 62% | 100% |
| puddles | 38% | 2% | 66% | 100% |
| wind | 50% | 96% | 82% | 100% |
| worn | 86% | 78% | 92% | 100% |
| laggy | 0% | 0% | 0% | 98% |
| heavy | 77% | 82% | 96% | 100% |
| combined | 28% | 24% | 33% | 97% |
| ice | 0% | 0% | 0% | 89% |
| extreme lag | 0% | 0% | 0% | 0% |
| noisy sensors | 0% | 12% | 0% | 100% |
| very heavy | 82% | 66% | 93% | 100% |

**Headline comparisons**

- Everything at once: 5% fewer crashes than the Basic AI (95% CI 0% to 10%)
- Share of the Teacher AI's advantage the Adaptive AI did not recover: 58% (95% CI 27% to 97%)
- Grip estimate error: ±0.079 μ (dry asphalt ≈ 1.08, ice 0.25)
- Single-puddle detection: 0 of 200 controlled trials (control without a puddle: 0 false alarms)
- Crash-risk estimate: ECE 0.019, Brier 0.061, AUROC 0.58
- Safety governor at τ = 0.15: crash rate 70% vs 72% without it (everything at once)
- Generalization (everything at once, laps per 100): training tracks vs unseen tracks: Adaptive 26% vs 28%, Basic 25% vs 24%

**Candidate bullets (generated; cut and reword freely)**

- Built TrustLap, a JAX vehicle-dynamics simulator that runs 65,536 cars in parallel on one T4 GPU (926x its CPU), with a TypeScript port held to the Python physics by golden-file parity tests and served as a live in-browser demo (Web Worker, Canvas 2D, hand-written MLP runtime).
- Trained PPO driving agents with population-based training and an automatic curriculum (3.0B environment steps in 83 min on NVIDIA T4 GPU (Kaggle)); an RMA-style adaptation module that infers road grip raised lap completion over a non-adaptive baseline on unseen tracks in puddles (38% vs 2%) and rain (39% vs 4%).
- Evaluated on 200 held-out tracks across 12 condition suites with paired cluster-bootstrap CIs and reported the losses too: 5% fewer crashes than the baseline in combined conditions (95% CI 0% to 10%), worse on dry roads (64% vs 97%), a crash-risk estimate that is calibrated (ECE 0.019) but weakly predictive (AUROC 0.58), and no detection of single puddles (0 of 200).
<!-- resume:end -->
