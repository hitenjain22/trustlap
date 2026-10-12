# TrustLap explained

Plain-English notes on every part of the project, written for Hiten to read after each milestone. Interviewers will ask about all of it, so the goal is that every section can be retold in your own words.

Each component gets: what it does (two sentences), why it's needed, the one equation or idea behind it, and what would break without it. The 20 interview questions come at the end once the components exist.

---

**Names on the website.** The site uses plain names for the four drivers: **Adaptive AI** = the student (RMA phase 2, feels the road from its motion history), **Basic AI** = the blind policy (same training, no grip sense), **Teacher AI** = the teacher (told the true conditions), **Rule-based driver** = the scripted controller (given the exact grip). This document and the code keep the technical names.

## The project layout (M0)

**What it is.** Two halves that never run at the same time. `sim/` is a Python package that simulates cars and trains the AI drivers offline on a GPU (Colab/Kaggle). `web/` is a static website that loads the trained networks as plain files and runs them in your browser.

**Why it's split like this.** Training needs a GPU and hours of compute; the website needs to be free to host and instant to load. So training produces files (weights, metrics, configs), and the website only reads files. There's no server doing work when someone visits.

**The one idea.** "Every number on the website comes from a file a script produced." The website never contains a hand-typed result. If the file is missing a number, the page says "Not computed yet."

**What would break without it.** If constants were copied by hand between Python and TypeScript, the browser car would drive slightly differently from the trained one, and the AI would crash in the browser for reasons no one could see. That's why `config.py` is the single source of truth, and why parity tests exist (coming in M2).

**Tooling, in one line each.**
- **JAX:** NumPy that runs on a GPU, compiles functions (`jit`), and turns a one-car function into a many-car function (`vmap`).
- **Vite + React + TypeScript:** the website's build tool, UI library and typed JavaScript.
- **Vitest / Playwright / pytest:** unit tests for the web, browser tests that click through the real site, and Python tests.
- **CI (GitHub Actions):** runs all the tests on every push, so a broken change is caught before it's deployed.
- **Vercel:** hosts the static site for free.

---

## The random number generator: PCG32 (M1, `rng.py`)

**What it is.** A tiny, well-known algorithm that turns a seed (any whole number) into an endless stream of random-looking numbers. Same seed, same stream, every time.

**Why it's here.** The website has to rebuild exactly the same track from a seed as Python did ("Hairpin Valley is seed 200,017"). Python's and JavaScript's built-in random functions are different algorithms, so we use one we implement ourselves in both languages, bit for bit.

**The one idea.** PCG = a simple "multiply and add" update of a 64-bit number (a linear congruential generator), plus a scrambling step on the output (xorshift + rotate) that hides the weaknesses of the plain update. The Python version uses explicit 64-bit masking so it does exactly what JavaScript's BigInt arithmetic does. The test checks our first six outputs for seed 42 against the published reference.

**What would break without it.** Shared links and showcase tracks wouldn't match between training and the browser.

---

## Procedural tracks (M1, `track.py`)

**What it does.** Makes a random race-track loop from a seed: 8–14 random points around a circle, some pulled inward to make hairpins, joined by a smooth curve, then checked for problems.

**Why it's built this way.**
- Points are placed at increasing angles around the center, so the rough shape can't cross itself.
- The smooth curve is a **centripetal Catmull–Rom spline**: it passes through every point and never makes loops or cusps (the "centripetal" version is the one proven not to).
- The curve is re-sampled every ~2 m, so "index i" means "about 2i meters along the track". For each point we store the direction of travel (tangent), the direction to the left (normal) and how sharply it turns (curvature: 1 / turning radius).
- **Validation** rejects tracks that are the wrong length, have a corner tighter than 18 m radius, have two parts that come too close ("pinch"), or whose edges cross.

**A problem we hit, and the fix.** With the spec's settings, 82% of random tracks had a corner tighter than 18 m, so most seeds needed several retries. The spec already had a fix for hand-drawn tracks, "Auto-fix": nudge each point toward the average of its neighbours (Laplacian smoothing), a few times, until the tight corner opens up. We now apply the same fix to generated tracks: 93% pass on the first try, and they keep their hairpins (90% still have a corner under 30 m).

**Track banks.** 2,000 training tracks, 100 validation tracks (for choosing settings and models) and 200 test tracks (only for final results, never for tuning). Keeping the test tracks unseen is what makes "it works on tracks it never saw" an honest claim.

---

## The car physics (M1, `dynamics.py`)

**What it is.** A **dynamic bicycle model**: the car is treated as one front wheel and one rear wheel (each standing for an axle), with a body that can slide sideways and rotate. Every 12.5 ms we add up the forces on the car and update its speed, sideways speed, rotation and position. Four of these "substeps" make one 50 ms control step, which is when the driver (AI or scripted) makes a decision.

**The forces, in plain words.**
- **Tires** are where all the interesting behaviour lives. When a tire rolls in a slightly different direction than it points, it pushes sideways. That angle is the **slip angle**. Small slip gives a sideways force proportional to the slip; large slip saturates at the grip limit. The **Pacejka "magic formula"** is the standard curve for this: `F = −μ · load · sin(C · atan(B·α − E·(B·α − atan(B·α))))`.
- **μ (grip)** multiplies everything. Rain lowers it, puddles lower it locally under one axle, worn tires lower it over the lap. That is why the AI has something to "feel".
- **Friction ellipse:** a tire has one grip budget, shared between accelerating/braking and turning. Use 80% of it to accelerate and only 60% is left for turning (`sqrt(1 − 0.8²)`).
- **Engine:** rear-wheel drive, power-limited (force = power / speed). **Brakes:** 60% front, 40% rear.
- **Air drag** grows with speed squared, and **wind** changes the air speed the car feels. Sideways wind pushes behind the center of gravity, so the car turns its nose into the wind like a weathervane (this surprised a test, and it's correct physics).
- **Steering lag** (the wheel takes time to turn) and **action delay** (commands arrive 0–4 steps late) model a laggy car.
- **Low-speed blending:** the tire formula divides by speed, which explodes near zero. Below 3 m/s we blend toward simple geometric ("kinematic") steering, where the car just follows its wheels.

**How we know it's right.** Ten sanity tests compare it with physics you can check by hand. Top speed matches the analytic value exactly (72.66 m/s). Dry braking from 30 m/s takes 40.6 m against v²/2μg = 42.5 m (air drag helps a little). Slow turns match the geometric radius. A puddle under the rear makes the car rotate more (oversteer), and under the front it rotates less (understeer). 10,000 random steps with every condition at its extreme stay finite. Running 4,096 cars at once gives the same answer as one at a time.

**One finding worth knowing for interviews.** With the spec's constants the car is almost exactly **neutral-steer**: front and rear tires lose grip at the same moment. It's also rear-wheel drive. So throttle in a corner uses up the rear tires' grip budget and the tail slides out (**power oversteer**). Coasting, the car is stable even past the limit. That makes throttle control part of what the AI must learn, especially in the rain.

---

## The environment (M1, `env.py`, `conditions.py`)

**What it is.** The "game" wrapped around the physics: where the car is on the track, what the driver gets to see, how it's scored, and when an episode ends.

- **Progress tracking.** Each step, find the nearest centerline point, but only search a window near where the car was last time (5 points back to 25 ahead). Without the window, a car near a hairpin could "teleport" its progress to the other side of the hairpin. Progress is the change in distance along the track, clipped to ±5 m per step.
- **Observations** (what the AI sees, in the car's own frame, scaled to about −1…1). 33 numbers for every driver: speed, sideways speed, rotation rate, steering angle, last action, distance from the center line, heading error, 12 points of the road ahead, and the track width. 14 "privileged" numbers that only the teacher and the training critic get: the true grip under each axle, puddles ahead, mass, wind and so on. A 1-second history (20 steps × 10 numbers) for the student.
- **Reward:** about 0.1 per meter of progress (so faster is better), small penalties for being off-center and for jerky inputs, +10 for finishing a lap, −10 for crashing.
- **Episodes end** on: off track, spinning for over 1 s, stalling for 3 s, driving the wrong way, finishing the lap (all "terminated": nothing comes after), or the 120 s time limit ("truncated": the future still had value, so training will estimate it instead of assuming zero). That difference matters for PPO in M3.
- **Conditions** are sampled per episode: rain, base grip, puddles, wind and gusts, tire wear, mass, power, brakes, drag, steering lag, delay, sensor noise. A **difficulty** number from 0 to 1 widens every range from "perfect day" to "full chaos" (the curriculum, M3).
- **Auto-reset:** when a car's episode ends, it immediately starts a new one on a new track. All cars always step together, which is what keeps the GPU busy.

---

## The scripted driver (M1, `scripted.py`)

**What it is.** A hand-written, non-learning driver that **knows the true grip**: a strong baseline, roughly a competent human with perfect road knowledge.

- **Steering: pure pursuit.** Pick a point on the center line ~1 s ahead and steer along the arc that hits it. Plus **yaw damping**: if the car is rotating faster than that arc needs, countersteer.
- **Speed plan:** for every point up to 160 m ahead, the fastest a corner allows is `sqrt(grip · g / curvature)`. Work backwards with a safe braking rate to get the speed allowed *now*.
- **Pedals:** ask for an acceleration proportional to the speed error, then convert it to throttle or brake using the car's real force limits. Throttle and brakes may only use the grip that cornering leaves free (the friction ellipse again: traction control and ABS).

**The debugging story (good interview material).** The first version crashed on every lap. Tracing one car showed: (1) it accelerated hard mid-corner and power-oversteered, so I made the pedals friction-circle aware; (2) the steering oscillated, because pure pursuit aimed only 7.7 m ahead, and each correction overshot more than the last until the car spun on a gentle curve, so I used a longer lookahead plus yaw damping; (3) it arrived at corners too fast because it braked too gently, so I planned with a conservative braking rate and a firmer speed controller; (4) the pedals then chattered between full brake and full throttle 20 times a second, because the controller's gain was in the wrong units, so I made it request an acceleration in m/s². Each fix came from looking at a trace, not from guessing. Result: 100% of dry laps and 99.1% of wet laps on 2,000 training tracks.

---

## The benchmark (M1, `bench.py`)

**What it measures.** How many car-steps per second the environment runs: on the GPU (JAX), on the CPU (JAX), and as a plain Python loop over cars (NumPy), using literally the same physics code for all three.

**Why JAX is fast here: three words.**
- `vmap`: write the code for one car; JAX turns it into code for 65,536 cars at once.
- `jit`: compile the whole step into one optimized GPU program instead of thousands of small operations.
- `lax.scan`: run 100 steps in a loop *on the GPU*, without coming back to Python between steps.

The Python loop is only measured up to 1,024 cars (the spec forbids extrapolating), so the speedup claim uses measured points only.

---

## The browser port and parity tests (M2, `web/src/sim/`, `web/tests/parity/`)

**What it is.** The tracks, physics, environment and scripted driver rewritten in TypeScript, line by line from the Python, so the website can run the very same simulation the AI trains in.

**Why it's needed.** A trained driver only works in the world it learned in. If the browser's car turned 1% differently, the AI would make mistakes that never happened in training, and nobody could tell why.

**How we prove they match ("parity tests").** Python computes thousands of random test cases: random cars, conditions and actions, plus the result after one step. These are saved as "golden" files. The TypeScript code replays every case and must land within 0.0001 (plus 0.01% relative) of Python. Measured: the worst of 2,000 physics cases uses **1.4% of that allowance**. A deliberately planted 1% change to rolling resistance makes the test fail, so the check is sensitive. Track generation matches to a millionth of a meter, and the random number generator matches bit for bit.

**The float32 trick.** Python/JAX computes in 32-bit floats; JavaScript computes in 64-bit. So after every 12.5 ms substep, the TypeScript rounds each state variable to 32 bits (`Math.fround`). Without that, tiny differences compound over thousands of steps.

**The guard.** If anyone changes the physics constants without regenerating the golden files, a Python test fails in CI and says which command to run.

---

## Running it live: worker, fixed timestep, interpolation (M2, `web/src/worker/`, `web/src/render/`)

**Web Worker.** The simulation runs on a separate browser thread, so heavy work never freezes scrolling or clicks.

**Fixed timestep.** The physics always steps exactly 20 times per simulated second, however fast the screen refreshes. The worker adds up elapsed time and takes as many 50 ms steps as fit (at most 4 at once, so a laggy moment can't snowball, the "spiral of death").

**Interpolation.** The screen refreshes 60+ times a second but physics only updates 20 times. So each frame blends between the last two physics snapshots: positions linearly, angles along the shorter way around the circle. That's why motion looks smooth.

**Rendering.** A 2D canvas, drawn in layers: textured runoff, track, kerbs, start line, puddles with ripples, skid marks, cars, headlights in the rain, and rain streaks blown by the wind. Things that rarely change are drawn once into a hidden canvas and copied each frame.

**A performance lesson.** The hero text panel had a slight blur of the canvas behind it. The browser has to re-blur that area every single frame because the canvas keeps changing, which cut the frame rate from 60 to 36 on a high-resolution screen. Removing one CSS property fixed it. Measure before guessing: the rain particles, which looked like the obvious suspect, weren't the problem.

**Results.** 60 fps with 48 cars, even on a retina-resolution canvas at full rain, with no main-thread task over 50 ms. Lighthouse (desktop): 100 performance, 100 accessibility, 100 best practices, 100 SEO.

---

## Learning to drive: PPO (M3, `ppo.py`, `networks.py`)

**What it is.** The driver is a small neural network (3 layers of 128) that takes the 33 observations and outputs steering and throttle/brake. PPO (Proximal Policy Optimization) improves it by trial and error: drive 1,024 cars for 64 steps, score how each decision turned out compared with what was expected, then nudge the network toward the better-than-expected decisions, but never far in one update (the "clip").

**The pieces, one line each.**
- **Actor and critic.** The actor picks actions. The critic predicts "how much reward is still coming from here", so the actor can tell good luck from good decisions.
- **Asymmetric critic.** The critic also sees the true conditions (grip, wind…). It only exists during training, so the deployed driver doesn't need them, but its predictions are better, which makes learning faster.
- **GAE (λ = 0.95).** A way to blend "what happened next step" with "what happened over the next many steps" when scoring a decision. λ near 1 trusts long-run outcomes; lower values trust the critic's guess more. 0.95 is the standard trade-off.
- **Terminated vs. truncated.** A crash ends the future (no more reward), so the score adds nothing after it. Hitting the 120 s time limit does *not* mean the future was worthless, so the score adds the critic's estimate of the state the car was in. Getting this wrong teaches the car that time limits are as bad as crashes.
- **Normalization.** Observations are rescaled using running averages, and rewards by the running spread of returns. These statistics are frozen and shipped with the weights, which is why the browser car drives like the Python one.
- **Why it's fast.** The whole loop (driving, scoring, updating) is one compiled JAX function, so it never returns to Python between steps (the PureJaxRL recipe).

**First run (overnight, laptop CPU).** At ~17k steps/s, the first 4M steps already took the car from crashing every episode to finishing ~90% of laps on dry roads.

## The automatic curriculum (M3, `curriculum.py`)

A single dial from 0 (perfect dry day) to 1 (all conditions at their worst training values). Every 10 updates: if more than 80% of episodes ended without a crash, twice in a row, the dial goes up 0.05; below 50% it goes down. 20% of cars always drive at difficulty 0 so easy skills aren't forgotten, and 10% get one condition pushed to the edge.

**The honest result.** With the overnight budget the dial reached 0.55 (blind) and 0.45 (teacher). So the drivers never trained on heavy rain, 2-step input delay or strong sensor noise, and they fail sharply there: the blind driver finishes 95% of laps at sensor noise 0.02 but 30% at 0.05 and 0% at 0.15. "Learned policies fall off a cliff outside their training range" is a real, measured finding, and the main reason to rerun at GPU scale.

**The GPU rerun.** At the design's budgets on one Kaggle T4 (blind 8 × 150M, teacher 16 × 100M), the dial reached 0.55 (blind) and 0.70 (teacher), still short of 1. The cliff beyond the reached range is still there; the Training lab and Results pages show the published run.

## Population-based training (M4, `pbt.py`)

Four teacher drivers trained side by side (16 in the full design). Every 25 updates they're ranked by progress per second. The worst one copies the best one's network, optimizer and normalization, then multiplies its learning rate and exploration bonus by 0.8 or 1.25 (or redraws them 25% of the time). Each copy gets a new id, so the history is a family tree (Training lab). It tunes hyperparameters during training instead of running dozens of separate experiments.

## The teacher and the student (M4–M5, `student.py`)

**Teacher.** Like the blind driver, but it's *told* the true conditions through a small encoder that squeezes 14 facts into 8 numbers ("z").

**Student (Rapid Motor Adaptation).** A second network reads the last second of motion (20 steps of speed, sideways slip, rotation, steering and pedals) and learns to output the same 8 numbers ("ẑ"), plus a direct estimate of the grip under each axle. The teacher's driving network then drives on ẑ instead of z. Training is DAgger-style: the car drives on the *student's* estimates, so the student learns in the situations it actually causes, not just the ones the teacher would reach.

**The honest result (overnight CPU run; the site now shows the GPU rerun, where the pattern is similar: better on wet, puddles, worn tires and a very heavy car, worse on dry, wind and noisy sensors, and still no puddle detection).** The grip estimate is good (error ≈ 0.05–0.12 μ). But on a straight at steady speed it does *not* notice a puddle (0 of 200 trials): the tires use almost no grip there, so the motion looks the same as on dry road. (An earlier version of my metric reported "350 ms"; it was measuring when the car left the puddle. Fixed. A second loophole showed up on the GPU run: it counted a dip in the estimate seconds after the puddle, which a no-puddle control showed was a knock-on wobble. Now only responses while grip is low, or within 0.5 s, count, and the control's false alarms are reported.) And it beats the blind driver exactly where grip changes: puddles (14.5% vs 0.7% of laps), wet (28% vs 5%), worn tires (84% vs 75%). But its estimate of the full 8-number summary stayed rough (loss ≈ 0.31). Things like mass, drag and wind are hard to feel in one second. Partly as a result, it does worse than the blind driver in dry, windy and heavy-car conditions. Also, each teacher trained on half as many steps as the blind driver, which confounds the comparison. Both are fixable with more compute.

## Knowing when it's unsure (M5–M6, `confidence.py`)

Five students trained from different random seeds. If their ẑ estimates (and the actions those would lead to) disagree, the car is probably in an unfamiliar situation. That disagreement score is turned into "probability of crashing in the next 2 s" with isotonic regression: a monotone staircase fit to real crash outcomes on the validation tracks. The safety governor eases off the throttle above a risk threshold.

**The honest result.** The probability is *calibrated*: expected calibration error 0.026, so "20% risk" really means about 20%. But it's only weakly *predictive* (AUROC 0.64, where 0.5 is a coin flip). Crashes mostly happen in conditions where all five students are confidently wrong together. The governor made no measurable difference at any threshold. An interviewer will like that you know the difference between calibration and discrimination.

## The evaluation (M6, `evaluate.py`)

- 200 test tracks the drivers never saw, 12 condition suites (4 never trained on), 2 condition draws per track = 400 laps per driver per suite.
- **Paired.** Every driver drives the identical track and condition pairs, so differences come from the drivers, not luck.
- **Cluster bootstrap.** To get a 95% interval, resample *tracks* (not laps) 2,000 times and recompute. Laps on the same track aren't independent, so resampling laps would make the intervals look falsely narrow.
- **Paired lap time.** Lap times are compared only on laps every driver finished; otherwise a cautious driver that crashes on hard tracks looks fast.
- **Hygiene.** The governor threshold and the calibration were chosen on the validation tracks. The test tracks were used once, for the final numbers.
- **A bug I caught.** The first evaluation showed 2.5% completion on training tracks vs 30% on test tracks, which made no sense. My cache of compiled rollout functions reused one compiled for the test tracks when running the training tracks. The fix: pass the track data as an argument instead of capturing it. Now there's a regression test for it.

---

## The Playground's tools (M7, `web/src/pages/Playground.tsx`, `web/src/render/layers/overlays.ts`)

- **Puddle test.** Loads the exact straight the evaluation uses (ported from Python and checked against its length and curvature), puts one puddle 200 m ahead, runs the student at half speed and scores it with the same rule as the evaluation (`state/puddleTest.ts` mirrors `notice_latencies`). It reports whatever happens; the published student doesn't notice.
- **Crash toast.** The worker remembers, per car, the speed when things went wrong (leaving the track, or the start of a spin) and a few causes (puddle in the last 1.5 s, mid-corner, rain, governor on), so the page can say what happened in one sentence.
- **Chase camera and tire forces.** Chase follows the focused car north-up with a critically damped spring, so it never jerks or rotates. The arrows come from a read-only buffer the physics writes each substep; nothing reads it back, so the parity tests still guard the physics.
- **Draw a track.** Your stroke is simplified (Ramer–Douglas–Peucker), scaled to a legal length, splined and checked by the same validator as generated tracks. Auto-fix smooths it and re-fits the length each pass, because smoothing shrinks the loop.
- **Phones.** Canvas first, a three-number telemetry strip, the conditions in a bottom sheet, 16 cars and fewer rain streaks.

**Bugs this stage found.** The slow-device fallback silently swapped the track back to the original; a closed stroke that ended exactly on its start collapsed to NaN and was then *accepted* as valid (NaN comparisons are false); and the Lab briefly said "Not computed yet" while its data was still loading.

## 21 likely interview questions

1. **Why PPO and not DQN?** Steering and throttle are continuous; DQN needs discrete actions. PPO is stable, simple, and works well with thousands of parallel environments.
2. **What is GAE and why λ = 0.95?** A weighted blend of short- and long-horizon advantage estimates. λ trades bias (trusting the critic) against variance (trusting noisy long returns); 0.95 is the well-tested default for continuous control.
3. **Terminated vs truncated, and why it matters?** Terminated (crash) means no future reward, so no bootstrap. Truncated (time limit) means the future still had value, so bootstrap from V(final state). Mixing them up teaches the agent that time limits are crashes.
4. **Why JAX? What do vmap, jit and scan do here?** vmap turns one-car code into many-car code; jit compiles the whole step into one GPU program; scan runs the rollout loop on the device. Result: 26M car-steps/s on a T4, 926× the same code on its CPU.
5. **How did you make the browser and Python physics match?** Same algorithms in the same operation order, 32-bit rounding after every substep, and "golden" test files computed in Python that the TypeScript must reproduce (worst case at 1.4% of a 1e-4 tolerance). A planted 1% physics change makes the tests fail.
6. **Can the student feel a puddle?** Not reliably. In controlled tests it noticed a single puddle in 0 of 200 runs on a straight and 0 of 100 mid-corner: the puddle passes in under a second and barely changes the motion the student reads. It still finishes more wet and puddle laps than the blind driver; I haven't isolated why. A plausible but untested explanation is caution inherited from the teacher it imitates, which also does better there. That's the next experiment.
7. **How does the student learn without being told the conditions?** It learns to infer the teacher's 8-number summary of the conditions from one second of motion, supervised by the teacher, while driving on its own estimates (DAgger).
8. **Why train a teacher first instead of the student directly?** Learning to drive and learning to sense at the same time is hard. With privileged information the teacher learns quickly; the student's job becomes plain supervised learning.
9. **What's the asymmetric critic?** The critic sees privileged conditions during training. Better value estimates, same deployable actor.
10. **How do you know the risk estimate is calibrated?** Reliability diagram on unseen test tracks: in each of 10 bins the predicted crash rate matches the observed one (ECE 0.026). Calibrated isn't the same as discriminative, though: AUROC is only 0.64.
11. **Why didn't the safety governor help?** Most crashes happen where all five students agree and are all wrong (out-of-distribution conditions), so disagreement doesn't flag them. Ensembles that share one teacher capture uncertainty about the road, not about the driving.
12. **Why a cluster bootstrap over tracks?** Laps on the same track are correlated; resampling laps would understate uncertainty.
13. **What's PBT doing that a grid search wouldn't?** It adapts hyperparameters during one training run, and the schedule itself can change over time (exploit/explore every 25 updates).
14. **What's the curriculum inspired by, and how is yours different?** Automatic Domain Randomization; mine is one global dial, not per-parameter ranges.
15. **Why does performance collapse outside the training range?** The network never saw those inputs; small out-of-range shifts (noise 0.02 → 0.05) cut completion from 95% to 30%. More curriculum (more compute) is the fix, plus domain randomization wider than the test range.
16. **Why does the student lose to the blind driver in some conditions?** Its estimate of the conditions is imperfect (mass, drag and wind are hard to feel in one second), and the teacher it imitates trained for fewer steps (two-thirds, in the GPU run). Fair comparisons need matched compute, and that's the first thing I'd fix.
17. **How did you avoid fooling yourself in evaluation?** Paired comparisons, held-out tracks, validation-only tuning, survivor-bias-free lap times, CIs, and reporting every suite, including the losses.
18. **What was the hardest bug?** The scripted driver crashed on every lap: power oversteer, then pure-pursuit oscillation, then late braking, then pedal chatter. Each was found from a per-step trace, not guessed. Also the evaluation cache bug that silently simulated the wrong tracks.
19. **How does the browser run 48 neural networks at 60 fps?** A Web Worker runs physics at a fixed 20 Hz and batches every car through a hand-written MLP (no libraries). The main thread only draws, blending between physics snapshots.
20. **What would you do with 10× more compute?** The full design budgets have been run once; next, repeat seeds, train long enough for the curriculum to reach full difficulty, match blind and teacher budgets, then test whether adaptation's advantage holds across all suites.
21. **What breaks in the real world?** Real tires, suspension, load transfer, sensor latency and failures, 3D roads, other cars. This is a simulated study of adaptive control, not a self-driving system.
