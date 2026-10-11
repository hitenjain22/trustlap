# Overnight log (2026-10-11 → morning)

Hiten went to sleep at ~2:40am and asked Claude Code to keep building autonomously: no approval gates, decisions logged in `DECISIONS.md`, self-push after local tests pass. This file is the morning summary: newest status at the top.

## Status (12:50)

**All seven milestones are built and live** at https://trustlap.vercel.app (build `3488268`, CI green: Python 89 tests, web 35 unit/parity tests, 4 browser tests; Lighthouse 100s at M2). Pages: Home (48 self-taught cars), Playground (pick driver, add ghosts, condition sliders), Training lab (live checkpoints, population, family tree, curriculum), Results (every metric with 95% CIs), How it works (method, model card, reproduction).

**Honest note:** the Claude session stalled at about 03:50 (most likely a usage limit) and the restart timer did *not* bring it back; work resumed at 11:40 when you messaged. The detached training chain kept going on its own and finished by 05:20.

### What the results say (all from metrics.json; laptop-CPU training, ~160M env steps total)
- **Simulator:** 65,536 cars at once on a T4, 926× the same code on its CPU.
- **Where grip changes, the adaptive student beats the non-adaptive (blind) driver:** puddles 14% vs 1% of laps, wet 28% vs 5%, worn tires 84% vs 75%.
- **Elsewhere it loses at this compute:** in "everything at once" it crashes 8% more than blind (95% CI 2–15% more); blind wins on dry, wind and heavy-car suites.
- **Why:** with CPU budgets the curriculum reached only ~half of full difficulty, so heavy rain, 2-step delay and strong sensor noise are beyond what any driver trained on, and they fail sharply there (verified not a bug: inside the range they're fine). Also each teacher had half the blind driver's steps (a confound).
- **The student does *not* notice single puddles** (0/200 on a straight, 0/100 mid-corner). I first reported "350 ms"; that metric was wrong (it also fired when the car left the puddle) and is corrected everywhere.
- **Crash-risk estimate:** calibrated (ECE 0.026) but weakly predictive (AUROC 0.64); the safety governor had no measurable effect.

## Things that need you

1. **Run the full-size GPU training** (the single biggest improvement available): Kaggle notebook `sim/notebooks/train_full.ipynb`, step-by-step in `docs/RUNBOOK.md` → "Full-size GPU run on Kaggle". It runs in the background (~several hours); then send me the zip and I'll publish the new results.
2. **Read** `docs/EXPLAINER.md` (21 interview questions with answers) and `docs/DECISIONS.md` (everything I decided overnight).
3. **Delete** `TRUSTLAP_SPEC.md` in the repo root (duplicate of `docs/SPEC.md`), if you like.

### Known follow-ups (not blocking)
- One-off main-thread long tasks of 50–150 ms during page load on retina screens (steady play is clean).
- Draw-a-track, URL state, the full telemetry panel and puddle-test button in the Playground (SPEC 10.2/10.5) aren't built yet.
- README GIF and demo video (SPEC 16) need a screen recording.

## Timeline

- 02:45 Overnight mode started: `caffeinate` keeps the Mac awake; a session timer (minutes 7/27/47) re-prompts Claude if it goes idle (e.g. after a usage limit).
- 02:47 Confirmed M2 on GitHub + CI green + live site working.
- 03:00 M3 code: networks, PureJaxRL-style PPO (GAE with terminated/truncated, obs/reward normalization), curriculum, PBT, training loop with checkpoints. Tests: GAE by hand, running stats, PBT copy/lineage, curriculum hysteresis, a learning smoke test.
- 03:05 CPU throughput ≈ 17k env steps/s (1,024 envs). A 4M-step trial already learned to lap at d = 0 (≈ 90% laps).
- 03:10 Blind training started (40M steps, seed 1, `sim/runs/blind/`).
- 03:20 Export of networks (SPEC 9.2: float32 .bin + json + sha256) and TypeScript inference (`web/src/policy/`); policy + closed-loop parity tests pass. Hero and Playground now drive with the network (Playground: pick driver, add ghosts).
- 03:35 Blind training at update 170/610: difficulty 0.25, 96% laps completed in training.
- 03:46 Blind training done (40M steps, 44 min, final difficulty 0.55).
- ~03:50 Session stalled (see note above).
- 04:55 Teacher PBT done (4 members × 20M steps, 68 min, final difficulty 0.45).
- 05:20 Students done (5 × 8M steps, 25 min; grip MAE ≈ 0.055; ẑ loss plateaued ≈ 0.31).
- 11:40 Resumed. Evaluation (M6) started on the 200 test tracks (2 condition draws per track, a SPEC 7.9 reduction).
- 12:00 Results, Training lab, How it works pages; OG image; README with generated numbers.
- 12:20 Found and fixed an evaluation bug (rollout cache reused one bank's geometry for another); reran the full evaluation.
- 12:40 Found the puddle-latency metric was wrong; corrected; the student does not detect single puddles. Pushed `3488268`; CI green; all five pages verified live.
