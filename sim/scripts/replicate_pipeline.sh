#!/bin/bash
# Equal-practice replication run (GPU): for each training seed, banks → Basic AI (blind, PBT 8 × 150M)
# → Teacher AI (PBT 8 × 150M, the same budget) → Adaptive AI students (5 × 80M) → evaluation →
# fine-tuning the driving network on the students' estimate → evaluation with the dry-road
# fallback; every evaluation uses the same test tracks and conditions. Seed 1 is the published
# one (fixed before any result). Every stage skips itself if its output exists, so a run attached
# from an earlier session continues where it stopped.
# Usage: scripts/replicate_pipeline.sh [OUT_DIR]   (default runs/rep; SEEDS="1 2 3")
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-runs/rep}
SEEDS=${SEEDS:-"1 2 3"}
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/pipeline.log"; }

[ -f runs/banks/test.npz ] || { log "building track banks"; python -m trustlap.track --out runs/banks; }
for s in $SEEDS; do
  D="$OUT/seed$s"
  [ -f "$D/blind/final.msgpack" ] || { log "seed $s: Basic AI"; python -m trustlap.train --agent blind --population 8 --env-steps 150e6 --out "$D/blind" --seed $((s * 10 + 1)); }
  [ -f "$D/teacher/final.msgpack" ] || { log "seed $s: Teacher AI"; python -m trustlap.train --agent teacher --population 8 --env-steps 150e6 --out "$D/teacher" --seed $((s * 10 + 2)); }
  [ -f "$D/students/done" ] || { log "seed $s: Adaptive AI"; python -m trustlap.student --teacher "$D/teacher/final.msgpack" --out "$D/students" --env-steps 80e6 --members 5 --envs 1024 --seed $((s * 10 + 3)) && touch "$D/students/done"; }
  [ -f "$D/eval/metrics_partial.json" ] || { log "seed $s: evaluation"; python -m trustlap.evaluate --blind "$D/blind/final.msgpack" --teacher "$D/teacher/final.msgpack" --students "$D/students/students.msgpack" --seeds 3 --seed 0 --out "$D/eval"; }
  # Stage 2: fine-tune the driving network on the students' own estimate, then evaluate it
  # together with the dry-road fallback (threshold chosen on the validation bank).
  [ -f "$D/finetune/actor.msgpack" ] || { log "seed $s: fine-tune on the students' estimate"; python -m trustlap.finetune --teacher "$D/teacher/final.msgpack" --students "$D/students/students.msgpack" --out "$D/finetune" --env-steps 100e6 --envs 1024 --seed $((s * 10 + 4)); }
  [ -f "$D/eval_ft/metrics_partial.json" ] || { log "seed $s: evaluation (fine-tuned + fallback)"; python -m trustlap.evaluate --blind "$D/blind/final.msgpack" --teacher "$D/teacher/final.msgpack" --students "$D/students/students.msgpack" --student-actor "$D/finetune/actor.msgpack" --hybrid --seeds 3 --seed 0 --out "$D/eval_ft"; }
done
log "combining seeds"
# Before fine-tuning (kept for the honest before/after comparison) and the published version.
python -m trustlap.replicate --root "$OUT" --seeds $SEEDS --primary 1 --eval-dir eval --out "$OUT/metrics_base.json"
python -m trustlap.replicate --root "$OUT" --seeds $SEEDS --primary 1 --eval-dir eval_ft --out "$OUT/metrics_partial.json"
log "done"
