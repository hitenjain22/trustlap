#!/bin/bash
# Equal-practice replication run (GPU): for each training seed, banks → Basic AI (blind, PBT 8 × 150M)
# → Teacher AI (PBT 8 × 150M, the same budget) → Adaptive AI students (5 × 80M) → evaluation on
# the same test tracks and conditions; then one combined metrics file. Seed 1 is the published
# one (fixed before any result). Every stage resumes from its checkpoint.
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
done
log "combining seeds"
python -m trustlap.replicate --root "$OUT" --seeds $SEEDS --primary 1 --out "$OUT/metrics_partial.json"
log "done"
