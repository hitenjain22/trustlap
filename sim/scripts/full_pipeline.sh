#!/bin/bash
# Full-size training at SPEC 7.9 budgets (GPU): banks → blind → teacher PBT → students → evaluation.
# Every stage resumes from its checkpoint, so re-running after a disconnect continues the work.
# Usage: scripts/full_pipeline.sh [OUT_DIR]   (default runs/full)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-runs/full}
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/pipeline.log"; }

[ -f runs/banks/test.npz ] || { log "building track banks"; python -m trustlap.track --out runs/banks; }
[ -f "$OUT/blind/final.msgpack" ] || { log "blind"; python -m trustlap.train --agent blind --population 8 --env-steps 150e6 --out "$OUT/blind" --seed 1; }
[ -f "$OUT/teacher/final.msgpack" ] || { log "teacher PBT"; python -m trustlap.train --agent teacher --population 16 --env-steps 100e6 --out "$OUT/teacher" --seed 2; }
[ -f "$OUT/students/done" ] || { log "students"; python -m trustlap.student --teacher "$OUT/teacher/final.msgpack" --out "$OUT/students" --env-steps 40e6 --members 5 --envs 1024 --seed 3 && touch "$OUT/students/done"; }
[ -f "$OUT/eval/metrics_partial.json" ] || { log "evaluation"; python -m trustlap.evaluate --blind "$OUT/blind/final.msgpack" --teacher "$OUT/teacher/final.msgpack" --students "$OUT/students/students.msgpack" --seeds 3 --out "$OUT/eval"; }
log "done"
