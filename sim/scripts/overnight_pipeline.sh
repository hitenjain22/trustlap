#!/bin/zsh
# Overnight CPU training chain (2026-10-11): blind (already running) → teacher PBT → students.
# Each stage resumes from its checkpoint if restarted. Logs in runs/*.log.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
while pgrep -f "trustlap.train --agent blind" >/dev/null; do sleep 30; done
echo "$(date) blind finished" >> runs/pipeline.log
if [ ! -f runs/teacher/final.msgpack ]; then
  python -m trustlap.train --agent teacher --population 4 --env-steps 20e6 --out runs/teacher --seed 2 >> runs/teacher_train.log 2>&1
  echo "$(date) teacher finished (exit $?)" >> runs/pipeline.log
fi
if [ ! -f runs/students/done ]; then
  python -m trustlap.student --teacher runs/teacher/final.msgpack --out runs/students --env-steps 8e6 --members 5 --envs 512 --seed 3 >> runs/students_train.log 2>&1 \
    && touch runs/students/done
  echo "$(date) students finished" >> runs/pipeline.log
fi
