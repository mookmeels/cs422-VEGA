#!/bin/bash
# The baselines' final test run: once. Refuses if a baseline test scoring is already logged.
set -euo pipefail
ROOT=/srv/ml/vega
PY=$ROOT/.venv/bin/python
CMP=$ROOT/runs/comparison
LOG=$CMP/baselines-final-2026-11-22.log
START=$(date +%s)
if grep -q "Baseline" "$CMP/FINAL-RUNS.log"; then echo "REFUSED: a baseline test scoring is already in FINAL-RUNS.log"; exit 1; fi
if [ -e "$LOG" ]; then echo "REFUSED: $LOG exists"; exit 1; fi
cd $ROOT/code/baselines


{
  echo "The baselines' final test run. Started $(date -u '+%Y-%m-%d %H:%M:%S UTC') on $(hostname). Andrew, 21 Nov about 11:10 pm Pacific: Yes, run them."
  echo "code (sha256):"
  sha256sum baselines.py ../shared/vega_data.py ../shared/vega_score.py
  echo
  $PY -B baselines.py --final
  echo
  $PY -B ../shared/vega_table.py
  echo "Finished $(date -u '+%Y-%m-%d %H:%M:%S UTC')."
} 2>&1 | tee "$LOG"
echo "---- FINAL-RUNS.log"
cat "$CMP/FINAL-RUNS.log"
echo "Time elapsed: $(( $(date +%s) - START )) s"
