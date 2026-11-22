#!/bin/bash
# The clustering model's final test run: once. Owner of the model: Edmond Luong (Si_mpl). Run by Andrew on ml-db.
# Runs Edmond's hclustering.py as committed (standardized vectors, cut 0.16), unedited, with the loader's final
# switch (fit = train + validation, eval = test), then scores the predictions on the test split once.
# Refuses if a clustering test scoring is already logged.
set -euo pipefail
ROOT=/srv/ml/vega
PY=$ROOT/.venv/bin/python
CMP=$ROOT/runs/comparison
OUT=$ROOT/runs/hclustering-final
LOG=$CMP/hclustering-final-2026-11-22.log
SCRIPT=/srv/ml/cs422-VEGA/src/models/hclustering/hclustering.py
WANT=8b883aaa08d24d1608f6458d7ea61a40ed5db3e79826afac21de9c18cfd5ae53
START=$(date +%s)
if grep -q "Hclustering" "$CMP/FINAL-RUNS.log"; then echo "REFUSED: a clustering test scoring is already in FINAL-RUNS.log"; exit 1; fi
if [ -e "$LOG" ]; then echo "REFUSED: $LOG exists"; exit 1; fi
if [ "$(sha256sum $SCRIPT | cut -d' ' -f1)" != "$WANT" ]; then echo "REFUSED: hclustering.py is not the version checked on validation"; exit 1; fi

{
  echo "The clustering model's final test run. Started $(date -u '+%Y-%m-%d %H:%M:%S UTC') on $(hostname)."
  echo "code (sha256):"
  sha256sum $SCRIPT $ROOT/code/experiments/hclustering_run.py $ROOT/code/shared/vega_data.py $ROOT/code/shared/vega_score.py
  echo
  $PY -B $ROOT/code/experiments/hclustering_run.py --final --out $OUT
  echo
  cd $ROOT/code/shared
  $PY -B vega_score.py --final --owner "Si_mpl" --pred $OUT/my-preds.npz --model "Hclustering(standardized 0.16)" \
      --note "final run: hclustering.py as committed (16bc914), unedited, loader switched to final; no randomness, so one run"
  echo
  $PY -B vega_table.py
  echo "Finished $(date -u '+%Y-%m-%d %H:%M:%S UTC')."
} 2>&1 | tee "$LOG"
echo "---- FINAL-RUNS.log"
cat "$CMP/FINAL-RUNS.log"
echo "Time elapsed: $(( $(date +%s) - START )) s"
