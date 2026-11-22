#!/bin/sh
# The NN's one final test run: the two commands written down on 21 Nov (WORKLOG, "BOTH NNs get a final test run"), then the table.
# It refuses to run before the freeze (22 Nov 02:00 UTC = Saturday 6 pm Pacific) and refuses to run a second time.
set -eu
ROOT=/srv/ml/vega
PY=$ROOT/.venv/bin/python
OUT=$ROOT/runs/comparison
LOG=$OUT/nn-final-2026-11-22.log
FREEZE=$(date -u -d "2026-11-22 02:00:00" +%s)
WAIT=$(( FREEZE - $(date -u +%s) ))
if [ "$WAIT" -gt 900 ]; then echo "more than 15 minutes before the freeze (22 Nov 02:00 UTC): not running"; exit 1; fi
if [ "$WAIT" -gt 0 ]; then sleep "$WAIT"; fi                  # started a few minutes early: wait for the freeze, then run
if [ -e "$OUT/FINAL-RUNS.log" ] && grep -q "NN (MLP" "$OUT/FINAL-RUNS.log"; then echo "the NN already has final rows: not running again"; exit 1; fi
if [ -e "$LOG" ]; then echo "$LOG exists: not running again"; exit 1; fi
cd $ROOT/code/nn
{
  echo "The NN's final test run. Started $(date -u '+%Y-%m-%d %H:%M:%S UTC') on $(hostname). Andrew, 21 Nov about 5:50 pm Pacific: Yes, run it."
  echo "code and tables (sha256):"
  sha256sum train_nn.py ../shared/vega_data.py ../shared/vega_score.py $ROOT/runs/family-v1/label_family.csv $ROOT/runs/family-v1/outside_family.csv
  echo "team repository: $(git -C /srv/ml/cs422-VEGA log --oneline -1)"
  echo
  $PY train_nn.py --seed 422 423 424 425 426 --final
  echo
  $PY train_nn.py --seed 422 423 424 425 426 --final --real-weight 0
  echo
  $PY ../shared/vega_table.py
  echo "Finished $(date -u '+%Y-%m-%d %H:%M:%S UTC')."
} > "$LOG" 2>&1
