#!/bin/bash
# The CNN's final test scoring: once. Owner of the model: Giancarlo Guevarra. Run by Andrew on ml-db.
# Scores the three CNN rows of the validation table on the test split, from the weights trained on Gianni's PC
# with train_cnn.py --final --fit-only (option A: no test picture left ml-db). Refuses if a CNN test scoring is
# already logged. The plain CNN has test predictions for seed 422 only, so seeds 423 to 426 are predicted first
# with Giancarlo's predict_cnn.py (which never scores).
set -euo pipefail
ROOT=/srv/ml/vega
PY=$ROOT/.venv/bin/python
CMP=$ROOT/runs/comparison
PRED=$ROOT/runs/cnn/final/preds
WEIGHTS=$ROOT/runs/cnn/final-from-pc
LOG=$CMP/cnn-final-2026-11-22.log
OWNER="Giancarlo Guevarra"
NOTE="final run, option A: weights trained on train + validation on Gianni's PC, test predicted on ml-db with predict_cnn.py"
START=$(date +%s)
if grep -q "CNN" "$CMP/FINAL-RUNS.log"; then echo "REFUSED: a CNN test scoring is already in FINAL-RUNS.log"; exit 1; fi
if [ -e "$LOG" ]; then echo "REFUSED: $LOG exists"; exit 1; fi

{
  echo "The CNN's final test scoring. Started $(date -u '+%Y-%m-%d %H:%M:%S UTC') on $(hostname)."
  echo "code (sha256):"
  sha256sum $ROOT/code/cnn/predict_cnn.py $ROOT/code/cnn/train_cnn.py $ROOT/code/shared/vega_data.py $ROOT/code/shared/vega_score.py
  echo

  missing=()
  for s in 423 424 425 426; do
    [ -e "$PRED/cnn-r18w64-224-test-merged-seed$s.npz" ] || missing+=("$WEIGHTS/cnn-r18w64-224-test-seed$s.pt")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "Predicting the plain CNN's missing seeds (${#missing[@]} weights files):"
    ( cd $ROOT/code/cnn && $PY -B predict_cnn.py --weights "${missing[@]}" )
    echo
  fi

  for tag in cnn-r18w64-224 cnn-r18w64-224-street cnn-r18w64-224-street-crop60-st15-mix-ls10-os; do
    for s in 422 423 424 425 426; do
      [ -e "$PRED/$tag-test-merged-seed$s.npz" ] || { echo "MISSING: $PRED/$tag-test-merged-seed$s.npz; nothing scored"; exit 1; }
    done
  done
  echo "prediction files (sha256):"
  sha256sum $PRED/*.npz
  echo

  cd $ROOT/code/shared
  for s in 422 423 424 425 426; do
    $PY -B vega_score.py --final --owner "$OWNER" --seed $s --note "$NOTE" \
        --pred "$PRED/cnn-r18w64-224-test-merged-seed$s.npz" --model "CNN (ResNet-18, 224 px)"
  done
  for s in 422 423 424 425 426; do
    $PY -B vega_score.py --final --owner "$OWNER" --seed $s --note "$NOTE" \
        --pred "$PRED/cnn-r18w64-224-street-test-merged-seed$s.npz" --model "CNN (ResNet-18, 224 px) + street photos"
  done
  for s in 422 423 424 425 426; do
    $PY -B vega_score.py --final --owner "$OWNER" --seed $s --note "$NOTE" \
        --pred "$PRED/cnn-r18w64-224-street-crop60-st15-mix-ls10-os-test-merged-seed$s.npz" \
        --model "CNN (ResNet-18, 224 px, random crop 0.6, stretch 0.15, mixup/CutMix, label smoothing 0.1, oversampling) + street photos"
  done
  echo
  $PY -B vega_table.py
  echo "Finished $(date -u '+%Y-%m-%d %H:%M:%S UTC')."
} 2>&1 | tee "$LOG"
echo "---- FINAL-RUNS.log"
cat "$CMP/FINAL-RUNS.log"
echo "Time elapsed: $(( $(date +%s) - START )) s"
