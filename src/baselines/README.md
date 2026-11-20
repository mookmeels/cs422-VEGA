# baselines: the simple methods every model is compared against

Owner: shared. `baselines.py` runs three baselines and scores them with the shared scorer, like any other model. None of them uses random numbers, so there are no seeds.

| Baseline | What it does |
|---|---|
| Popularity | Always suggests the labels with the most training images (the "majority class" baseline). |
| Nearest centroid | Averages the Qwen vectors of each label's training images, then suggests the labels whose average is closest to the image. |
| Zero-shot text | Compares the image vector with the Qwen vector of each aesthetic's wiki introduction. Uses no training labels. |

`baselines.py` is also the shortest example of how a model plugs into the shared code in `../common`.

## Running (on ml-db, from this folder)

    PY=/srv/ml/vega/.venv/bin/python
    $PY baselines.py                  # validation, a few seconds
    $PY ../common/vega_table.py       # rebuild the comparison table

- Runs append rows to `/srv/ml/vega/runs/comparison/results.jsonl`. Add `--out <folder>` to try it without touching the team's table.
- `--final` scores the **test split**. It is for the one final run, after the team has frozen every model. It has not been run.

## Results so far

Validation split (1,351 images), 503 labels, measured 20 Nov 2026, after one validation image that was a near-copy of a test image moved to test:

| | top-5 | top-1 | MRR | per-label top-5 |
|---|---|---|---|---|
| Nearest centroid | 80.2 | 52.2 | 0.647 | 80.6 |
| Zero-shot text | 42.8 | 22.9 | 0.330 | 43.4 |
| Popularity | 5.8 | 1.4 | 0.047 | 1.0 |

Nearest centroid is the number a trained model has to beat. **The test split has not been scored.**

## This is a copy

Copied on 20 Nov 2026 from `/srv/ml/vega/code/baselines/baselines.py` on ml-db, where the recorded runs were made. One difference: this copy looks for the shared code in `../common` as well as in `../shared` (three lines near the top). Run from this folder on the validation split, it gave the same scores as the recorded run, to the last digit (checked again on 20 Nov, after the split fix).

Versions: Python 3.12.3, numpy 2.5.5, polars 2.1.0.

Generative-AI use: this file was written by Claude Code (Anthropic) at Andrew's direction. `/srv/ml/vega/handoff/WORKLOG.md` (17 Nov) records what was run and how it was checked.
