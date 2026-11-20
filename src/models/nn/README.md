# NN: a small neural network on the Qwen image vectors

Owner: Andrew Nguyen. One of the team's three models. It reads its data from the team's shared loader and is scored by the shared scorer, so its numbers sit in the same table as the other models.

## What it is

Every image has already been turned into 2,048 numbers by the frozen Qwen3-VL-Embedding-2B model (the team's shared vectors). The NN takes those numbers and gives a score to each of the 503 labels. It is a multilayer perceptron:

    2,048 inputs -> scale each input -> one hidden layer of 1,024 units (ReLU, dropout 0.7) -> 503 scores

- **Input scaling:** mean and standard deviation of each input, measured on the training images only.
- **Loss:** cross-entropy against the image's labels. An image in two galleries has half its weight on each label. Every label counts alike, whatever its number of images.
- **Training:** AdamW, peak learning rate 1e-3 (linear warm-up, then a cosine down to zero), weight decay 1, batch 512, 40 epochs. No early stopping.

The docstring at the top of `train_nn.py` explains each choice.

## Files

| File | What it does |
|---|---|
| `train_nn.py` | Trains and scores the NN, one run per `--seed`. The chosen settings are its defaults. |
| `sweep_nn.py` | How the settings were chosen: the full list of configurations tried, on validation only. |
| `check_nn.py` | Checks `train_nn.py` on made-up data. Reads no real image, label or split. |
| `analyse_errors.py` | Error analysis on validation: tables, and example images with `--sheets`. |

These are the same files as `/srv/ml/vega/code/nn/` on ml-db, where the recorded runs were made.

## Running it (on ml-db, from this folder)

    PY=/srv/ml/vega/.venv/bin/python
    $PY check_nn.py                                # should end with "all checks passed"
    $PY train_nn.py --seed 422 423 424 425 426     # validation, about 1 minute per seed
    $PY ../../common/vega_table.py                 # rebuild the comparison table
    $PY analyse_errors.py --sheets                 # needs the five runs above

- Runs append a row to `/srv/ml/vega/runs/comparison/results.jsonl`. Add `--out <folder>` to try something without touching the team's table.
- `--final` scores the **test split**. It is for the one final run, after the team has frozen every model. It has not been run.
- The loader and scorer (`vega_data.py`, `vega_score.py`) are in `src/common`, and the scripts find them there. The data is not in git: it is read from `/srv/ml/vega` on ml-db, or from the folder `VEGA_ROOT` names on another machine.

## Results so far

Validation split, 503 labels, seeds 422 to 426 (mean ± standard deviation over the five seeds), measured 17 Nov 2026:

| | top-5 | top-1 | MRR | per-label top-5 |
|---|---|---|---|---|
| NN | 87.6 ± 0.2 | 62.2 ± 0.6 | 0.731 ± 0.003 | 85.7 ± 0.5 |
| Baseline: nearest centroid | 80.3 | 52.1 | 0.647 | 80.6 |

**The test split has not been scored.** The full table is `/srv/ml/vega/runs/comparison/table.md`.

## How the settings were chosen

On the validation split only: 146 runs over 32 configurations, changing one setting at a time and then combining the best. The winning combination was then run on five fresh seeds (427 to 431) under a rule written into `sweep_nn.py` before that run, and it passed. Record: `/srv/ml/vega/runs/nn/sweep-qwen2beos/summary.md`.

The gain from tuning is small (about half a point of top-5). The settings are frozen unless the vectors or the label set change.

## Repeating a run

Training uses one CPU thread by default. On the same machine with the same package versions, the same seed gives the same scores to the last digit. `--threads 0` uses every core and is faster, but a run may then end a few tenths of a point away. Another machine can differ in the last decimals.

## Versions

Python 3.12.3, torch 2.15.0 (CPU build), numpy 2.5.5, polars 2.1.0, and Pillow 13.0.0 for `analyse_errors.py --sheets`. These are the versions in `/srv/ml/vega/.venv`.

## Where the outputs are (ml-db)

| Path | What |
|---|---|
| `/srv/ml/vega/runs/comparison/` | scored runs (`results.jsonl`), the table, saved predictions (`preds/`) |
| `/srv/ml/vega/runs/nn/sweep-qwen2beos/` | the settings sweep |
| `/srv/ml/vega/runs/nn/curve-qwen2beos/` | loss and accuracy per epoch |
| `/srv/ml/vega/runs/nn/errors/` | error analysis (`summary.md` first) |
| `/srv/ml/vega/runs/nn/repeatability/` | why a seed did not repeat with two threads, and the fix |
| `/srv/ml/vega/handoff/WORKLOG.md` | what was run and checked, entry by entry (17 Nov for the NN) |

## Generative-AI use

The code in this folder was written by Claude Code (Anthropic) at Andrew's direction. The WORKLOG above records what it ran and how each result was checked. The team's disclosure table is `docs/ai-disclosure.md`.
