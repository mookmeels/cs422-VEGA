# common: the same data and the same scoring for every model

These files hand every model its images and labels and score its predictions, so all models are compared on the same held-out images with the same metrics. **Start with `README.txt`**: it shows how to plug a model in (about ten lines of code) and lists the rules the code enforces.

| File | What it does |
|---|---|
| `vega_data.py` | The loader. `load()` gives the training split to fit on and the validation split to score. `load(final=True)` gives train + validation to fit on and the test images without their labels. |
| `vega_score.py` | The scorer. Checks a predictions file, works out the metrics and appends one row to `results.jsonl`. It reads test labels only with `--final`, and logs every such use. |
| `vega_table.py` | Builds the comparison table (`table.md`, `table.csv`) from `results.jsonl`. |
| `vega_selftest.py` | Checks the three files above on made-up data. |
| `vega_transfer.py` | The transfer test, a second evaluation: a model fitted on the wiki's training + validation images is scored on FashionStyle14 outfit photos from outside the wiki. `load()` gives the images to fit on and the photos to score; `--pred` appends one row to `/srv/ml/vega/runs/transfer/results.jsonl`; `--table` rebuilds `table.md` there. Same rule as the test split: run it only on a frozen model. |
| `README.txt` | How to plug a model in, the rules, the vectors, the metrics, and the transfer test. |

## Where the data is

The code is here; the data is not in git. The loader reads it from the project folder on ml-db, `/srv/ml/vega`. Set `VEGA_ROOT` to use a copy somewhere else.

- `splits/wiki-split-v2/split.csv`: train, validation or test for every image
- `emb/qwen2beos-aesthetic-448/` and `emb/qwen2beos-aesthetic-text/`: the image vectors and the text vectors, since 20 Nov 2026 (images embedded with the instruction "Represent the fashion aesthetic of this image."). `VEGA_VECTORS=qwen2beos` loads the earlier ones, `emb/qwen2beos-448/`.
- `wiki/manifest.csv` and `wiki/images/`: which gallery each image comes from, and the image files
- `runs/labeler-wearable/catalog.csv` and `runs/labeler-wearable-merged/label_map.csv`: the 503 labels
- for the transfer test: `emb/fashionstyle14/` (the photos' vectors), `runs/transfer/rows.csv` (which styles count as which wiki labels), and the photo files in `/srv/ml/datasets/fashionstyle14/` (`VEGA_DATASETS` changes that folder)

Results go to `/srv/ml/vega/runs/comparison/` unless a script is given `--out`; the transfer test's go to `/srv/ml/vega/runs/transfer/`.

## Running (on ml-db, from this folder)

    PY=/srv/ml/vega/.venv/bin/python
    $PY vega_selftest.py          # should end with "all checks passed"
    $PY vega_score.py --pred my-preds.npz --model "My model" --owner "My Name" --seed 422
    $PY vega_table.py

`--final` scores the **test split**. It is for the one final run of each model, after the team has frozen everything. Nothing has been scored on the test split yet.

The transfer test, once a model is frozen (`vega_transfer.py` with no options also rebuilds the table):

    $PY vega_transfer.py --pred my-transfer.npz --model "My model" --owner "My Name" --seed 422
    $PY vega_transfer.py --table

The usage example in `vega_transfer.py`'s docstring adds `/srv/ml/vega/code/shared` to the path. From this repo, add this folder (`src/common`) instead; the code is the same.

## These are copies

Copied unchanged on 20 Nov 2026 from `/srv/ml/vega/code/shared/` on ml-db, where the recorded runs were made (`vega_transfer.py` added later that day; `vega_data.py` and `README.txt` copied again after the vector switch). Change the files there, then copy them here again. To check that the copies are current (no output means identical):

    for f in vega_*.py README.txt; do cmp /srv/ml/vega/code/shared/$f $f; done

Not copied: `make_split.py` (it made the split) and `audit_catalog.py`.

Versions: Python 3.12.3, numpy 2.5.5, polars 2.1.0.

Generative-AI use: these files were written by Claude Code (Anthropic) at Andrew's direction. `/srv/ml/vega/handoff/WORKLOG.md` (17 Nov) records what was run and how it was checked. The team's disclosure table is `docs/ai-disclosure.md`.
