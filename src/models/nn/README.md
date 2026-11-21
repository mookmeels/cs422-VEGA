# NN: a small neural network on the Qwen image vectors

Owner: Andrew Nguyen. One of the team's three models. It reads its data from the team's shared loader and is scored by the shared scorer, so its numbers sit in the same table as the other models.

**Since 21 Nov 2026 the NN also learns from real outfit photos** (Andrew's decision). The network without them, the "wiki-only NN", is kept as a reference and is one flag away (`--real-weight 0`). Both get a final test run.

## What it is

Every image has already been turned into 2,048 numbers by the frozen Qwen3-VL-Embedding-2B model, with the instruction "Represent the fashion aesthetic of this image." (the team's shared vectors, `emb/qwen2beos-aesthetic-448`, since 20 Nov 2026). The NN takes those numbers and gives a score to each of the 503 labels. It is a multilayer perceptron:

    2,048 inputs -> scale each input -> one hidden layer of 1,024 units (ReLU, dropout 0.7) -> 503 scores

- **Input scaling:** mean and standard deviation of each input, measured on the wiki training images only.
- **Loss on wiki images:** cross-entropy against the image's labels. An image in two galleries has half its weight on each label. Every label counts alike, whatever its number of images.
- **Training:** AdamW, peak learning rate 1e-3 (linear warm-up, then a cosine down to zero), weight decay 1, batch 512, 40 epochs. No early stopping.

### Real outfit photos

The wiki galleries are mostly moodboards; the input the project is about is a photo of a person's outfit. So each training step also takes 512 photos from Street Fashion Style (outfit posts, `emb/street-fashion-style`).

- A real photo has no aesthetic label. It has the style tag its poster typed, and that tag stands for one of the team's 20 style **families** (`runs/family-v1/outside_family.csv`). Its loss is minus the log of the probability the network gives to that family in total, summed over the family's aesthetics. The network is told which family, never which aesthetic.
- How a real photo is drawn: a family, each equally often; then one of that family's tags, each equally often; then one of the tag's photos (`--real-draw tag`, the default since 21 Nov). So within a family a narrow tag such as Formal (1,300 photos) has the same say as a broad one. Until then the draw was family, then photo (`--real-draw photo`), which let a family's largest tag supply most of its photos.
- Their loss is added with weight 0.25 (`--real-weight`).
- 30 tags are used. Three broad ones (Elegant, Urban, Androgynous) were taken out on 21 Nov because they diluted their families; see "How the settings were chosen".
- Trained on: 149,686 photos in 12 of the 20 families. Never trained on: the 519 posters with one of those tags whose name hashes to 0 mod 10 (18,610 photos), and all of FashionStyle14.
- The wiki images, their order and their loss are exactly as in the wiki-only NN. With `--real-weight 0` the script gives the wiki-only NN to the last digit.

The docstring at the top of `train_nn.py` explains each choice.

## Files

| File | What it does |
|---|---|
| `train_nn.py` | Trains and scores the NN, one run per `--seed`. The chosen settings are its defaults. |
| `sweep_nn.py` | How the wiki-only settings were chosen: the full list of configurations tried, on validation only. |
| `check_nn.py` | Checks `train_nn.py` on made-up data, with and without (made-up) real photos. Reads no real image, label or split. |
| `analyse_errors.py` | Error analysis on validation: tables, and example images with `--sheets`. |
| `real_photo_folds.py` | Part of the final run: scores every tagged real photo with a network that never trained on its poster (10 folds by poster). |

These are the same files as `/srv/ml/vega/code/nn/` on ml-db, where the recorded runs were made.

## Running it (on ml-db, from this folder)

    PY=/srv/ml/vega/.venv/bin/python
    $PY check_nn.py                                                 # should end with "all checks passed"
    $PY train_nn.py --seed 422 423 424 425 426                      # the NN (with real photos), validation, about 2 minutes per seed
    $PY train_nn.py --seed 422 423 424 425 426 --real-weight 0      # the wiki-only NN, about 1 minute per seed
    $PY ../../common/vega_table.py                                  # rebuild the comparison table
    $PY analyse_errors.py --pattern "nn-mlp1x1024real-validation-merged-seed*.npz" --out /srv/ml/vega/runs/nn/errors-real-photos --sheets

- Runs append a row to `/srv/ml/vega/runs/comparison/results.jsonl`. Add `--out <folder>` to try something without touching the team's table.
- `--final` scores the **test split**. It is for the one final run, after the team has frozen every model (freeze: Saturday 21 Nov, 6 pm Pacific). It has not been run. Both networks will be run with it.
- **The real-photo test, also for the final run only:** `$PY real_photo_folds.py --final` (about 40 minutes for one seed). Posters fall into 10 groups by name; for each group a network is trained without it and then scores that group's photos, so all 168,296 photos with a tag in use are scored once by a network that never saw their poster. The wiki-only NN is scored on the same photos. It has not been run. `--check` shows that it runs (two folds, one epoch, no score shown, nothing kept). A fold that is saved is never trained again.
- The real photos and the family tables must be under the project root (`emb/street-fashion-style/qwen2beos-aesthetic-448/`, `emb/street-fashion-style/manifest.csv`, `runs/family-v1/`). If they are missing the script stops and says so; it does not fall back to the wiki-only NN.
- The predictions folder now holds two networks, so `analyse_errors.py` needs `--pattern` (`nn-mlp1x1024real-...` for the NN, `nn-mlp1x1024-...` for the wiki-only one).
- The loader and scorer (`vega_data.py`, `vega_score.py`) are in `src/common`, and the scripts find them there. The data is not in git: it is read from `/srv/ml/vega` on ml-db, or from the folder `VEGA_ROOT` names on another machine.

## Results so far

Validation split (1,351 images), 503 labels, seeds 422 to 426 (mean ± standard deviation over the five seeds), measured 21 Nov 2026:

| | top-5 | top-1 | MRR | per-label top-5 | family top-1 |
|---|---|---|---|---|---|
| NN (with real outfit photos) | 87.9 ± 0.1 | 62.3 ± 0.6 | 0.736 ± 0.003 | 86.1 ± 0.1 | 75.3 ± 0.4 |
| Wiki-only NN | 88.3 ± 0.4 | 62.7 ± 0.5 | 0.741 ± 0.002 | 86.6 ± 0.5 | 75.6 ± 0.5 |
| Baseline: nearest centroid | 82.0 | 53.5 | 0.660 | 82.6 | 69.6 |

(The NN's row is the network as it is since the change of 21 Nov to the tags and the draw. The family column uses the family table as amended on 21 Nov.)

On real outfit photos the network never trained on (family top-1, 20 families). These two rows were measured when real photos were first added, with the first tag mapping and the family-then-photo draw; they have not been re-measured for the network as it is now:

| | FashionStyle14, 11 styles, 10,312 photos | Street Fashion Style, 530 held-out posters, 21,024 photos |
|---|---|---|
| NN (with real outfit photos), first version | 49.4 ± 0.8 | 31.6 ± 0.3 |
| Wiki-only NN | 33.1 ± 0.5 | 23.7 ± 0.4 |

What the change of 21 Nov added, measured once on half of those FashionStyle14 photos that had taken no part in choosing it (5,145 photos, 10 seeds, GPU runs of the same network and settings): 51.2 ± 0.6 before, 54.5 ± 0.6 after, a gain of 3.3 points (95% interval +2.6 to +3.9, paired bootstrap over photos). Wiki top-5 in those runs: 87.9 and 87.8.

**The test split has not been scored.** The full table is `/srv/ml/vega/runs/comparison/table.md`.

How to read this (the intervals in the first three points were measured for the first version of the NN with real photos):

- On the wiki images the two networks are level: top-5 differs by -0.4 points (95% interval -0.9 to +0.1, paired bootstrap).
- On real outfit photos the NN is 16.3 points better at the family (interval +15.4 to +17.1). Even so, about half of the FashionStyle14 photos get the wrong family.
- Only 12 of the 20 families have real photos. Wiki outfit images of the other 8 families are guessed as one of the 12 covered families 2.2 points more often than before (interval +0.5 to +4.2).
- The NN has training data the team's other models do not have. For a like-for-like comparison of algorithms, use the wiki-only NN's row.
- It is not a wiki-only model for the transfer test (bohemian street photos are in its training).
- FashionStyle14's 11 styles are not an untouched test for this model: they were used to choose the weight, between two values, and on 21 Nov one half of them was used to choose the tag mapping and the draw and the other half to check that choice once. The held-out posters above were scored in the first experiment. The untouched real-photo figure will come from `real_photo_folds.py` in the final run.
- Two FashionStyle14 styles stay poor whatever was tried: "gal" (about 30% right) and "mode" (about 22%). No street tag teaches either look well.

The family level (20 style families above the 503 labels) is scored by `vega_score.py` from the same predictions. Training the network directly on the 20 families was tried and is 7.5 points worse than grouping the 503-label network's guesses (`/srv/ml/vega/runs/family-v1/nn/results.md`), so the network keeps its 503 outputs.

## How the settings were chosen

**The wiki-only settings.** On the validation split only, with the vectors of 17 Nov (Qwen's default instruction): 146 runs over 32 configurations, changing one setting at a time and then combining the best. The winning combination was then run on five fresh seeds (427 to 431) under a rule written into `sweep_nn.py` before that run, and it passed. Record: `/srv/ml/vega/runs/nn/sweep-qwen2beos/summary.md`. The gain from tuning is small (about half a point of top-5). These settings were frozen on 17 Nov. They were not re-tuned when the vectors changed on 20 Nov, and not when real photos were added on 21 Nov.

**The real photos' weight.** Two values were tried, 0.25 and 1.0, and nothing else, under rules written before the runs (`/srv/ml/vega/runs/real-photos-v1/RULES.md`): a weight is a candidate if FashionStyle14 family accuracy rises beyond its interval and neither wiki top-5 nor the uncovered families' outfit images fall beyond theirs. 0.25 passed. 1.0 gained 2.7 more points on real photos and lost 1.3 points of wiki top-5. Record: `/srv/ml/vega/runs/real-photos-v1/results.md`.

**Which tags, and how photos are drawn (21 Nov).** Tested on a GPU with the same network and settings, about 6 seconds per network, on the even-sha1 half of FashionStyle14's 11 styles. The network was retrained with each of the 33 tags left out in turn. A narrow tag teaches its family and a broad tag on the same family dilutes it: leaving out Formal lowers the "dressy" style by 5.6 points while leaving out Elegant raises it by 13.7; Sports -9.2 against Urban +8.9 for "street". Two changes followed: Elegant, Urban and Androgynous are no longer used, and a family's tags are drawn equally often. Also tried and not adopted: adding the unused "Chic" photos (56,000; "dressy" fell from 71% to 10%), moving tags to other families, keeping only the photos a wiki-only network half-agrees with (worse), and other weights (the same trade as before, so 0.25 stays). The two changes together were then checked once on the odd-sha1 half, with the comparison written down first: +3.3 points. The three tags were picked from 33 leave-one-out results, so the choosing half flatters them; the checking half does not. Record: `/srv/ml/vega/runs/tag-iter/`, and Amendment 5 in `/srv/ml/vega/runs/family-v1/FROZEN.md`. The weight and the wiki-only settings were not re-tuned.

## Repeating a run

Training uses one CPU thread by default. On the same machine with the same package versions, the same seed gives the same scores to the last digit. `--threads 0` uses every core and is faster, but a run may then end a few tenths of a point away. Another machine can differ in the last decimals. To repeat a run made before 20 Nov ~13:10 UTC, with the earlier vectors, set `VEGA_VECTORS=qwen2beos`.

## Versions

Python 3.12.3, torch 2.15.0 (CPU build), numpy 2.5.5, polars 2.1.0, and Pillow 13.0.0 for `analyse_errors.py --sheets`. These are the versions in `/srv/ml/vega/.venv`.

## Where the outputs are (ml-db)

| Path | What |
|---|---|
| `/srv/ml/vega/runs/comparison/` | scored runs (`results.jsonl`), the table, saved predictions (`preds/`) |
| `/srv/ml/vega/runs/real-photos-v1/` | the experiment that added real photos: rules, results, predictions |
| `/srv/ml/vega/runs/family-v1/` | the 20 families (frozen table), and the network trained on families (`nn/`) |
| `/srv/ml/vega/runs/nn/errors-real-photos/` | error analysis of the NN (`summary.md` first) |
| `/srv/ml/vega/runs/nn/errors/` | error analysis of the wiki-only NN |
| `/srv/ml/vega/runs/nn/sweep-qwen2beos/` | the settings sweep (wiki-only) |
| `/srv/ml/vega/runs/nn/curve-qwen2beos/` | loss and accuracy per epoch (wiki-only) |
| `/srv/ml/vega/runs/nn/repeatability/` | why a seed did not repeat with two threads, and the fix |
| `/srv/ml/vega/handoff/WORKLOG.md` | what was run and checked, entry by entry (17 Nov and 21 Nov for the NN) |

## Generative-AI use

The code in this folder was written by Claude Code (Anthropic) at Andrew's direction. The WORKLOG above records what it ran and how each result was checked. The team's disclosure table is `docs/ai-disclosure.md`.
