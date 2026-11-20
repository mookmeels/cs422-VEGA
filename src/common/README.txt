Shared code: how every VEGA model gets its data and its score
==============================================================

Why this exists: the assignment wants all models compared on the same held-out images, with the same
metrics, and with the test split used once. These three files make that automatic.

  vega_data.py      hands a model its images and labels
  vega_score.py     scores a model's predictions and adds them to runs/comparison/results.jsonl
  vega_table.py     turns results.jsonl into the comparison table (table.md, table.csv)
  vega_selftest.py  checks the three files above on made-up data (run it after changing them)
  ../baselines/baselines.py   popularity, nearest centroid, zero-shot text; also the shortest full example

Run with the project's Python:  /srv/ml/vega/.venv/bin/python
On another machine, point at your copy of the project first:  set VEGA_ROOT (default /srv/ml/vega)


PLUGGING IN A MODEL (about ten lines)

    import sys, numpy as np
    sys.path.insert(0, "/srv/ml/vega/code/shared")
    from vega_data import load

    d = load()                              # fit = training split, eval = validation split
    classes = d["classes"]                  # 503 label names; your score columns must follow this order
    x, y = d["fit"]["x"], d["fit"]["y"]     # Qwen image vectors (n x 2048); labels (n x 503, each row sums to 1)
    files = d["fit"]["paths"]               # image files, if your model reads pixels (CNN)

    ... train on d["fit"] only ...
    scores = ...                            # one row per image in d["eval"], one column per label, higher = better

    np.savez("my-preds.npz", ids=d["eval"]["ids"], scores=scores, classes=np.asarray(classes))

then:

    python vega_score.py --pred my-preds.npz --model "CNN (ResNet-18)" --owner "Your Name" --seed 422
    python vega_table.py

A model that gives one answer per image (k-means, for example) still needs a score for every label:
use the share of each label among the training images in the image's cluster, or 1 for its answer and 0 elsewhere.
An image can have more than one true label (41 of the 1,352 validation images do); y splits the weight between them.


THE RULES THE CODE ENFORCES

  1. Everything a model learns (weights, scaling, averages, clusters) comes from d["fit"] only.
  2. Choose settings by their validation score. d["eval"]["y"] is there for that.
  3. The test split: load(final=True) fits on train + validation and returns the test images WITHOUT labels.
     Only `vega_score.py --final` reads test labels, and it logs each use in runs/comparison/FINAL-RUNS.log.
     Do this once per model, after the team has frozen every setting. Do not change a model after seeing it.
  4. Run stochastic models with several seeds (the team has used 422, 423, 424, 425, 426) and pass --seed each
     time. The table shows mean ± spread across seeds.
  5. The scorer refuses a file that is missing eval images, repeats an image, or has labels in another order.


THE VECTORS

  d["fit"]["x"] and d["eval"]["x"] come from emb/qwen2beos-448: Qwen3-VL-Embedding-2B, each image shrunk to at most
  448 px on its longest edge, 2048 numbers per image, read at the model's end-of-text token.
  Until 17 Nov 2026 the loader used emb/qwen2b-448, which read the vector one token early (a setup error; see
  handoff/WORKLOG.md, 17 Nov). Runs from before the change stay in results.jsonl; the table shows each model's
  latest runs and flags any model that has not been re-run on the current vectors.
  Never mix vectors from two folders. New images (Fashionpedia, for example) must be embedded the same way:
  code/experiments/embed_eos_check.py on the PC, NOT handoff/scripts/vega_embed.py as it stands.


WHAT IS SCORED

  Label set "merged": the wearable catalogue (544 aesthetics) with look-alike aesthetics merged, 503 labels.
  Which aesthetics were merged, and why: runs/labeler-wearable-merged/label_map.csv and groups.csv.
  (--labels wearable scores the 544 unmerged labels instead. Use one label set for the whole comparison.)

  Training split: 9,359 images. Validation: 1,352. Half the labels have 14 or fewer training images;
  274 of the 503 have under 16 ("thin" in the table).

  top5              share of images with a true label among the 5 best-scored labels   <- proposed primary metric
  top1              share whose best-scored label is a true label
  mrr               mean of 1 / (rank of the best true label)
  label_top5        top5 worked out per label, then averaged, so small labels count as much as big ones
  thin / rest top5  the same, for labels with under 16 training images and for the others
  coverage5         share of the 503 labels that show up in at least one image's top 5

The primary metric is a proposal (it is the proposal document's recall@5). The team should confirm it,
and the label set, before anyone runs --final.
