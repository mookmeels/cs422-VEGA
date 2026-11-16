# cs422-VEGA

VEGA is a CS 422 team project: an internet-aesthetic classifier (cottagecore, dark academia, gorpcore, and so on). The team compares three models on one shared task and one shared evaluation protocol:

| Model | Member | Folder |
|---|---|---|
| Neural network (NN) | TBD | `src/models/nn/` |
| Convolutional neural network (CNN) | TBD | `src/models/cnn/` |
| K-Means | TBD | `src/models/kmeans/` |
| Baseline (most common aesthetic) | shared | `src/baselines/` |

Before the models can be trained we need a labeled dataset. The plan: score new images with the existing labeler, hand-check a sample to choose a confidence cutoff, and keep only labels above that cutoff. See `docs/` for details.

## What lives where

The repo holds code, docs, configs, small label files and result tables. Images, embeddings (`.npy`, `.npz`), model weights and any `.env` file stay out of git (see `.gitignore`). They live on the team machine `ml-db` under `/srv/ml`, described in `docs/data-layout.md`.

```
cs422-VEGA/
  README.md
  requirements.txt     Python packages, with pinned versions
  docs/                proposal, labeling guide, data layout, discussion log,
                       contributions, generative-AI disclosure
  configs/             embedding settings and per-model configs (seeds included)
  labels/              labels and hand-check sheets (small text files)
  splits/              image-ID lists for train / validation / test, shared by every model
  results/             result tables and figures used in the report
  src/
    common/            shared code: data loading, splits, metrics, evaluation
    baselines/         simple baseline(s)
    embed/             embedding script (vectors made with Qwen3-VL-Embedding-2B)
    labeler/           scoring with the labeler, choosing the cutoff
    review/            hand-check tool, rater agreement (Cohen's kappa)
    models/            nn/, cnn/, kmeans/ (one per member)
  notebooks/           exploration only; final results must come from .py scripts
```

## Working in the shared copy on ml-db

The team works in one shared copy at `/srv/ml/cs422-VEGA`. Each member needs to run these once on ml-db:

```bash
git config --global --add safe.directory /srv/ml/cs422-VEGA
git config --global user.name "<your name>"
git config --global user.email "<your GitHub no-reply address>"
```

The first line tells git to trust a folder owned by another user. Files are group-writable for `mlteam`. Each member pushes with their own GitHub SSH key. Since everyone edits the same files, say in the team chat before larger changes, and commit small and often.

## Reproducing the results

To be filled in as the work lands (required by the course): dependencies and versions, random seeds, data access steps, and the commands that reproduce the main comparison.

## Course submission notes

- Scripts are submitted as `.py.txt` files; rename to `.py` to run. Notebooks are not submitted.
- Upload the report, code, README and slides separately, not as a ZIP.
