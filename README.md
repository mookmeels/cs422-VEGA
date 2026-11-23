# cs422-VEGA

VEGA is a CS 422 team project: an internet-aesthetic classifier (cottagecore, dark academia, gorpcore, and so on). The team compares three models on one shared task and one shared evaluation protocol:

| Model | Member | Folder |
|---|---|---|
| Neural network (NN): an MLP on frozen image vectors | Andrew Nguyen | `src/models/nn/` |
| Convolutional neural network (CNN): a ResNet-18 trained from pixels | Giancarlo Guevarra | `src/models/cnn/` |
| Hierarchical clustering on the image vectors | Edmond Luong | `src/models/hclustering/` |
| Baselines: popularity, zero-shot text, nearest centroid | shared | `src/baselines/` |

The task: given an image, rank 503 aesthetics from the Aesthetics Wiki. The labels are the wiki's own galleries (an image on an aesthetic's page is an example of it), so nothing was labelled by hand. Every model is fitted and scored on the same split with the same scorer (`src/common/`). The consolidated table, validation and test, is `results/comparison/table.md`.

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
    models/            nn/, cnn/, hclustering/ (one per member)
    app/               the application layer on top of the NN: profile from several photos,
                       nearby and surprising suggestions, the local demo page, the rating page
    experiments/       one-off measurements quoted in the report, and the final-run scripts
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

Everything below ran on the team machine `ml-db` (2 CPU cores, about 8 GB of memory, no GPU) unless it says otherwise. On ml-db the same code lives in `/srv/ml/vega/code/` (`shared` = `src/common`, `nn` = `src/models/nn`, and so on) and the commands were run there; the final-run scripts keep those paths.

### Dependencies and versions

Python 3.12.3. `pip install -r requirements.txt` gives numpy 2.5.5, polars 2.1.0, torch 2.15.0 (CPU build), pillow 13.0.0 and scikit-learn 1.9.1. The CNN was trained on Giancarlo's PC with PyTorch 2.9.1 + ROCm 7.2.1 (see `src/models/cnn/README.txt`). The image vectors were made on a PC with an RTX 4080 using llama.cpp and the Qwen3-VL-Embedding-2B f16 GGUF file (`configs/embedding.json`).

### Seeds, split and saved configurations

- Stochastic models (NN, CNN) are run with seeds 422, 423, 424, 425 and 426 and reported as mean ± standard deviation. The clustering model and the baselines have no randomness and are run once.
- The split is `splits/wiki-split-v2/split.csv` (sha256 in `configs/protocol.json`): 9,359 training, 1,351 validation and 2,618 test images. Near-duplicate images are kept in the same part. `src/common/make_split.py` made it.
- `configs/` holds the settings of the protocol, the embedding, each model and the baselines, as they were run.

### Data access

The data is not in git (see `docs/data-layout.md`). To rebuild it elsewhere, set `VEGA_ROOT` to a folder with the same layout as `/srv/ml/vega`:

1. **Aesthetics Wiki images and pages** (aesthetics.fandom.com), a snapshot from November 2026: 1,161 pages, 23,056 images. The images are not redistributed; `wiki/manifest.csv` on ml-db lists each image and the page it came from.
2. **Street Fashion Style** (Gu et al., 2017), Zenodo record 833051, CC BY-NC 4.0: extra training photos for the NN and the CNN's street-photo versions.
3. **FashionStyle14** (Takagi et al., 2017): outfit photos with expert style labels, used only for checks on real photos. Its lolita, fairy and ethnic photos are set aside as a transfer test.
4. **Image vectors**: `src/embed/vega_embed.py` and `src/experiments/embed_eos_check.py` turn images into vectors with the settings in `configs/embedding.json` (about 12 images a second on an RTX 4080). Set `VEGA_LLAMA_DIR` to the folder that holds `runtime/llama-server.exe` and `VEGA_LMSTUDIO_DIR` to the LM Studio folder with the model files; the demo page takes `--embed-python`.

### Commands for the main comparison

Validation (fit on the training part, score the validation part). Each command appends rows to `runs/comparison/results.jsonl`; the last one rebuilds the table.

```bash
PY=/srv/ml/vega/.venv/bin/python
cd src/common   && $PY vega_selftest.py                      # the loader and scorer check themselves
cd ../baselines && $PY baselines.py                          # popularity, zero-shot text, nearest centroid
cd ../models/nn && $PY train_nn.py --seed 422 423 424 425 426 --real-weight 0     # NN, wiki images only
                   $PY train_nn.py --seed 422 423 424 425 426                     # NN + real outfit photos
cd ../hclustering && $PY hclustering.py                      # writes my-preds.npz (about 9 minutes)
                   $PY ../../common/vega_score.py --pred my-preds.npz --model "Hclustering(standardized 0.16)" --owner "Edmond Luong"
                   # (the recorded rows carry the owner name Si_mpl and seeds 422 to 426, all from this one file)
# CNN: trained on a GPU, predictions copied back and scored here; commands in src/models/cnn/README.txt
cd ../../common && $PY vega_table.py                         # table.md and table.csv
```

Test (fit on training + validation, score the test part once). Each script refuses to run a second time, and every test scoring is written to `results/comparison/FINAL-RUNS.log.txt`.

```bash
sh   src/models/nn/nn_final_run.sh                  # both NNs, five seeds each
bash src/baselines/baselines_final_run.sh           # the three baselines
bash src/experiments/cnn_final_score.sh             # the CNN's three rows, from weights trained with train_cnn.py --final --fit-only
bash src/experiments/hclustering_final_run.sh       # the clustering model, its script unedited
```

The logs of these four runs, with the sha256 of the code and of each prediction file, are in `results/comparison/`.

### Other results quoted in the report

| Result | Script | Saved output |
|---|---|---|
| NN error analysis (final network, validation) | `src/models/nn/analyse_errors.py` | `results/nn-errors-final/` |
| Real outfit photos in training | `src/experiments/real_photos_nn.py` | `results/real-photos.md` |
| Encoders compared (Qwen, CLIP ViT-B/32, CLIP ViT-L/14) | `src/experiments/clip_compare.py` | `results/encoders-wiki.md`, `results/encoders-outfit-photos.md` |
| The 20 style families | `src/experiments/family_v1.py`, `family_v1_freeze.py` | `results/family-table.csv`, `results/family-table-frozen.md` |
| Profiles from several photos, surprise score | `src/app/persona_eval.py`, `poster_eval.py`, `fs14_persona_eval.py`, `weight_variants.py`, `many_photos.py` | `results/personas-*.md` |
| The three input settings for every model | `src/experiments/context_all_models.py` | `results/context-all-models.md` |
| Fairness checks | `src/experiments/fairness_spread.py`, `fairness_head.py` | `results/fairness-spread.md`, `results/fairness-head.md` |
| The demo page and the rating page | `src/app/vega_app.py`, `rate_build.py`, `rate_score.py` | `src/app/README.md` |
| Human ratings, page 1 (do suggestions fit) | `src/app/rate_build.py`, `rate_score.py` | `results/ratings-page-1.md`, `results/ratings-page-1-answers.txt` |
| Human ratings, page 2 (precision@5) | `src/experiments/rate5_build.py`, `rate5_score.py` | `results/ratings-page-2.md`, `results/ratings-page-2-answers.txt` |

Scripts that need a GPU say so at the top; they ran on the PC that made the vectors.

Rescoring the rating answers needs each page's key file. The keys name the dataset posters whose photos are shown, so they stay on the team server and out of this repository.

## Course submission notes

- Scripts are submitted as `.py.txt` files; rename to `.py` to run. Notebooks are not submitted.
- Upload the report, code, README and slides separately, not as a ZIP.
