"""One data loader for every VEGA model, so all models train and are scored on exactly the same images.

    from vega_data import load
    d = load()                    # fit on the training split, evaluate on validation
    d["classes"]                  # label names, in the column order every model must use
    d["fit"]["x"], d["fit"]["y"]  # Qwen image vectors (n x 2048) and labels (n x K, rows sum to 1)
    d["fit"]["paths"]             # image files, for models that read pixels
    d["eval"]["ids"], d["eval"]["x"], d["eval"]["paths"]

    d = load(final=True)          # the one final run: fit on train + validation, evaluate on test

Rules this file enforces:
  * the split comes from splits/wiki-split-v2/split.csv and nowhere else;
  * validation labels are returned (tuning on validation is allowed);
  * TEST LABELS ARE NEVER RETURNED. With final=True a model gets the test images only. vega_score.py is the
    only code that reads test labels, and it records every time it does.

Label sets: "merged" (default; the wearable catalogue with the audit's merges applied, 503 labels) or
"wearable" (544 labels, unmerged). An image keeps every label it has in the chosen set; images with none are
left out, as is the one wiki image that has no Qwen vector, so every model sees the same rows.

The vectors come from emb/qwen2beos-448 (see VECTORS below for what that is and why it changed on 17 Nov).
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(os.environ.get("VEGA_ROOT", "/srv/ml/vega"))
THIN = 16                      # labels with fewer fit images than this are reported separately as "thin"

# Which saved vectors every model gets: <root>/emb/<VECTORS>-448 (images) and <root>/emb/<VECTORS>-text (texts).
#   "qwen2beos"  Qwen3-VL-Embedding-2B, vector read at the end-of-text token, as the model is built to be used.
#                The team's vectors since 17 Nov 2026.
#   "qwen2b"     the same model with the vector read one token early (a setup error found on 17 Nov). Every
#                result before that date used these. Set VEGA_VECTORS=qwen2b only to reproduce those results.
# Vectors from different folders must never be mixed in one model or one table.
VECTORS = os.environ.get("VEGA_VECTORS", "qwen2beos")


def image_dir(root=None):
    return Path(root or ROOT) / "emb" / f"{VECTORS}-448"


def text_dir(root=None):
    return Path(root or ROOT) / "emb" / f"{VECTORS}-text"


def split_sha256(root=None):
    root = Path(root or ROOT)
    return hashlib.sha256((root / "splits" / "wiki-split-v2" / "split.csv").read_bytes()).hexdigest()


def label_map(labels="merged", root=None):
    """aesthetic name -> label name, and the sorted label list."""
    root = Path(root or ROOT)
    cat = pl.read_csv(root / "runs" / "labeler-wearable" / "catalog.csv", infer_schema=False)
    keep = cat.filter(~pl.col("catalog").str.starts_with("dropped"))["aesthetic"].to_list()
    if labels == "wearable":
        m = {a: a for a in keep}
    elif labels == "merged":
        lm = pl.read_csv(root / "runs" / "labeler-wearable-merged" / "label_map.csv", infer_schema=False)
        m = dict(zip(lm["aesthetic"].to_list(), lm["merged_label"].to_list()))
        if set(m) != set(keep):
            raise SystemExit("label_map.csv and catalog.csv disagree on which aesthetics are in the catalogue")
    else:
        raise SystemExit(f"unknown label set {labels!r}: use 'merged' or 'wearable'")
    return m, sorted(set(m.values()))


def _rows(root, labels):
    """Every usable image: sha1, split, label indices, file path, row in the embedding matrix."""
    m, classes = label_map(labels, root)
    index = {c: i for i, c in enumerate(classes)}
    man = pl.read_csv(root / "wiki" / "manifest.csv", infer_schema=False).select("sha1", "aesthetic", "path")
    names, path = {}, {}
    for sha, a, p in man.iter_rows():
        path.setdefault(sha, p)
        if a in m:
            names.setdefault(sha, set()).add(index[m[a]])
    ids = (image_dir(root) / "ids.txt").read_text(encoding="utf-8").split()
    row = {sha: i for i, sha in enumerate(ids)}
    out = []
    split = pl.read_csv(root / "splits" / "wiki-split-v2" / "split.csv", infer_schema=False).select("sha1", "split")
    for sha, sp in split.iter_rows():
        if sp in ("train", "validation", "test") and sha in names and sha in row:
            out.append((sha, sp, np.asarray(sorted(names[sha])), str(root / "wiki" / path[sha]), row[sha]))
    out.sort(key=lambda r: r[0])
    return out, classes


def _part(rows, splits, k, mat, with_labels):
    sel = [r for r in rows if r[1] in splits]
    part = {"ids": np.asarray([r[0] for r in sel]), "split": np.asarray([r[1] for r in sel]),
            "paths": [r[3] for r in sel]}
    if mat is not None:
        part["x"] = np.ascontiguousarray(mat[[r[4] for r in sel]], np.float32)
    if with_labels:
        part["truth"] = [r[2] for r in sel]
        y = np.zeros((len(sel), k), np.float32)
        for i, r in enumerate(sel):
            y[i, r[2]] = 1 / len(r[2])
        part["y"] = y
    return part


def load(final=False, labels="merged", root=None, vectors=True):
    """The fit and eval images for one run. See the module docstring."""
    root = Path(root or ROOT)
    rows, classes = _rows(root, labels)
    mat = np.load(image_dir(root) / "embeddings.npy", mmap_mode="r") if vectors else None
    k = len(classes)
    if final:
        print("vega_data: FINAL RUN. Fitting on train + validation; test images are returned without labels.",
              file=sys.stderr, flush=True)
        fit, ev = _part(rows, ("train", "validation"), k, mat, True), _part(rows, ("test",), k, mat, False)
    else:
        fit, ev = _part(rows, ("train",), k, mat, True), _part(rows, ("validation",), k, mat, True)
    return {"classes": classes, "labels": labels, "final": final, "eval_split": "test" if final else "validation",
            "fit": fit, "eval": ev}


def _eval_truth(final, labels="merged", root=None):
    """For vega_score.py only: eval ids, their true labels, and how many fit images each label has."""
    root = Path(root or ROOT)
    rows, classes = _rows(root, labels)
    k = len(classes)
    fit_splits, ev_split = (("train", "validation"), "test") if final else (("train",), "validation")
    ev = [r for r in rows if r[1] == ev_split]
    fit_count = np.zeros(k, np.int64)
    for r in rows:
        if r[1] in fit_splits:
            fit_count[r[2]] += 1
    return np.asarray([r[0] for r in ev]), [r[2] for r in ev], classes, fit_count


def text_vectors(labels="merged", root=None, field="intro"):
    """Qwen text vectors of each aesthetic's wiki description, grouped by label: a list of (m_i x 2048) arrays."""
    root = Path(root or ROOT)
    m, classes = label_map(labels, root)
    titles = [t for t in (text_dir(root) / "titles.txt").read_text(encoding="utf-8").split("\n") if t]
    vec = np.load(text_dir(root) / f"text-{field}.npy")
    where = {t: i for i, t in enumerate(titles)}
    members = {c: [] for c in classes}
    for a, c in m.items():
        members[c].append(where[a])
    return [np.asarray(vec[members[c]], np.float32) for c in classes]


if __name__ == "__main__":
    for final in (False,):
        d = load(final=final)
        f, e = d["fit"], d["eval"]
        per = (f["y"] > 0).sum(0)
        print(f"labels: {d['labels']} ({len(d['classes'])})   split file sha256 {split_sha256()[:12]}   vectors: emb/{VECTORS}-448")
        print(f"fit: {len(f['ids'])} images ({d['fit']['split'][0]})   eval: {len(e['ids'])} images ({d['eval_split']})")
        print(f"fit images per label: min {per.min()}, median {int(np.median(per))}, max {per.max()}; "
              f"labels under {THIN}: {int((per < THIN).sum())}; labels with no fit image: {int((per == 0).sum())}")
