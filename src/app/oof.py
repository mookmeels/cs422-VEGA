"""Held-out NN predictions for every wiki training and validation image, for building personas. The test split is never read.

    python oof.py [--root E:/vega] [--device cuda] [--seed 422]      -> runs/personas-v1/oof.npz

A persona needs several images of one aesthetic, and the validation split has under three per label. So the training and
validation images (10,710) are put into 5 folds, and each fold is scored by a network trained on the other four: every image
gets a prediction from a network that never saw it. Near-duplicate images share a fold (the split file's groups).

The network is the NN as it stands (code/nn/train_nn.py: frozen settings, with real outfit photos, poster group 0 held out).
These are GPU runs for the application layer; they are not rows of the team's table.
"""
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

T0 = time.time()
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROOT, DEVICE, SEED = Path(arg("--root", "E:/vega")), arg("--device", "cuda"), int(arg("--seed", 422))
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "nn"))
import train_nn  # noqa: E402
import vega_data  # noqa: E402

d = vega_data.load(root=ROOT)                                          # training and validation only
classes = [str(c) for c in d["classes"]]
ids = np.asarray([str(a) for a in list(d["fit"]["ids"]) + list(d["eval"]["ids"])])
x = np.concatenate([d["fit"]["x"], d["eval"]["x"]]).astype(np.float32)
y = np.concatenate([d["fit"]["y"], d["eval"]["y"]]).astype(np.float32)
part = np.asarray(["train"] * len(d["fit"]["ids"]) + ["validation"] * len(d["eval"]["ids"]))
group = {r["sha1"]: r["group"] for r in csv.DictReader(open(ROOT / "splits" / "wiki-split-v2" / "split.csv", encoding="utf-8", newline=""))}
fold = np.asarray([int(hashlib.sha1(group[i].encode("utf-8")).hexdigest(), 16) % 5 for i in ids])
real = train_nn.real_photos(ROOT, classes)
print(f"{len(ids):,} images, {len(classes)} labels; folds {np.bincount(fold).tolist()}; street photos {real['photos']:,}, drawn by {real['draw']}", flush=True)

probs = np.zeros((len(ids), len(classes)), np.float32)
for k in range(5):
    tr, te = fold != k, fold == k
    z = train_nn.train(x[tr], y[tr], x[te], train_nn.SETTINGS, SEED, DEVICE, real=real).astype(np.float64)
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    probs[te] = (p / p.sum(1, keepdims=True)).astype(np.float32)
    top5 = np.argsort(-probs[te], 1)[:, :5]
    hit = [(y[te][i, top5[i]] > 0).any() for i in range(te.sum())]
    print(f"  fold {k}: {te.sum():,} images scored by a network trained on the other {tr.sum():,}; top-5 {np.mean(hit):.1%}", flush=True)
top5 = np.argsort(-probs, 1)[:, :5]
hit5 = np.asarray([(y[i, top5[i]] > 0).any() for i in range(len(ids))])
hit1 = np.asarray([y[i, top5[i, 0]] > 0 for i in range(len(ids))])
out = ROOT / "runs" / "personas-v1"
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "oof.npz", ids=ids, probs=probs.astype(np.float16), fold=fold, part=part, classes=np.asarray(classes), seed=SEED)
print(f"held-out top-5 {hit5.mean():.1%}, top-1 {hit1.mean():.1%} over all {len(ids):,} images; saved {out / 'oof.npz'}; time elapsed {time.time() - T0:.0f}s", flush=True)
