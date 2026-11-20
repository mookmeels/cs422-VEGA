"""The simple baselines every VEGA model is compared against, scored with the shared scorer.

    python baselines.py              # fit on train, score on validation
    python baselines.py --final      # the one final run: fit on train + validation, score on test

  popularity        always suggests the labels with the most training images (the "majority class" baseline)
  nearest centroid  average the Qwen vectors of each label's training images; suggest the closest averages
  zero-shot text    compare the image vector with the Qwen vector of each aesthetic's wiki introduction.
                    Uses no training labels at all: what the pretrained model knows on its own.

None of them has random numbers, so there are no seeds. Predictions go to <out>/preds/, results to
<out>/results.jsonl. This file is also the shortest example of how a model plugs into the shared code.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np

# The team's loader and scorer: ../common in the team repo, ../shared in /srv/ml/vega/code on ml-db.
for folder in ("shared", "common"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / folder))
import vega_data                                                # noqa: E402
import vega_score                                               # noqa: E402

OWNER = "shared (baseline)"


def popularity(d):
    count = (d["fit"]["y"] > 0).sum(0).astype(np.float64)
    return np.tile(count, (len(d["eval"]["ids"]), 1))


def nearest_centroid(d):
    x, y = d["fit"]["x"], d["fit"]["y"]
    mean = x.mean(0)                                             # learned from the fit images only
    cen = ((y > 0).T.astype(np.float32) @ (x - mean))
    cen /= np.linalg.norm(cen, axis=1, keepdims=True) + 1e-12   # a label with no fit image keeps a zero vector
    e = d["eval"]["x"] - mean
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e @ cen.T


def zero_shot_text(d, root):
    text = vega_data.text_vectors(d["labels"], root)             # per label: the vectors of its member aesthetics
    e = d["eval"]["x"]
    return np.stack([(e @ t.T).max(1) for t in text], axis=1)    # a merged label scores as its best member


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true", help="score the TEST split (only once everything is frozen)")
    ap.add_argument("--labels", default="merged", choices=("merged", "wearable"))
    ap.add_argument("--root", default=None, help="default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--out", default=None, help="default <root>/runs/comparison")
    a = ap.parse_args()
    t0 = time.perf_counter()
    root = Path(a.root or vega_data.ROOT)
    out = Path(a.out) if a.out else root / "runs" / "comparison"
    (out / "preds").mkdir(parents=True, exist_ok=True)
    d = vega_data.load(final=a.final, labels=a.labels, root=root)
    print(f"fit {len(d['fit']['ids'])} images, eval {len(d['eval']['ids'])} ({d['eval_split']}), {len(d['classes'])} labels", flush=True)
    for name, fn in (("Baseline: popularity", lambda: popularity(d)),
                     ("Baseline: nearest centroid", lambda: nearest_centroid(d)),
                     ("Baseline: zero-shot text", lambda: zero_shot_text(d, root))):
        scores = fn()
        slug = name.split(": ")[1].replace(" ", "-")
        pred = out / "preds" / f"baseline-{slug}-{d['eval_split']}-{a.labels}.npz"
        np.savez_compressed(pred, ids=d["eval"]["ids"], scores=scores.astype(np.float32), classes=np.asarray(d["classes"]))
        vega_score.score(pred, name, OWNER, seed=None, final=a.final, labels=a.labels, root=root, out=out)
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
