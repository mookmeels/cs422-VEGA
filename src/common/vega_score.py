"""Score any model's predictions the same way, and add the result to the team's comparison file.

A model saves one file per run:

    np.savez("my-preds.npz", ids=d["eval"]["ids"], scores=scores, classes=np.asarray(d["classes"]))

where `scores` has one row per eval image and one column per label (higher = better fit; any scale), in the
order of d["classes"]. Then:

    python vega_score.py --pred my-preds.npz --model "CNN (ResNet-18)" --owner "Name" --seed 422

Validation is scored by default. The test split is scored only with --final, which is for the one run after
every setting is frozen; each such run is written to FINAL-RUNS.log.

Metrics (an image may have several true labels; the best-ranked one counts):
  top5        share of images with a true label in the top 5 (recall@5)      <- proposed primary metric
  top1        share whose first guess is a true label
  mrr         mean of 1 / rank of the best true label
  label_top5  recall@5 computed per label, then averaged (every label counts equally, big or small)
  thin_top5 / rest_top5   the same per-label average, for labels with under 16 fit images and for the rest
  coverage5   share of all labels that appear in at least one image's top 5
Equal scores are ordered by a fixed random order of the labels, the same for every model, so a model that
gives many labels the same score gains nothing from it.

Appends one line to <out>/results.jsonl (default <root>/runs/comparison). vega_table.py turns that into the table.
"""
import argparse, datetime, getpass, hashlib, json, platform, socket, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vega_data                                                # noqa: E402

TIE_SEED = 422


def metrics(scores, truth, fit_count, thin=vega_data.THIN):
    n, k = scores.shape
    tie = np.random.default_rng(TIE_SEED).permutation(k)
    order = np.lexsort((np.broadcast_to(tie, (n, k)), -scores), axis=1)       # best first; ties by the fixed order
    pos = np.empty_like(order)
    pos[np.arange(n)[:, None], order] = np.arange(k)[None, :]
    rank = np.asarray([pos[i, t].min() + 1 for i, t in enumerate(truth)])
    hit5 = rank <= 5
    cnt, hit = np.zeros(k), np.zeros(k)
    for i, t in enumerate(truth):                                              # per label: is THIS label in the top 5
        cnt[t] += 1
        hit[t] += pos[i, t] < 5
    seen = cnt > 0
    per = hit[seen] / cnt[seen]
    is_thin = (fit_count < thin)[seen]
    return {"top5": float(hit5.mean()), "top1": float((rank == 1).mean()), "mrr": float((1 / rank).mean()),
            "label_top5": float(per.mean()),
            "thin_top5": float(per[is_thin].mean()) if is_thin.any() else None,
            "rest_top5": float(per[~is_thin].mean()) if (~is_thin).any() else None,
            "coverage5": float(len(np.unique(order[:, :5])) / k),
            "labels_scored": int(seen.sum()), "thin_labels_scored": int(is_thin.sum())}


def score(pred, model, owner, seed=None, final=False, labels="merged", root=None, out=None, note="", quiet=False):
    """Check a predictions file against the shared eval set, compute the metrics, append them to results.jsonl."""
    root = Path(root or vega_data.ROOT)
    out = Path(out) if out else root / "runs" / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    ids, truth, classes, fit_count = vega_data._eval_truth(final, labels, root)
    z = np.load(pred, allow_pickle=False)
    for key in ("ids", "scores", "classes"):
        if key not in z:
            raise SystemExit(f"{pred}: missing '{key}'. Save with np.savez(path, ids=..., scores=..., classes=...).")
    if [str(c) for c in z["classes"]] != classes:
        raise SystemExit(f"{pred}: label names or their order differ from vega_data.load(labels={labels!r})['classes'] "
                         f"({len(z['classes'])} given, {len(classes)} expected).")
    s = np.asarray(z["scores"], np.float64)
    where = {str(a): i for i, a in enumerate(z["ids"])}
    if len(where) != len(z["ids"]) or s.shape != (len(z["ids"]), len(classes)):
        raise SystemExit(f"{pred}: scores must be (images x labels) = ({len(z['ids'])} x {len(classes)}) with no repeated id; got {s.shape}.")
    missing = [a for a in ids if a not in where]
    if missing:
        split = "test" if final else "validation"
        raise SystemExit(f"{pred}: {len(missing)} of the {len(ids)} {split} images have no prediction. Every model must "
                         f"score every eval image" + ("" if final else " (did you mean --final?)") + ".")
    s = s[[where[a] for a in ids]]
    if not np.isfinite(s).all():
        raise SystemExit(f"{pred}: scores contain NaN or infinity.")
    m = metrics(s, truth, fit_count)
    row = {"model": model, "owner": owner, "seed": seed, "split": "test" if final else "validation", "labels": labels,
           "n_images": len(ids), "n_labels": len(classes), **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()},
           "note": note, "pred_file": str(pred), "pred_sha256": hashlib.sha256(Path(pred).read_bytes()).hexdigest(),
           "split_sha256": vega_data.split_sha256(root), "vectors": vega_data.VECTORS,
           "when_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
           "host": socket.gethostname(), "user": getpass.getuser(),
           "python": platform.python_version(), "numpy": np.__version__}
    with open(out / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    if final:
        with open(out / "FINAL-RUNS.log", "a", encoding="utf-8") as f:
            f.write(f"{row['when_utc']} UTC  {row['user']}@{row['host']}  test split scored for: {model} (owner {owner}, seed {seed})\n")
    if not quiet:
        thin = "n/a" if m["thin_top5"] is None else f"{m['thin_top5']:.1%}"
        print(f"{model:<34} {row['split']:<10} seed {str(seed):<5} top-5 {m['top5']:.1%}  top-1 {m['top1']:.1%}  MRR {m['mrr']:.3f}  "
              f"per-label top-5 {m['label_top5']:.1%} (thin {thin}, rest {m['rest_top5']:.1%})  coverage {m['coverage5']:.1%}  "
              f"[{len(ids)} images, {len(classes)} labels]", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help=".npz with ids, scores, classes")
    ap.add_argument("--model", required=True, help='name for the table, e.g. "CNN (ResNet-18)"')
    ap.add_argument("--owner", required=True, help="the team member responsible for this model")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--final", action="store_true", help="score the TEST split: only for the one run after everything is frozen")
    ap.add_argument("--labels", default="merged", choices=("merged", "wearable"))
    ap.add_argument("--root", default=None, help="default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--out", default=None, help="default <root>/runs/comparison")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    score(a.pred, a.model, a.owner, a.seed, a.final, a.labels, a.root, a.out, a.note)


if __name__ == "__main__":
    main()
