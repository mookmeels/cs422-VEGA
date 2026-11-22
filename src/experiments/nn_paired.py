"""Is NN run B better than NN run A on validation, beyond noise?

Each run is a folder of prediction files written by code/nn/train_nn.py (one .npz per seed: ids, scores, classes).
For every validation image the script takes the share of seeds that put a true label in the top 5 (and at rank 1),
then re-draws the validation images 2,000 times (the same re-drawn images for both runs, so the comparison is
paired) and reports the difference B minus A with a 95% interval. Validation only; the test split is never read.

    python nn_paired.py <preds folder A> <preds folder B> [<preds folder C> ...] --root E:\\vega

Every folder after the first is compared with the first.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

DRAWS, DRAW_SEED = 2000, 422


def hits(folder, ids, truth, tie):
    """(seeds, images) arrays: true label in the top 5, true label first."""
    files = sorted(Path(folder).glob("*-validation-*.npz"))
    if not files:
        sys.exit(f"no validation prediction files in {folder}")
    h5, h1 = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        where = {s: i for i, s in enumerate(z["ids"].tolist())}
        s = z["scores"][[where[i] for i in ids]]
        n, k = s.shape
        order = np.lexsort((np.broadcast_to(tie, (n, k)), -s), axis=1)
        pos = np.empty_like(order)
        pos[np.arange(n)[:, None], order] = np.arange(k)[None, :]
        rank = np.asarray([pos[i, t].min() + 1 for i, t in enumerate(truth)])
        h5.append(rank <= 5)
        h1.append(rank == 1)
    return np.asarray(h5, float), np.asarray(h1, float), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
    import vega_data
    import vega_score
    ids, truth, classes, _ = vega_data._eval_truth(False, root=a.root)
    ids = ids.tolist()
    tie = np.random.default_rng(vega_score.TIE_SEED).permutation(len(classes))
    pick = np.random.default_rng(DRAW_SEED).integers(0, len(ids), size=(DRAWS, len(ids)))
    base5, base1, nb = hits(a.folders[0], ids, truth, tie)
    print(f"{len(ids)} validation images; A = {a.folders[0]} ({nb} seeds): top-5 {base5.mean():.1%} (seed spread {base5.mean(1).std():.1%}), top-1 {base1.mean():.1%}")
    for folder in a.folders[1:]:
        h5, h1, n = hits(folder, ids, truth, tie)
        out = [f"B = {folder} ({n} seeds): top-5 {h5.mean():.1%} (seed spread {h5.mean(1).std():.1%}), top-1 {h1.mean():.1%}"]
        for name, b, base in (("top-5", h5, base5), ("top-1", h1, base1)):
            d = b.mean(0) - base.mean(0)                              # per image, averaged over seeds
            draws = d[pick].mean(1) * 100
            lo, hi = np.percentile(draws, [2.5, 97.5])
            out.append(f"  B - A, {name}: {d.mean() * 100:+.1f} points (95% interval {lo:+.1f} to {hi:+.1f})")
        print("\n".join(out))


if __name__ == "__main__":
    main()
