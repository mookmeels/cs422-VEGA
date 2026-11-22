r"""Re-check the split's near-duplicate rule on the CORRECTED vectors (the split was built with the old ones).

The rule (splits/wiki-split-v2/report.json): two images are near-duplicates when their vectors' cosine similarity is
>= 0.97, or >= 0.93 with a pixel correlation >= 0.7, or >= 0.85 with a pixel correlation >= 0.9 (Pearson correlation
of 32 x 32 greyscale thumbnails). This script finds every pair of images in DIFFERENT splits (train, validation,
test) whose corrected vectors pass the cosine part, and writes them out; pixel correlations for the pairs that need
one are added on ml-db by the second step. Reads only; the split is not changed.

    python recheck_near_duplicates.py candidates   (PC, GPU: E:\vega)          -> runs/split-recheck/candidates.csv
    python recheck_near_duplicates.py pixels        (ml-db: reads the images)  -> runs/split-recheck/crossing.csv + summary
"""
import csv, json, sys
from pathlib import Path

import numpy as np

STEPS = [(0.97, -1.0), (0.93, 0.7), (0.85, 0.9)]


def candidates(root):
    import torch
    ids = (root / "emb" / "qwen2beos-448" / "ids.txt").read_text().split()
    x = torch.from_numpy(np.load(root / "emb" / "qwen2beos-448" / "embeddings.npy")).cuda()
    split = {r["sha1"]: r["split"] for r in csv.DictReader(open(root / "splits" / "wiki-split-v2" / "split.csv", encoding="utf-8"))}
    code = {"train": 0, "validation": 1, "test": 2}
    s = torch.tensor([code.get(split.get(i, ""), -1) for i in ids], device="cuda")
    out = []
    for a in range(0, len(ids), 2048):
        sim = x[a:a + 2048] @ x.T
        rows = torch.arange(a, min(a + 2048, len(ids)), device="cuda")[:, None]
        cols = torch.arange(len(ids), device="cuda")[None, :]
        mask = (sim >= 0.85) & (cols > rows) & (s[a:a + 2048][:, None] != s[None, :]) & (s[a:a + 2048][:, None] >= 0) & (s[None, :] >= 0)
        for i, j in mask.nonzero().tolist():
            out.append((ids[a + i], ids[j], float(sim[i, j])))
    d = root / "runs" / "split-recheck"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "candidates.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["a", "b", "split_a", "split_b", "cosine"])
        for i, j, c in sorted(out, key=lambda t: -t[2]):
            w.writerow([i, j, split[i], split[j], f"{c:.4f}"])
    print(f"cross-split pairs with cosine >= 0.85: {len(out)}; >= 0.93: {sum(c >= 0.93 for *_, c in out)}; >= 0.97: {sum(c >= 0.97 for *_, c in out)}")


def pixels(root):
    from PIL import Image, ImageOps
    d = root / "runs" / "split-recheck"
    rows = list(csv.DictReader(open(d / "candidates.csv", encoding="utf-8")))
    paths = {}
    for r in csv.DictReader(open(root / "wiki" / "manifest.csv", encoding="utf-8")):
        paths.setdefault(r["sha1"], r["path"])

    def thumb(sha):
        with Image.open(root / "wiki" / paths[sha]) as im:
            g = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((32, 32), Image.Resampling.BILINEAR), np.float64).ravel()
        return (g - g.mean()) / (g.std() + 1e-9)

    cache, crossing = {}, []
    for r in rows:
        c = float(r["cosine"])
        if c >= 0.97:
            pc = None
        else:
            for k in (r["a"], r["b"]):
                if k not in cache:
                    try:
                        cache[k] = thumb(k)
                    except Exception:
                        cache[k] = None
            ta, tb = cache[r["a"]], cache[r["b"]]
            pc = float(np.mean(ta * tb)) if ta is not None and tb is not None else None
        dup = any(c >= cmin and (pmin < 0 or (pc is not None and pc >= pmin)) for cmin, pmin in STEPS)
        if dup:
            crossing.append({**r, "pixel_corr": "" if pc is None else f"{pc:.3f}"})
    with open(d / "crossing.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["a", "b", "split_a", "split_b", "cosine", "pixel_corr"], lineterminator="\n")
        w.writeheader()
        w.writerows(crossing)
    summary = {"candidates_cosine_085": len(rows), "near_duplicate_pairs_crossing_splits": len(crossing),
               "by_splits": {k: sum(1 for c in crossing if tuple(sorted((c["split_a"], c["split_b"]))) == k) for k in
                             {tuple(sorted((c["split_a"], c["split_b"]))) for c in crossing}}}
    summary["by_splits"] = {" / ".join(k): v for k, v in summary["by_splits"].items()}
    (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"E:\vega")
    {"candidates": candidates, "pixels": pixels}[sys.argv[1]](root)
