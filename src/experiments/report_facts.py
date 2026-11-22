"""Recompute the descriptive figures the report quotes that no other run folder records.

Read-only on the data; validation only (the test split's labels are never read). Run on ml-db:

    /srv/ml/vega/.venv/bin/python code/experiments/report_facts.py

Writes runs/report-facts/summary.md. About a minute on two cores.
"""
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/srv/ml/vega")
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "baselines"))
import vega_data  # noqa: E402
import vega_score  # noqa: E402
import baselines  # noqa: E402

t0 = time.time()
out = ["# Figures quoted in the report, recomputed", "",
       f"Root {ROOT}; split file sha256 {vega_data.split_sha256(ROOT)[:12]}; vectors {vega_data.image_dir(ROOT).name}. "
       "Validation only; the test split's labels are not read.", ""]

# --- the wiki download
csv.field_size_limit(10 ** 9)
rows = list(csv.DictReader(open(ROOT / "wiki" / "manifest.csv", encoding="utf-8", newline="")))
by_sha = defaultdict(set)
for r in rows:
    by_sha[r["sha1"]].add(r["aesthetic"])
first = {}
for r in rows:
    first.setdefault(r["sha1"], r)
shared = Counter(len(v) for v in by_sha.values())
w = [int(r["width"]) for r in first.values() if r["width"]]
h = [int(r["height"]) for r in first.values() if r["height"]]
mime = Counter(r["mime"] for r in first.values())
pages, stamps = 0, []
for line in open(ROOT / "wiki" / "pages.jsonl", encoding="utf-8"):
    p = json.loads(line)
    pages += 1
    if p.get("timestamp"):
        stamps.append(p["timestamp"])
out += ["## The wiki download (wiki/manifest.csv, wiki/pages.jsonl)", "",
        f"- page-image entries (manifest rows): {len(rows):,}",
        f"- distinct images by SHA-1: {len(by_sha):,}",
        f"- images in more than one gallery: {sum(n for k, n in shared.items() if k > 1):,} "
        f"(galleries per image: {dict(sorted(shared.items()))})",
        f"- median width x height of the distinct images: {statistics.median(w):g} x {statistics.median(h):g}",
        f"- formats: {dict(mime.most_common())}",
        f"- article pages: {pages:,}; page revisions dated {min(stamps)} to {max(stamps)}", ""]

# --- labels and split (training and validation only)
d = vega_data.load(root=ROOT)
fit, ev = d["fit"], d["eval"]
per_label = (fit["y"] > 0).sum(0)
multi = sum(len(t) > 1 for t in ev["truth"])
out += ["## Labels (vega_data.load(), training and validation)", "",
        f"- training images {len(fit['ids']):,}; validation images {len(ev['ids']):,}; labels {len(d['classes'])}",
        f"- training images per label: lowest {int(per_label.min())}, median {float(np.median(per_label)):g}, highest {int(per_label.max())} "
        f"({d['classes'][int(per_label.argmax())]})",
        f"- labels with 14 or fewer training images: {int((per_label <= 14).sum())}; with under 16: {int((per_label < 16).sum())}",
        f"- validation images with more than one label: {multi}", ""]

# --- what the 2,048 numbers hold
x = fit["x"].astype(np.float64)
s = np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2
share = np.cumsum(s) / s.sum()
half, most = int(np.searchsorted(share, 0.5) + 1), int(np.searchsorted(share, 0.9) + 1)
_, _, _, fit_count = vega_data._eval_truth(False, root=ROOT)


def cut(part, m):
    q = dict(part)
    v = np.asarray(part["x"][:, :m], np.float32)
    q["x"] = v / np.linalg.norm(v, axis=1, keepdims=True)
    return q


line = []
for m in (256, 512, 1024, 2048):
    dd = dict(d)
    dd["fit"], dd["eval"] = cut(fit, m), cut(ev, m)
    sc = vega_score.metrics(baselines.nearest_centroid(dd), ev["truth"], fit_count)
    line.append(f"first {m:,}: {sc['top5']:.1%}")
out += ["## The vectors (training images for the variation, validation images for the scores)", "",
        f"- values run from {float(fit['x'].min()):+.2f} to {float(fit['x'].max()):+.2f}; every vector has length {float(np.linalg.norm(fit['x'], axis=1).mean()):.3f}",
        f"- directions that carry half of the variation between training images: {half}; 90%: {most} (principal components of the centred training vectors)",
        "- nearest-centroid top-5 on validation using only the leading numbers of each vector (re-scaled to length 1): " + "; ".join(line), ""]

# --- how often a true label is in the zero-shot top 50
z = baselines.zero_shot_text(d, ROOT)
n, k = z.shape
tie = np.random.default_rng(vega_score.TIE_SEED).permutation(k)
order = np.lexsort((np.broadcast_to(tie, (n, k)), -z), axis=1)
pos = np.empty_like(order)
pos[np.arange(n)[:, None], order] = np.arange(k)[None, :]
rank = np.asarray([pos[i, t].min() + 1 for i, t in enumerate(ev["truth"])])
out += ["## Zero-shot text ranking (validation)", "",
        "- a true label is in the zero-shot top 5 / 10 / 50 / 100 for "
        + " / ".join(f"{(rank <= c).mean():.1%}" for c in (5, 10, 50, 100)) + " of validation images",
        "- so a model that only re-orders the zero-shot top 50 could not pass the top-50 figure", "",
        f"Time elapsed: {time.time() - t0:.0f}s."]

dst = ROOT / "runs" / "report-facts"
dst.mkdir(parents=True, exist_ok=True)
(dst / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
