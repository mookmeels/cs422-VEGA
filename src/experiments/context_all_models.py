"""The proposal's three input settings for EVERY model of the team table, from saved validation predictions.

    python context_all_models.py            -> runs/context-all-models/results.md, results.json, personas.csv

The proposal: "Each member runs their model in all three input settings (outfit only, equal average, weighted
profile), so all results land in one comparison table." Until now this was measured for the NN only (code/app).
Nothing is trained here and the test split is never read: each model's VALIDATION prediction file, the one scored in
runs/comparison/results.jsonl (same sha256), is pooled over sets of validation images.

A persona is one outfit image plus context images (no person at half the image's height), as in code/app/persona_eval.py,
but drawn from validation images only, so every model has a held-out prediction for every image.
  aesthetic personas  1 outfit + 2 context images (3 where there are 3), all with the same single label (the "home")
  family personas     1 outfit + 3 context images of one style family (any aesthetics of it)
The models' scores are on different scales (probabilities, cosines, mostly-zero similarities), so each photo's scores
are first turned into the same kind of vector for every model: 1 / rank, scaled to sum to 1, with the team scorer's
ranking (and so its rule for equal scores). The three settings are then code/app/vega_profile.pooled as it stands:
"outfit" (the outfit photo alone), "equal" (all photos) and "agree" (the app's weighted profile for up to 16 photos:
agreement weights, geometric mean). "agree_mean" (the same weights, ordinary average) is shown beside it. No setting is tuned here.
Intervals: 2,000 resamples of the outfit images (seed 422), per-persona hits averaged over a model's seeds first.
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

T0 = time.time()
ROOT = Path("/srv/ml/vega")
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "app"))
import vega_data  # noqa: E402
import vega_score  # noqa: E402
import vega_profile as vp  # noqa: E402

OUT = ROOT / "runs" / "context-all-models"
OUT.mkdir(exist_ok=True)
MODELS = [  # (name in results.jsonl, short name for the table)
    ("NN (MLP, 1 hidden layer of 1024)", "NN, wiki images only"),
    ("NN (MLP, 1 hidden layer of 1024) + real outfit photos", "NN + real outfit photos"),
    ("Baseline: nearest centroid", "Baseline: nearest centroid"),
    ("Hclustering(standardized 0.16)", "Hierarchical clustering"),
    ("Baseline: zero-shot text", "Baseline: zero-shot text"),
    ("CNN (ResNet-18, 224 px) + street photos", "CNN + street photos"),
    ("CNN (ResNet-18, 224 px)", "CNN"),
    ("CNN (ResNet-18, 224 px, random crop 0.6, stretch 0.15, mixup/CutMix, label smoothing 0.1, oversampling) + street photos",
     "CNN, adopted recipe + street photos"),
    ("Baseline: popularity", "Baseline: popularity"),
]
HCLUST_FILE = Path("/srv/ml/cs422-VEGA/src/models/hclustering/my-preds.npz")
MODES = ("outfit", "equal", "agree", "agree_mean")

d = vega_data.load(root=ROOT, vectors=False)
ids = [str(a) for a in d["eval"]["ids"]]
classes = [str(c) for c in d["classes"]]
y = np.asarray(d["eval"]["y"]) > 0
k = len(classes)
lab_fam, _ = vega_score.family_of(classes, ROOT)
n_fam = int(lab_fam.max()) + 1
one = y.sum(1) == 1
label = y.argmax(1)
with open(ROOT / "runs" / "person-crops" / "wiki-trainval.csv", encoding="utf-8", newline="") as f:
    boxes = {r["id"]: (float(r["height"]), json.loads(r["people"]) if int(r["n_people"]) > 0 else []) for r in csv.DictReader(f)}
outfit = np.asarray([any((b[3] - b[1]) >= 0.5 * boxes[i][0] for b in boxes[i][1]) for i in ids])
print(f"validation images {len(ids)}; one label {int(one.sum())}; outfit {int((one & outfit).sum())}, context {int((one & ~outfit).sum())}", flush=True)

rng = np.random.default_rng(422)
personas = []                                                    # (kind, home label or family, rows; the outfit image first)
for a in range(k):
    o, c = np.flatnonzero(one & outfit & (label == a)), np.flatnonzero(one & ~outfit & (label == a))
    if len(o) >= 1 and len(c) >= 2:
        for _ in range(6):
            personas.append(("aesthetic", a, [int(rng.choice(o))] + [int(v) for v in rng.choice(c, min(3, len(c)), replace=False)]))
for fm in range(n_fam):
    o, c = np.flatnonzero(one & outfit & (lab_fam[label] == fm)), np.flatnonzero(one & ~outfit & (lab_fam[label] == fm))
    if len(o) >= 1 and len(c) >= 3:
        for _ in range(100):
            personas.append(("family", fm, [int(rng.choice(o))] + [int(v) for v in rng.choice(c, 3, replace=False)]))
kind = np.asarray([p[0] for p in personas])
home = np.asarray([p[1] for p in personas])
first = np.asarray([p[2][0] for p in personas])
n_a, n_f = int((kind == "aesthetic").sum()), int((kind == "family").sum())
print(f"personas: {n_a} aesthetic ({len(set(home[kind == 'aesthetic'].tolist()))} home labels), "
      f"{n_f} family ({len(set(home[kind == 'family'].tolist()))} families)", flush=True)
with open(OUT / "personas.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kind", "home", "outfit_image", "context_images"])
    fam_names = sorted({r["family"] for r in csv.DictReader(open(ROOT / "runs" / "family-v1" / "label_family.csv", encoding="utf-8", newline=""))})
    for kd, h, rows in personas:
        w.writerow([kd, classes[h] if kd == "aesthetic" else fam_names[h], ids[rows[0]], " ".join(ids[r] for r in rows[1:])])

latest = {}
for line in open(ROOT / "runs" / "comparison" / "results.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["split"] == "validation" and r["labels"] == "merged":
        latest[(r["model"], r["seed"])] = r


def photo_vectors(row):
    """1 / rank per photo, summing to 1, from the prediction file of a results.jsonl row (sha256 checked)."""
    p = Path(row["pred_file"])
    if not p.exists():
        p = ROOT / "code" / "shared" / row["pred_file"]
    if not p.exists() and row["model"].startswith("Hclustering"):
        p = HCLUST_FILE
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != row["pred_sha256"]:
        cands = [q for q in [HCLUST_FILE] if q.exists() and hashlib.sha256(q.read_bytes()).hexdigest() == row["pred_sha256"]]
        if not cands:
            return None, str(p)
        p = cands[0]
    z = np.load(p, allow_pickle=False)
    assert [str(c) for c in z["classes"]] == classes
    where = {str(a): i for i, a in enumerate(z["ids"])}
    s = np.asarray(z["scores"], np.float64)[[where[a] for a in ids]]
    order = vega_score.ranking(s)
    R = np.empty_like(s)
    R[np.arange(len(ids))[:, None], order] = 1.0 / (np.arange(k)[None, :] + 1)
    return R / R.sum(1, keepdims=True), str(p)


def hits(R):
    """Per persona and setting: home in top 1, home in top 5 (aesthetic); family first, family in first 3 (family)."""
    out = {}
    for mode in MODES:
        Q = np.stack([vp.pooled(R[rows], np.arange(len(rows)) == 0, mode=mode, lab_fam=lab_fam) for _, _, rows in personas])
        order = vega_score.ranking(Q)
        h1, h5 = np.zeros(len(personas)), np.zeros(len(personas))
        for i, (kd, h, _) in enumerate(personas):
            if kd == "aesthetic":
                h1[i], h5[i] = order[i, 0] == h, h in order[i, :5]
            else:
                fams = []
                for fm in lab_fam[order[i]].tolist():
                    if fm not in fams:
                        fams.append(fm)
                        if len(fams) == 3:
                            break
                h1[i], h5[i] = fams[0] == h, h in fams
        out[mode] = (h1, h5)
    return out


results, per_persona = [], {}
for name, short in MODELS:
    rows = sorted((r for (m, s), r in latest.items() if m == name), key=lambda r: str(r["seed"]))
    if name.startswith("Hclustering"):
        rows = rows[:1]                                          # no randomness: the five rows are one file
    got = []
    for r in rows:
        R, path = photo_vectors(r)
        if R is None:
            print(f"SKIPPED {name} seed {r['seed']}: prediction file not found or changed ({path})", flush=True)
            continue
        got.append((r["seed"], hits(R)))
    if not got:
        continue
    res = {"model": short, "seeds": [s for s, _ in got]}
    for kd in ("aesthetic", "family"):
        sel = kind == kd
        for j, m in enumerate(("first", "top")):
            for mode in MODES:
                per_seed = np.asarray([h[mode][j][sel].mean() for _, h in got])
                res[f"{kd}_{m}_{mode}"] = [float(per_seed.mean()), float(per_seed.std(ddof=1)) if len(got) > 1 else None]
                per_persona[(short, kd, m, mode)] = np.mean([h[mode][j][sel] for _, h in got], axis=0)
    results.append(res)
    print(f"{short}: {len(got)} run(s), {time.time() - T0:.0f}s", flush=True)


def interval(short, kd, m, a, b):
    """95% interval of (setting a - setting b), resampling outfit images."""
    diff = per_persona[(short, kd, m, a)] - per_persona[(short, kd, m, b)]
    g = first[kind == kd]
    groups = [np.flatnonzero(g == u) for u in np.unique(g)]
    r = np.random.default_rng(422)
    draws = []
    for _ in range(2000):
        pick = r.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        draws.append(diff[idx].mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


pc = lambda v: f"{100 * v[0]:.1f}" + (f" ± {100 * v[1]:.1f}" if v[1] is not None else "")
ci = lambda t: f"{100 * t[0]:+.1f} ({100 * t[1]:+.1f} to {100 * t[2]:+.1f})"
lines = ["# The three input settings for every model (validation images only)", "",
         f"Made {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by code/experiments/context_all_models.py. Nothing trained; the test split is not read.",
         f"Personas from validation images: {n_a} aesthetic personas over {len(set(home[kind == 'aesthetic'].tolist()))} home labels "
         f"(1 outfit + 2 or 3 context images of one label), {n_f} family personas over {len(set(home[kind == 'family'].tolist()))} families "
         "(1 outfit + 3 context images of one family).",
         "Each photo's scores become 1 / rank (the scorer's ranking), then code/app/vega_profile.pooled: outfit, equal, agree, agree_mean.",
         "± is the spread over a model's seeds; intervals are 95%, resampling outfit images.", ""]
for kd, m, title in (("aesthetic", "top", "Aesthetic personas: home aesthetic in the top 5"),
                     ("aesthetic", "first", "Aesthetic personas: home aesthetic first"),
                     ("family", "first", "Family personas: home family first"),
                     ("family", "top", "Family personas: home family in the first 3")):
    lines += [f"## {title}", "", "| model | runs | outfit photo only | all photos, equal | weighted by agreement (geometric mean, the app's rule) | weighted by agreement (ordinary average) | all equal - outfit only | weighted (app's rule) - equal |", "|---|---|---|---|---|---|---|---|"]
    for res in results:
        s = res["model"]
        res[f"{kd}_{m}_equal_minus_outfit"] = interval(s, kd, m, "equal", "outfit")
        res[f"{kd}_{m}_agree_minus_equal"] = interval(s, kd, m, "agree", "equal")
        lines.append(f"| {s} | {len(res['seeds'])} | {pc(res[f'{kd}_{m}_outfit'])} | {pc(res[f'{kd}_{m}_equal'])} | {pc(res[f'{kd}_{m}_agree'])} | {pc(res[f'{kd}_{m}_agree_mean'])} | "
                     f"{ci(res[f'{kd}_{m}_equal_minus_outfit'])} | {ci(res[f'{kd}_{m}_agree_minus_equal'])} |")
    lines.append("")
lines.append(f"Time elapsed: {time.time() - T0:.0f} s")
(OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print("\n".join(lines))
