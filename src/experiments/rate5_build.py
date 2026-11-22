r"""The second rating page: the proposal's precision@5, for every system that reads the same vectors.

    E:\vega\.venv\Scripts\python.exe -B rate5_build.py [--root E:/vega] [--sets 5] [--device cuda]
        -> runs/ratings-v2/vega-rate5.html, key.csv, build.md

The proposal: for each persona, pool the top 5 aesthetics of every system, hide which system produced each, and have
every team member rate each pooled aesthetic "fits" or "doesn't fit"; report precision@5, nDCG@5 and Fleiss' kappa.
The first page (code/app/rate_build.py) rates the app's first, nearby and surprising suggestion. This page rates top fives.

Photo sets, --sets of each, real outfit photos no system was fitted on, drawn with a seed of their own (522):
  expert    four FashionStyle14 photos of ONE style (a different style per set)
  posters   four photos of one Street Fashion Style poster of the held-out group 0
Systems, all fitted on the wiki training and validation images as the app's network is (the test split is not read):
  nn          the NN with real outfit photos (train_nn.train, frozen settings, seed 422)
  centroid    the nearest-centroid baseline       (code/baselines/baselines.py, as it stands)
  zeroshot    the zero-shot text baseline          (the same)
  popularity  the popularity baseline              (the same; its five are the same for every set)
Each photo's scores become 1 / rank with the team scorer's ranking, the four photos are averaged, and the first five are
the system's suggestions: the same pooling for every system (as in context_all_models.py). The CNN and the clustering
model are not on the page: running them on new photos is their owners' call.
The page is the first page's template (code/app/rate_page.html), unedited: one aesthetic per screen with three example
pictures; the photo marked "closest match" is the one the NN matches most with that aesthetic.
"""
import base64
import csv
import hashlib
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

T0 = time.time()
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROOT, DEVICE = Path(arg("--root", "E:/vega")), arg("--device", "cuda")
IMAGES = Path(arg("--images", "E:/pack-test/wiki"))
N_EACH = int(arg("--sets", 5))
OUT = ROOT / "runs" / "ratings-v2"
TRANSFER = {"lolita", "fairy", "ethnic"}
SYSTEMS = ["nn", "centroid", "zeroshot", "popularity"]
for sub in ("shared", "nn", "baselines"):
    sys.path.insert(0, str(ROOT / "code" / sub))
import baselines  # noqa: E402
import train_nn  # noqa: E402
import vega_data  # noqa: E402
import vega_score  # noqa: E402

read = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
rng = np.random.default_rng(522)

d = vega_data.load(root=ROOT)                                          # training and validation; the test split is not read
classes = [str(c) for c in d["classes"]]
X = np.concatenate([d["fit"]["x"], d["eval"]["x"]]).astype(np.float32)
Y = np.concatenate([d["fit"]["y"], d["eval"]["y"]]).astype(np.float32)
PATHS = [IMAGES / Path(str(p)).relative_to(ROOT / "wiki") for p in list(d["fit"]["paths"]) + list(d["eval"]["paths"])]
FAMILY = {r["wiki_label"]: r["family_name"] for r in read(ROOT / "runs" / "family-v1" / "label_family.csv")}
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
CEN = ((Y > 0).T.astype(np.float64) @ Xn) / np.maximum((Y > 0).sum(0), 1)[:, None]
CEN /= np.linalg.norm(CEN, axis=1, keepdims=True)
one = (Y > 0).sum(1) == 1


def examples(j):
    rows = np.flatnonzero((Y[:, j] > 0) & one)
    rows = rows if len(rows) else np.flatnonzero(Y[:, j] > 0)
    return [PATHS[int(r)] for r in rows[np.argsort(-(Xn[rows] @ CEN[j]))][:3]]


def b64(path, box):
    im = Image.open(path)
    im.seek(0)
    im = im.convert("RGB")
    im.thumbnail(box)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=74)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------- the photo sets
vec = f"{vega_data.VECTORS}-448"
sdir, simg = ROOT / "emb" / "street-fashion-style", ROOT / "datasets" / "street-fashion-style"
srow = {s: i for i, s in enumerate((sdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
held = sorted((r["user"], srow[r["id"]], r["path"], r["style"]) for r in read(sdir / "manifest.csv")
              if r["id"] in srow and int(hashlib.sha1(r["user"].encode("utf-8")).hexdigest(), 16) % train_nn.REAL_GROUPS == 0
              and (simg / r["path"]).exists())
S_all = np.load(sdir / vec / "embeddings.npy", mmap_mode="r")
by_user = defaultdict(list)
for i, (u, _, _, _) in enumerate(held):
    by_user[u].append(i)
users = sorted(u for u, idx in by_user.items() if len(idx) >= 4)
rng.shuffle(users)

fdir, fimg = ROOT / "emb" / "fashionstyle14", ROOT / "datasets" / "fashionstyle14"
fams = [r["family"] for r in read(ROOT / "runs" / "family-v1" / "families.csv")]
style_ok = {r["style_or_tag"] for r in read(ROOT / "runs" / "family-v1" / "outside_family.csv")
            if r["source"] == "FashionStyle14" and r["family"] in fams and r["style_or_tag"] not in TRANSFER}
frow = {s: i for i, s in enumerate((fdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
styles, fpath = {}, {}
for r in read(fdir / "manifest.csv"):
    styles.setdefault(r["sha1"], set()).add(r["style"])
    fpath.setdefault(r["sha1"], r["path"])
ph = sorted((frow[s], next(iter(st)), fpath[s]) for s, st in styles.items()
            if len(st) == 1 and s in frow and next(iter(st)) in style_ok and (fimg / fpath[s]).exists())
F_all = np.load(fdir / vec / "embeddings.npy", mmap_mode="r")
kinds = sorted({st for _, st, _ in ph})

sets = []                                                              # (source, who, [(vector, image path, tag)] x 4)
for k in rng.choice(kinds, N_EACH, replace=False):
    rows = rng.choice([i for i, p in enumerate(ph) if p[1] == k], 4, replace=False)
    sets.append(("expert", str(k), [(np.asarray(F_all[ph[i][0]]), fimg / ph[i][2], ph[i][1]) for i in rows]))
for u in users[:N_EACH]:
    rows = rng.choice(by_user[u], 4, replace=False)
    sets.append(("posters", u, [(np.asarray(S_all[held[i][1]]), simg / held[i][2], held[i][3]) for i in rows]))
P = np.stack([v for _, _, photos in sets for v, _, _ in photos]).astype(np.float32)
print(f"{len(sets)} sets, {len(P)} photos", flush=True)

# ---------------------------------------------------------------- every system's scores for the photos
z = train_nn.train(X, Y, P, train_nn.SETTINGS, 422, DEVICE, real=train_nn.real_photos(ROOT, classes)).astype(np.float64)
z -= z.max(1, keepdims=True)
nn_p = np.exp(z)
nn_p /= nn_p.sum(1, keepdims=True)
dd = {"fit": {"x": X, "y": Y}, "eval": {"x": P.copy(), "ids": np.arange(len(P))}, "labels": "merged"}
raw = {"nn": nn_p, "centroid": baselines.nearest_centroid(dd), "zeroshot": baselines.zero_shot_text(dd, ROOT), "popularity": baselines.popularity(dd)}
K = len(classes)


def recip(s):
    order = vega_score.ranking(np.asarray(s, np.float64))
    R = np.empty((len(s), K))
    R[np.arange(len(s))[:, None], order] = 1.0 / (np.arange(K)[None, :] + 1)
    return R / R.sum(1, keepdims=True)


rr = {k: recip(v) for k, v in raw.items()}
rated = []
for si, (source, who, photos) in enumerate(sets):
    rows = np.arange(4 * si, 4 * si + 4)
    top = {k: vega_score.ranking(rr[k][rows].mean(0, keepdims=True))[0, :5].tolist() for k in SYSTEMS}
    for a in sorted({a for v in top.values() for a in v}):
        rated.append({"set": si, "source": source, "who": who, "aesthetic": classes[a], "family": FAMILY[classes[a]], "a": a,
                      "marked_photo": int(nn_p[rows, a].argmax()) + 1,
                      **{f"{k}_rank": (top[k].index(a) + 1 if a in top[k] else "") for k in SYSTEMS},
                      "photo_tags": " | ".join(t for _, _, t in photos),
                      "photo_files": " | ".join(str(q.relative_to(ROOT / "datasets")).replace("\\", "/") for _, q, _ in photos)})

# ---------------------------------------------------------------- order: rounds holding one screen of every set, so stopping early stays balanced
per_set = {si: [r for r in rated if r["set"] == si] for si in range(len(sets))}
for v in per_set.values():
    rng.shuffle(v)
order = []
for j in range(max(len(v) for v in per_set.values())):
    block = [v[j] for v in per_set.values() if j < len(v)]
    order += [block[i] for i in rng.permutation(len(block))]
for n, r in enumerate(order, 1):
    r["n"] = n
version = "v2-" + hashlib.sha1(json.dumps([(r["n"], r["set"], r["aesthetic"]) for r in order]).encode("utf-8")).hexdigest()[:6]
ex_cache = {}
data = {"key": version,
        "sets": [[b64(q, (380, 440)) for _, q, _ in photos] for _, _, photos in sets],
        "items": [{"n": r["n"], "set": r["set"], "name": r["aesthetic"], "photo": r["marked_photo"] - 1,
                   "ex": ex_cache.setdefault(r["a"], [b64(q, (260, 260)) for q in examples(r["a"])])} for r in order]}
page = (ROOT / "code" / "app" / "rate_page.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "vega-rate5.html").write_text(page, encoding="utf-8")
cols = ["n", "set", "source", "who", "aesthetic", "family", "marked_photo"] + [f"{k}_rank" for k in SYSTEMS] + ["photo_tags", "photo_files"]
with open(OUT / "key.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(order)
overlap = {k: sum(1 for r in rated if r[f"{k}_rank"] != "" and r["nn_rank"] != "") for k in SYSTEMS[1:]}
(OUT / "build.md").write_text("\n".join([
    f"# The second rating page (precision@5 per system), version {version}", "",
    f"Made by `code/experiments/rate5_build.py` on {DEVICE}. {len(sets)} sets of four real outfit photos ({N_EACH} of one FashionStyle14 style each, "
    f"{N_EACH} of one held-out poster each), {len(order)} screens: every aesthetic in the top five of any of the four systems, once per set.",
    f"Of the NN's {5 * len(sets)} suggestions, also in the top five of: " + ", ".join(f"{k} {v}" for k, v in overlap.items()) + ".",
    "`key.csv` says which systems suggested each aesthetic and at which rank. Paste returned answer lines into `answers.txt` and run `rate5_score.py`.",
    f"time elapsed {time.time() - T0:.0f}s"]) + "\n", encoding="utf-8")
print(f"{version}: {len(order)} screens, page {len(page) / 1e6:.1f} MB -> {OUT / 'vega-rate5.html'}; time elapsed {time.time() - T0:.0f}s", flush=True)
