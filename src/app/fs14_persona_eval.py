"""A control for poster_eval.py: the same two questions on FashionStyle14's expert-labelled outfit photos.

    python fs14_persona_eval.py [--root E:/vega] [--seeds 3] [--device cuda]      -> runs/personas-v1/fs14-sets.md

poster_eval.py judges by the posters' own style words, which are coarse and 73% one family. If the profile and the surprise
score look weak there, is it the labels or the photos? FashionStyle14's photos are real outfit photos too, but each carries
one of 11 styles given by experts, in near-equal numbers. Sets are built by style, not by person:
  pure set    four photos of one style
  mixed set   three photos of one style and one of a style from another family
Judged at family level (runs/family-v1/outside_family.csv). The network (train_nn.train, frozen settings, wiki training
images + street photos) is never trained on FashionStyle14. Its lolita, fairy and ethnic photos are not read; nor is the
wiki test split. FashionStyle14's 11 styles were used on 21 Nov to choose and check the street-photo settings.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

T0 = time.time()
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROOT, DEVICE = Path(arg("--root", "E:/vega")), arg("--device", "cuda")
SEEDS = [422 + i for i in range(int(arg("--seeds", 3)))]
N = int(arg("--sets", 3000))
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "nn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_nn  # noqa: E402
import vega_data  # noqa: E402
import vega_profile as vp  # noqa: E402

read = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
V1, OUT = ROOT / "runs" / "family-v1", ROOT / "runs" / "personas-v1"
TRANSFER = {"lolita", "fairy", "ethnic"}
d = vega_data.load(root=ROOT)
fit, classes = d["fit"], [str(c) for c in d["classes"]]
fams = [r["family"] for r in read(V1 / "families.csv")]
of_label = {r["wiki_label"]: r["family"] for r in read(V1 / "label_family.csv")}
lab_fam = np.asarray([fams.index(of_label[c]) for c in classes])
M = np.zeros((len(classes), len(fams)))
M[np.arange(len(classes)), lab_fam] = 1
style_fam = {r["style_or_tag"]: fams.index(r["family"]) for r in read(V1 / "outside_family.csv")
             if r["source"] == "FashionStyle14" and r["family"] in fams and r["style_or_tag"] not in TRANSFER}
fx = np.asarray(fit["x"], np.float64)
SIM = vp.aesthetic_map(fx / np.linalg.norm(fx, axis=1, keepdims=True), np.asarray(fit["y"]) > 0)

fdir = ROOT / "emb" / "fashionstyle14"
frow = {s: i for i, s in enumerate((fdir / f"{vega_data.VECTORS}-448" / "ids.txt").read_text(encoding="utf-8").split())}
styles = {}
for r in read(fdir / "manifest.csv"):
    styles.setdefault(r["sha1"], set()).add(r["style"])
photos = sorted((frow[s], next(iter(st))) for s, st in styles.items() if len(st) == 1 and s in frow and next(iter(st)) in style_fam)
X = np.ascontiguousarray(np.load(fdir / f"{vega_data.VECTORS}-448" / "embeddings.npy", mmap_mode="r")[[r for r, _ in photos]])
style = np.asarray([st for _, st in photos])
truef = np.asarray([style_fam[st] for st in style])
kinds = sorted(set(style.tolist()))

real = train_nn.real_photos(ROOT, classes)
Ps = []
for seed in SEEDS:
    z = train_nn.train(fit["x"], fit["y"], X, train_nn.SETTINGS, seed, DEVICE, real=real).astype(np.float64)
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    Ps.append((p / p.sum(1, keepdims=True)).astype(np.float32))
    print(f"  seed {seed}: {len(X):,} photos scored; per photo, family right {((Ps[-1] @ M).argmax(1) == truef).mean():.1%}", flush=True)
del real

rng = np.random.default_rng(422)
of_style = {k: np.flatnonzero(style == k) for k in kinds}
pure = [(k, rng.choice(of_style[k], 4, replace=False)) for k in rng.choice(kinds, N)]
mixed = []
for k in rng.choice(kinds, N):
    t = str(rng.choice([q for q in kinds if style_fam[q] != style_fam[k]]))
    mixed.append((k, t, np.append(rng.choice(of_style[k], 3, replace=False), rng.choice(of_style[t]))))


def share(v):
    v = np.asarray(v, np.float64)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
    return f"{v.mean():.1%} ({np.percentile(b, 2.5):.1%} to {np.percentile(b, 97.5):.1%})"


L = ["# A control on expert-labelled outfit photos (FashionStyle14, 11 styles)", "",
     f"Made by `code/app/fs14_persona_eval.py`, seeds {SEEDS} on {DEVICE}. {len(X):,} photos never trained on; per photo the network names the right family "
     f"{np.mean([((P @ M).argmax(1) == truef).mean() for P in Ps]):.1%} of the time. {N:,} pure and {N:,} mixed sets, built by style.", "",
     "## Question 1: more photos of one style", "", "| photos pooled | names the style's family | the style's family among the first three |", "|---|---|---|"]
q1 = {}
for k in (1, 2, 3, 4):
    a, b = np.zeros(N), np.zeros(N)
    for P in Ps:
        for i, (st, rows) in enumerate(pure):
            order = np.argsort(-(P[rows[:k]].mean(0) @ M))
            a[i] += order[0] == style_fam[st]
            b[i] += style_fam[st] in order[:3]
    q1[k] = a / len(Ps)
    L.append(f"| {k} | {share(a / len(Ps))} | {share(b / len(Ps))} |")
    print(L[-1], flush=True)
keys = ("top_is_a", "s_fam", "rel_fam", "second_fam", "rand", "explain", "seen", "s_seen")
res = {k: np.zeros(N) for k in keys}
best_m, best_p = np.zeros(N), np.zeros(N)
for P in Ps:
    for i, (st, t, rows) in enumerate(mixed):
        a, b = style_fam[st], style_fam[t]
        prof = vp.pooled(P[rows], np.ones(4, bool), "equal")
        s, rel, unexp = vp.surprise(P[rows], prof, SIM)
        first, pick = int(prof.argmax()), int(s.argmax())
        rel_x = rel.copy()
        rel_x[first] = -1
        res["top_is_a"][i] += lab_fam[first] == a
        res["s_fam"][i] += lab_fam[pick] == b
        res["rel_fam"][i] += lab_fam[int(rel_x.argmax())] == b
        res["second_fam"][i] += int(np.argsort(-(prof @ M))[1]) == b
        res["rand"][i] += np.mean(lab_fam[unexp >= unexp[pick] - 1e-12] == b)
        res["explain"][i] += int(P[rows][:, pick].argmax()) == 3
        seen = int((P[rows[3]] @ M).argmax()) == b
        res["seen"][i] += seen
        res["s_seen"][i] += seen and lab_fam[pick] == b
        best_m[i] += s.max()
    for i, (st, rows) in enumerate(pure):
        best_p[i] += vp.surprise(P[rows], vp.pooled(P[rows], np.ones(4, bool), "equal"), SIM)[0].max()
for k in keys:
    res[k] /= len(Ps)
best_m /= len(Ps)
best_p /= len(Ps)
auc = float((best_m[:, None] > best_p[None, :]).mean())
L += ["", "## Question 2: three photos of one style and one of a style from another family", "", "| | share of sets |", "|---|---|",
      f"| the profile's first aesthetic is of the main style's family | {share(res['top_is_a'])} |",
      f"| **the surprise score's first suggestion is of the other style's family** | {share(res['s_fam'])} |",
      f"| relevance alone picks that family | {share(res['rel_fam'])} |",
      f"| the profile's second family is that family (no surprise score) | {share(res['second_fam'])} |",
      f"| a random aesthetic at least as unexpected is of that family | {res['rand'].mean():.1%} |",
      f"| the photo given as the reason is the odd one out | {share(res['explain'])} |", "",
      f"- On its own the network puts the odd photo in its style's family {res['seen'].mean():.1%} of the time; in those cases the surprise score's first suggestion "
      f"is of that family {res['s_seen'].sum() / max(res['seen'].sum(), 1e-9):.1%} of the time.",
      f"- The largest S of a mixed set is higher than that of a pure set {auc:.1%} of the time (50% is chance); medians {np.median(best_m):.2f} and {np.median(best_p):.2f}. "
      f"At the app's cut of 0.25, {np.mean(best_m >= 0.25):.1%} of mixed sets and {np.mean(best_p >= 0.25):.1%} of pure sets would show a surprise.",
      "", "Intervals resample sets (1,000 draws). Sets are by style, not by person.", f"time elapsed {time.time() - T0:.0f}s"]
(OUT / "fs14-sets.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L[L.index("## Question 2: three photos of one style and one of a style from another family"):]), flush=True)
