"""Real people's photo sets: the profile and the surprise score on Street Fashion Style posters the network never trained on.

    python poster_eval.py [--root E:/vega] [--seeds 3] [--device cuda]      -> runs/personas-v1/posters.md

Street Fashion Style is outfit posts by real people, each with the style word its poster chose. A tag stands for one of the
20 families (runs/family-v1/outside_family.csv), so everything here is judged at FAMILY level: nobody has said which
aesthetic a street photo is. Only posters of group 0 are used: the network (train_nn.train, frozen settings, wiki training
images + the other nine groups' street photos) has seen none of their photos. The wiki test split is not read.

Question 1, more photos: take k of a poster's photos, pool them, and name a family. Is it the family the poster uses most
in their OTHER photos? k = 1, 2, 3, 5. (The other photos are the reference, so a photo is never judged by its own tag.)

Question 2, surprise: take three photos the poster tagged with their main family A and one they tagged with another family
B. Does the surprise score (vega_profile.surprise) point at B's family, and at the right photo?
"""
import csv
import hashlib
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

T0 = time.time()
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROOT, DEVICE = Path(arg("--root", "E:/vega")), arg("--device", "cuda")
SEEDS = [422 + i for i in range(int(arg("--seeds", 3)))]
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "nn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_nn  # noqa: E402
import vega_data  # noqa: E402
import vega_profile as vp  # noqa: E402

read = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
V1, OUT = ROOT / "runs" / "family-v1", ROOT / "runs" / "personas-v1"
d = vega_data.load(root=ROOT)                                          # training and validation; only the training images are fitted
fit, classes = d["fit"], [str(c) for c in d["classes"]]
fams = [r["family"] for r in read(V1 / "families.csv")]
fname = {r["family"]: r["name"] for r in read(V1 / "families.csv")}
of_label = {r["wiki_label"]: r["family"] for r in read(V1 / "label_family.csv")}
lab_fam = np.asarray([fams.index(of_label[c]) for c in classes])
M = np.zeros((len(classes), len(fams)))
M[np.arange(len(classes)), lab_fam] = 1
tag_fam = {r["style_or_tag"]: fams.index(r["family"]) for r in read(V1 / "outside_family.csv") if r["source"] == "Street Fashion Style" and r["family"] in fams}
fx = np.asarray(fit["x"], np.float64)
SIM = vp.aesthetic_map(fx / np.linalg.norm(fx, axis=1, keepdims=True), np.asarray(fit["y"]) > 0)

sdir, vec = ROOT / "emb" / "street-fashion-style", f"{vega_data.VECTORS}-448"
srow = {s: i for i, s in enumerate((sdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
tagged = [(r["user"], srow[r["id"]], tag_fam[r["style"]], int(hashlib.sha1(r["user"].encode("utf-8")).hexdigest(), 16) % train_nn.REAL_GROUPS)
          for r in read(sdir / "manifest.csv") if r["style"] in tag_fam and r["id"] in srow]
held = sorted((u, r, f) for u, r, f, g in tagged if g == 0)
# how common each family is among the posts of the OTHER nine groups (the ones the network trains on): a prior for street photos
PRIOR = np.bincount([f for _, _, f, g in tagged if g != 0], minlength=len(fams)).astype(np.float64)
PRIOR /= PRIOR.sum()
X = np.ascontiguousarray(np.load(sdir / vec / "embeddings.npy", mmap_mode="r")[[r for _, r, _ in held]])
user = np.asarray([u for u, _, _ in held])
tagf = np.asarray([f for _, _, f in held])
by_user = defaultdict(list)
for i, u in enumerate(user):
    by_user[u].append(i)
print(f"held-out posters: {len(by_user):,} with {len(held):,} photos carrying a tag in use; photos per poster: median {int(np.median([len(v) for v in by_user.values()]))}, "
      f"{sum(len(v) >= 8 for v in by_user.values())} posters have 8 or more", flush=True)

real = train_nn.real_photos(ROOT, classes)                             # group 0 held out, as always
P_real, P_wiki = [], []
for seed in SEEDS:
    for store, r in ((P_real, real), (P_wiki, None)):
        z = train_nn.train(fit["x"], fit["y"], X, train_nn.SETTINGS, seed, DEVICE, real=r).astype(np.float64)
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        store.append((p / p.sum(1, keepdims=True)).astype(np.float32))
    print(f"  seed {seed}: both networks trained and the held-out photos scored; per photo, family right: "
          f"NN {((P_real[-1] @ M).argmax(1) == tagf).mean():.1%}, wiki-only {((P_wiki[-1] @ M).argmax(1) == tagf).mean():.1%}", flush=True)
del real

# ---------------------------------------------------------------- question 1: k photos against the poster's other photos
rng = np.random.default_rng(422)
KS, REPEATS = (1, 2, 3, 5), 5
sets = []                                                              # (poster, 5 persona photos, the reference family, the set of reference families)
for u, idx in by_user.items():
    if len(idx) < 8:
        continue
    for _ in range(REPEATS):
        pick = rng.choice(idx, 5, replace=False)
        rest = Counter(tagf[[i for i in idx if i not in set(pick.tolist())]].tolist())
        (f1, n1), *more = rest.most_common(2) if len(rest) > 1 else (rest.most_common(1)[0],)
        if more and more[0][1] == n1:
            continue                                                   # no single most-used family among the other photos
        sets.append((u, pick, f1, set(rest)))
users_q1 = sorted({s[0] for s in sets})
draws = [rng.choice(users_q1, len(users_q1)) for _ in range(2000)]
rows_of = defaultdict(list)
for i, s in enumerate(sets):
    rows_of[s[0]].append(i)


def ci(v):
    v = np.asarray(v, np.float64)
    boot = [np.concatenate([v[rows_of[u]] for u in dr]).mean() for dr in draws]
    return f"{v.mean():.1%} ({np.percentile(boot, 2.5):.1%} to {np.percentile(boot, 97.5):.1%})"


def hits(Ps, k, how="main"):
    """Per set, the share of seeds that are right. how: main = names the poster's main family; prior = the same after weighting
    families by how common they are in the other groups' posts; top3 = the main family is among the first three families;
    any = names a family the poster uses at all."""
    out = np.zeros(len(sets))
    for P in Ps:
        for i, (_, pick, f1, allf) in enumerate(sets):
            pf = P[pick[:k]].mean(0) @ M
            if how == "prior":
                pf = pf * PRIOR
            order = np.argsort(-pf)
            out[i] += (order[0] in allf) if how == "any" else (f1 in order[:3]) if how == "top3" else (order[0] == f1)
    return out / len(Ps)


ref = np.asarray([s[2] for s in sets])
common = Counter(ref.tolist()).most_common(1)[0]
macro = lambda v: float(np.mean([v[ref == f].mean() for f in sorted(set(ref.tolist()))]))    # every main family counts the same
L = ["# Real people's photo sets: Street Fashion Style posters the network never trained on", "",
     f"Made by `code/app/poster_eval.py`, seeds {SEEDS} on {DEVICE}. {len(by_user):,} held-out posters (group 0), {len(held):,} photos with a tag in use. "
     "Judged at family level against the posters' own tags. The wiki test split was not read.", "",
     "## Question 1: do more photos of a person give a better profile?", "",
     f"{len(users_q1)} posters with 8 or more photos; {len(sets):,} draws of five photos. The reference is the family the poster uses most in their OTHER photos.", "",
     f"**Read this first:** that family is '{fname[fams[common[0]]]}' for {common[1] / len(sets):.1%} of the draws, so always answering it would be right {common[1] / len(sets):.1%} of the time. "
     f"The network is trained with every family drawn equally often, so it does not know that. Two columns correct for it: 'with the street prior' multiplies the network's family "
     f"shares by how common each family is in the other nine groups' posts, and 'every family counted the same' averages the hit rate over the {len(set(ref.tolist()))} main families.", "",
     "| photos pooled | names the poster's main family | with the street prior | every family counted the same | main family among the first three | names a family the poster uses at all |",
     "|---|---|---|---|---|---|"]
h = {}
for k in KS:
    h[k] = {how: hits(P_real, k, how) for how in ("main", "prior", "top3", "any")}
    L.append(f"| {k} | {ci(h[k]['main'])} | {ci(h[k]['prior'])} | {macro(h[k]['main']):.1%} | {ci(h[k]['top3'])} | {ci(h[k]['any'])} |")
    print(L[-1], flush=True)
L += ["", "Five photos against one, same draws:"]
for how, name in (("main", "names the main family"), ("prior", "with the street prior"), ("top3", "main family among the first three"), ("any", "names a family the poster uses")):
    dlt = (h[5][how] - h[1][how]) * 100
    boot = [np.concatenate([dlt[rows_of[u]] for u in dr]).mean() for dr in draws]
    L.append(f"- {name}: {dlt.mean():+.1f} points ({np.percentile(boot, 2.5):+.1f} to {np.percentile(boot, 97.5):+.1f})")
    print(L[-1], flush=True)
wk = {k: hits(P_wiki, k, "top3") for k in (1, 5)}
L += [f"- wiki-only NN, main family among the first three: {wk[1].mean():.1%} with one photo, {wk[5].mean():.1%} with five (the NN: {h[1]['top3'].mean():.1%} and {h[5]['top3'].mean():.1%})",
      "", "Intervals resample posters (2,000 draws, seed 422).", ""]

# ---------------------------------------------------------------- question 2: three photos of the main family, one of another
mixed, pure = [], []                                                   # (poster, photos, A, B, position of the planted photo)
for u, idx in by_user.items():
    c = Counter(tagf[idx].tolist())
    (a, na), *rest = c.most_common()
    if na < 4 or (rest and rest[0][1] == na):
        continue
    mine = [i for i in idx if tagf[i] == a]
    for _ in range(REPEATS):
        pure.append((u, rng.choice(mine, 4, replace=False), a, -1, -1))
        if rest:
            b = int(rng.choice([f for f, _ in rest]))
            planted = int(rng.choice([i for i in idx if tagf[i] == b]))
            mixed.append((u, np.append(rng.choice(mine, 3, replace=False), planted), a, b, 3))
users_q2 = sorted({m[0] for m in mixed})
rows2 = defaultdict(list)
for i, m in enumerate(mixed):
    rows2[m[0]].append(i)
draws2 = [rng.choice(users_q2, len(users_q2)) for _ in range(2000)]


def ci2(v):
    v = np.asarray(v, np.float64)
    boot = [np.concatenate([v[rows2[u]] for u in dr]).mean() for dr in draws2]
    return f"{v.mean():.1%} ({np.percentile(boot, 2.5):.1%} to {np.percentile(boot, 97.5):.1%})"


res = {k: np.zeros(len(mixed)) for k in ("s_fam", "rel_fam", "explain", "top_is_a", "second_fam", "rand", "seen", "s_seen")}
best_mixed, best_pure = np.zeros(len(mixed)), np.zeros(len(pure))
for P in P_real:
    for i, (_, rows, a, b, pos) in enumerate(mixed):
        prof = vp.pooled(P[rows], np.ones(4, bool), "equal")
        s, rel, unexp = vp.surprise(P[rows], prof, SIM)
        first = int(prof.argmax())
        pick = int(s.argmax())
        rel_x = rel.copy()
        rel_x[first] = -1
        res["s_fam"][i] += lab_fam[pick] == b
        res["rel_fam"][i] += lab_fam[int(rel_x.argmax())] == b
        res["explain"][i] += int(P[rows][:, pick].argmax()) == pos
        res["top_is_a"][i] += lab_fam[first] == a
        pf = prof @ M
        res["second_fam"][i] += int(np.argsort(-pf)[1]) == b
        res["rand"][i] += np.mean(lab_fam[unexp >= unexp[pick] - 1e-12] == b)
        seen = int((P[rows[pos]] @ M).argmax()) == b                   # does the network itself put the planted photo in family B?
        res["seen"][i] += seen
        res["s_seen"][i] += seen and lab_fam[pick] == b
        best_mixed[i] += s.max()
    for i, (_, rows, a, _, _) in enumerate(pure):
        prof = vp.pooled(P[rows], np.ones(4, bool), "equal")
        best_pure[i] += vp.surprise(P[rows], prof, SIM)[0].max()
for k in res:
    res[k] /= len(P_real)
best_mixed /= len(P_real)
best_pure /= len(P_real)
auc = float((best_mixed[:, None] > best_pure[None, :]).mean())
dlt = (res["s_fam"] - res["rel_fam"]) * 100
boot = [np.concatenate([dlt[rows2[u]] for u in dr]).mean() for dr in draws2]
L += ["## Question 2: does the surprise score find the photo that is off the poster's usual style?", "",
      f"{len(users_q2)} posters whose most-used family A has four or more photos and who tagged at least one photo with another family B. "
      f"{len(mixed):,} sets: three photos tagged A and one tagged B.", "",
      "| | share of sets |", "|---|---|",
      f"| the profile's first aesthetic is of family A | {ci2(res['top_is_a'])} |",
      f"| **the surprise score's first suggestion is of family B** | {ci2(res['s_fam'])} |",
      f"| relevance alone picks an aesthetic of family B | {ci2(res['rel_fam'])} |",
      f"| the profile's second family is B (no surprise score) | {ci2(res['second_fam'])} |",
      f"| a random aesthetic at least as unexpected is of family B | {res['rand'].mean():.1%} |",
      f"| the photo given as the reason is the one tagged B | {ci2(res['explain'])} |", "",
      f"- Surprise score against relevance alone, same sets: {dlt.mean():+.1f} points ({np.percentile(boot, 2.5):+.1f} to {np.percentile(boot, 97.5):+.1f}).",
      f"- **The ceiling is the network.** On its own it puts the B-tagged photo in family B only {res['seen'].mean():.1%} of the time. In those cases the surprise score's "
      f"first suggestion is of family B {res['s_seen'].sum() / max(res['seen'].sum(), 1e-9):.1%} of the time.",
      f"- The largest S of a set with a B photo is higher than that of a set of four A photos {auc:.1%} of the time (50% is chance); medians {np.median(best_mixed):.2f} and {np.median(best_pure):.2f}. "
      f"At the app's cut of 0.25, {np.mean(best_mixed >= 0.25):.1%} of mixed sets and {np.mean(best_pure >= 0.25):.1%} of pure sets would show a surprise.", "",
      "## What this is and is not", "",
      "- The ground truth is the poster's own style word, mapped to one of 12 families by our table. It is coarse and noisy: a photo-by-photo check gives the same network about a third right.",
      "- All photos are outfit posts. There is no room, car or object photo here, so this tests 'more photos of a person', not 'context'.",
      "- A photo tagged with another family is off the poster's usual style by their own word. Whether the aesthetic the score names would feel right to them is not measured.",
      "", f"time elapsed {time.time() - T0:.0f}s"]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "posters.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L[L.index("## Question 2: does the surprise score find the photo that is off the poster's usual style?"):]), flush=True)
