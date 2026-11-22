"""What changes when a person gives many photos (up to 50) instead of 3 to 5?

    python many_photos.py [--root E:/vega] [--device cuda] [--agree-up-to 16]
        -> runs/personas-v1/many-photos.md      the tables
           code/app/surprise_cuts.json          the cut per number of photos, read by the app

Everything about the app was measured with sets of 3 to 5 photos. Two things depend on the number of photos n:
  the profile   more photos should help; does weighting by agreement still beat the plain average?
  the surprise  relevance was the largest share ANY ONE photo gives a family, and the largest of 50 is larger than the largest
                of 4, so with one fixed cut most large sets would be shown a "surprise".

Part A compares three ways to pool at each size: the plain average, agreement weights with a geometric mean (the app's
"agree"), and the same weights with an ordinary average.
Part B uses the rule the app then follows ("agree" up to --agree-up-to photos, "agree_mean" beyond; relevance read from
the photo a tenth of the way down, vega_profile.support_rank) and finds, for every n from 2 to 50, the cut at which 1 set of
ONE style in 10 is shown a surprise. Up to 5 photos the app keeps its validated cut; from 6 on it reads this table.

Sets of n expert-labelled outfit photos (FashionStyle14's 11 non-transfer styles, never trained on):
  one style        n photos of one style
  one odd photo    n - 1 of one style and 1 of a style of another family
  a side style     about a fifth of the photos (at least 1) are of ONE style of another family
and of real posters: n photos of one poster (held-out group 0), no truth, only how often a surprise would be shown.
The network is the app's (frozen settings, street photos, seed 422, wiki training + validation images). The wiki test split
and FashionStyle14's lolita, fairy and ethnic photos are not read. No model or setting is changed.
"""
import csv
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

T0 = time.time()
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROOT, DEVICE = Path(arg("--root", "E:/vega")), arg("--device", "cuda")
HERE = Path(__file__).resolve().parent
NS = [2, 3, 4, 5, 8, 12, 16, 20, 30, 50]
PER = 1000                                                             # sets per size and kind in the tables
PER_CUT = 2000                                                         # one-style sets per size for the cuts
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "nn"))
sys.path.insert(0, str(HERE))
import train_nn  # noqa: E402
import vega_data  # noqa: E402
import vega_profile as vp  # noqa: E402

app = (HERE / "vega_app.py").read_text(encoding="utf-8")
found = re.search(r"^AGREE_UP_TO = (\d+)", app, re.M)
AGREE_UP_TO = int(arg("--agree-up-to", found.group(1) if found else 16))
SMALL_CUT = float(re.search(r"^SURPRISE_MIN = ([0-9.]+)", app, re.M).group(1))
read = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
V1, OUT = ROOT / "runs" / "family-v1", ROOT / "runs" / "personas-v1"
TRANSFER = {"lolita", "fairy", "ethnic"}
rng = np.random.default_rng(422)
d = vega_data.load(root=ROOT)
classes = [str(c) for c in d["classes"]]
X = np.concatenate([d["fit"]["x"], d["eval"]["x"]]).astype(np.float32)
Y = np.concatenate([d["fit"]["y"], d["eval"]["y"]]).astype(np.float32)
fams = [r["family"] for r in read(V1 / "families.csv")]
of_label = {r["wiki_label"]: r["family"] for r in read(V1 / "label_family.csv")}
lab_fam = np.asarray([fams.index(of_label[c]) for c in classes])
M = np.zeros((len(classes), len(fams)))
M[np.arange(len(classes)), lab_fam] = 1
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
SIM = vp.aesthetic_map(Xn.astype(np.float64), Y > 0)
SIMF = vp.family_map(SIM, lab_fam, len(fams))

vec = f"{vega_data.VECTORS}-448"
fdir = ROOT / "emb" / "fashionstyle14"
style_fam = {r["style_or_tag"]: fams.index(r["family"]) for r in read(V1 / "outside_family.csv")
             if r["source"] == "FashionStyle14" and r["family"] in fams and r["style_or_tag"] not in TRANSFER}
frow = {s: i for i, s in enumerate((fdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
styles = {}
for r in read(fdir / "manifest.csv"):
    styles.setdefault(r["sha1"], set()).add(r["style"])
ph = sorted((frow[s], next(iter(st))) for s, st in styles.items() if len(st) == 1 and s in frow and next(iter(st)) in style_fam)
XF = np.ascontiguousarray(np.load(fdir / vec / "embeddings.npy", mmap_mode="r")[[r for r, _ in ph]])
style = np.asarray([st for _, st in ph])
kinds = sorted(set(style.tolist()))
of_style = {k: np.flatnonzero(style == k) for k in kinds}

sdir = ROOT / "emb" / "street-fashion-style"
srow = {s: i for i, s in enumerate((sdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
held = sorted((r["user"], srow[r["id"]]) for r in read(sdir / "manifest.csv")
              if r["id"] in srow and int(hashlib.sha1(r["user"].encode("utf-8")).hexdigest(), 16) % train_nn.REAL_GROUPS == 0)
XP = np.ascontiguousarray(np.load(sdir / vec / "embeddings.npy", mmap_mode="r")[[r for _, r in held]])
by_user = defaultdict(list)
for i, (u, _) in enumerate(held):
    by_user[u].append(i)

print(f"training the app's network on {DEVICE} and scoring {len(XF) + len(XP):,} outfit photos ...", flush=True)
z = train_nn.train(X, Y, np.concatenate([XF, XP]), train_nn.SETTINGS, 422, DEVICE, real=train_nn.real_photos(ROOT, classes)).astype(np.float64)
z -= z.max(1, keepdims=True)
PALL = np.exp(z)
PALL /= PALL.sum(1, keepdims=True)
PF_, PP_ = PALL[:len(XF)], PALL[len(XF):]


def other(k):
    return str(rng.choice([q for q in kinds if style_fam[q] != style_fam[k]]))


def pure_set(n):
    k = str(rng.choice(kinds))
    return PF_[rng.choice(of_style[k], n, replace=False)], style_fam[k], -1


def odd_set(n):
    k = str(rng.choice(kinds))
    t = other(k)
    return PF_[np.append(rng.choice(of_style[k], n - 1, replace=False), rng.choice(of_style[t]))], style_fam[k], style_fam[t]


def side_set(n):
    k = str(rng.choice(kinds))
    t, m = other(k), max(1, round(n / 5))
    return PF_[np.append(rng.choice(of_style[k], n - m, replace=False), rng.choice(of_style[t], m, replace=False))], style_fam[k], style_fam[t]


def poster_set(n, users):
    return PP_[rng.choice(by_user[str(rng.choice(users))], n, replace=False)], -1, -1


POOLS = [("plain average", lambda P: P.mean(0)), ("agreement, geometric", lambda P: vp.pooled(P, np.ones(len(P), bool), "agree", lab_fam=lab_fam)),
         ("agreement, ordinary average", lambda P: vp.pooled(P, np.ones(len(P), bool), "agree_mean", lab_fam=lab_fam))]
app_pool = lambda P: vp.pooled(P, np.ones(len(P), bool), "agree" if len(P) <= AGREE_UP_TO else "agree_mean", lab_fam=lab_fam)


def best_surprise(P):
    """The app's best surprise score for a set, and its family."""
    s = vp.surprise_family(P, app_pool(P), SIMF, lab_fam, rank=vp.support_rank(len(P)))[0]
    return float(s.max()), int(s.argmax())


L = ["# Many photos: what changes from 4 photos to 50", "",
     f"Made by `code/app/many_photos.py` on {DEVICE}; the app's network (seed 422). Expert-labelled outfit photos unless it says posters; {PER:,} sets of each kind and size.", "",
     "## A. The profile: is the pooled top family the main style's family?", "",
     "Differences are against the plain average on the same sets, in points, with 95% intervals.", "",
     "| photos | sets | plain average | agreement, geometric | difference | agreement, ordinary average | difference |", "|---|---|---|---|---|---|---|"]
for n in NS:
    for label, make in (("one style", pure_set), ("a side style", side_set)):
        hit = np.zeros((PER, len(POOLS)))
        for i in range(PER):
            P, main, _ = make(n)
            hit[i] = [int((pool(P) @ M).argmax()) == main for _, pool in POOLS]
        cells = [f"{hit[:, 0].mean():.1%}"]
        for j in (1, 2):
            dlt = (hit[:, j] - hit[:, 0]) * 100
            se = dlt.std(ddof=1) / np.sqrt(PER)
            cells += [f"{hit[:, j].mean():.1%}", f"{dlt.mean():+.1f} ({dlt.mean() - 1.96 * se:+.1f} to {dlt.mean() + 1.96 * se:+.1f})"]
        L.append(f"| {n} | {label} | " + " | ".join(cells) + " |")
        print(L[-1], flush=True)

# ---------------------------------------------------------------- B. the surprise under the app's rule
cuts_raw = {}
for n in range(2, 51):
    cuts_raw[n] = float(np.percentile([best_surprise(pure_set(n)[0])[0] for _ in range(PER_CUT)], 90))
ns = sorted(cuts_raw)
smooth = {n: float(np.mean([cuts_raw[m] for m in (n - 1, n, n + 1) if m in cuts_raw])) for n in ns}       # each size with its two neighbours
CUT = lambda n: SMALL_CUT if n <= 5 else smooth[n]
(HERE / "surprise_cuts.json").write_text(json.dumps(
    {"made_by": "code/app/many_photos.py", "what": "surprise cut per number of photos: 1 expert-labelled set of ONE style in 10 is shown a surprise; "
     "each size averaged with its two neighbours. The app uses its own SURPRISE_MIN up to 5 photos and this table from 6.",
     "agree_up_to": AGREE_UP_TO, "small_cut": SMALL_CUT, "cut": {str(n): round(smooth[n], 3) for n in ns},
     "raw": {str(n): round(cuts_raw[n], 3) for n in ns}}, indent=1) + "\n", encoding="utf-8")
L += ["", f"## B. The surprise, as the app now works it out", "",
      f"Pooling: agreement weights with a geometric mean up to {AGREE_UP_TO} photos, with an ordinary average beyond. Relevance of a family: its share in the photo a tenth of the way down "
      "(the strongest up to 10 photos, the 2nd strongest of 20, the 5th of 50). 'Old rule' is the strongest photo with the one cut for small sets.", "",
      "| photos | old rule: one-style sets shown a surprise | cut now | one style: shown | one odd photo: shown | a side style: shown | of those, its family named | posters: shown |",
      "|---|---|---|---|---|---|---|---|"]
for n in NS:
    users = [u for u, idx in by_user.items() if len(idx) >= n]
    old = np.mean([vp.surprise_family(P, app_pool(P), SIMF, lab_fam)[0].max() >= SMALL_CUT for P in (pure_set(n)[0] for _ in range(PER))])
    res = {}
    for label, make in (("pure", pure_set), ("odd", odd_set), ("side", side_set), ("post", lambda m: poster_set(m, users))):
        rows = []
        for _ in range(PER):
            P, _, b = make(n)
            s, f = best_surprise(P)
            rows.append((s >= CUT(n), f == b))
        res[label] = np.asarray(rows, float)
    shown = res["side"][:, 0] > 0
    L.append(f"| {n} | {old:.1%} | {CUT(n):.2f} | {res['pure'][:, 0].mean():.1%} | {res['odd'][:, 0].mean():.1%} | {res['side'][:, 0].mean():.1%} | "
             f"{res['side'][shown, 1].mean():.1%} | {res['post'][:, 0].mean():.1%} |")
    print(L[-1], flush=True)
L += ["", "The cut for every number of photos is in `code/app/surprise_cuts.json`. A side style is about a fifth of the photos; with 10 photos or fewer one photo is "
      "enough to raise a surprise, beyond that a single odd photo is deliberately no longer enough.", "", f"time elapsed {time.time() - T0:.0f}s"]
(OUT / "many-photos.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"time elapsed {time.time() - T0:.0f}s", flush=True)
