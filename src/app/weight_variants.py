"""Should some of a person's photos count more than others? Ways to weight the photos of one set, against the plain average.

    python weight_variants.py [--root E:/vega] [--seeds 3] [--device cuda]      -> runs/personas-v1/weight-variants.md

The question (Andrew, 21 Nov): a person uploads 3 to 5 photos; which of them form their actual aesthetic, and should some
photos weigh more than others? persona_eval.py already tried two kinds of weight, an outfit photo counting more or less
than a context photo (x0.5 to x4) and a photo counting by how sure the network is of it, and the plain average won. This
tries the kind it did not: a photo counts by how much it AGREES with the person's other photos, so the odd one out counts
less. No weight here uses a label.

Ways to pool (P is photos x aesthetics, one row of probabilities per photo):
  equal                     the plain average (what the app does)
  confidence ^p             a photo's weight is its strongest match, to the power p
  agreement on the map ^p   a photo's weight is its average closeness to the other photos, where the closeness of two photos
                            is the expected similarity of their aesthetics on the map (p_i' SIM p_j), divided by the largest, to the power p
  agreement on family ^p    the same with closeness = how much two photos' family shares overlap
  drop the odd one out      the photo that agrees least on family is left out, the rest average
  geometric mean            the average of the photos' log-probabilities: an aesthetic must suit every photo
  geometric, by agreement   the same average of log-probabilities, each photo weighted by its agreement on family ^p
  ORACLE                    the average of the photos that truly are of the main style: what perfect weights would reach

Sets, each pure (all photos of one style) and mixed (one photo of another family's style, always the last row):
  wiki personas        runs/personas-v1/personas.csv, report half and, as a repeat, the tune half (nothing here is tuned on
                       either); held-out predictions of oof.py; is the home aesthetic first, in the top five
  expert-labelled      FashionStyle14's 11 non-transfer styles, four photos; is the pooled top family the main style's family
The wiki test split and FashionStyle14's lolita, fairy and ethnic photos are not read. No model or setting is changed.
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
M[np.arange(len(classes)), lab_fam] = 1                                # aesthetics x families
fx = np.asarray(fit["x"], np.float64)
SIM = vp.aesthetic_map(fx / np.linalg.norm(fx, axis=1, keepdims=True), np.asarray(fit["y"]) > 0)
rng = np.random.default_rng(422)

# ---------------------------------------------------------------- the sets: (rows, target, number of photos of the main style)
z = np.load(OUT / "oof.npz", allow_pickle=False)
wid = {str(a): i for i, a in enumerate(z["ids"])}
at = {c: i for i, c in enumerate(classes)}
W, w_home = {}, {}                                                       # (half, kind) -> sets, and their home aesthetics' names
for r in read(OUT / "personas.csv"):
    rows = np.asarray([wid[i] for i in r[[k for k in r if k.startswith("images")][0]].split()])
    W.setdefault((r["half"], r["kind"]), []).append((rows, at[r["home"]], len(rows) - (r["kind"] == "mixed")))
    w_home.setdefault((r["half"], r["kind"]), []).append(r["home"])
PW = [z["probs"].astype(np.float64)]

fdir, vec = ROOT / "emb" / "fashionstyle14", f"{vega_data.VECTORS}-448"
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
f_pure = [(rng.choice(of_style[k], 4, replace=False), style_fam[k], 4) for k in rng.choice(kinds, 3000)]
f_mixed, f_other = [], []                                              # f_other: the family of the off-style photo
for k in rng.choice(kinds, 3000):
    t = str(rng.choice([q for q in kinds if style_fam[q] != style_fam[k]]))
    f_mixed.append((np.append(rng.choice(of_style[k], 3, replace=False), rng.choice(of_style[t])), style_fam[k], 3))
    f_other.append(style_fam[t])
real = train_nn.real_photos(ROOT, classes)
PF14 = []
for seed in SEEDS:
    zz = train_nn.train(fit["x"], fit["y"], XF, train_nn.SETTINGS, seed, DEVICE, real=real).astype(np.float64)
    zz -= zz.max(1, keepdims=True)
    p = np.exp(zz)
    PF14.append(p / p.sum(1, keepdims=True))
    print(f"  seed {seed}: FashionStyle14 photos scored", flush=True)
del real


# ---------------------------------------------------------------- the ways to pool; each returns one profile (a row over aesthetics)
def others(A):
    """Each photo's average closeness to the other photos."""
    return (A.sum(1) - np.diag(A)) / (len(A) - 1)


def on_map(P):
    return others(P @ SIM @ P.T)


def on_family(P):
    F = P @ M
    return others(np.minimum(F[:, None, :], F[None, :, :]).sum(2))


def by(weight, power=1.0):
    def pool(P, n_main):
        w = weight(P)
        w = (w / w.max()) ** power
        return (w[:, None] * P).sum(0) / w.sum()
    return pool


def drop_odd(P, n_main):
    keep = np.ones(len(P), bool)
    if len(P) >= 3:
        keep[int(on_family(P).argmin())] = False
    return P[keep].mean(0)


def geometric(P, n_main):
    g = np.exp(np.log(P + 1e-9).mean(0))
    return g / g.sum()


def geometric_by(power):
    def pool(P, n_main):
        w = on_family(P)
        w = (w / w.max()) ** power
        g = np.exp((w[:, None] * np.log(P + 1e-9)).sum(0) / w.sum())
        return g / g.sum()
    return pool


WAYS = [("equal (the app)", lambda P, n: P.mean(0)),
        ("confidence ^1", by(lambda P: P.max(1), 1)), ("confidence ^2", by(lambda P: P.max(1), 2)),
        ("agreement on the map ^2", by(on_map, 2)), ("agreement on the map ^8", by(on_map, 8)),
        ("agreement on family ^1", by(on_family, 1)), ("agreement on family ^2", by(on_family, 2)), ("agreement on family ^4", by(on_family, 4)),
        ("drop the odd one out", drop_odd), ("geometric mean", geometric),
        ("geometric, by agreement on family ^1 (vega_profile's \"agree\")", lambda P, n: vp.pooled(P, np.ones(len(P), bool), "agree", lab_fam=lab_fam)), ("geometric, by agreement on family ^2", geometric_by(2)),
        ("ORACLE: only the photos of the main style", lambda P, n: P[:n].mean(0))]


def hits(sets, Ps, pool, family):
    """Per set, averaged over the networks: is the target first, and is it in the top five (aesthetic) or top three (family)."""
    first, among = np.zeros(len(sets)), np.zeros(len(sets))
    for P in Ps:
        for i, (rows, target, n_main) in enumerate(sets):
            prof = pool(P[rows], n_main)
            prof = prof @ M if family else prof
            order = np.argsort(-prof)
            first[i] += order[0] == target
            among[i] += target in order[:3 if family else 5]
    return first / len(Ps), among / len(Ps)


def interval(v, groups=None):
    """Mean with a 95% interval; resamples sets, or whole groups of sets (the personas of one home aesthetic)."""
    if groups is None:
        b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
    else:
        names = sorted(set(groups))
        idx = {g: np.flatnonzero(np.asarray(groups) == g) for g in names}
        b = [v[np.concatenate([idx[names[j]] for j in rng.integers(0, len(names), len(names))])].mean() for _ in range(1000)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


L = ["# Should some photos count more than others? Ways to weight a person's photos", "",
     f"Made by `code/app/weight_variants.py`, seeds {SEEDS} on {DEVICE} for the outfit photos; wiki personas use the held-out predictions of oof.py. "
     "No weight uses a label, except the ORACLE row, which shows what perfect weights would reach. Differences are against the plain average on the same sets, "
     "in points, with 95% intervals (wiki: home aesthetics resampled; outfit photos: sets resampled).", ""]
BLOCKS = [("Wiki personas, all photos of one aesthetic: the home aesthetic", W["report", "pure"], PW, False, w_home["report", "pure"], False),
          ("Wiki personas, one photo of another family: the home aesthetic", W["report", "mixed"], PW, False, w_home["report", "mixed"], True),
          ("Expert-labelled outfit photos, four of one style: the style's family", f_pure, PF14, True, None, False),
          ("Expert-labelled outfit photos, three of one style and one of another family: the main style's family", f_mixed, PF14, True, None, True),
          ("Repeat on the other half of the wiki personas, all photos of one aesthetic", W["tune", "pure"], PW, False, w_home["tune", "pure"], False),
          ("Repeat on the other half of the wiki personas, one photo of another family", W["tune", "mixed"], PW, False, w_home["tune", "mixed"], True)]
for title, sets, Ps, family, groups, has_odd in BLOCKS:
    among = "in the top three" if family else "in the top five"
    L += [f"## {title} ({len(sets):,} sets)", "", f"| photos weighted by | is first | against equal | {among} | against equal |", "|---|---|---|---|---|"]
    base = None
    for name, pool in WAYS:
        if name.startswith("ORACLE") and not has_odd:
            continue                                                   # with no off-style photo the oracle is the plain average
        first, top = hits(sets, Ps, pool, family)
        if base is None:
            base = (first, top)
            L.append(f"| {name} | {first.mean():.1%} | | {top.mean():.1%} | |")
        else:
            cells = []
            for v, b in ((first, base[0]), (top, base[1])):
                lo, hi = interval((v - b) * 100, groups)
                cells += [f"{v.mean():.1%}", f"{(v - b).mean() * 100:+.1f} ({lo:+.1f} to {hi:+.1f})"]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        print(title[:40], "|", L[-1], flush=True)
    L.append("")
# ---------------------------------------------------------------- what the profile's weighting does to the surprise score (per family, as in the app)
SIMF = vp.family_map(SIM, lab_fam, len(fams))
CUT = 0.38
L += [f"## What it does to the surprise score (expert-labelled outfit photos; per family, shown from {CUT} as in the app)", "",
      "| profile | sets of one style shown a surprise | sets with an off-style photo shown a surprise | of those, it names the off-style photo's family | names that family, all such sets | the cut that shows 1 set of one style in 10 | at that cut: sets with an off-style photo shown | of those, right family |",
      "|---|---|---|---|---|---|---|---|"]
four = np.ones(4, bool)
for name, mode in (("equal (the app)", "equal"), ("geometric, by agreement on family ^1", "agree")):
    sp, sm, hit = [], [], []
    for P in PF14:
        for rows, _, _ in f_pure:
            sp.append(vp.surprise_family(P[rows], vp.pooled(P[rows], four, mode, lab_fam=lab_fam), SIMF, lab_fam)[0].max())
        for (rows, _, _), b in zip(f_mixed, f_other):
            s = vp.surprise_family(P[rows], vp.pooled(P[rows], four, mode, lab_fam=lab_fam), SIMF, lab_fam)[0]
            sm.append(s.max())
            hit.append(int(s.argmax()) == b)
    sp, sm, hit = np.asarray(sp), np.asarray(sm), np.asarray(hit, float)
    own = float(np.percentile(sp, 90))
    L.append(f"| {name} | {np.mean(sp >= CUT):.1%} | {np.mean(sm >= CUT):.1%} | {hit[sm >= CUT].mean():.1%} | {hit.mean():.1%} | {own:.3f} | "
             f"{np.mean(sm >= own):.1%} | {hit[sm >= own].mean():.1%} |")
    print(L[-1], flush=True)
L += ["", f"time elapsed {time.time() - T0:.0f}s"]
(OUT / "weight-variants.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"time elapsed {time.time() - T0:.0f}s", flush=True)
