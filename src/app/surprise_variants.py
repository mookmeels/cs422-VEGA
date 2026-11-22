"""Which surprise score? The proposal's (per aesthetic) against one worked out per family, on three kinds of photo sets.

    python surprise_variants.py [--root E:/vega] [--seeds 3] [--device cuda]      -> runs/personas-v1/surprise-variants.md

Why: on real outfit photos the network spreads a photo's probability over several aesthetics of one family, so "the strongest
match of any one photo with ONE aesthetic" is small and the proposal's score has little to work with (poster_eval.py).

Ways to pick the surprising suggestion for a set of four photos, one of which is off the set's main style:
  per aesthetic        the proposal's score: S(a) = strongest single-photo match with a  x  (1 - similarity of a to the top aesthetic)
  per family           S(F) = largest share one photo gives family F  x  (1 - similarity of F to the top family); then the
                       aesthetic of F that one photo matches most
  family match only    the per-family score without the distance part (top family excluded)
  second family        no score: the profile's second family

Settings (each with mixed sets, which hold one off-style photo, and pure sets, which do not):
  wiki personas        runs/personas-v1/personas.csv, report half; held-out predictions from oof.py; the planted AESTHETIC is known
  expert-labelled      FashionStyle14's 11 non-transfer styles, sets by style (as fs14_persona_eval.py)
  real posters         Street Fashion Style group 0, three photos tagged with the poster's main family and one with another
The wiki test split and FashionStyle14's lolita, fairy and ethnic photos are not read.
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
TRANSFER = {"lolita", "fairy", "ethnic"}
d = vega_data.load(root=ROOT)
fit, classes = d["fit"], [str(c) for c in d["classes"]]
fams = [r["family"] for r in read(V1 / "families.csv")]
of_label = {r["wiki_label"]: r["family"] for r in read(V1 / "label_family.csv")}
lab_fam = np.asarray([fams.index(of_label[c]) for c in classes])
fx = np.asarray(fit["x"], np.float64)
SIM = vp.aesthetic_map(fx / np.linalg.norm(fx, axis=1, keepdims=True), np.asarray(fit["y"]) > 0)
SIMF = vp.family_map(SIM, lab_fam, len(fams))
out_rows = read(V1 / "outside_family.csv")
rng = np.random.default_rng(422)

# ---------------------------------------------------------------- the three settings: (mixed sets, pure sets, probabilities per seed)
# a mixed set is (rows, family of the off-style photo, its aesthetic or -1); the off-style photo is always the last row
z = np.load(OUT / "oof.npz", allow_pickle=False)
wid = {str(a): i for i, a in enumerate(z["ids"])}
at = {c: i for i, c in enumerate(classes)}
w_mixed, w_pure = [], []
for r in read(OUT / "personas.csv"):
    if r["half"] != "report":
        continue
    rows = np.asarray([wid[i] for i in r[[k for k in r if k.startswith("images")][0]].split()])
    if r["kind"] == "mixed":
        w_mixed.append((rows, int(lab_fam[at[r["planted"]]]), at[r["planted"]]))
    else:
        w_pure.append(rows)
SETTINGS = {"wiki personas": (w_mixed, w_pure, [z["probs"].astype(np.float32)])}

sdir, vec = ROOT / "emb" / "street-fashion-style", f"{vega_data.VECTORS}-448"
tag_fam = {r["style_or_tag"]: fams.index(r["family"]) for r in out_rows if r["source"] == "Street Fashion Style" and r["family"] in fams}
srow = {s: i for i, s in enumerate((sdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
held = sorted((r["user"], srow[r["id"]], tag_fam[r["style"]]) for r in read(sdir / "manifest.csv")
              if r["style"] in tag_fam and r["id"] in srow and int(hashlib.sha1(r["user"].encode("utf-8")).hexdigest(), 16) % train_nn.REAL_GROUPS == 0)
XP = np.ascontiguousarray(np.load(sdir / vec / "embeddings.npy", mmap_mode="r")[[r for _, r, _ in held]])
tagf = np.asarray([f for _, _, f in held])
by_user = defaultdict(list)
for i, (u, _, _) in enumerate(held):
    by_user[u].append(i)
p_mixed, p_pure = [], []
for u, idx in by_user.items():
    (a, na), *rest = Counter(tagf[idx].tolist()).most_common()
    if na < 4 or (rest and rest[0][1] == na):
        continue
    mine = [i for i in idx if tagf[i] == a]
    for _ in range(5):
        p_pure.append(rng.choice(mine, 4, replace=False))
        if rest:
            b = int(rng.choice([f for f, _ in rest]))
            p_mixed.append((np.append(rng.choice(mine, 3, replace=False), int(rng.choice([i for i in idx if tagf[i] == b]))), b, -1))

fdir = ROOT / "emb" / "fashionstyle14"
style_fam = {r["style_or_tag"]: fams.index(r["family"]) for r in out_rows if r["source"] == "FashionStyle14" and r["family"] in fams and r["style_or_tag"] not in TRANSFER}
frow = {s: i for i, s in enumerate((fdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
styles = {}
for r in read(fdir / "manifest.csv"):
    styles.setdefault(r["sha1"], set()).add(r["style"])
ph = sorted((frow[s], next(iter(st))) for s, st in styles.items() if len(st) == 1 and s in frow and next(iter(st)) in style_fam)
XF = np.ascontiguousarray(np.load(fdir / vec / "embeddings.npy", mmap_mode="r")[[r for r, _ in ph]])
style = np.asarray([st for _, st in ph])
kinds = sorted(set(style.tolist()))
of_style = {k: np.flatnonzero(style == k) for k in kinds}
f_pure = [rng.choice(of_style[k], 4, replace=False) for k in rng.choice(kinds, 3000)]
f_mixed = []
for k in rng.choice(kinds, 3000):
    t = str(rng.choice([q for q in kinds if style_fam[q] != style_fam[k]]))
    f_mixed.append((np.append(rng.choice(of_style[k], 3, replace=False), rng.choice(of_style[t])), style_fam[t], -1))

real = train_nn.real_photos(ROOT, classes)
PP, PFS = [], []
for seed in SEEDS:
    zz = train_nn.train(fit["x"], fit["y"], np.concatenate([XP, XF]), train_nn.SETTINGS, seed, DEVICE, real=real).astype(np.float64)
    zz -= zz.max(1, keepdims=True)
    p = np.exp(zz)
    p = (p / p.sum(1, keepdims=True)).astype(np.float32)
    PP.append(p[:len(XP)])
    PFS.append(p[len(XP):])
    print(f"  seed {seed}: posters' and FashionStyle14 photos scored", flush=True)
del real
SETTINGS["expert-labelled outfit photos"] = (f_mixed, f_pure, PFS)
SETTINGS["real posters"] = (p_mixed, p_pure, PP)


# ---------------------------------------------------------------- the ways to pick: each returns (family, aesthetic, photo given as the reason, score)
def per_aesthetic(P):
    prof = P.mean(0)
    s, _, _ = vp.surprise(P, prof, SIM)
    a = int(s.argmax())
    return int(lab_fam[a]), a, int(P[:, a].argmax()), float(s[a])


def _within(P, PF, f, score):
    inside = np.flatnonzero(lab_fam == f)
    return f, int(inside[P[:, inside].max(0).argmax()]), int(PF[:, f].argmax()), float(score)


def per_family(P):
    s, rel, unexp, top, PF = vp.surprise_family(P, P.mean(0), SIMF, lab_fam)
    f = int(s.argmax())
    return _within(P, PF, f, s[f])


def family_match_only(P):
    s, rel, unexp, top, PF = vp.surprise_family(P, P.mean(0), SIMF, lab_fam)
    rel = rel.copy()
    rel[top] = -1
    f = int(rel.argmax())
    return _within(P, PF, f, rel[f])


def second_family(P):
    s, rel, unexp, top, PF = vp.surprise_family(P, P.mean(0), SIMF, lab_fam)
    pf = PF.mean(0)
    f = int(np.argsort(-pf)[1])
    return _within(P, PF, f, pf[f])


WAYS = [("per aesthetic (the proposal's)", per_aesthetic), ("per family", per_family), ("family match only", family_match_only), ("second family, no score", second_family)]
L = ["# Which surprise score? Per aesthetic against per family", "",
     f"Made by `code/app/surprise_variants.py`, seeds {SEEDS} on {DEVICE} for the outfit photos; wiki personas use the held-out predictions of oof.py. "
     "Each mixed set has four photos, the last one off the set's main style. Intervals resample sets (1,000 draws).", ""]
PICK = {}
for name, (mixed, pure, Ps) in SETTINGS.items():
    has_a = mixed[0][2] >= 0
    L += [f"## {name}: {len(mixed):,} mixed sets, {len(pure):,} pure sets", "",
          "| picked by | names the off-style photo's family |" + (" names its exact aesthetic |" if has_a else "") + " the reason given is that photo | tells mixed from pure | mixed sets shown when 1 pure set in 10 is |",
          "|---|---|" + ("---|" if has_a else "") + "---|---|---|"]
    keep = {}
    for way, f in WAYS:
        fam_hit, a_hit, why, sm, sp = np.zeros(len(mixed)), np.zeros(len(mixed)), np.zeros(len(mixed)), np.zeros(len(mixed)), np.zeros(len(pure))
        for P in Ps:
            for i, (rows, b, a) in enumerate(mixed):
                g, ga, photo, score = f(P[rows].astype(np.float64))
                fam_hit[i] += g == b
                a_hit[i] += ga == a
                why[i] += photo == 3
                sm[i] += score
            for i, rows in enumerate(pure):
                sp[i] += f(P[rows].astype(np.float64))[3]
        fam_hit, a_hit, why, sm, sp = fam_hit / len(Ps), a_hit / len(Ps), why / len(Ps), sm / len(Ps), sp / len(Ps)
        keep[way] = fam_hit
        auc = float((sm[:, None] > sp[None, :]).mean())
        cut = float(np.percentile(sp, 90))

        def share(v):
            b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
            return f"{v.mean():.1%} ({np.percentile(b, 2.5):.1%} to {np.percentile(b, 97.5):.1%})"

        L.append(f"| {way} | {share(fam_hit)} |" + (f" {share(a_hit)} |" if has_a else "") + f" {share(why)} | {auc:.1%} | {np.mean(sm >= cut):.1%} (cut {cut:.2f}); "
                 f"of those, right family {fam_hit[sm >= cut].mean():.1%} |")
        print(name, "|", L[-1], flush=True)
        PICK[(name, way)] = cut
    dlt = (keep["per family"] - keep["per aesthetic (the proposal's)"]) * 100
    b = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(1000)]
    L += ["", f"Per family against per aesthetic, same sets, family named: {dlt.mean():+.1f} points ({np.percentile(b, 2.5):+.1f} to {np.percentile(b, 97.5):+.1f}).", ""]
    print(L[-2], flush=True)
L += ["'Tells mixed from pure': how often a mixed set's score is higher than a pure set's (50% is chance). The last column sets the cut where 10% of pure sets would be shown "
      "a surprise, and says how many mixed sets clear it.", f"time elapsed {time.time() - T0:.0f}s"]
(OUT / "surprise-variants.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"time elapsed {time.time() - T0:.0f}s", flush=True)
