"""Personas from wiki images, and the proposal's two questions tested on them. The test split is never read.

    python oof.py                      # first: held-out predictions for every training and validation image
    python persona_eval.py [--root E:/vega] [--per-label 6]      -> runs/personas-v1/results.md, personas.csv

A persona is one outfit image plus context images (rooms, objects, scenery: images with no person at half the image's
height), as the proposal describes. Every image's probabilities come from a network that never trained on it (oof.py).
  pure persona    1 outfit + 2 or 3 context images, all from one aesthetic A (its "home")
  mixed persona   1 outfit + 2 context images from A, plus ONE context image from an aesthetic B of another family: the
                  planted surprise. We know B is relevant (one photo really is B) and off the person's main style.
Only images with exactly one label are used. Home aesthetics are split in two halves by name: settings are chosen on one
half ("tune") and every figure in the results is from the other half ("report").

Question 1, context: does the person's home aesthetic rank higher from all photos than from the outfit photo alone?
Question 2, serendipity: does S = relevance x unexpectedness put the planted aesthetic B first, and does the explanation
point at the planted photo?
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

T0 = time.time()
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROOT, PER = Path(arg("--root", "E:/vega")), int(arg("--per-label", 6))
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vega_data  # noqa: E402
import vega_profile as vp  # noqa: E402

OUT = ROOT / "runs" / "personas-v1"
read = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
z = np.load(OUT / "oof.npz", allow_pickle=False)
ids, P, classes = [str(a) for a in z["ids"]], z["probs"].astype(np.float64), [str(c) for c in z["classes"]]
d = vega_data.load(root=ROOT)
assert ids == [str(a) for a in list(d["fit"]["ids"]) + list(d["eval"]["ids"])] and classes == [str(c) for c in d["classes"]]
y = np.concatenate([d["fit"]["y"], d["eval"]["y"]]) > 0
one = y.sum(1) == 1                                                   # images with exactly one label
label = y.argmax(1)
fx = np.asarray(d["fit"]["x"], np.float64)
SIM = vp.aesthetic_map(fx / np.linalg.norm(fx, axis=1, keepdims=True), np.asarray(d["fit"]["y"]) > 0)     # training images only
fam = {r["wiki_label"]: r["family"] for r in read(ROOT / "runs" / "family-v1" / "label_family.csv")}
fam_of = np.asarray([fam[c] for c in classes])
boxes = {r["id"]: (float(r["height"]), json.loads(r["people"]) if int(r["n_people"]) > 0 else []) for r in read(ROOT / "runs" / "person-crops" / "wiki-trainval.csv")}
outfit = np.asarray([any((b[3] - b[1]) >= 0.5 * boxes[i][0] for b in boxes[i][1]) for i in ids])

rng = np.random.default_rng(422)
O = {a: np.flatnonzero(one & (label == a) & outfit) for a in range(len(classes))}
C = {a: np.flatnonzero(one & (label == a) & ~outfit) for a in range(len(classes))}
homes = [a for a in range(len(classes)) if len(O[a]) >= 1 and len(C[a]) >= 2]
half = lambda a: "tune" if int(hashlib.sha1(classes[a].encode("utf-8")).hexdigest(), 16) % 2 == 0 else "report"
personas = []                                                         # (kind, home, planted or -1, image rows, planted position or -1)
for a in homes:
    others = [b for b in range(len(classes)) if fam_of[b] != fam_of[a] and len(C[b]) >= 1]
    for _ in range(PER):
        o = int(rng.choice(O[a]))
        ctx = rng.choice(C[a], min(3, len(C[a])), replace=False).tolist()
        personas.append(("pure", a, -1, [o] + ctx, -1))
        b = int(rng.choice(others))
        two = rng.choice(C[a], 2, replace=False).tolist()
        personas.append(("mixed", a, b, [o] + two + [int(rng.choice(C[b]))], 3))
with open(OUT / "personas.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["persona", "kind", "half", "home", "planted", "images (outfit first; the planted image is last in a mixed persona)"])
    for k, (kind, a, b, rows, _) in enumerate(personas):
        w.writerow([k, kind, half(a), classes[a], classes[b] if b >= 0 else "", " ".join(ids[r] for r in rows)])
is_out = lambda rows: outfit[rows]
tune = [p for p in personas if half(p[1]) == "tune"]
rep = [p for p in personas if half(p[1]) == "report"]


def rank_of(profile, a):
    return int((profile > profile[a]).sum()) + 1


def mrr(ps, mode, **kw):
    return float(np.mean([1.0 / rank_of(vp.pooled(P[p[3]], is_out(p[3]), mode, **kw), p[1]) for p in ps]))


# ---------------------------------------------------------------- settings of the weighted profile, chosen on the tune half
grid = [(wo, pw) for wo in (0.5, 1.0, 2.0, 4.0) for pw in (0.0, 1.0, 2.0)]
scores = {g: mrr(tune, "weighted", w_outfit=g[0], power=g[1]) for g in grid}
WO, PW = max(scores, key=scores.get)

# ---------------------------------------------------------------- question 1 on the report half
eq = lambda rows: vp.pooled(P[rows], is_out(rows), "equal")
WEIGHTED = (f"weighted profile (outfit x{WO:g}, strength ^{PW:g})", lambda r: vp.pooled(P[r], is_out(r), "weighted", w_outfit=WO, power=PW))
# pure personas also get rows that separate "more photos" from "context": the image rows are [outfit, context, context, (context)]
MODES_PURE = [("outfit photo only", lambda r: eq(r[:1])), ("one context image only", lambda r: eq(r[1:2])),
              ("outfit + 1 context image", lambda r: eq(r[:2])), ("outfit + 2 context images", lambda r: eq(r[:3])),
              ("context images only, no outfit", lambda r: eq(r[1:])), ("all photos, equal", eq), WEIGHTED]
MODES_MIXED = [("outfit photo only", lambda r: eq(r[:1])), ("all photos, equal", eq), WEIGHTED]
labels_rep = sorted({p[1] for p in rep})
draws = [rng.choice(labels_rep, len(labels_rep)) for _ in range(2000)]      # resample home aesthetics, with all their personas


def table(kind):
    ps = [p for p in rep if p[0] == kind]
    MODES = MODES_PURE if kind == "pure" else MODES_MIXED
    by = {a: [i for i, p in enumerate(ps) if p[1] == a] for a in labels_rep}
    ranks = {name: np.asarray([rank_of(f(p[3]), p[1]) for p in ps]) for name, f in MODES}
    base = ranks[MODES[0][0]]
    rows = []
    for name, _ in MODES:
        r = ranks[name]
        cell = f"| {name} | {(r == 1).mean():.1%} | {(r <= 5).mean():.1%} | {(1 / r).mean():.3f} |"
        if name != MODES[0][0]:
            dlt = ((r <= 5).astype(float) - (base <= 5)) * 100
            boot = [np.concatenate([dlt[by[a]] for a in dr]).mean() for dr in draws]
            cell += f" {dlt.mean():+.1f} ({np.percentile(boot, 2.5):+.1f} to {np.percentile(boot, 97.5):+.1f}) |"
        else:
            cell += " |"
        rows.append(cell)
    return len(ps), rows


# ---------------------------------------------------------------- question 2 on the report half
mixed = [p for p in rep if p[0] == "mixed"]
pure = [p for p in rep if p[0] == "pure"]
res = {k: [] for k in ("s1", "s3", "rel1", "rel3", "same3", "rand", "explain", "explain_any", "in_top5", "fam1", "seen", "s1_seen")}
top_s_mixed, top_s_pure = [], []
for kind, a, b, rows, pos in mixed:
    pr = vp.pooled(P[rows], is_out(rows), "weighted", w_outfit=WO, power=PW)
    s, rel, unexp = vp.surprise(P[rows], pr, SIM)
    order_s = np.argsort(-s)
    first = int(np.argmax(pr))
    rel_x = rel.copy()
    rel_x[first] = -1
    order_rel = np.argsort(-rel_x)
    order_p = np.argsort(-pr)
    res["s1"].append(order_s[0] == b)
    res["s3"].append(b in order_s[:3])
    res["rel1"].append(order_rel[0] == b)
    res["rel3"].append(b in order_rel[:3])
    res["same3"].append(b in order_p[1:4])
    band = np.flatnonzero(unexp >= unexp[order_s[0]] - 1e-12)             # aesthetics at least as unexpected as the one chosen
    res["rand"].append(float(b in band) / len(band))
    res["in_top5"].append(b in order_p[:5])
    res["fam1"].append(fam_of[order_s[0]] == fam_of[b])
    res["explain_any"].append(int(np.argmax(P[rows][:, order_s[0]])) == pos)
    seen = int(np.argmax(P[rows[pos]])) == b                             # does the network itself call the planted photo B?
    res["seen"].append(seen)
    if seen:
        res["s1_seen"].append(order_s[0] == b)
    if order_s[0] == b:
        res["explain"].append(int(np.argmax(P[rows][:, b])) == pos)
    top_s_mixed.append(s[order_s[0]])
for kind, a, b, rows, pos in pure:
    pr = vp.pooled(P[rows], is_out(rows), "weighted", w_outfit=WO, power=PW)
    top_s_pure.append(vp.surprise(P[rows], pr, SIM)[0].max())
m, q = np.asarray(top_s_mixed), np.asarray(top_s_pure)
auc = float((m[:, None] > q[None, :]).mean() + 0.5 * (m[:, None] == q[None, :]).mean())
by_m = {a: [i for i, p in enumerate(mixed) if p[1] == a] for a in labels_rep}


def rate(key):
    v = np.asarray(res[key], float)
    boot = [np.concatenate([v[by_m[a]] for a in dr]).mean() for dr in draws] if len(v) == len(mixed) else None
    return f"{v.mean():.1%}" + (f" ({np.percentile(boot, 2.5):.1%} to {np.percentile(boot, 97.5):.1%})" if boot else "")


_d = (np.asarray(res["s1"], float) - np.asarray(res["rel1"], float)) * 100
_b = [np.concatenate([_d[by_m[a]] for a in dr]).mean() for dr in draws]
paired_s_rel = f"{_d.mean():+.1f} ({np.percentile(_b, 2.5):+.1f} to {np.percentile(_b, 97.5):+.1f})"

# ---------------------------------------------------------------- write
n_pure, rows_pure = table("pure")
n_mixed, rows_mixed = table("mixed")
L = ["# Personas from wiki images: the context question and the serendipity question", "",
     f"Made by `code/app/persona_eval.py`. {len(homes)} aesthetics have an outfit image and two context images with one label; {PER} pure and {PER} mixed personas each: "
     f"{len(personas):,} personas. Settings were chosen on {len(tune):,} personas of {len({p[1] for p in tune})} home aesthetics; **every figure below is from the other "
     f"{len(rep):,} personas of {len(labels_rep)} home aesthetics.** Each image is scored by a network that never trained on it. The test split was not read.", "",
     f"The weighted profile's settings, chosen on the tune half: an outfit photo counts x{WO:g}; a photo's strongest match is raised to the power {PW:g}.", "",
     "## Question 1: does context help? Where the home aesthetic ranks", "",
     f"### Pure personas ({n_pure:,}): 1 outfit and 2 or 3 context images of one aesthetic", "",
     "| profile from | home is first | home in top 5 | MRR | top 5 against outfit only (95% interval) |", "|---|---|---|---|---|"] + rows_pure + ["",
     f"### Mixed personas ({n_mixed:,}): one of the context images is from another family's aesthetic", "",
     "| profile from | home is first | home in top 5 | MRR | top 5 against outfit only (95% interval) |", "|---|---|---|---|---|"] + rows_mixed + ["",
     "Intervals resample home aesthetics with all their personas (2,000 draws, seed 422).", "",
     "## Question 2: does the surprise score find the planted aesthetic?", "",
     f"On the {len(mixed):,} mixed personas. The planted aesthetic B is one photo of four, from another family than the home.", "",
     "| how the suggestion is picked | B is the first suggestion | B is among the first three |", "|---|---|---|",
     f"| **S = relevance x unexpectedness** (the proposal's score) | {rate('s1')} | {rate('s3')} |",
     f"| relevance alone (strongest single-photo match, top aesthetic excluded) | {rate('rel1')} | {rate('rel3')} |",
     f"| more of the same: ranks 2 to 4 of the profile | | {rate('same3')} |",
     f"| a random aesthetic at least as unexpected as the one S picked | {np.mean(res['rand']):.2%} | |", "",
     f"- **What unexpectedness adds over relevance alone:** {paired_s_rel} points on 'B is the first suggestion' (95% interval, same personas).",
     f"- **The ceiling is the network, not the score.** The network's own first guess for the planted photo is B in {rate('seen')} of mixed personas. "
     f"In those {len(res['s1_seen']):,} personas S puts B first {np.mean(res['s1_seen']):.1%} of the time.",
     f"- B is already inside the profile's top 5 for {rate('in_top5')} of mixed personas, so most planted aesthetics are visible before any surprise score.",
     f"- S's first suggestion is in B's family for {rate('fam1')} of mixed personas (that includes the cases where it is B itself).",
     f"- **Explanation:** when S picks B, the photo named as the reason is the planted photo in {np.mean(res['explain']):.1%} of cases ({len(res['explain']):,} personas). "
     f"Over all mixed personas, the reason given for S's first suggestion is the planted photo in {rate('explain_any')}.",
     f"- **Can the score tell when there is something to find?** The largest S of a mixed persona is higher than that of a pure persona {auc:.1%} of the time "
     f"(50% would be chance; median {np.median(m):.2f} against {np.median(q):.2f}).", "",
     "## What this is and is not", "",
     "- The personas are synthetic and their images come from wiki galleries, as the proposal planned for training and validation personas. Real people's photo sets are not tested here.",
     "- 'Context' means non-outfit images of the same aesthetic (rooms, objects, scenery, collages), not a person's own room or car.",
     "- The planted aesthetic is relevant by construction. Whether a surprising suggestion FEELS fitting to a person needs human ratings, which this does not have.",
     "- Aesthetics without an outfit image and two context images cannot be a home, so the personas cover " + f"{len(homes)} of the 503 labels.", "",
     f"time elapsed {time.time() - T0:.0f}s"]
(OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
