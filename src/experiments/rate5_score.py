r"""Score the second rating page: precision@5 and nDCG@5 per system, and how much the raters agree (Fleiss' kappa).

    python rate5_score.py [--root E:/vega]        reads runs/ratings-v2/answers.txt and key.csv -> runs/ratings-v2/results.md

answers.txt holds the lines people copied from the page's last screen, one per line:
    VEGA ratings v2-xxxxxx | name | ffnu-f...        (f fits, n doesn't fit, u can't tell, - not answered; in screen order)
Lines of another version than key.csv are refused; a person who sends two lines counts once (the later line).

An aesthetic counts as fitting a photo set by majority, as the proposal says: more raters said "fits" than "doesn't fit"
(with one rater, that rater's answer). "Can't tell" and unanswered count for neither side; an aesthetic nobody could
judge counts as not fitting, and their number is reported.
  precision@5   the share of a system's five suggestions that fit, averaged over the photo sets
  nDCG@5        the same with higher places counting more: sum of fit / log2(place + 1), divided by its value if all five fit
  Fleiss' kappa over the screens every rater answered, three categories (fits, doesn't fit, can't tell), and again for
                fits against doesn't fit on the screens where nobody answered "can't tell"
Intervals resample the photo sets (2,000 draws, seed 422).
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
ROOT = Path(arg("--root", "E:/vega"))
OUT = ROOT / "runs" / "ratings-v2"
SYSTEMS = [("nn", "NN + real outfit photos"), ("centroid", "Baseline: nearest centroid"), ("zeroshot", "Baseline: zero-shot text"), ("popularity", "Baseline: popularity")]
key = sorted(csv.DictReader(open(OUT / "key.csv", encoding="utf-8", newline="")), key=lambda r: int(r["n"]))
version = "v2-" + hashlib.sha1(json.dumps([(int(r["n"]), int(r["set"]), r["aesthetic"]) for r in key]).encode("utf-8")).hexdigest()[:6]

people, refused = {}, []
for line in (OUT / "answers.txt").read_text(encoding="utf-8").splitlines():
    if "VEGA ratings" not in line:
        continue
    parts = [p.strip() for p in line[line.index("VEGA ratings") + len("VEGA ratings"):].split("|")]
    if len(parts) != 3 or parts[0] != version or len(parts[2]) != len(key) or set(parts[2]) - set("fnu-"):
        refused.append(line[:60])
        continue
    people[parts[1] if parts[1] != "no name" else f"no name {len(people) + 1}"] = parts[2]
if not people:
    raise SystemExit(f"no usable answer line for {version} ({len(refused)} refused)")
names = sorted(people)
A = np.asarray([list(people[n]) for n in names])                      # raters x screens
fits, nots = (A == "f").sum(0), (A == "n").sum(0)
judged = (fits + nots) > 0
fit = fits > nots                                                      # majority; a tie or nobody able to judge counts as not fitting
sets = sorted({int(r["set"]) for r in key})
source = {int(r["set"]): r["source"] for r in key}
DISC = 1 / np.log2(np.arange(5) + 2)


def per_set(system, vec):
    """Per photo set: precision@5 and nDCG@5 of this system, from a 0/1 vector over screens."""
    p, g = [], []
    for s in sets:
        rel = np.zeros(5)
        for i, r in enumerate(key):
            if int(r["set"]) == s and r[f"{system}_rank"] != "":
                rel[int(r[f"{system}_rank"]) - 1] = vec[i]
        p.append(rel.mean())
        g.append((rel * DISC).sum() / DISC.sum())
    return np.asarray(p), np.asarray(g)


rng = np.random.default_rng(422)
draws = [rng.integers(0, len(sets), len(sets)) for _ in range(2000)]
ci = lambda v: f"{v.mean():.2f} ({np.percentile([v[d].mean() for d in draws], 2.5):.2f} to {np.percentile([v[d].mean() for d in draws], 97.5):.2f})"


def kappa(M):
    """Fleiss' kappa. M: items x categories, the number of raters choosing each category (the same total in every row)."""
    n = M.sum(1)[0]
    if len(M) == 0 or n < 2:
        return None
    pj = M.sum(0) / M.sum()
    Pi = ((M ** 2).sum(1) - n) / (n * (n - 1))
    Pe = (pj ** 2).sum()
    return float((Pi.mean() - Pe) / (1 - Pe)) if Pe < 1 else None


L = [f"# Precision@5 per system by people's ratings, page version {version}", "",
     f"Made by `code/experiments/rate5_score.py` from {len(names)} rater(s): {', '.join(names)}. {len(key)} screens over {len(sets)} sets of four real outfit photos; "
     f"{int((~judged).sum())} screens nobody could judge. An aesthetic fits by majority of the raters.", "",
     "| system | precision@5 (95% interval) | nDCG@5 | expert sets, precision@5 | posters' sets, precision@5 |", "|---|---|---|---|---|"]
res = {}
for k, name in SYSTEMS:
    p, g = per_set(k, fit.astype(float))
    ex = np.asarray([source[s] == "expert" for s in sets])
    res[k] = p
    L.append(f"| {name} | {ci(p)} | {ci(g)} | {p[ex].mean():.2f} | {p[~ex].mean():.2f} |")
L += ["", "Against the popularity baseline, precision@5, same sets:", ""]
for k, name in SYSTEMS[:3]:
    dv = res[k] - res["popularity"]
    L.append(f"- {name}: {dv.mean():+.2f} ({np.percentile([dv[d].mean() for d in draws], 2.5):+.2f} to {np.percentile([dv[d].mean() for d in draws], 97.5):+.2f})")
dv = res["nn"] - res["centroid"]
L.append(f"- NN minus nearest centroid: {dv.mean():+.2f} ({np.percentile([dv[d].mean() for d in draws], 2.5):+.2f} to {np.percentile([dv[d].mean() for d in draws], 97.5):+.2f})")
dv = res["nn"] - res["zeroshot"]
L.append(f"- NN minus zero-shot text: {dv.mean():+.2f} ({np.percentile([dv[d].mean() for d in draws], 2.5):+.2f} to {np.percentile([dv[d].mean() for d in draws], 97.5):+.2f})")
L += ["", "## Agreement between raters", ""]
if len(names) < 2:
    L.append("One rater so far: no agreement figure.")
else:
    full = (A != "-").all(0)
    M3 = np.stack([(A[:, full] == c).sum(0) for c in "fnu"], axis=1).astype(float)
    two = full & (A != "u").all(0)
    M2 = np.stack([(A[:, two] == c).sum(0) for c in "fn"], axis=1).astype(float)
    k3, k2 = kappa(M3), kappa(M2)
    L.append(f"- Fleiss' kappa, three answers (fits, doesn't fit, can't tell), {int(full.sum())} screens answered by all {len(names)} raters: " + (f"{k3:.2f}" if k3 is not None else "n/a"))
    L.append(f"- Fleiss' kappa, fits against doesn't fit, {int(two.sum())} screens where nobody said can't tell: " + (f"{k2:.2f}" if k2 is not None else "n/a"))
    for i, n in enumerate(names):
        p, _ = per_set("nn", (A[i] == "f").astype(float))
        L.append(f"- {n} alone: NN precision@5 {p.mean():.2f}; said can't tell on {int((A[i] == 'u').sum())} screens")
if refused:
    L += ["", f"Refused lines (wrong version or length): {len(refused)}"]
L += ["", f"time elapsed {time.time() - T0:.0f}s"]
(OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
