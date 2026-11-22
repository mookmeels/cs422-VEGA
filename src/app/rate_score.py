"""Score the rating page's answers: how often do people say a suggestion fits, by kind of suggestion?

    python rate_score.py [--root E:/vega] [--answers runs/ratings-v1/answers.txt] [--out runs/ratings-v1/results.md]

answers.txt holds the lines people copied from the page's last screen, one per line:
    VEGA ratings v1-xxxxxx | name | ffnu-f...        (f fits, n doesn't fit, u can't tell, - not answered; in screen order)
Lines of another version of the page than key.csv are refused. A person who sends two lines counts once (the later line).

"Fits" is the share of Fits among Fits + Doesn't fit; "can't tell" is reported beside it. Intervals resample the photo
sets (2,000 draws, seed 422), since one set gives four screens and several people rate the same set.
"""
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

T0 = time.time()
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROOT = Path(arg("--root", "E:/vega"))
OUT = ROOT / "runs" / "ratings-v1"
ANSWERS = Path(arg("--answers", str(OUT / "answers.txt")))
KINDS = ["top", "nearby", "surprising", "random"]
key = sorted(csv.DictReader(open(OUT / "key.csv", encoding="utf-8", newline="")), key=lambda r: int(r["n"]))
version = "v1-" + hashlib.sha1(json.dumps([(int(r["n"]), int(r["set"]), r["kind"], r["aesthetic"]) for r in key]).encode("utf-8")).hexdigest()[:6]

people, refused = {}, []
for line in ANSWERS.read_text(encoding="utf-8").splitlines():
    if "VEGA ratings" not in line:
        continue
    parts = [p.strip() for p in line[line.index("VEGA ratings") + len("VEGA ratings"):].split("|")]
    if len(parts) != 3 or parts[0] != version or len(parts[2]) != len(key) or set(parts[2]) - set("fnu-"):
        refused.append(line[:60])
        continue
    name = parts[1] if parts[1] != "no name" else f"no name {len(people) + 1}"
    people[name] = parts[2]
if not people:
    sys.exit(f"no usable line for version {version} in {ANSWERS}" + (f"; refused: {refused}" if refused else ""))

n_sets = 1 + max(int(r["set"]) for r in key)
source = {int(r["set"]): r["source"] for r in key}
fit, known, cant = (defaultdict(lambda: np.zeros(n_sets)) for _ in range(3))      # per kind: per set, summed over people
for letters in people.values():
    for r, a in zip(key, letters):
        k, s = r["kind"], int(r["set"])
        fit[k][s] += a == "f"
        known[k][s] += a in "fn"
        cant[k][s] += a == "u"
rng = np.random.default_rng(422)
draws = rng.integers(0, n_sets, (2000, n_sets))


def rate(k, sets):
    """Share of Fits among Fits + Doesn't fit for kind k over these sets, with an interval over sets."""
    f, n = fit[k][sets], known[k][sets]
    if n.sum() == 0:
        return "no answers"
    pick = draws[:, :len(sets)] % len(sets)
    b = f[pick].sum(1) / np.maximum(n[pick].sum(1), 1)
    return f"{f.sum() / n.sum():.0%} ({np.percentile(b, 2.5):.0%} to {np.percentile(b, 97.5):.0%}), {int(f.sum())} of {int(n.sum())}"


def gap(a, b, sets):
    """Kind a minus kind b, same sets, in points, with an interval over sets."""
    pick = draws[:, :len(sets)] % len(sets)
    one = lambda k, p: fit[k][sets][p].sum(-1) / np.maximum(known[k][sets][p].sum(-1), 1)
    d = (one(a, pick) - one(b, pick)) * 100
    here = (fit[a][sets].sum() / max(known[a][sets].sum(), 1) - fit[b][sets].sum() / max(known[b][sets].sum(), 1)) * 100
    return f"{here:+.0f} points ({np.percentile(d, 2.5):+.0f} to {np.percentile(d, 97.5):+.0f})"


every = np.arange(n_sets)
groups = [("all sets", every)] + [(f"{s} sets", np.asarray([i for i in every if source[i] == s])) for s in sorted(set(source.values()))]
who = ", ".join(f"{n} ({len(a) - a.count('-')} screens)" for n, a in people.items())
L = [f"# Do the app's suggestions fit? People's ratings, page version {version}", "",
     f"Made by `code/app/rate_score.py` from {len(people)} person(s): {who}. "
     "Each screen showed four real outfit photos and one suggested aesthetic with three example pictures; the rater could not tell which kind of "
     "suggestion it was. Only sets for which the app shows a surprise are on the page (see build.md).", "",
     "| kind of suggestion | " + " | ".join(g for g, _ in groups) + " | can't tell |", "|---|" + "---|" * (len(groups) + 1)]
for k in KINDS:
    L.append(f"| {k} | " + " | ".join(rate(k, s) for _, s in groups) + f" | {int(cant[k].sum())} |")
L += ["", "Against the random control, same sets:", ""]
for k in KINDS[:3]:
    L.append(f"- {k}: " + "; ".join(f"{g}: {gap(k, 'random', s)}" for g, s in groups))
if len(people) > 1:
    names = list(people)
    same = both = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for a, b in zip(people[names[i]], people[names[j]]):
                if a in "fn" and b in "fn":
                    both += 1
                    same += a == b
    L += ["", f"Two people who both answered Fits or Doesn't fit on a screen agreed on {same} of {both} ({same / max(both, 1):.0%})."]
if refused:
    L += ["", f"Refused lines (another version of the page, or damaged): {len(refused)}."]
L += ["", f"time elapsed {time.time() - T0:.0f}s"]
Path(arg("--out", str(OUT / "results.md"))).write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
