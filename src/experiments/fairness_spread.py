"""Who is served worse? Family accuracy on real outfit photos, by poster, by style and by where the poster lives.

    python fairness_spread.py [--root E:/vega] [--seeds 3] [--device cuda]      -> runs/fairness-v1/spread.md, spread.json

The proposal planned a fairness check: do similar outfits get different predictions depending on who wears them. No
dataset we hold records who is in a photo. What Street Fashion Style does record is the poster and the location the
poster typed. This script measures how unevenly the recommended NN (frozen settings, wiki training images + street
photos of the other nine poster groups) names the family of photos by posters it never saw (group 0):
  1. by family of the poster's own tag, and by FashionStyle14 style (expert labels, never trained on);
  2. by poster: the spread of per-poster accuracy, and how much of it is more than chance and tag mix explain;
  3. by region of the poster's typed location, raw and with every region given the same mix of tag families.
Nothing is tuned and no model changes. The wiki test split is not read; FashionStyle14's lolita, fairy and ethnic photos
are not read. Region is read from free text by the rules in REGION below; unmatched locations are reported, not guessed.
"""
import csv
import hashlib
import json
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
import train_nn  # noqa: E402
import vega_data  # noqa: E402

read = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
V1, OUT = ROOT / "runs" / "family-v1", ROOT / "runs" / "fairness-v1"
OUT.mkdir(parents=True, exist_ok=True)
TRANSFER = {"lolita", "fairy", "ethnic"}

US = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split())
REGION = {  # lower-case words looked for in the typed location, most specific first
    "United States and Canada": ["united states", "usa", "u.s.", "pennsylvania", "baltimore", "california", "new york", "texas", "florida", "chicago",
                                 "los angeles", "san francisco", "seattle", "boston", "canada", "british columbia", "ontario", "quebec", "alberta",
                                 "toronto", "vancouver", "montreal", "hawaii", "brooklyn", "manhattan"],
    "Western and Northern Europe": ["belgium", "united kingdom", "england", "scotland", "wales", "ireland", "dublin", "london", "milton keynes", "spain",
                                    "madrid", "barcelona", "italy", "toscana", "milan", "roma", "france", "ile de france", "paris", "limoges", "norway",
                                    "denmark", "sweden", "finland", "iceland", "germany", "berlin", "netherlands", "amsterdam", "greece", "attiki",
                                    "athens", "portugal", "lisbon", "switzerland", "austria", "vienna", "luxembourg"],
    "Eastern Europe and Central Asia": ["romania", "cluj", "satu mare", "bucuresti", "bucharest", "poland", "mazowieckie", "warsaw", "russia", "tatarstan",
                                        "moscow", "kazakhstan", "slovenia", "ukraine", "bulgaria", "serbia", "croatia", "hungary", "czech", "slovakia",
                                        "lithuania", "latvia", "estonia", "belarus", "moldova", "bosnia", "macedonia", "georgia (country)"],
    "East and Southeast Asia": ["philippines", "manila", "indonesia", "jakarta", "seoul", "korea", "japan", "tokyo", "singapore", "hong kong", "malaysia",
                                "thailand", "bangkok", "china", "taiwan", "vietnam", "cebu"],
    "South Asia and Middle East": ["india", "gujarat", "pakistan", "bangladesh", "sri lanka", "israel", "turkey", "istanbul", "dubai", "emirates",
                                   "saudi", "lebanon", "iran", "egypt"],
    "Latin America": ["panama", "mexico", "brazil", "brasil", "argentina", "chile", "colombia", "peru", "venezuela", "puerto rico", "costa rica",
                      "ecuador", "guatemala", "dominican"],
    "Africa and Oceania": ["south africa", "nigeria", "kenya", "ghana", "australia", "sydney", "melbourne", "new zealand", "auckland"],
}


def region_of(loc):
    low = loc.lower().strip()
    if not low:
        return "not given"
    last = loc.split(",")[-1].strip()
    if last.upper() in US and len(last) == 2 and last.upper() not in {"IN", "ME", "OR", "DE", "LA", "PA", "CA", "AL", "GA"}:
        return "United States and Canada"
    for region, words in REGION.items():
        if any(w in low for w in words):
            return region
    if last.upper() in US and len(last) == 2 and "," in loc:        # "City, CA": a state code after a comma
        return "United States and Canada"
    return "not matched"


d = vega_data.load(root=ROOT)
fit, classes = d["fit"], [str(c) for c in d["classes"]]
fams = [r["family"] for r in read(V1 / "families.csv")]
fname = {r["family"]: r["name"] for r in read(V1 / "families.csv")}
of_label = {r["wiki_label"]: r["family"] for r in read(V1 / "label_family.csv")}
lab_fam = np.asarray([fams.index(of_label[c]) for c in classes])
M = np.zeros((len(classes), len(fams)))
M[np.arange(len(classes)), lab_fam] = 1
outside = read(V1 / "outside_family.csv")
tag_fam = {r["style_or_tag"]: fams.index(r["family"]) for r in outside if r["source"] == "Street Fashion Style" and r["family"] in fams}
style_fam = {r["style_or_tag"]: fams.index(r["family"]) for r in outside
             if r["source"] == "FashionStyle14" and r["family"] in fams and r["style_or_tag"] not in TRANSFER}

sdir, vec = ROOT / "emb" / "street-fashion-style", f"{vega_data.VECTORS}-448"
srow = {s: i for i, s in enumerate((sdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
held = sorted((r["user"], srow[r["id"]], tag_fam[r["style"]], r["location"]) for r in read(sdir / "manifest.csv")
              if r["style"] in tag_fam and r["id"] in srow
              and int(hashlib.sha1(r["user"].encode("utf-8")).hexdigest(), 16) % train_nn.REAL_GROUPS == 0)
XS = np.ascontiguousarray(np.load(sdir / vec / "embeddings.npy", mmap_mode="r")[[h[1] for h in held]])
user = np.asarray([h[0] for h in held])
tagf = np.asarray([h[2] for h in held])
loc_of_user = {u: Counter(h[3] for h in held if h[0] == u).most_common(1)[0][0] for u in sorted(set(user.tolist()))}
region = np.asarray([region_of(loc_of_user[u]) for u in user])

fdir = ROOT / "emb" / "fashionstyle14"
frow = {s: i for i, s in enumerate((fdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
styles = {}
for r in read(fdir / "manifest.csv"):
    styles.setdefault(r["sha1"], set()).add(r["style"])
photos = sorted((frow[s], next(iter(st))) for s, st in styles.items() if len(st) == 1 and s in frow and next(iter(st)) in style_fam)
XF = np.ascontiguousarray(np.load(fdir / vec / "embeddings.npy", mmap_mode="r")[[p[0] for p in photos]])
style = np.asarray([p[1] for p in photos])
truef = np.asarray([style_fam[s] for s in style])
print(f"held-out posters {len(set(user.tolist()))}, photos {len(held):,}; FashionStyle14 photos {len(photos):,} in {len(set(style.tolist()))} styles", flush=True)

real = train_nn.real_photos(ROOT, classes)
hit_s, hit_f = [], []
for seed in SEEDS:
    z = train_nn.train(fit["x"], fit["y"], np.concatenate([XS, XF]), train_nn.SETTINGS, seed, DEVICE, real=real).astype(np.float64)
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    pf = (p / p.sum(1, keepdims=True)) @ M
    hit_s.append(pf[:len(XS)].argmax(1) == tagf)
    hit_f.append(pf[len(XS):].argmax(1) == truef)
    print(f"  seed {seed}: family right, street {hit_s[-1].mean():.1%}, FashionStyle14 {hit_f[-1].mean():.1%}", flush=True)
del real
hs, hf = np.mean(hit_s, axis=0), np.mean(hit_f, axis=0)             # per photo: the share of seeds that are right

users = sorted(set(user.tolist()))
rows_of = {u: np.flatnonzero(user == u) for u in users}
rng = np.random.default_rng(422)
DRAWS = [rng.integers(0, len(users), len(users)) for _ in range(2000)]


def boot(stat):
    """95% interval of stat(rows), resampling posters."""
    v = [stat(np.concatenate([rows_of[users[i]] for i in dr])) for dr in DRAWS]
    v = [x for x in v if x is not None and not np.isnan(x)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


res = {"seeds": SEEDS, "street_photos": len(held), "posters": len(users), "street_family_top1": float(hs.mean()),
       "fs14_photos": len(photos), "fs14_family_top1": float(hf.mean())}
L = ["# Who is served worse: family accuracy on real outfit photos by style, poster and region", "",
     f"Made {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by code/experiments/fairness_spread.py. NN with real outfit photos, frozen settings, "
     f"seeds {SEEDS}, trained on the wiki training images and the street photos of poster groups 1 to 9. Nothing tuned; test split not read.",
     f"Street Fashion Style, held-out posters (group 0): {len(held):,} photos by {len(users)} posters, family right for {hs.mean():.1%}. "
     f"FashionStyle14, 11 styles: {len(photos):,} photos, family right for {hf.mean():.1%}.", ""]

# 1. by style and by tag family
L += ["## 1. By style (FashionStyle14, expert labels)", "", "| style | family it maps to | photos | family right |", "|---|---|---|---|"]
res["fs14_by_style"] = {}
for s in sorted(set(style.tolist()), key=lambda s: -hf[style == s].mean()):
    res["fs14_by_style"][s] = [int((style == s).sum()), float(hf[style == s].mean())]
    L.append(f"| {s} | {fname[fams[style_fam[s]]]} | {int((style == s).sum()):,} | {hf[style == s].mean():.1%} |")
v = [a for _, a in res["fs14_by_style"].values()]
L += ["", f"Best style {max(v):.1%}, worst {min(v):.1%}: a gap of {100 * (max(v) - min(v)):.0f} points.", "",
      "## 2. By family of the poster's own tag (Street Fashion Style, held-out posters)", "", "| family | photos | share of photos | family right |", "|---|---|---|---|"]
res["street_by_family"] = {}
for f in sorted(set(tagf.tolist()), key=lambda f: -(tagf == f).sum()):
    res["street_by_family"][fname[fams[f]]] = [int((tagf == f).sum()), float(hs[tagf == f].mean())]
    L.append(f"| {fname[fams[f]]} | {int((tagf == f).sum()):,} | {(tagf == f).mean():.1%} | {hs[tagf == f].mean():.1%} |")

# 2. by poster
acc_u = np.asarray([hs[rows_of[u]].mean() for u in users])
n_u = np.asarray([len(rows_of[u]) for u in users])
big = n_u >= 20
fam_rate = np.asarray([hs[tagf == f].mean() if (tagf == f).any() else 0.0 for f in range(len(fams))])
expected = np.asarray([fam_rate[tagf[rows_of[u]]].mean() for u in users])          # what the poster's tag mix alone predicts
sim_sd = []
for _ in range(1000):                                                               # spread if only tag mix and chance mattered
    sim = np.asarray([(rng.random(len(rows_of[u])) < fam_rate[tagf[rows_of[u]]]).mean() for u in np.asarray(users)[big]])
    sim_sd.append(sim.std())
q = lambda a, p: float(np.percentile(a, p))
res["by_poster"] = {"posters_20_plus": int(big.sum()), "p10": q(acc_u[big], 10), "median": q(acc_u[big], 50), "p90": q(acc_u[big], 90),
                    "sd": float(acc_u[big].std()), "sd_if_tag_mix_and_chance_only": float(np.mean(sim_sd)),
                    "sd_of_residual": float((acc_u[big] - expected[big]).std()),
                    "under_10pct": int((acc_u[big] < 0.10).sum()), "over_50pct": int((acc_u[big] > 0.50).sum())}
b = res["by_poster"]
L += ["", "## 3. By poster (posters with 20 or more held-out photos)", "",
      f"- {b['posters_20_plus']} posters. Family right per poster: 10th percentile {b['p10']:.1%}, median {b['median']:.1%}, 90th percentile {b['p90']:.1%}.",
      f"- {b['under_10pct']} posters are under 10% and {b['over_50pct']} are over 50%.",
      f"- Spread between posters (standard deviation) {100 * b['sd']:.1f} points. If only each poster's mix of tags and chance mattered it would be "
      f"{100 * b['sd_if_tag_mix_and_chance_only']:.1f} points (1,000 simulations). After taking out what the tag mix predicts, {100 * b['sd_of_residual']:.1f} points remain.", ""]

# 3. by region
L += ["## 4. By region of the location the poster typed", "",
      "Raw, and with every region given the same mix of tag families as all held-out photos (families with under 20 photos in a region are left out of that region's adjusted figure, and the weights rescaled). Intervals are 95%, resampling posters.",
      "", "| region | posters | photos | family right | adjusted for tag mix (95% interval) |", "|---|---|---|---|---|"]
overall_mix = np.bincount(tagf, minlength=len(fams)) / len(tagf)
res["by_region"] = {}


def adjusted(rows, reg):
    rows = rows[region[rows] == reg]
    if len(rows) == 0:
        return None
    num = den = 0.0
    for f in range(len(fams)):
        m = rows[tagf[rows] == f]
        if len(m) >= 20:
            num += overall_mix[f] * hs[m].mean()
            den += overall_mix[f]
    return num / den if den > 0.5 else None


allrows = np.arange(len(held))
for reg in sorted(set(region.tolist()), key=lambda r: -(region == r).sum()):
    m = region == reg
    adj = adjusted(allrows, reg)
    lo, hi = boot(lambda rows, reg=reg: adjusted(rows, reg)) if adj is not None else (None, None)
    res["by_region"][reg] = {"posters": len(set(user[m].tolist())), "photos": int(m.sum()), "raw": float(hs[m].mean()), "adjusted": adj, "interval": [lo, hi]}
    L.append(f"| {reg} | {len(set(user[m].tolist()))} | {int(m.sum()):,} | {hs[m].mean():.1%} | " + (f"{adj:.1%} ({lo:.1%} to {hi:.1%})" if adj is not None else "too few photos") + " |")
REF = "United States and Canada"
L += ["", f"Difference from {REF}, adjusted for tag mix (95% interval, resampling posters):", ""]
res["region_minus_reference"] = {}
for reg in ("Western and Northern Europe", "East and Southeast Asia", "Eastern Europe and Central Asia", "Latin America"):
    def diff(rows, reg=reg):
        a, b = adjusted(rows, reg), adjusted(rows, REF)
        return None if a is None or b is None else a - b
    dv = diff(allrows)
    lo, hi = boot(diff)
    res["region_minus_reference"][reg] = [dv, lo, hi]
    L.append(f"- {reg}: {100 * dv:+.1f} points ({100 * lo:+.1f} to {100 * hi:+.1f})")
train_rows = [r for r in read(sdir / "manifest.csv") if r["style"] in tag_fam and r["id"] in srow
              and int(hashlib.sha1(r["user"].encode("utf-8")).hexdigest(), 16) % train_nn.REAL_GROUPS != 0]
tu = defaultdict(Counter)
for r in train_rows:
    tu[r["user"]][r["location"]] += 1
treg = Counter()
for r in train_rows:
    treg[region_of(tu[r["user"]].most_common(1)[0][0])] += 1
res["training_photos_by_region"] = {k: v / len(train_rows) for k, v in treg.items()}
L += ["", f"Where the street photos the network trains on come from ({len(train_rows):,} photos of the other nine poster groups): "
      + "; ".join(f"{k} {v / len(train_rows):.1%}" for k, v in treg.most_common()) + "."]
unm = Counter(loc_of_user[u] for u in users if region_of(loc_of_user[u]) == "not matched")
L += ["", f"Locations not matched to a region ({sum(unm.values())} posters), most common: " + "; ".join(f"{k} ({v})" for k, v in unm.most_common(12)) + ".", "",
      f"Time elapsed: {time.time() - T0:.0f} s"]
(OUT / "spread.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(OUT / "spread.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("\n".join(L))
