"""Which Street Fashion Style tags should stand for which family? Fast tests on a GPU. Iteration only.

    python tag_iter.py --diagnose                 where a wiki-only NN puts each tag's photos        -> runs/tag-iter/diagnose.md
    python tag_iter.py --loo [--seeds 3]          retrain with each mapped tag left out, one at a time -> runs/tag-iter/loo.md
    python tag_iter.py --variants [--seeds 5]     retrain with the mapping changes in VARIANTS         -> runs/tag-iter/variants-<time>.md
    (all: [--root E:/vega] [--device cuda])

The family table is the one on disk. Only the street-tag mapping (runs/family-v1/outside_family.csv) is varied, in memory;
nothing on disk is changed and nothing is written to the team's results. The model of record is code/nn/train_nn.py on ml-db.

What a run is judged on, all with the same network and settings as train_nn.py:
  wiki top-5, wiki family top-1   the wiki validation split
  FashionStyle14 (dev half)       family with the largest summed probability against the style's family, on HALF of the
                                  11 non-transfer styles' photos: those whose sha1 is even. The other half (odd sha1) is
                                  NEVER read by this script. It is kept so that a mapping chosen here can be checked once on
                                  photos that took no part in choosing it. FashionStyle14 is never trained on.
  held-out posters                Street Fashion Style posters of group 0 (never trained on), tags as in the mapping on disk
Never read: the wiki test split; FashionStyle14 lolita, fairy, ethnic (the transfer test); the odd-sha1 half above.
"""
import csv
import datetime
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

T0 = time.time()
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROOT = Path(arg("--root", "E:/vega"))
DEVICE = arg("--device", "cuda")
SEEDS = [int(arg("--seed-base", 422)) + i for i in range(int(arg("--seeds", 3 if "--loo" in sys.argv else 5)))]   # --seed-base 427: fresh seeds
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "nn"))
import train_nn  # noqa: E402
import vega_data  # noqa: E402
import vega_score  # noqa: E402

V1, OUT = ROOT / "runs" / "family-v1", ROOT / "runs" / "tag-iter"
OUT.mkdir(parents=True, exist_ok=True)
# The one-time check (21 Nov, Andrew: "Yes, check both"): --check-half reads the ODD-sha1 half of FashionStyle14 instead of
# the even one, runs exactly the comparison written down beforehand, and refuses to run a second time.
CHECK = "--check-half" in sys.argv
DRY = "--dry-on-dev" in sys.argv                                  # rehearse the check's code on the dev half; reads nothing new
HALF = 1 if CHECK and not DRY else 0
CHECK_FILE = "check-half-rehearsal-on-dev.md" if DRY else "check-half.md"
if CHECK and not DRY:
    assert not any(m in sys.argv for m in ("--diagnose", "--loo", "--variants", "--round3", "--robust", "--weights")), "the check half is for the one check only"
    assert not (OUT / "check-half.md").exists(), "the check half has been used; it is not read again"
TRANSFER_STYLES = {"lolita", "fairy", "ethnic"}
read = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
log = lambda *a: print(*a, flush=True)

# Mapping changes to try with --variants: tag -> family key, or None to leave the tag's photos out.
VARIANTS = {
    "mapping on disk": {},
    # remaps suggested by --diagnose (where a wiki-only NN puts the tag's photos) or by the FashionStyle14 "natural" lesson
    "Natural -> Casual": {"Natural": "casual"},
    "Menswear -> Casual": {"Menswear": "casual"},
    "Menswear -> Classic": {"Menswear": "classic"},
    "Androgynous -> Casual": {"Androgynous": "casual"},
    "Elegant, Formal -> Classic": {"Elegant": "classic", "Formal": "classic"},
    "decade tags (50s to 90s) left out": {"50s": None, "60s": None, "70s": None, "80s": None, "90s": None},
    # tags not used today
    "+ Chic -> Elegant": {"Chic": "elegant"},
    "+ Chic -> Classic": {"Chic": "classic"},
    "+ Geek -> Classic": {"Geek": "classic"},
    "+ Grandma -> Vintage": {"Grandma": "vintage"},
    "+ Sporty -> Street": {"Sporty": "street"},
    # round 2b: the broad tags that --loo says dilute their family (run with --only "r2b")
    "r2b: Elegant left out": {"Elegant": None},
    "r2b: Urban left out": {"Urban": None},
    "r2b: Androgynous left out": {"Androgynous": None},
    "r2b: Elegant, Urban, Androgynous left out": {"Elegant": None, "Urban": None, "Androgynous": None},
    "r2b: Elegant, Urban left out; Androgynous -> Casual": {"Elegant": None, "Urban": None, "Androgynous": "casual"},
    "r2b: every tag whose removal helped (+ 90s, Preppy, Natural, Casual)": {"Elegant": None, "Urban": None, "Androgynous": None, "90s": None, "Preppy": None, "Natural": None, "Casual": None},
}

d = vega_data.load(root=ROOT)
classes = [str(c) for c in d["classes"]]
fit, ev = d["fit"], d["eval"]
_ids, truth, _classes, FIT_COUNT = vega_data._eval_truth(False, "merged", ROOT)
assert [str(a) for a in _ids] == [str(a) for a in ev["ids"]] and [str(c) for c in _classes] == classes
nw = len(ev["ids"])
fam_rows = read(V1 / "families.csv")
fams = [r["family"] for r in fam_rows]
fname = {r["family"]: r["name"] for r in fam_rows}
short = {f: fname[f].split(",")[0].split(" and")[0] for f in fams}
fi = {f: i for i, f in enumerate(fams)}
lab = {r["wiki_label"]: r["family"] for r in read(V1 / "label_family.csv")}
lab_fam = np.asarray([fi[lab[c]] for c in classes])
outside = read(V1 / "outside_family.csv")
BASE = {r["style_or_tag"]: r["family"] for r in outside if r["source"] == "Street Fashion Style" and r["family"] in fi}
FS_MAP = {r["style_or_tag"]: fi[r["family"]] for r in outside if r["source"] == "FashionStyle14" and r["family"] in fi and r["style_or_tag"] not in TRANSFER_STYLES}
MASK = torch.full((len(fams), len(classes)), -1e9)
MASK[torch.from_numpy(lab_fam), torch.arange(len(classes))] = 0.0

sdir, fdir, vec = ROOT / "emb" / "street-fashion-style", ROOT / "emb" / "fashionstyle14", f"{vega_data.VECTORS}-448"
srow = {s: i for i, s in enumerate((sdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
sfs = [(r["style"], srow[r["id"]], int(hashlib.sha1(r["user"].encode("utf-8")).hexdigest(), 16) % train_nn.REAL_GROUPS)
       for r in read(sdir / "manifest.csv") if r["id"] in srow]
SFS_X = np.load(sdir / vec / "embeddings.npy")                                  # all in memory: 2.3 GB
TAG_COUNT = {}
for st, _, _ in sfs:
    TAG_COUNT[st] = TAG_COUNT.get(st, 0) + 1
frow = {s: i for i, s in enumerate((fdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
styles = {}
for r in read(fdir / "manifest.csv"):
    styles.setdefault(r["sha1"], set()).add(r["style"])
fs = sorted((frow[s], next(iter(st))) for s, st in styles.items()
            if len(st) == 1 and s in frow and next(iter(st)) in FS_MAP and int(s, 16) % 2 == HALF)       # one half only
FS_X = np.ascontiguousarray(np.load(fdir / vec / "embeddings.npy", mmap_mode="r")[[r for r, _ in fs]])
FS_STYLE = np.asarray([st for _, st in fs])
FS_F = np.asarray([FS_MAP[st] for st in FS_STYLE])
he = sorted((r, fi[BASE[st]]) for st, r, g in sfs if st in BASE and g == 0)        # held-out posters, tags as on disk
HELD_X, HELD_F = SFS_X[[r for r, _ in he]], np.asarray([f for _, f in he])
log(f"wiki fit {len(fit['ids']):,}, validation {nw:,}; Street Fashion Style {len(sfs):,} photos, {len(BASE)} tags mapped; "
    f"FashionStyle14 dev half {len(fs):,} photos of {len(set(FS_STYLE))} styles; held-out posters {len(he):,} photos; loaded in {time.time() - T0:.0f}s")


def family_probs(z):
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    pf = np.zeros((len(p), len(fams)))
    np.add.at(pf.T, lab_fam, (p / p.sum(1, keepdims=True)).T)
    return pf


LAST = {}                                                        # the last run's FashionStyle14 dev family probabilities (seed average)


def run(mapping, seeds=SEEDS, extra=None, even_tags=False, keep=None, weight=None):
    """Train through this tag mapping (tag -> family key). Returns per-seed rows of figures in %, the FashionStyle14 dev
    figure by style, and the seed-averaged family probabilities for `extra` photos if given.

    even_tags: a street photo is drawn as family -> one of the family's tags -> one of its photos, so a small tag has the
               same say within its family as a large one. (train_nn.py draws family -> photo.)
    keep:      if given, only street photos whose row in the vectors is in this set are trained on."""
    tr = sorted((fi[mapping[st]], st, r) for st, r, g in sfs if st in mapping and g != 0 and (keep is None or r in keep))
    if not tr:
        real = None
    elif not even_tags:
        fam = np.asarray([f for f, _, _ in tr])
        covered = sorted(set(fam.tolist()))
        real = {"x": SFS_X[[r for _, _, r in tr]], "starts": torch.tensor([int(np.searchsorted(fam, f)) for f in covered]),
                "sizes": torch.tensor([int((fam == f).sum()) for f in covered]), "covered": torch.tensor(covered), "mask": MASK}
    else:
        groups, start = [], 0                                    # (family, where the tag's photos start, how many)
        for k in range(1, len(tr) + 1):
            if k == len(tr) or tr[k][:2] != tr[start][:2]:
                groups.append((tr[start][0], start, k - start))
                start = k
        per_fam = {f: sum(1 for g in groups if g[0] == f) for f in {g[0] for g in groups}}
        lcm = int(np.lcm.reduce(list(per_fam.values())))
        rep = [g for g in groups for _ in range(lcm // per_fam[g[0]])]      # every family gets lcm entries, split evenly over its tags
        real = {"x": SFS_X[[r for _, _, r in tr]], "starts": torch.tensor([g[1] for g in rep]), "sizes": torch.tensor([g[2] for g in rep]),
                "covered": torch.tensor([g[0] for g in rep]), "mask": MASK}
    E = np.concatenate([ev["x"], FS_X, HELD_X] + ([extra] if extra is not None else []))
    c1, c2, c3 = nw, nw + len(FS_X), nw + len(FS_X) + len(HELD_X)
    rows, per_style, pf_extra = [], [], 0
    LAST["fs"], LAST["n_train"], LAST["hits"] = 0, len(tr), []
    for seed in seeds:
        s = train_nn.SETTINGS if weight is None else dict(train_nn.SETTINGS, real_weight=weight)     # weight: how much the street-photo loss counts
        z = train_nn.train(fit["x"], fit["y"], E, s, seed, DEVICE, real=real).astype(np.float64)
        m, fm = vega_score.metrics(z[:c1], truth, FIT_COUNT), vega_score.family_metrics(z[:c1], truth, lab_fam)
        pf = family_probs(z[c1:])
        hit = pf[:len(FS_X)].argmax(1) == FS_F
        rows.append([m["top5"], fm["family_top1"], hit.mean(), (pf[len(FS_X):c3 - c1].argmax(1) == HELD_F).mean()])
        per_style.append({st: hit[FS_STYLE == st].mean() for st in sorted(set(FS_STYLE))})
        LAST["fs"] = LAST["fs"] + pf[:len(FS_X)] / len(seeds)
        LAST["hits"].append(hit.astype(np.float64))
        if extra is not None:
            pf_extra = pf_extra + pf[c3 - c1:] / len(seeds)
    a = np.asarray(rows) * 100
    return a, {st: 100 * np.mean([p[st] for p in per_style]) for st in per_style[0]}, pf_extra


fmt = lambda a: " | ".join(f"{a[:, k].mean():.1f} ± {a[:, k].std():.1f}" for k in range(a.shape[1]))

if "--diagnose" in sys.argv:
    rng = np.random.default_rng(422)
    tags = [t for t, n in sorted(TAG_COUNT.items(), key=lambda kv: -kv[1]) if t in BASE or n >= 300]
    pick, where = [], {}
    for t in tags:
        rows = np.asarray([r for st, r, _ in sfs if st == t])
        rows = rng.choice(rows, 2000, replace=False) if len(rows) > 2000 else rows
        where[t] = (len(pick), len(pick) + len(rows))
        pick += rows.tolist()
    _, _, pf = run({}, SEEDS, extra=SFS_X[pick])                    # a wiki-only NN: no street photo in training
    top = pf.argmax(1)
    overall = np.bincount(top, minlength=len(fams)) / len(top)
    L = ["# Where a wiki-only NN puts each street tag's photos", "",
         f"Made by `code/experiments/tag_iter.py --diagnose`. A network trained on wiki images only ({len(SEEDS)} seeds, probabilities averaged) "
         "scores up to 2,000 photos of each tag; a photo's family is the one with the largest summed probability. "
         "'All tags' is the same over every photo scored, so a share is only telling when it is well above that.", "",
         "All tags: " + ", ".join(f"{short[fams[j]]} {overall[j]:.0%}" for j in np.argsort(-overall)[:8]), "",
         "| tag | photos | mapped to | share the NN puts there (all tags) | where the NN puts them most | then |", "|---|---|---|---|---|---|"]
    for t in tags:
        a, b = where[t]
        share = np.bincount(top[a:b], minlength=len(fams)) / (b - a)
        o = np.argsort(-share)
        m = BASE.get(t)
        L.append(f"| {t} | {TAG_COUNT[t]:,} | {fname[m] if m else '(not used)'} | " + (f"{share[fi[m]]:.0%} ({overall[fi[m]]:.0%})" if m else "") +
                 f" | {short[fams[o[0]]]} {share[o[0]]:.0%} ({overall[o[0]]:.0%}) | {short[fams[o[1]]]} {share[o[1]]:.0%} ({overall[o[1]]:.0%}) |")
    L += ["", f"time elapsed {time.time() - T0:.0f}s"]
    (OUT / "diagnose.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("\n".join(L))

if "--loo" in sys.argv:
    base, base_style, _ = run(BASE)
    wiki_only, _, _ = run({})
    L = ["# Leave one street tag out", "",
         f"Made by `code/experiments/tag_iter.py --loo`, seeds {SEEDS} on {DEVICE}. Each row retrains with that one tag's photos removed; "
         "figures are the change from the mapping on disk, in points. Photos are drawn family by family, so removing a tag changes "
         "what a family's street photos look like, not how often the family is drawn; a family whose only tag is removed is no longer taught by street photos.", "",
         "| left out | photos | its family | only tag of its family | wiki top-5 | wiki family top-1 | FashionStyle14 dev | held-out posters | FashionStyle14 style that moved most |",
         "|---|---|---|---|---|---|---|---|---|",
         f"| (nothing: mapping on disk) | | | | {fmt(base)} | |", f"| (every tag: wiki-only NN) | | | | {fmt(wiki_only)} | |"]
    log(L[-2]); log(L[-1])
    noise = base.std(0) * np.sqrt(2 / len(SEEDS))
    for t in sorted(BASE, key=lambda t: -TAG_COUNT[t]):
        a, style, _ = run({k: v for k, v in BASE.items() if k != t})
        dlt = a.mean(0) - base.mean(0)
        ds = max(style, key=lambda s: abs(style[s] - base_style[s]))
        only = sum(1 for k, v in BASE.items() if v == BASE[t]) == 1
        L.append(f"| {t} | {TAG_COUNT[t]:,} | {short[BASE[t]]} | {'yes' if only else ''} | " + " | ".join(f"{x:+.1f}" for x in dlt) + f" | {ds} {style[ds] - base_style[ds]:+.1f} |")
        log(L[-1])
    L += ["", "Difference of two means of " + f"{len(SEEDS)} seeds: one standard error is about " + ", ".join(f"{x:.1f}" for x in noise) + " points (same column order). "
          "Changes inside about twice that are noise.", f"time elapsed {time.time() - T0:.0f}s"]
    (OUT / "loo.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(L[-2]); log(L[-1])

if "--variants" in sys.argv:
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
    L = [f"# Street-tag mapping variants, {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", "", f"Seeds {SEEDS} on {DEVICE}. FashionStyle14 is the dev half only.", "",
         "| variant | training photos | wiki top-5 | wiki family top-1 | FashionStyle14 dev | held-out posters (tags as on disk) | FashionStyle14 dev by style |", "|---|---|---|---|---|---|---|"]
    for n, change in VARIANTS.items():
        if only and not any(o.lower() in n.lower() for o in only):
            continue
        mapping = {k: v for k, v in {**BASE, **change}.items() if v is not None}
        a, style, _ = run(mapping)
        n_tr = sum(1 for st, _, g in sfs if st in mapping and g != 0)
        L.append(f"| {n} | {n_tr:,} | {fmt(a)} | " + ", ".join(f"{s} {v:.0f}" for s, v in style.items()) + " |")
        log(L[-1])
    L += ["", f"time elapsed {time.time() - T0:.0f}s"]
    p = OUT / f"variants-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"wrote {p}; time elapsed {time.time() - T0:.0f}s")

if "--round3" in sys.argv:
    THREE = {k: v for k, v in BASE.items() if k not in ("Elegant", "Urban", "Androgynous")}
    L = [f"# Round 3: how street photos are drawn and filtered, {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", "", f"Seeds {SEEDS} on {DEVICE}. FashionStyle14 is the dev half only.", ""]

    def confusion(title):
        top = LAST["fs"].argmax(1)
        L.extend(["", f"## Where each FashionStyle14 style's photos go: {title}", "", "| style | its family | right | goes instead to |", "|---|---|---|---|"])
        for st in sorted(set(FS_STYLE)):
            share = np.bincount(top[FS_STYLE == st], minlength=len(fams)) / (FS_STYLE == st).sum()
            own = FS_MAP[st]
            L.append(f"| {st} | {short[fams[own]]} | {share[own]:.0%} | " + ", ".join(f"{short[fams[j]]} {share[j]:.0%}" for j in np.argsort(-share) if j != own and share[j] >= 0.05) + " |")

    head = ["| run | street photos trained on | wiki top-5 | wiki family top-1 | FashionStyle14 dev | held-out posters | FashionStyle14 dev by style |", "|---|---|---|---|---|---|---|"]
    table = []

    def go(name, mapping, **kw):
        a, style, pf = run(mapping, **kw)
        table.append(f"| {name} | {LAST['n_train']:,} | {fmt(a)} | " + ", ".join(f"{s} {v:.0f}" for s, v in style.items()) + " |")
        log(table[-1])
        return pf

    go("mapping on disk", BASE)
    confusion("mapping on disk")
    go("three broad tags left out", THREE)
    confusion("Elegant, Urban, Androgynous left out")
    go("mapping on disk, tags drawn evenly within a family", BASE, even_tags=True)
    go("three broad tags left out, tags drawn evenly", THREE, even_tags=True)
    # Keep a street photo only if a wiki-only NN ranks its tag's family among its first k families (noisy-tag filter).
    rows = sorted({r for st, r, g in sfs if st in BASE and g != 0})
    pf = run({}, extra=SFS_X[rows])[2]
    tag_of = {r: st for st, r, g in sfs if st in BASE and g != 0}
    rank = {r: int((pf[i] > pf[i, fi[BASE[tag_of[r]]]]).sum()) for i, r in enumerate(rows)}
    for k in (3, 5, 8):
        keep = {r for r, v in rank.items() if v < k}
        go(f"mapping on disk, only photos whose family is in the wiki-only NN's first {k} families", BASE, keep=keep)
    keep = {r for r, v in rank.items() if v < 5}
    go("three broad tags left out + first-5 filter", THREE, keep=keep)
    go("mapping on disk, first-5 filter, tags drawn evenly", BASE, keep=keep, even_tags=True)
    L = L[:4] + head + table + L[4:] + ["", f"time elapsed {time.time() - T0:.0f}s"]
    p = OUT / f"round3-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    log("\n".join(L[4 + len(head) + len(table):]))
    log(f"wrote {p}")

if "--robust" in sys.argv:          # the four runs that matter, for repeating on other seeds: --robust --seed-base 427
    THREE = {k: v for k, v in BASE.items() if k not in ("Elegant", "Urban", "Androgynous")}
    L = [f"# The four runs that matter, seeds {SEEDS} on {DEVICE}. FashionStyle14 is the dev half only.", "",
         "| run | wiki top-5 | wiki family top-1 | FashionStyle14 dev | held-out posters | FashionStyle14 dev by style |", "|---|---|---|---|---|---|"]
    for name, mapping, even in (("mapping on disk", BASE, False), ("three broad tags left out", THREE, False),
                                ("mapping on disk, tags drawn evenly", BASE, True), ("three broad tags left out, tags drawn evenly", THREE, True)):
        a, style, _ = run(mapping, even_tags=even)
        L.append(f"| {name} | {fmt(a)} | " + ", ".join(f"{s} {v:.0f}" for s, v in style.items()) + " |")
        log(L[-1])
    L += ["", f"time elapsed {time.time() - T0:.0f}s"]
    (OUT / f"robust-seeds-{SEEDS[0]}-{SEEDS[-1]}.md").write_text("\n".join(L) + "\n", encoding="utf-8")

if CHECK:
    # Written down before it was run (WORKLOG, 21 Nov): one comparison, "mapping on disk, drawn family -> photo" against
    # "Elegant, Urban and Androgynous left out, tags drawn evenly", weight 0.25, the given seeds, on the odd-sha1 half.
    # Per photo: the share of seeds that get its family right; paired bootstrap over photos, 2,000 draws, seed 422.
    THREE = {k: v for k, v in BASE.items() if k not in ("Elegant", "Urban", "Androgynous")}
    a0, s0, _ = run(BASE)
    h0 = np.mean(LAST["hits"], axis=0)
    a1, s1, _ = run(THREE, even_tags=True)
    h1 = np.mean(LAST["hits"], axis=0)
    rng = np.random.default_rng(422)
    draws = np.asarray([(h1[i] - h0[i]).mean() for i in (rng.integers(0, len(h0), len(h0)) for _ in range(2000))]) * 100
    L = [f"# The one check on the unread half of FashionStyle14, {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
         f"Odd-sha1 half: {len(fs):,} photos of {len(set(FS_STYLE))} styles, read here for the first and only time. Seeds {SEEDS} on {DEVICE}, weight 0.25.", "",
         "| run | wiki top-5 | wiki family top-1 | FashionStyle14 check half | held-out posters |", "|---|---|---|---|---|",
         f"| mapping on disk, drawn family -> photo | {fmt(a0)} |", f"| Elegant, Urban, Androgynous left out; tags drawn evenly | {fmt(a1)} |", "",
         f"**Difference on the check half: {100 * (h1 - h0).mean():+.1f} points (95% interval {np.percentile(draws, 2.5):+.1f} to {np.percentile(draws, 97.5):+.1f}; paired bootstrap over photos, 2,000 draws).**", "",
         "| style | before | after | change |", "|---|---|---|---|"] + [f"| {st} | {s0[st]:.0f} | {s1[st]:.0f} | {s1[st] - s0[st]:+.0f} |" for st in s0] + ["", f"time elapsed {time.time() - T0:.0f}s"]
    if DRY:
        L[0] = "# REHEARSAL on the dev half (not the check): " + L[0][2:]
    (OUT / CHECK_FILE).write_text("\n".join(L) + "\n", encoding="utf-8")
    log("\n".join(L))

if "--weights" in sys.argv:         # how much the street-photo loss counts, with tags drawn evenly and the three broad tags left out
    THREE = {k: v for k, v in BASE.items() if k not in ("Elegant", "Urban", "Androgynous")}
    L = [f"# Street-photo weight, tags drawn evenly, three broad tags left out; seeds {SEEDS} on {DEVICE}. FashionStyle14 is the dev half only.", "",
         "| weight | wiki top-5 | wiki family top-1 | FashionStyle14 dev | held-out posters | FashionStyle14 dev by style |", "|---|---|---|---|---|---|"]
    for wgt in (0.1, 0.25, 0.5, 1.0):
        a, style, _ = run(THREE, even_tags=True, weight=wgt)
        L.append(f"| {wgt:g} | {fmt(a)} | " + ", ".join(f"{s} {v:.0f}" for s, v in style.items()) + " |")
        log(L[-1])
    L += ["", f"time elapsed {time.time() - T0:.0f}s"]
    (OUT / f"weights-seeds-{SEEDS[0]}-{SEEDS[-1]}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
