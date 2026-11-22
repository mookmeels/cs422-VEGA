"""Build the rating page: do people think the app's suggestions fit?

    E:\\vega\\.venv\\Scripts\\python.exe -B rate_build.py [--root E:/vega] [--sets 10] [--device cuda]
        -> runs/ratings-v1/vega-rate.html   the page: one suggestion per screen, click Fits / Doesn't fit / Can't tell
           runs/ratings-v1/key.csv          what each screen was (the page itself does not say)
           runs/ratings-v1/build.md         how the sets were drawn

The proposal's own check of the suggestions is people's ratings. Each screen shows four photos and ONE suggested aesthetic
with three example pictures. For every set of photos four suggestions are rated, on separate screens and in mixed order:

  top          the first aesthetic of the profile
  nearby       the first "nearby" aesthetic
  surprising   the app's surprising suggestion (per family, as in the app)
  random       a control: an aesthetic drawn at random from a family other than the set's main family

The rater cannot tell which is which. The photo marked "closest match" is, for every kind, the photo the network matches
most with the suggested aesthetic (for the surprising suggestion: with its family, as the app does).

Photo sets, --sets of each, both of real outfit photos the network has never trained on:
  posters      four photos of one poster, drawn at random (Street Fashion Style, the held-out poster group 0; only photos
               whose image file is on this PC, which is under half of them)
  expert       three photos of one FashionStyle14 style and one of a style of another family, in shuffled order
Only sets for which the app SHOWS a surprise (score at or above vega_app.SURPRISE_MIN) are used; build.md says how many
sets were drawn to find them. The network is the app's: frozen settings, street photos, seed 422, trained on the wiki
training and validation images. The wiki test split and FashionStyle14's lolita, fairy and ethnic photos are not read.
"""
import base64
import csv
import hashlib
import io
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

T0 = time.time()
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROOT, DEVICE = Path(arg("--root", "E:/vega")), arg("--device", "cuda")
IMAGES = Path(arg("--images", "E:/pack-test/wiki"))
N_EACH = int(arg("--sets", 10))
HERE = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "ratings-v1"
CUT = float(re.search(r"^SURPRISE_MIN = ([0-9.]+)", (HERE / "vega_app.py").read_text(encoding="utf-8"), re.M).group(1))
POOL = re.search(r'^POOL = "([a-z]+)"', (HERE / "vega_app.py").read_text(encoding="utf-8"), re.M).group(1)      # how the app combines a person's photos
KINDS = ["top", "nearby", "surprising", "random"]
TRANSFER = {"lolita", "fairy", "ethnic"}
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "nn"))
sys.path.insert(0, str(HERE))
import train_nn  # noqa: E402
import vega_data  # noqa: E402
import vega_profile as vp  # noqa: E402

read = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
rng = np.random.default_rng(422)

# ---------------------------------------------------------------- the app's data, as in vega_app.py
d = vega_data.load(root=ROOT)                                          # training and validation; the test split is not read
classes = [str(c) for c in d["classes"]]
X = np.concatenate([d["fit"]["x"], d["eval"]["x"]]).astype(np.float32)
Y = np.concatenate([d["fit"]["y"], d["eval"]["y"]]).astype(np.float32)
PATHS = [IMAGES / Path(str(p)).relative_to(ROOT / "wiki") for p in list(d["fit"]["paths"]) + list(d["eval"]["paths"])]
fam_rows = read(ROOT / "runs" / "family-v1" / "label_family.csv")
FAMILY = {r["wiki_label"]: r["family_name"] for r in fam_rows}
keys = sorted({r["family"] for r in fam_rows})
key_of = {r["wiki_label"]: r["family"] for r in fam_rows}
LAB_FAM = np.asarray([keys.index(key_of[c]) for c in classes])
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
SIM = vp.aesthetic_map(Xn.astype(np.float64), Y > 0)
SIMF = vp.family_map(SIM, LAB_FAM, len(keys))
CEN = ((Y > 0).T.astype(np.float64) @ Xn) / np.maximum((Y > 0).sum(0), 1)[:, None]
CEN /= np.linalg.norm(CEN, axis=1, keepdims=True)
one = (Y > 0).sum(1) == 1


def examples(j):
    """Three example images of an aesthetic: those closest to its average (as the app shows)."""
    rows = np.flatnonzero((Y[:, j] > 0) & one)
    rows = rows if len(rows) else np.flatnonzero(Y[:, j] > 0)
    return [PATHS[int(r)] for r in rows[np.argsort(-(Xn[rows] @ CEN[j]))][:3]]


# ---------------------------------------------------------------- candidate sets of four photos
vec = f"{vega_data.VECTORS}-448"
sdir, simg = ROOT / "emb" / "street-fashion-style", ROOT / "datasets" / "street-fashion-style"
srow = {s: i for i, s in enumerate((sdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
held = sorted((r["user"], srow[r["id"]], r["path"], r["style"]) for r in read(sdir / "manifest.csv")
              if r["id"] in srow and int(hashlib.sha1(r["user"].encode("utf-8")).hexdigest(), 16) % train_nn.REAL_GROUPS == 0
              and (simg / r["path"]).exists())                         # the PC holds the image files of under half of these photos; sets are drawn from those
XP = np.ascontiguousarray(np.load(sdir / vec / "embeddings.npy", mmap_mode="r")[[r for _, r, _, _ in held]])
by_user = defaultdict(list)
for i, (u, _, _, _) in enumerate(held):
    by_user[u].append(i)
users = sorted(u for u, idx in by_user.items() if len(idx) >= 4)
rng.shuffle(users)
poster_sets = [(u, rng.choice(by_user[u], 4, replace=False)) for u in users]         # one set per poster

fdir, fimg = ROOT / "emb" / "fashionstyle14", ROOT / "datasets" / "fashionstyle14"
fams = [r["family"] for r in read(ROOT / "runs" / "family-v1" / "families.csv")]
style_fam = {r["style_or_tag"]: r["family"] for r in read(ROOT / "runs" / "family-v1" / "outside_family.csv")
             if r["source"] == "FashionStyle14" and r["family"] in fams and r["style_or_tag"] not in TRANSFER}
frow = {s: i for i, s in enumerate((fdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
styles, fpath = {}, {}
for r in read(fdir / "manifest.csv"):
    styles.setdefault(r["sha1"], set()).add(r["style"])
    fpath.setdefault(r["sha1"], r["path"])
ph = sorted((frow[s], next(iter(st)), fpath[s]) for s, st in styles.items() if len(st) == 1 and s in frow and next(iter(st)) in style_fam)
XF = np.ascontiguousarray(np.load(fdir / vec / "embeddings.npy", mmap_mode="r")[[r for r, _, _ in ph]])
style = np.asarray([st for _, st, _ in ph])
kinds = sorted(set(style.tolist()))
of_style = {k: np.flatnonzero(style == k) for k in kinds}
expert_sets = []
for k in rng.choice(kinds, 600):
    t = str(rng.choice([q for q in kinds if style_fam[q] != style_fam[k]]))
    rows = np.append(rng.choice(of_style[k], 3, replace=False), rng.choice(of_style[t]))
    expert_sets.append((f"{k} x3 + {t}", rng.permutation(rows)))                     # shuffled: the off-style photo is not always last
rng2 = np.random.default_rng(423)                                      # sets with no off-style photo, only for the note in build.md
expert_pure = [rng2.choice(of_style[k], 4, replace=False) for k in rng2.choice(kinds, 600)]

# ---------------------------------------------------------------- the app's network, once
print(f"training the app's network on {DEVICE} and scoring {len(XP) + len(XF):,} outfit photos ...", flush=True)
z = train_nn.train(X, Y, np.concatenate([XP, XF]), train_nn.SETTINGS, 422, DEVICE, real=train_nn.real_photos(ROOT, classes)).astype(np.float64)
z -= z.max(1, keepdims=True)
PALL = np.exp(z)
PALL /= PALL.sum(1, keepdims=True)
SOURCES = {"posters": (poster_sets, PALL[:len(XP)], lambda i: (simg / held[i][2], held[i][3])),
           "expert": (expert_sets, PALL[len(XP):], lambda i: (fimg / ph[i][2], ph[i][1]))}


def b64(path, box):
    im = Image.open(path)
    im.seek(0)
    im = im.convert("RGB")
    im.thumbnail(box)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=74)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------- keep the sets the app shows a surprise for, and their four suggestions
sets, rated, L = [], [], []                                            # sets: photos per set; rated: one dict per (set, kind)
for source, (cands, P, where) in SOURCES.items():
    kept = looked = 0
    for who, rows in cands:
        if kept == N_EACH:
            break
        looked += 1
        if not all(where(int(i))[0].exists() for i in rows):
            continue
        p = P[rows]
        out = vp.suggest(p, np.ones(len(p), bool), SIM, mode=POOL, lab_fam=LAB_FAM, simf=SIMF, cut=CUT)
        a_s, s, rel, un, photo_s = out["surprising"][0]
        if s < CUT:
            continue
        a_top, share, photo_top = out["profile"][0]
        a_near = out["nearest"][0][0]
        top_fam = int(np.bincount(LAB_FAM, weights=p.mean(0), minlength=len(keys)).argmax())
        used = {a for a, _, _ in out["profile"]} | {a for a, _ in out["nearest"]} | {a for a, *_ in out["surprising"]}
        a_rand = int(rng.choice([a for a in range(len(classes)) if LAB_FAM[a] != top_fam and a not in used]))
        picks = {"top": (a_top, photo_top), "nearby": (a_near, int(p[:, a_near].argmax())),
                 "surprising": (a_s, photo_s), "random": (a_rand, int(p[:, a_rand].argmax()))}
        k = len(sets)
        sets.append([where(int(i)) for i in rows])
        for kind, (a, photo) in picks.items():
            rated.append({"set": k, "source": source, "who": who, "kind": kind, "aesthetic": classes[a], "family": FAMILY[classes[a]], "a": a,
                          "marked_photo": photo + 1, "surprise_score": round(s, 3), "top_aesthetic": classes[a_top],
                          "photo_tags": " | ".join(t for _, t in sets[k]), "photo_files": " | ".join(str(q.relative_to(ROOT / "datasets")).replace("\\", "/") for q, _ in sets[k])})
        kept += 1
    L.append(f"- **{source}**: {kept} sets kept, the first {kept} of {looked} drawn for which the app shows a surprise.")
    print(L[-1], flush=True)

# ---------------------------------------------------------------- over ALL candidate sets (not only the kept ones): how often a surprise is shown,
# and what leaving the surprising family out of the profile does (the cut argument of vega_profile.suggest)
ones = np.ones(4, bool)
L += ["", "## Over all the sets drawn", ""]
for source, (cands, P, where) in SOURCES.items():
    shown = same_first = in_five = lead_old = lead_new = 0
    for who, rows in cands:
        p = P[rows]
        old = vp.suggest(p, ones, SIM, mode=POOL, lab_fam=LAB_FAM, simf=SIMF)
        if old["surprising"][0][1] < CUT:
            continue
        new = vp.suggest(p, ones, SIM, mode=POOL, lab_fam=LAB_FAM, simf=SIMF, cut=CUT)
        shown += 1
        same_first += old["surprising"][0][0] == old["profile"][0][0]
        in_five += old["surprising"][0][0] in [a for a, _, _ in old["profile"]]
        if source == "expert":                                         # is the profile's first aesthetic of the family of the three photos' style?
            major = style_fam[who.split(" x3")[0]]
            lead_old += keys[LAB_FAM[old["profile"][0][0]]] == major
            lead_new += keys[LAB_FAM[new["profile"][0][0]]] == major
    L.append(f"- **{source}**: a surprise is shown for {shown} of {len(cands)} sets ({shown / len(cands):.1%}). Before the surprising family was left out of the profile, "
             f"the surprising aesthetic was also the profile's FIRST aesthetic in {same_first} of them ({same_first / shown:.0%}) and among its five in {in_five} ({in_five / shown:.0%}).")
    if source == "expert":
        L.append(f"  - In those {shown} sets the profile's first aesthetic is of the family of the three photos' style in {lead_old / shown:.1%} before, {lead_new / shown:.1%} after.")
pure = sum(vp.suggest(SOURCES["expert"][1][rows], ones, SIM, mode=POOL, lab_fam=LAB_FAM, simf=SIMF)["surprising"][0][1] >= CUT for rows in expert_pure)
L.append(f"- **expert, four photos of ONE style** (no off-style photo): a surprise is shown for {pure} of {len(expert_pure)} sets ({pure / len(expert_pure):.1%}). "
         "The cut was set for 10%, with networks trained on the training images only; the app's network also saw the validation images.")
L.append(f"- Photos are combined as in the app: `{POOL}` (vega_profile.pooled).")
for line in L[-5:]:
    print(line, flush=True)

# ---------------------------------------------------------------- order: blocks of eight screens, two of each kind, so stopping early stays balanced
n_sets = len(sets)
perm = {kind: rng.permutation(n_sets) for kind in KINDS}
by = {(r["set"], r["kind"]): r for r in rated}
order = []
for j in range(0, n_sets, 2):
    block = [by[(int(s), kind)] for kind in KINDS for s in perm[kind][j:j + 2]]
    order += [block[i] for i in rng.permutation(len(block))]
for n, r in enumerate(order, 1):
    r["n"] = n
version = "v1-" + hashlib.sha1(json.dumps([(r["n"], r["set"], r["kind"], r["aesthetic"]) for r in order]).encode("utf-8")).hexdigest()[:6]

ex_cache = {}
data = {"key": version,
        "sets": [[b64(q, (380, 440)) for q, _ in s] for s in sets],
        "items": [{"n": r["n"], "set": r["set"], "name": r["aesthetic"], "photo": r["marked_photo"] - 1,
                   "ex": ex_cache.setdefault(r["a"], [b64(q, (260, 260)) for q in examples(r["a"])])} for r in order]}
page = (HERE / "rate_page.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "vega-rate.html").write_text(page, encoding="utf-8")
cols = ["n", "set", "source", "who", "kind", "aesthetic", "family", "marked_photo", "surprise_score", "top_aesthetic", "photo_tags", "photo_files"]
with open(OUT / "key.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(order)
(OUT / "build.md").write_text("\n".join(
    [f"# The rating page, version {version}", "",
     f"Made by `code/app/rate_build.py` on {DEVICE}. {n_sets} sets of four real outfit photos, {len(order)} screens: for each set its top, nearby and surprising "
     f"suggestion and a random control, one per screen, in blocks of eight screens holding two of each kind.",
     f"Only sets for which the app shows a surprise (score at or above {CUT}) are used. A family shown as surprising is left out of the profile and the nearby list (as in the app):", ""] + L +
    ["", "`key.csv` says what each screen was. Paste returned answer lines into `answers.txt`, one per line, and run `rate_score.py`.",
     f"time elapsed {time.time() - T0:.0f}s"]) + "\n", encoding="utf-8")
print(f"{version}: {len(order)} screens, page {len(page) / 1e6:.1f} MB -> {OUT / 'vega-rate.html'}; time elapsed {time.time() - T0:.0f}s", flush=True)
