"""Same outfit, wearer hidden: do the suggestions change when the head is covered?

    python fairness_head.py [--root E:/vega] [--per-style 180] [--seeds 3] [--device cuda]   -> runs/fairness-v1/head.md, head.json

The proposal's fairness question is whether similar outfits get different predictions depending on who wears them. No
dataset we hold says who is in a photo, so this holds the outfit fixed and hides the wearer instead. For FashionStyle14
photos with one clearly found, tall person (runs/person-crops/fs14.csv), five versions are embedded with the team's
embedder and scored by the recommended NN (frozen settings, never trained on FashionStyle14):
  orig   the photo as it is
  head   a grey block over the top 15% of the person box (the head)
  blur   the same region heavily blurred instead (shapes gone, colours left)
  mid    the same grey block over 42 to 57% of the person box (waist and hips: clothing)
  low    the same grey block over the bottom 15% (feet and shoes)
"mid" and "low" give the scale: how much does covering a clothing region of the same size change the answer?
If "head" changes the family about as often as a clothing region, the model reads the head as much as it reads clothes.
Limit, stated in the report: hair, make-up and headwear are part of some aesthetics, so some change is legitimate; and
this says nothing direct about race, body type or age. Lolita, fairy and ethnic photos are not read; nor is the wiki test split.
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

T0 = time.time()
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROOT, DEVICE = Path(arg("--root", "E:/vega")), arg("--device", "cuda")
PER, SEEDS = int(arg("--per-style", 180)), [422 + i for i in range(int(arg("--seeds", 3)))]
EMBED_PY = arg("--embed-python", sys.executable)       # the Python that runs the embedder (the 22 Nov run used the PC's miniconda Python)
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "nn"))
import train_nn  # noqa: E402
import vega_data  # noqa: E402

read = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
V1, OUT = ROOT / "runs" / "family-v1", ROOT / "runs" / "fairness-v1"
WORK = OUT / "head"
TRANSFER = {"lolita", "fairy", "ethnic"}
VARIANTS = ("orig", "head", "blur", "mid", "low")
BAND = {"head": (0.0, 0.15), "blur": (0.0, 0.15), "mid": (0.42, 0.57), "low": (0.85, 1.0)}

d = vega_data.load(root=ROOT)
fit, classes = d["fit"], [str(c) for c in d["classes"]]
fams = [r["family"] for r in read(V1 / "families.csv")]
fname = {r["family"]: r["name"] for r in read(V1 / "families.csv")}
of_label = {r["wiki_label"]: r["family"] for r in read(V1 / "label_family.csv")}
lab_fam = np.asarray([fams.index(of_label[c]) for c in classes])
M = np.zeros((len(classes), len(fams)))
M[np.arange(len(classes)), lab_fam] = 1
style_fam = {r["style_or_tag"]: fams.index(r["family"]) for r in read(V1 / "outside_family.csv")
             if r["source"] == "FashionStyle14" and r["family"] in fams and r["style_or_tag"] not in TRANSFER}

fdir = ROOT / "emb" / "fashionstyle14"
styles, path_of = {}, {}
for r in read(fdir / "manifest.csv"):
    styles.setdefault(r["sha1"], set()).add(r["style"])
    path_of[r["sha1"]] = r["path"]
boxes = {}
for r in read(ROOT / "runs" / "person-crops" / "fs14.csv"):
    people = [b for b in json.loads(r["people"])] if int(r["n_people"]) > 0 else []
    strong = [b for b in people if b[4] >= 0.9]
    if len(people) == 1 and len(strong) == 1 and (strong[0][3] - strong[0][1]) >= 0.6 * float(r["height"]):
        boxes[r["id"]] = strong[0][:4]
rng = np.random.default_rng(422)
chosen = []
for st in sorted(style_fam):
    pool = sorted(s for s, v in styles.items() if v == {st} and s in boxes)
    chosen += [(s, st) for s in rng.choice(pool, min(PER, len(pool)), replace=False)]
print(f"{len(chosen):,} photos of {len(style_fam)} styles with one tall, clearly found person", flush=True)

# ------------------------------------------------------------------ the five versions of each photo
src = ROOT / "datasets" / "fashionstyle14"
made = 0
for v in VARIANTS:
    (WORK / "img" / v).mkdir(parents=True, exist_ok=True)
for s, _ in chosen:
    if all((WORK / "img" / v / f"{s}.jpg").exists() for v in VARIANTS):
        continue
    with Image.open(next(p for p in (src / path_of[s], src / "FashionStyle14_v1" / path_of[s]) if p.exists())) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    x0, y0, x1, y1 = boxes[s]
    h = y1 - y0
    for v in VARIANTS:
        out = im.copy()
        if v != "orig":
            a, b = BAND[v]
            region = (int(x0), int(y0 + a * h), int(x1), int(y0 + b * h))
            if v == "blur":
                out.paste(out.crop(region).filter(ImageFilter.GaussianBlur(max(6, 0.06 * h))), region)
            else:
                ImageDraw.Draw(out).rectangle(region, fill=(128, 128, 128))
        out.save(WORK / "img" / v / f"{s}.jpg", quality=95)
    made += 1
print(f"versions written for {made:,} photos ({time.time() - T0:.0f}s)", flush=True)

# ------------------------------------------------------------------ embed them as the app embeds an upload
vecdir = WORK / "emb" / "head-448"
want = [f"{v}-{s}" for v in VARIANTS for s, _ in chosen]
if not (vecdir / "embeddings.npy").exists() or set((vecdir / "ids.txt").read_text(encoding="utf-8").split()) != set(want):
    (WORK / "list.tsv").write_text("".join(f"{v}-{s}\t{v}/{s}.jpg\n" for v in VARIANTS for s, _ in chosen), encoding="utf-8")
    cmd = [EMBED_PY, "-u", str(ROOT / "code" / "experiments" / "embed_eos_check.py"), "2b", "--list", str(WORK / "list.tsv"), "--out", str(WORK / "emb"),
           "--wiki", str(WORK / "img"), "--no-text", "--name", "head", "--port", "18084", "--instruction", "Represent the fashion aesthetic of this image."]
    t1 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT / "code" / "experiments"))
    if r.returncode != 0 or not (vecdir / "embeddings.npy").exists():
        raise SystemExit("the embedder failed")
    print(f"embedded {len(want):,} images in {time.time() - t1:.0f}s", flush=True)
ids = (vecdir / "ids.txt").read_text(encoding="utf-8").split()
row = {a: i for i, a in enumerate(ids)}
E = np.load(vecdir / "embeddings.npy")
X = {v: E[[row[f"{v}-{s}"] for s, _ in chosen]] for v in VARIANTS}
style = np.asarray([st for _, st in chosen])
truef = np.asarray([style_fam[st] for st in style])
stored = np.load(fdir / f"{vega_data.VECTORS}-448" / "embeddings.npy", mmap_mode="r")
srow = {s: i for i, s in enumerate((fdir / f"{vega_data.VECTORS}-448" / "ids.txt").read_text(encoding="utf-8").split())}
S0 = np.asarray(stored[[srow[s] for s, _ in chosen]], np.float64)
cos = (X["orig"] * S0).sum(1) / (np.linalg.norm(X["orig"], axis=1) * np.linalg.norm(S0, axis=1))
print(f"check: the re-embedded original against the stored vector, cosine median {np.median(cos):.4f}, lowest {cos.min():.4f}", flush=True)

# ------------------------------------------------------------------ score
real = train_nn.real_photos(ROOT, classes)
allx = np.concatenate([X[v] for v in VARIANTS])
n = len(chosen)
right = {v: [] for v in VARIANTS}
changed = {v: [] for v in VARIANTS}
changed_a = {v: [] for v in VARIANTS}
moved = {v: [] for v in VARIANTS}
for seed in SEEDS:
    z = train_nn.train(fit["x"], fit["y"], allx, train_nn.SETTINGS, seed, DEVICE, real=real).astype(np.float64)
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    pf = p @ M
    base_f, base_a, base_pf = pf[:n].argmax(1), p[:n].argmax(1), pf[:n]
    for k, v in enumerate(VARIANTS):
        q, qa = pf[k * n:(k + 1) * n], p[k * n:(k + 1) * n]
        right[v].append(q.argmax(1) == truef)
        changed[v].append(q.argmax(1) != base_f)
        changed_a[v].append(qa.argmax(1) != base_a)
        moved[v].append(0.5 * np.abs(q - base_pf).sum(1))
    print(f"  seed {seed}: family right on the originals {right['orig'][-1].mean():.1%}", flush=True)
del real
mean = lambda dct, v: np.mean(dct[v], axis=0)
rs = np.random.default_rng(422)
draws = [rs.integers(0, n, n) for _ in range(2000)]


def ci(vec):
    b = [vec[dr].mean() for dr in draws]
    return float(vec.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


f3 = lambda t: f"{100 * t[0]:.1f}% ({100 * t[1]:.1f} to {100 * t[2]:.1f})"
d3 = lambda t: f"{100 * t[0]:+.1f} ({100 * t[1]:+.1f} to {100 * t[2]:+.1f})"
NAME = {"orig": "the photo as it is", "head": "head covered (grey block)", "blur": "head blurred", "mid": "waist and hips covered", "low": "feet covered"}
res = {"photos": n, "seeds": SEEDS, "cosine_reembedded_median": float(np.median(cos))}
L = ["# Same outfit, wearer hidden: what changes when the head is covered", "",
     f"Made {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by code/experiments/fairness_head.py. {n:,} FashionStyle14 photos ({PER} per style at most, 11 styles) with one "
     f"clearly found person at least 0.6 of the photo tall. NN with real outfit photos, frozen settings, seeds {SEEDS}; it never trained on FashionStyle14. Intervals are 95%, resampling photos.", "",
     "| version | family right | first family differs from the original's | first aesthetic differs | how far the family shares move (0 to 1) |", "|---|---|---|---|---|"]
for v in VARIANTS:
    res[v] = {"family_right": ci(mean(right, v)), "family_changed": ci(mean(changed, v)), "aesthetic_changed": ci(mean(changed_a, v)), "moved": float(mean(moved, v).mean())}
    L.append(f"| {NAME[v]} | {f3(res[v]['family_right'])} | " + ("" if v == "orig" else f3(res[v]["family_changed"])) + " | "
             + ("" if v == "orig" else f3(res[v]["aesthetic_changed"])) + " | " + ("" if v == "orig" else f"{res[v]['moved']:.3f}") + " |")
res["head_minus_mid_changed"] = ci(mean(changed, "head") - mean(changed, "mid"))
res["head_minus_low_changed"] = ci(mean(changed, "head") - mean(changed, "low"))
res["head_minus_orig_right"] = ci(mean(right, "head") - mean(right, "orig"))
res["mid_minus_orig_right"] = ci(mean(right, "mid") - mean(right, "orig"))
L += ["", f"- Covering the head changes the first family {d3(res['head_minus_mid_changed'])} points more often than covering the waist and hips, "
      f"and {d3(res['head_minus_low_changed'])} more often than covering the feet.",
      f"- Family right: head covered {d3(res['head_minus_orig_right'])} points against the original; waist and hips covered {d3(res['mid_minus_orig_right'])}.", "",
      "## By style: how often the first family changes", "", "| style | photos | head covered | head blurred | waist and hips covered | feet covered | family right, original | family right, head covered |", "|---|---|---|---|---|---|---|---|"]
res["by_style"] = {}
for st in sorted(set(style.tolist()), key=lambda s: -mean(changed, "head")[style == s].mean()):
    m = style == st
    res["by_style"][st] = {v: float(mean(changed, v)[m].mean()) for v in VARIANTS[1:]} | {"right_orig": float(mean(right, "orig")[m].mean()), "right_head": float(mean(right, "head")[m].mean())}
    L.append(f"| {st} | {int(m.sum())} | " + " | ".join(f"{mean(changed, v)[m].mean():.1%}" for v in VARIANTS[1:])
             + f" | {mean(right, 'orig')[m].mean():.1%} | {mean(right, 'head')[m].mean():.1%} |")
L += ["", f"Check: the re-embedded originals match the stored vectors (cosine median {np.median(cos):.4f}, lowest {cos.min():.4f}).", "", f"Time elapsed: {time.time() - T0:.0f} s"]
(OUT / "head.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(OUT / "head.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("\n".join(L))
