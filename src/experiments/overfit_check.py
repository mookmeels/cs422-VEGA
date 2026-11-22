"""How overfitted is the NN? Training images against validation images, epoch by epoch. Validation data only.

    python overfit_check.py [--root /srv/ml/vega]      -> runs/nn/overfit-check/summary.md, curves.csv

Trains the NN exactly as train_nn.py does (seed 422, one CPU thread), once without real photos and once with them
(weight 0.25), and after every epoch scores, with dropout switched off:
  - the wiki TRAINING images it is learning from, and the wiki VALIDATION images it never sees;
  - 20,000 of the real photos it trains on, and the 21,024 photos of the 530 posters it never trains on (family level).
The gap between the two sides is the overfitting. The test split is not read.
"""
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

t0 = time.perf_counter()
ROOT = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else Path("/srv/ml/vega")
sys.path.insert(0, str(ROOT / "code" / "nn"))
import train_nn  # noqa: E402

vega_data, vega_score = train_nn.vega_data, train_nn.vega_score
OUT = ROOT / "runs" / "nn" / "overfit-check"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
SEED, S = 422, train_nn.SETTINGS

d = vega_data.load(root=ROOT)
classes, fit, ev = list(d["classes"]), d["fit"], d["eval"]
count = (fit["y"] > 0).sum(0)
real = train_nn.real_photos(ROOT, classes)
member = (real["mask"].numpy() == 0).T.astype(np.float64)                   # labels x families
real_fam = real["photo_family"]                                               # the family of every training photo
pick = np.sort(np.random.default_rng(422).choice(len(real_fam), 20000, replace=False))

# the posters never trained on (name hashes to 0 mod 10), read the same way real_photos reads the others
fam_dir, sfs = ROOT / "runs" / "family-v1", ROOT / "emb" / "street-fashion-style"
vec = sfs / f"{vega_data.VECTORS}-448"
read = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
fi = {r["family"]: i for i, r in enumerate(read(fam_dir / "families.csv"))}
tag = {r["style_or_tag"]: fi[r["family"]] for r in read(fam_dir / "outside_family.csv") if r["source"] == "Street Fashion Style" and r["family"] in fi}
row = {s: i for i, s in enumerate((vec / "ids.txt").read_text(encoding="utf-8").split())}
held = sorted((row[r["id"]], tag[r["style"]]) for r in read(sfs / "manifest.csv") if r["style"] in tag and r["id"] in row
              and int(hashlib.sha1(r["user"].encode("utf-8")).hexdigest(), 16) % train_nn.REAL_HELD_OUT == 0)
held_x = np.ascontiguousarray(np.load(vec / "embeddings.npy", mmap_mode="r")[[r for r, _ in held]])
held_f = np.asarray([f for _, f in held])

E = np.concatenate([fit["x"], ev["x"], real["x"][pick], held_x])
c1, c2, c3 = len(fit["ids"]), len(fit["ids"]) + len(ev["ids"]), len(fit["ids"]) + len(ev["ids"]) + len(pick)


def loss(z, y):                                                              # plain cross-entropy against the label rows
    z = z.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    return float(-(y * (z - np.log(np.exp(z).sum(1, keepdims=True)))).sum(1).mean())


def family_top1(z, f):
    z = z.astype(np.float64)
    p = np.exp(z - z.max(1, keepdims=True))
    return float((((p / p.sum(1, keepdims=True)) @ member).argmax(1) == f).mean())


curves = []
for name, r in (("wiki-only NN", None), ("NN with real outfit photos", real)):
    t1 = time.perf_counter()

    def each_epoch(epoch, _, z, name=name):
        a = vega_score.metrics(z[:c1], fit["truth"], count)
        b = vega_score.metrics(z[c1:c2], ev["truth"], count)
        curves.append((name, epoch, loss(z[:c1], fit["y"]), loss(z[c1:c2], ev["y"]), a["top1"], b["top1"], a["top5"], b["top5"],
                       family_top1(z[c2:c3], real_fam[pick]), family_top1(z[c3:], held_f)))

    train_nn.train(fit["x"], fit["y"], E, S, SEED, "cpu", each_epoch, real=r)
    print(f"{name}: trained in {time.perf_counter() - t1:.0f}s", flush=True)

with open(OUT / "curves.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "epoch", "train_loss", "validation_loss", "train_top1", "validation_top1", "train_top5", "validation_top5",
                "real_training_photos_family_top1", "held_out_posters_family_top1"])
    for c in curves:
        w.writerow([c[0], c[1]] + [f"{v:.4f}" for v in c[2:]])

L = ["# How overfitted is the NN? (seed 422, validation data only, dropout off when scoring)", "",
     f"Wiki: {c1:,} training images, {c2 - c1:,} validation images. Real photos: 20,000 of the {len(real_fam):,} trained on, "
     f"and {len(held_f):,} from posters never trained on. The wiki-only NN never sees a real photo, so both its real-photo columns are unseen photos.", ""]
for name in ("wiki-only NN", "NN with real outfit photos"):
    rows = [c for c in curves if c[0] == name]
    best = min(rows, key=lambda c: c[3])
    L += [f"## {name}", "",
          "| epoch | training loss | validation loss | training top-1 | validation top-1 | training top-5 | validation top-5 | real photos trained on: family top-1 | posters never trained on: family top-1 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in rows:
        if c[1] in (1, 2, 5, 10, 15, 20, 25, 30, 35, 40):
            L.append(f"| {c[1]} | {c[2]:.3f} | {c[3]:.3f} | {100 * c[4]:.1f} | {100 * c[5]:.1f} | {100 * c[6]:.1f} | {100 * c[7]:.1f} | {100 * c[8]:.1f} | {100 * c[9]:.1f} |")
    last = rows[-1]
    L += ["", f"Lowest validation loss: {best[3]:.3f} at epoch {best[1]}; at the last epoch it is {last[3]:.3f}. "
              f"Best validation top-5: {100 * max(c[7] for c in rows):.1f} at epoch {max(rows, key=lambda c: c[7])[1]}; last epoch {100 * last[7]:.1f}.", ""]
L += [f"Time elapsed: {time.perf_counter() - t0:.0f} s. One seed, so differences of a few tenths between epochs are noise."]
(OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
