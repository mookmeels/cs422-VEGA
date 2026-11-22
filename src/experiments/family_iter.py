"""Try candidate family tables quickly: score each one several ways, and retrain the NN with real outfit photos through it on a GPU.

    python family_iter.py [--root E:/vega] [--seeds 3] [--device cuda] [--only name ...] [--no-train]     -> runs/family-iter/<time>.md

For ITERATION ONLY. Nothing here writes to the team's results; the model of record is trained by code/nn/train_nn.py on
ml-db. Validation and training images only; the wiki test split and the transfer-test photos (FashionStyle14 lolita, fairy,
ethnic) are never read.

A candidate is the table on disk plus changes (VARIANTS below): labels moved, families merged, or every label sent to its
nearest family centre. For each candidate:

  families, chance        how many families; chance = family top-1 of always guessing the most common family (validation)
  at home                 share of labels whose average training photo is nearest their own family's centre (size-neutral)
  family top-1, rescored  saved validation predictions of the wiki-only NN, nearest centroid, zero-shot text and the CNN, with
                          each guess replaced by its family under the candidate. No training. The CNN never saw the vectors,
                          so it is the one judge here that the embedder did not shape.
  above chance            (family top-1 - chance) / (1 - chance), so tables with fewer families can be compared
  trained through it      the NN with real outfit photos (train_nn.train, same settings) retrained with the candidate deciding
                          which labels a street photo's tag stands for: wiki top-5 and family top-1 on validation; on
                          FashionStyle14 (11 styles, never trained on) and held-out posters (group 0), the family with the
                          largest summed probability, as in runs/real-photos-v1

Merging families raises family top-1 by construction; read "above chance", and do not pick a table by its score alone.
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
SEEDS = [422, 423, 424, 425, 426][:int(arg("--seeds", 3))]
DEVICE = arg("--device", "cuda")
TRAIN = "--no-train" not in sys.argv
ONLY = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "nn"))
import train_nn  # noqa: E402
import vega_data  # noqa: E402
import vega_score  # noqa: E402

V1 = ROOT / "runs" / "family-v1"
TRANSFER_STYLES = {"lolita", "fairy", "ethnic"}
read = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
log = lambda *a: print(*a, flush=True)

# ------------------------------------------------------------------ candidates
VARIANTS = {
    "table as amended": {},
    "+ 4 moves that hold": {"move": {"Clowncore": "kawaii", "Vampire": "goth", "Goth Punk": "rock", "Old Hollywood": "vintage"}},
    "Goth + Rock merged": {"merge": {"goth": "rock"}},
    "Elegant + Classic merged": {"merge": {"elegant": "classic"}},
    "Glam + Club merged": {"merge": {"club": "glam"}},
    "all three merges": {"merge": {"goth": "rock", "elegant": "classic", "club": "glam"}},
    "nearest centre, repeated (embedder-drawn, not a proposal)": {"auto": True},
}

# ------------------------------------------------------------------ data, loaded once
d = vega_data.load(root=ROOT)
classes = [str(c) for c in d["classes"]]
fit, ev = d["fit"], d["eval"]
_ids, truth, _classes, FIT_COUNT = vega_data._eval_truth(False, "merged", ROOT)     # as the shared scorer reads them
assert [str(a) for a in _ids] == [str(a) for a in ev["ids"]] and [str(c) for c in _classes] == classes
nw = len(ev["ids"])
base_lab = {r["wiki_label"]: r["family"] for r in read(V1 / "label_family.csv")}
base_out = [(r["source"], r["style_or_tag"], r["family"]) for r in read(V1 / "outside_family.csv")]
fam_name = {r["family"]: r["name"] for r in read(V1 / "families.csv")}

xf = np.asarray(fit["x"], np.float32)
yf = np.asarray(fit["y"]) > 0
CEN = np.stack([xf[yf[:, j]].mean(0) for j in range(len(classes))])
CEN /= np.linalg.norm(CEN, axis=1, keepdims=True)


def nearest_centre(lab_fam, n_fam):
    """Each label's nearest family centre (centre = average of the family's label averages, the label itself left out)."""
    sums = np.zeros((n_fam, CEN.shape[1]))
    np.add.at(sums, lab_fam, CEN)
    count = np.bincount(lab_fam, minlength=n_fam)
    out = np.empty(len(lab_fam), int)
    for j in range(len(lab_fam)):
        s, c = sums.copy(), count.copy()
        s[lab_fam[j]] -= CEN[j]
        c[lab_fam[j]] -= 1
        fc = s / np.maximum(c, 1)[:, None]
        sim = (fc / (np.linalg.norm(fc, axis=1, keepdims=True) + 1e-12)) @ CEN[j]
        sim[c == 0] = -np.inf
        out[j] = int(np.argmax(sim))
    return out


def build(spec):
    merge, move = spec.get("merge", {}), spec.get("move", {})
    lab = {c: merge.get(move.get(c, f), move.get(c, f)) for c, f in base_lab.items()}
    fams = sorted(set(lab.values()))
    fi = {f: i for i, f in enumerate(fams)}
    lab_fam = np.asarray([fi[lab[c]] for c in classes])
    if spec.get("auto"):
        for _ in range(30):
            new = nearest_centre(lab_fam, len(fams))
            if (new == lab_fam).all():
                break
            lab_fam = new
    out = {(s, t): fi[merge.get(f, f)] for s, t, f in base_out if merge.get(f, f) in fi}
    return fams, lab_fam, out


def saved(pattern):
    files = sorted((ROOT / "runs" / "comparison" / "preds").glob(pattern)) + sorted((ROOT / "runs" / "cnn-validation-preds").glob(pattern))
    out = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        assert [str(c) for c in z["classes"]] == classes, f.name
        where = {str(a): i for i, a in enumerate(z["ids"])}
        out.append(np.asarray(z["scores"], np.float64)[[where[str(a)] for a in ev["ids"]]])
    assert out, pattern
    return out


SAVED = {"NN wiki-only": saved("nn-mlp1x1024-validation-merged-seed*.npz"), "nearest centroid": saved("baseline-nearest-centroid-*.npz"),
         "zero-shot text": saved("baseline-zero-shot-text-*.npz"), "CNN": saved("cnn-r18w64-224-validation-merged-seed*.npz"),
         "popularity": saved("baseline-popularity-*.npz")}
log(f"wiki fit {len(xf):,}, validation {nw:,}; saved predictions: " + ", ".join(f"{k} x{len(v)}" for k, v in SAVED.items()))

if TRAIN:
    sdir, fdir = ROOT / "emb" / "street-fashion-style", ROOT / "emb" / "fashionstyle14"
    vec = f"{vega_data.VECTORS}-448"
    srow = {s: i for i, s in enumerate((sdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
    sfs = [(r["style"], srow[r["id"]], int(hashlib.sha1(r["user"].encode("utf-8")).hexdigest(), 16) % train_nn.REAL_GROUPS)
           for r in read(sdir / "manifest.csv") if r["id"] in srow]
    SFS_X = np.load(sdir / vec / "embeddings.npy", mmap_mode="r")
    frow = {s: i for i, s in enumerate((fdir / vec / "ids.txt").read_text(encoding="utf-8").split())}
    styles = {}
    for r in read(fdir / "manifest.csv"):
        styles.setdefault(r["sha1"], set()).add(r["style"])
    fs = sorted((frow[s], next(iter(st))) for s, st in styles.items() if len(st) == 1 and s in frow and next(iter(st)) not in TRANSFER_STYLES)
    FS_X = np.ascontiguousarray(np.load(fdir / vec / "embeddings.npy", mmap_mode="r")[[r for r, _ in fs]])
    log(f"Street Fashion Style {len(sfs):,} photos; FashionStyle14 {len(fs):,} photos of {len({s for _, s in fs})} styles (never trained on); loaded in {time.time() - T0:.0f}s")


def trained(fams, lab_fam, out):
    """Retrain the NN with real outfit photos through this table; mean and sd over seeds of each figure, in %."""
    tag = {t: f for (s, t), f in out.items() if s == "Street Fashion Style"}
    tr = sorted((tag[st], r) for st, r, g in sfs if st in tag and g != 0)
    he = sorted((r, tag[st]) for st, r, g in sfs if st in tag and g == 0)
    fam = np.asarray([f for f, _ in tr])
    covered = sorted(set(fam.tolist()))
    mask = torch.full((len(fams), len(classes)), -1e9)
    mask[torch.from_numpy(lab_fam), torch.arange(len(classes))] = 0.0
    real = {"x": np.ascontiguousarray(SFS_X[[r for _, r in tr]]), "starts": torch.tensor([int(np.searchsorted(fam, f)) for f in covered]),
            "sizes": torch.tensor([int((fam == f).sum()) for f in covered]), "covered": torch.tensor(covered), "mask": mask}
    held_x, held_f = np.ascontiguousarray(SFS_X[[r for r, _ in he]]), np.asarray([f for _, f in he])
    fs_keep = [k for k, (_, st) in enumerate(fs) if ("FashionStyle14", st) in out]
    fs_f = np.asarray([out[("FashionStyle14", fs[k][1])] for k in fs_keep])
    E = np.concatenate([ev["x"], FS_X[fs_keep], held_x])
    rows = []
    for seed in SEEDS:
        z = train_nn.train(fit["x"], fit["y"], E, train_nn.SETTINGS, seed, DEVICE, real=real).astype(np.float64)
        m, fm = vega_score.metrics(z[:nw], truth, FIT_COUNT), vega_score.family_metrics(z[:nw], truth, lab_fam)
        zo = z[nw:] - z[nw:].max(1, keepdims=True)              # outside photos: the family with the largest summed probability,
        p = np.exp(zo)                                          # as in runs/real-photos-v1 (real_photos_nn.py), so figures compare
        pf = np.zeros((len(p), len(fams)))
        np.add.at(pf.T, lab_fam, (p / p.sum(1, keepdims=True)).T)
        first = pf.argmax(1)
        rows.append([m["top5"], fm["family_top1"], (first[:len(fs_f)] == fs_f).mean(), (first[len(fs_f):] == held_f).mean()])
    a = np.asarray(rows) * 100
    return a.mean(0), a.std(0), len(tr), len(covered)


# ------------------------------------------------------------------ run
names = [n for n in VARIANTS if not ONLY or any(o.lower() in n.lower() for o in ONLY)]
L = [f"# Candidate family tables, {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} (iteration only; not the model of record)", "",
     f"Table on disk sha256 {hashlib.sha256((V1 / 'label_family.csv').read_bytes()).hexdigest()[:12]}. Seeds for the trained column: {SEEDS} on {DEVICE}. "
     "Validation and training images only.", "",
     "| candidate | families | chance | at home | NN wiki-only: family top-1 (above chance) | nearest centroid | zero-shot text | CNN | "
     "trained through it: wiki top-5 | family top-1 | FashionStyle14 | held-out posters |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for n in names:
    t1 = time.time()
    fams, lab_fam, out = build(VARIANTS[n])
    chance = vega_score.family_metrics(SAVED["popularity"][0], truth, lab_fam)["family_top1"]
    home = (nearest_centre(lab_fam, len(fams)) == lab_fam).mean()
    cells = []
    for k in ("NN wiki-only", "nearest centroid", "zero-shot text", "CNN"):
        v = np.mean([vega_score.family_metrics(s, truth, lab_fam)["family_top1"] for s in SAVED[k]])
        cells.append(f"{v:.1%} ({(v - chance) / (1 - chance):.1%})")
    moved = int((lab_fam != build({})[1]).sum()) if len(fams) == len(build({})[0]) else None
    if TRAIN:
        mean, sd, n_real, n_cov = trained(fams, lab_fam, out)
        tcells = [f"{mean[0]:.1f} ± {sd[0]:.1f}", f"{mean[1]:.1f} ± {sd[1]:.1f} ({(mean[1] / 100 - chance) / (1 - chance):.1%})", f"{mean[2]:.1f} ± {sd[2]:.1f}", f"{mean[3]:.1f} ± {sd[3]:.1f}"]
    else:
        tcells = ["", "", "", ""]
    L.append(f"| {n}{f' [{moved} labels differ]' if moved else ''} | {len(fams)} | {chance:.1%} | {home:.1%} | " + " | ".join(cells + tcells) + " |")
    log(L[-1], f"   [{time.time() - t1:.0f}s]")
L += ["", "Figures in brackets are above chance: (family top-1 - chance) / (1 - chance).", f"time elapsed {time.time() - T0:.0f}s"]
out_dir = ROOT / "runs" / "family-iter"
out_dir.mkdir(parents=True, exist_ok=True)
p = out_dir / f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
p.write_text("\n".join(L) + "\n", encoding="utf-8")
log(f"wrote {p}; time elapsed {time.time() - T0:.0f}s")
