"""Real photos in training: the 503-label NN with Street Fashion Style photos added through their family. EXPERIMENT.

The rules were fixed first: runs/real-photos-v1/RULES.md. Read them before reading the results.

    python real_photos_nn.py [--root /srv/ml/vega]     -> runs/real-photos-v1/{results.md, results.json, preds/, check.txt}

What is trained: code/nn/train_nn.py's network and SETTINGS, unchanged, 503 outputs, the wiki loss unchanged. In each
training step 512 real photos are added, with loss = minus the log of the summed probability of the photo's family
(frozen table runs/family-v1), times `weight`. Exactly three weights: 0 (the NN as it is), 0.25, 1.0. Seeds 422-426.

What is held out: Street Fashion Style posters whose name hashes to 0 mod 10; all of FashionStyle14 (its lolita, fairy
and ethnic photos, the transfer test, are not read at all); the wiki validation split. The wiki test split is not read.

Self-check: with weight 0 this file's training loop must give the predictions train_nn.py saved for the team table.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

t0 = time.perf_counter()
ROOT = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else Path("/srv/ml/vega")
sys.path.insert(0, str(ROOT / "code" / "nn"))
import train_nn  # noqa: E402

vega_data = train_nn.vega_data
SEEDS = [422, 423, 424, 425, 426]
WEIGHTS = [0.0, 0.25, 1.0]
REAL_BATCH = 512
TRANSFER_STYLES = {"lolita", "fairy", "ethnic"}
S = train_nn.SETTINGS
V1, OUT = ROOT / "runs" / "family-v1", ROOT / "runs" / "real-photos-v1"
assert (OUT / "RULES.md").exists(), "the rules come first"
SMOKE = "--smoke" in sys.argv                               # does the code run end to end? 2 seeds, 1 epoch, results thrown away
if SMOKE:
    SEEDS, S, OUT = SEEDS[:2], dict(S, epochs=1), OUT / "smoke"
(OUT / "preds").mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
log = lambda *a: print(*a, flush=True)

# ------------------------------------------------------------------ wiki
d = vega_data.load(root=ROOT)
classes = list(d["classes"])
fit, ev = d["fit"], d["eval"]
table = pl.read_csv(V1 / "label_family.csv")
lab2fam = dict(zip(table["wiki_label"], table["family"]))
fam_table = pl.read_csv(V1 / "families.csv")
families = list(fam_table["family"])
fam_name = dict(zip(fam_table["family"], fam_table["name"]))
fi = {f: i for i, f in enumerate(families)}
lab_fam = np.asarray([fi[lab2fam[c]] for c in classes])
M = np.zeros((len(classes), len(families)), np.float64)
M[np.arange(len(classes)), lab_fam] = 1
truth = ev["truth"]
truth_f = [set(lab_fam[t].tolist()) for t in truth]
wiki_ids = ev["ids"].tolist()
nw = len(wiki_ids)

det = pl.read_csv(ROOT / "runs" / "person-crops" / "wiki-trainval.csv")
outfit = {}
for i, h, k, boxes in zip(det["id"], det["height"], det["n_people"], det["people"]):
    outfit[i] = any((b[3] - b[1]) >= 0.5 * h for b in json.loads(boxes)) if k > 0 else False
is_outfit = np.asarray([outfit[i] for i in wiki_ids])

# ------------------------------------------------------------------ real photos
om = pl.read_csv(V1 / "outside_family.csv", infer_schema=False)
sfs_map = {t: f for s, t, f in zip(om["source"], om["style_or_tag"], om["family"]) if s == "Street Fashion Style" and f in fi}
fs_map = {t: f for s, t, f in zip(om["source"], om["style_or_tag"], om["family"])
          if s == "FashionStyle14" and f in fi and t not in TRANSFER_STYLES}
covered = sorted({fi[f] for f in sfs_map.values()})
is_cov = np.zeros(len(families), bool)
is_cov[covered] = True

sdir = ROOT / "emb" / "street-fashion-style"
srow = {s: i for i, s in enumerate((sdir / "qwen2beos-aesthetic-448" / "ids.txt").read_text(encoding="utf-8").split())}
man = pl.read_csv(sdir / "manifest.csv", infer_schema=False).select("id", "user", "style")
tr, he = [], []                                             # (family, row) for training; (row, family, poster) held out
for i, u, st in man.iter_rows():
    if st in sfs_map and i in srow:
        held = int(hashlib.sha1(u.encode("utf-8")).hexdigest(), 16) % 10 == 0
        (he if held else tr).append((fi[sfs_map[st]], srow[i], u))
tr.sort()
mm = np.load(sdir / "qwen2beos-aesthetic-448" / "embeddings.npy", mmap_mode="r")
real_x = np.ascontiguousarray(mm[[r for _, r, _ in tr]])
real_f = np.asarray([f for f, _, _ in tr])
starts = torch.tensor([int(np.searchsorted(real_f, f)) for f in covered])
sizes = torch.tensor([int((real_f == f).sum()) for f in covered])
cov_t = torch.tensor(covered)
he.sort(key=lambda r: r[1])
held_x = np.ascontiguousarray(mm[[r for _, r, _ in he]])
held_f = np.asarray([f for f, _, _ in he])
posters = sorted({u for _, _, u in he})
pidx = {u: k for k, u in enumerate(posters)}
held_p = np.asarray([pidx[u] for _, _, u in he])
del mm

fdir = ROOT / "emb" / "fashionstyle14"
frow = {s: i for i, s in enumerate((fdir / "qwen2beos-aesthetic-448" / "ids.txt").read_text(encoding="utf-8").split())}
fman = pl.read_csv(fdir / "manifest.csv", infer_schema=False).group_by("sha1").agg(pl.col("style").unique())
fs = sorted((frow[s], st[0]) for s, st in fman.iter_rows() if len(st) == 1 and st[0] in fs_map and s in frow)
fm = np.load(fdir / "qwen2beos-aesthetic-448" / "embeddings.npy", mmap_mode="r")
fs_x = np.ascontiguousarray(fm[[r for r, _ in fs]])        # only the 11 styles that are not the transfer test
fs_style = np.asarray([st for _, st in fs])
fs_f = np.asarray([fi[fs_map[st]] for st in fs_style])
del fm

log(f"wiki: fit {len(fit['ids']):,}, validation {nw:,}. Street Fashion Style: {len(tr):,} training photos in {len(covered)} families, "
    f"{len(he):,} held out from {len(posters):,} posters. FashionStyle14: {len(fs):,} photos of {len(set(fs_style))} styles (never trained on). "
    f"loaded in {time.perf_counter() - t0:.0f}s")

scale = train_nn.input_scaling(fit["x"], S["prep"])
E = torch.from_numpy(np.concatenate([scale(ev["x"]), scale(fs_x), scale(held_x)]))
cut1, cut2 = nw, nw + len(fs)
MASK = torch.full((len(families), len(classes)), -1e9)
MASK[torch.from_numpy(lab_fam), torch.arange(len(classes))] = 0.0


def train_mixed(seed, weight):
    """train_nn.train, line for line, plus the real-photo term when weight > 0."""
    torch.manual_seed(seed)
    X, Y = torch.from_numpy(scale(fit["x"])), torch.from_numpy(fit["y"])
    W = torch.from_numpy(train_nn.image_weights(fit["y"], S["balance"]))
    net = train_nn.build(X.shape[1], Y.shape[1], S)
    opt = torch.optim.AdamW(net.parameters(), lr=S["lr"], weight_decay=S["weight_decay"])
    steps = S["epochs"] * math.ceil(len(X) / S["batch"])
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / warm if i < warm else
                                              0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, steps - warm))))
    shuffle = torch.Generator().manual_seed(seed)
    pick = torch.Generator().manual_seed(seed + 100_000)    # real photos have their own generator: the wiki order is untouched
    for _ in range(S["epochs"]):
        net.train()
        order = torch.randperm(len(X), generator=shuffle)
        for i in range(0, len(X), S["batch"]):
            idx = order[i:i + S["batch"]]
            if weight > 0:
                f = torch.randint(len(covered), (REAL_BATCH,), generator=pick)
                rows = starts[f] + (torch.rand(REAL_BATCH, generator=pick) * sizes[f]).long()
                xr = torch.from_numpy(scale(real_x[rows.numpy()]))
                logp = F.log_softmax(net(torch.cat([X[idx], xr])), dim=1)
                loss_wiki = (W[idx] * -(Y[idx] * logp[:len(idx)]).sum(1)).mean()
                loss_real = -torch.logsumexp(logp[len(idx):] + MASK[cov_t[f]], dim=1).mean()
                loss = loss_wiki + weight * loss_real
            else:
                loss = (W[idx] * -(Y[idx] * F.log_softmax(net(X[idx]), dim=1)).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return train_nn.predict(net, E)


def softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------ runs
keys = ("l1", "l5", "f1", "wf", "fs", "sh")
res = {w: {k: [] for k in keys} for w in WEIGHTS}
check = []
for seed in SEEDS:
    for w in WEIGHTS:
        t1 = time.perf_counter()
        z = train_mixed(seed, w)
        zw = z[:cut1]
        np.savez_compressed(OUT / "preds" / f"nn-real{w:g}-validation-merged-seed{seed}.npz", ids=ev["ids"],
                            scores=zw.astype(np.float32), classes=np.asarray(classes))
        if w == 0 and not SMOKE:
            saved = np.load(ROOT / "runs" / "comparison" / "preds" / f"nn-mlp1x1024-validation-merged-seed{seed}.npz", allow_pickle=True)
            assert saved["ids"].tolist() == wiki_ids
            gap = float(np.abs(saved["scores"] - zw).max())
            same = bool((np.argsort(-saved["scores"], 1)[:, :5] == np.argsort(-zw, 1)[:, :5]).all())
            check.append(f"seed {seed}: largest difference from the team table's saved scores {gap:.2e}; the same five first guesses for every image: {same}")
            log("  check,", check[-1])
            assert same and gap < 1e-2, "this loop does not reproduce train_nn.py"
        order = np.argsort(-zw, axis=1)[:, :5]
        pf = softmax(z) @ M
        wf = pf[:cut1].argmax(1)
        res[w]["l1"].append(np.asarray([o[0] in t for o, t in zip(order, truth)], np.float64))
        res[w]["l5"].append(np.asarray([bool(set(o.tolist()) & set(t.tolist())) for o, t in zip(order, truth)], np.float64))
        res[w]["f1"].append(np.asarray([g in t for g, t in zip(wf, truth_f)], np.float64))
        res[w]["wf"].append(wf)
        res[w]["fs"].append((pf[cut1:cut2].argmax(1) == fs_f).astype(np.float64))
        res[w]["sh"].append((pf[cut2:].argmax(1) == held_f).astype(np.float64))
        np.savez_compressed(OUT / "preds" / f"nn-real{w:g}-realphotos-family-seed{seed}.npz", fs14=pf[cut1:cut2].astype(np.float16),
                            sfs_held=pf[cut2:].astype(np.float16), families=np.asarray(families))
        log(f"  seed {seed} weight {w:g}: wiki top-5 {100 * res[w]['l5'][-1].mean():.1f}, family {100 * res[w]['f1'][-1].mean():.1f}; "
            f"FashionStyle14 family {100 * res[w]['fs'][-1].mean():.1f}; held-out posters {100 * res[w]['sh'][-1].mean():.1f}; {time.perf_counter() - t1:.0f}s")
(OUT / "check.txt").write_text("Weight 0 in real_photos_nn.py against the predictions train_nn.py saved for the team table:\n" + "\n".join(check) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ tables
uncov_out = np.asarray([o and not (t & set(covered)) for o, t in zip(is_outfit, truth_f)])
per = {w: {k: np.mean(res[w][k], axis=0) for k in ("l1", "l5", "f1", "fs", "sh")} for w in WEIGHTS}
for w in WEIGHTS:
    per[w]["pull"] = np.mean([is_cov[g] for g in res[w]["wf"]], axis=0).astype(np.float64)   # first family guess is a covered family
rng = np.random.default_rng(422)


def paired(a, b, keep=None, cluster=None):
    diff = a - b if keep is None else (a - b)[keep]
    if cluster is None:
        boot = diff[rng.integers(0, len(diff), size=(2000, len(diff)))].mean(1)
    else:                                                    # resample posters, not photos
        tot = np.bincount(cluster, weights=diff)
        cnt = np.bincount(cluster).astype(np.float64)
        pick_ = rng.integers(0, len(tot), size=(2000, len(tot)))
        boot = tot[pick_].sum(1) / cnt[pick_].sum(1)
    return 100 * diff.mean(), 100 * float(np.percentile(boot, 2.5)), 100 * float(np.percentile(boot, 97.5))


def ms(w, k, keep=None):
    v = [100 * (r if keep is None else r[keep]).mean() for r in res[w][k]]
    return f"{np.mean(v):.1f} ± {np.std(v):.1f}"


def style_mean(w):
    return np.mean([100 * np.mean([r[fs_style == st].mean() for r in res[w]["fs"]]) for st in sorted(set(fs_style))])


name = {0.0: "weight 0 (the NN as it is)", 0.25: "weight 0.25", 1.0: "weight 1.0"}
L = ["# Real photos in training: results (validation data only; rules in RULES.md)", "",
     f"Wiki: {len(fit['ids']):,} training images, {nw:,} validation images ({int(is_outfit.sum())} show an outfit). Street Fashion Style: "
     f"{len(tr):,} training photos in {len(covered)} of the 20 families; {len(he):,} photos of {len(posters):,} held-out posters. "
     f"FashionStyle14: {len(fs):,} photos of {len(set(fs_style))} styles, never trained on. Seeds 422 to 426, mean ± spread.", "",
     "| model | wiki aesthetic top-1 | wiki aesthetic top-5 | wiki family top-1 | FashionStyle14 family top-1 | the same, each style equal | held-out posters family top-1 |",
     "|---|---|---|---|---|---|---|"]
for w in WEIGHTS:
    L.append(f"| {name[w]} | {ms(w, 'l1')} | {ms(w, 'l5')} | {ms(w, 'f1')} | {ms(w, 'fs')} | {style_mean(w):.1f} | {ms(w, 'sh')} |")
L += ["", "## Differences from weight 0 (paired bootstrap, 2,000 draws, seed 422; 95% interval)", "",
      "| model | wiki aesthetic top-5 | wiki aesthetic top-1 | wiki family top-1 | FashionStyle14 family top-1 | held-out posters family top-1 |", "|---|---|---|---|---|---|"]
fmt = lambda c: f"{c[0]:+.1f} ({c[1]:+.1f} to {c[2]:+.1f})"
D = {}
for w in WEIGHTS[1:]:
    D[w] = {"l5": paired(per[w]["l5"], per[0.0]["l5"]), "l1": paired(per[w]["l1"], per[0.0]["l1"]), "f1": paired(per[w]["f1"], per[0.0]["f1"]),
            "fs": paired(per[w]["fs"], per[0.0]["fs"]), "sh": paired(per[w]["sh"], per[0.0]["sh"], cluster=held_p),
            "out": paired(per[w]["f1"], per[0.0]["f1"], is_outfit), "oth": paired(per[w]["f1"], per[0.0]["f1"], ~is_outfit),
            "unc": paired(per[w]["f1"], per[0.0]["f1"], uncov_out), "pull": paired(per[w]["pull"], per[0.0]["pull"], uncov_out)}
    L.append(f"| {name[w]} | {fmt(D[w]['l5'])} | {fmt(D[w]['l1'])} | {fmt(D[w]['f1'])} | {fmt(D[w]['fs'])} | {fmt(D[w]['sh'])} |")
L += ["", "## Wiki validation by kind of image, and the shortcut check", "",
      f"{int(uncov_out.sum())} validation images show an outfit and belong only to families that have no real photos "
      "(cute, lolita, historical, fantasy, club, regional, seasons, design). If real photos teach the network that "
      "\"a photographed outfit means one of the 12 covered families\", these images are where it shows.", "",
      "| model | outfit images: family top-1 | other images: family top-1 | uncovered-family outfit images: family top-1 | their first guess is a covered family |",
      "|---|---|---|---|---|"]
for w in WEIGHTS:
    pull = [100 * np.mean(is_cov[g][uncov_out]) for g in res[w]["wf"]]
    L.append(f"| {name[w]} | {ms(w, 'f1', is_outfit)} | {ms(w, 'f1', ~is_outfit)} | {ms(w, 'f1', uncov_out)} | {np.mean(pull):.1f} ± {np.std(pull):.1f} |")
for w in WEIGHTS[1:]:
    L.append(f"| {name[w]} minus weight 0 | {fmt(D[w]['out'])} | {fmt(D[w]['oth'])} | {fmt(D[w]['unc'])} | {fmt(D[w]['pull'])} |")
L += ["", "## How the rules read it (rule 7)", ""]
for w in WEIGHTS[1:]:
    helps = D[w]["fs"][1] > 0
    cost = D[w]["l5"][2] < 0 or D[w]["unc"][2] < 0
    verdict = ("helps on real photos without a cost on wiki: a candidate for adoption" if helps and not cost else
               "helps on real photos, with a cost on wiki: a trade-off" if helps else "no help on real photos: a negative result")
    L.append(f"- **{name[w]}:** helps on real photos: {'yes' if helps else 'no'}; costs on wiki: {'yes' if cost else 'no'}. {verdict}.")
L += ["", "## FashionStyle14 by style (family top-1, mean of 5 seeds)", "", "| style | family | photos | " + " | ".join(name[w] for w in WEIGHTS) + " |",
      "|---|---|---|" + "---|" * len(WEIGHTS)]
for st in sorted(set(fs_style)):
    k = fs_style == st
    L.append(f"| {st} | {fam_name[fs_map[st]]} | {int(k.sum())} | " + " | ".join(f"{100 * per[w]['fs'][k].mean():.1f}" for w in WEIGHTS) + " |")
L += ["", "## Held-out Street Fashion Style posters by family (family top-1, mean of 5 seeds)", "",
      "| family | photos | " + " | ".join(name[w] for w in WEIGHTS) + " |", "|---|---|" + "---|" * len(WEIGHTS)]
for f in covered:
    k = held_f == f
    L.append(f"| {fam_name[families[f]]} | {int(k.sum())} | " + " | ".join(f"{100 * per[w]['sh'][k].mean():.1f}" for w in WEIGHTS) + " |")
L += ["", "Self-check (weight 0 against the team table's saved predictions): " + "; ".join(check), "",
      f"Time elapsed: {time.perf_counter() - t0:.0f} s. torch {torch.__version__}, one CPU thread."]
(OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(OUT / "results.json").write_text(json.dumps({
    "seeds": SEEDS, "weights": WEIGHTS, "settings": S, "real_batch": REAL_BATCH, "sfs_training_photos": len(tr), "sfs_held_out_photos": len(he),
    "held_out_posters": len(posters), "fs14_photos": len(fs), "covered_families": [families[f] for f in covered],
    "scores": {str(w): {k: [float(100 * r.mean()) for r in res[w][k]] for k in ("l1", "l5", "f1", "fs", "sh")} for w in WEIGHTS},
    "differences_from_weight_0": {str(w): {k: list(v) for k, v in D[w].items()} for w in WEIGHTS[1:]}, "torch": torch.__version__}, indent=1), encoding="utf-8")
log("\n".join(L))
