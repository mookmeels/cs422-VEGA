"""The NN: a small neural network on the Qwen image vectors. Owner: Andrew Nguyen.

    python train_nn.py --seed 422 423 424 425 426     # fit on train, score on validation, one run per seed
    python train_nn.py --seed 422 --curve             # also write the per-epoch learning curve
    python train_nn.py --seed 422 423 424 425 426 --final    # the one final run: fit on train + validation, score on test
    python train_nn.py --seed 422 --real-weight 0     # the wiki-only network (the NN as it was until 21 Nov)

Since 21 Nov the NN also learns from real outfit photos (Andrew's decision; see "Real outfit photos" below).

What it is. Every image has already been turned into 2048 numbers by the frozen Qwen3-VL-Embedding-2B model
(the same vectors every VEGA model may use). This network takes those 2048 numbers and gives one score per
label (503 labels). It is a multilayer perceptron:

    2048 inputs -> scale each input -> [hidden layer: Linear, ReLU, Dropout] x --layers -> Linear -> 503 scores

With --layers 0 there is no hidden layer and the model is plain softmax (multinomial logistic) regression,
which is the reference the hidden layer has to beat.

How it learns.
  * Input scaling (mean and standard deviation of each of the 2048 inputs) is measured on the fit images only
    and then applied unchanged to the images being scored.
  * Loss: cross-entropy between the network's softmax and the image's label row. An image that sits in two
    galleries has half its weight on each label (the rows of y from vega_data sum to 1).
  * --balance: labels have very different numbers of images (half have 14 or fewer). Each image's loss is
    weighted by (images of its label) ** -balance, so 0 treats every image alike and 1 treats every label alike.
  * Optimiser: AdamW (Adam with weight decay). The learning rate rises linearly for the first tenth of the
    steps and then falls to zero along a cosine.
  * The number of epochs is a fixed setting chosen on validation. Nothing here stops early by looking at the
    eval images, because the final run has no labelled held-out images to look at.

Real outfit photos (--real-weight, default 0.25).
  * The wiki galleries are mostly moodboards; the app's input is a photo of a person's outfit. So besides the
    wiki training images the network sees Street Fashion Style photos (outfit posts, emb/street-fashion-style).
  * Such a photo has no aesthetic label. It has the style tag its poster typed, and that tag stands for one of
    the 20 style FAMILIES (frozen tables runs/family-v1/label_family.csv and outside_family.csv). So a real
    photo's loss is minus the log of the probability the network gives to its family IN TOTAL, summed over
    that family's aesthetics: it is told which family, never which aesthetic.
  * Each training step takes the usual batch of wiki images plus 512 real photos and adds real_weight x their
    mean loss. The wiki images, their order and their loss are exactly as without real photos.
  * How a real photo is drawn (--real-draw, default "tag"): a family that has photos, each equally often; then one
    of that family's TAGS, each equally often; then one of that tag's photos. So within a family a narrow tag
    (Formal, 1,300 photos) has the same say as a broad one. Until 21 Nov the draw was family -> photo
    (--real-draw photo), which let the largest tag supply most of a family's photos. Adopted by Andrew on 21 Nov
    together with leaving three broad tags out of the mapping (Elegant, Urban, Androgynous: Amendment 5 in
    runs/family-v1/FROZEN.md). Evidence: runs/tag-iter (chosen on one half of FashionStyle14's photos, then
    checked once on the other: +3.3 points of family accuracy, 95% interval +2.6 to +3.9, wiki scores unchanged).
  * Posters fall into 10 groups by name (sha1 mod 10). Group 0 is never trained on; its photos stay available
    for checking. real_photo_folds.py rotates which group is left out to score every photo once (10 folds).
    FashionStyle14 is never trained on.
  * 0.25 was chosen on validation data under rules written before the runs (runs/real-photos-v1/RULES.md and
    results.md): family accuracy on FashionStyle14 photos rose from 33.1% to 49.4% and top-5 on the wiki
    validation images did not fall beyond its margin (88.3 to 87.9). Weight 1.0 was the only other value tried.
  * --real-weight 0 gives the wiki-only network, the same to the last digit as the runs made before 21 Nov.
  * Limits to state with any result: 12 of the 20 families have real photos and 8 have none; the tags are the
    posters' own; and this network has training data the team's other models do not have.

The defaults below are the settings chosen on the validation split (see runs/nn/sweep-qwen2beos/summary.md;
runs/nn/sweep is the earlier sweep on the vectors the team used before 17 Nov). Each run saves its
predictions to <out>/preds/ and is scored by the shared scorer, which appends a row to <out>/results.jsonl.

Where it runs. The same file sits in /srv/ml/vega/code/nn on ml-db and in src/models/nn of the team repo. The
data and the scorer come from the team's shared code (vega_data.py, vega_score.py), looked for in ../shared,
then ../../common, then $VEGA_ROOT/code/shared (see shared_code below). The data itself is under $VEGA_ROOT
(default /srv/ml/vega).

Repeating a run. Torch seeds the weights, the dropout masks and the order of the images from --seed, and
training uses ONE CPU thread by default. With one thread, the same seed on the same machine and the same
package versions gives the same scores to the last digit, run after run. --threads 0 uses every core and is
faster, but then the first training in a new process sometimes rounds differently and ends a few tenths of a
point away (seen on ml-db on 17 Nov: seed 422 gave 87.0 or 86.7 top-5 with 2 threads). Another machine can
still differ in the last decimals, because its processor adds the same numbers in another order.
"""
import argparse, csv, hashlib, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def shared_code():
    """The folder holding the team's loader and scorer (vega_data.py, vega_score.py), wherever this file sits."""
    here = Path(__file__).resolve().parent
    places = [here.parent / "shared",                            # ml-db: /srv/ml/vega/code/nn, next to code/shared
              here.parent.parent / "common",                     # team repo: src/models/nn, shared code in src/common
              Path(os.environ.get("VEGA_ROOT", "/srv/ml/vega")) / "code" / "shared"]
    if os.environ.get("VEGA_SHARED"):                            # or say where it is
        places.insert(0, Path(os.environ["VEGA_SHARED"]))
    for p in places:
        if (p / "vega_data.py").exists():
            return p
    raise SystemExit("vega_data.py not found; set VEGA_SHARED to its folder. Looked in: " + ", ".join(str(p) for p in places))


SHARED = shared_code()
sys.path.insert(0, str(SHARED))
import vega_data                                                # noqa: E402
import vega_score                                               # noqa: E402

OWNER = "Andrew Nguyen"
# Chosen on validation, 17 Nov, on the corrected vectors (runs/nn/sweep-qwen2beos/summary.md). The search started
# from dropout 0.5, weight decay 0.01, balance 0.5; the three values below that differ from that were adopted
# together after a confirmation on five fresh seeds (stage 4 of sweep_nn.py).
# real_weight was chosen on 21 Nov between 0.25 and 1.0, the only two values tried (runs/real-photos-v1).
SETTINGS = {"layers": 1, "hidden": 1024, "dropout": 0.7, "input_dropout": 0.0, "weight_decay": 1.0, "lr": 1e-3,
            "epochs": 40, "batch": 512, "balance": 1.0, "prep": "standardize", "real_weight": 0.25, "real_draw": "tag"}
REAL_BATCH = 512                                                 # real photos added to every training step
REAL_GROUPS = 10                                                 # posters fall into 10 groups: sha1(name) % 10


def real_photo_table(root, classes):
    """Every Street Fashion Style photo whose style tag stands for a family.

    Returns rows = (family, row in the vectors, poster group, poster, photo id), sorted; the folder holding the
    vectors; each label's family; and the families' names. A poster's group is sha1(name) % 10, so all photos of
    one poster are in one group.
    """
    root = Path(root)
    fam_dir, sfs = root / "runs" / "family-v1", root / "emb" / "street-fashion-style"
    vec = sfs / f"{vega_data.VECTORS}-448"
    need = [fam_dir / "label_family.csv", fam_dir / "families.csv", fam_dir / "outside_family.csv",
            sfs / "manifest.csv", vec / "ids.txt", vec / "embeddings.npy"]
    if any(not p.exists() for p in need):
        raise SystemExit("The real outfit photos are part of this model and were not found:\n  "
                         + "\n  ".join(str(p) for p in need if not p.exists())
                         + "\nFor the wiki-only network, run with --real-weight 0.")

    def read(path):
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    families = [r["family"] for r in read(fam_dir / "families.csv")]
    fi = {f: i for i, f in enumerate(families)}
    lab = {r["wiki_label"]: r["family"] for r in read(fam_dir / "label_family.csv")}
    if any(c not in lab for c in classes):
        raise SystemExit("The family table covers the merged label set only; with other labels run with --real-weight 0.")
    lab_fam = np.asarray([fi[lab[c]] for c in classes])
    tag = {r["style_or_tag"]: fi[r["family"]] for r in read(fam_dir / "outside_family.csv")
           if r["source"] == "Street Fashion Style" and r["family"] in fi}
    row = {s: i for i, s in enumerate((vec / "ids.txt").read_text(encoding="utf-8").split())}
    rows = sorted((tag[r["style"]], row[r["id"]], int(hashlib.sha1(r["user"].encode("utf-8")).hexdigest(), 16) % REAL_GROUPS,
                   r["user"], r["id"]) for r in read(sfs / "manifest.csv") if r["style"] in tag and r["id"] in row)
    if not rows:
        raise SystemExit(f"No Street Fashion Style photo with a mapped style tag in {sfs}.")
    return rows, vec, lab_fam, families


def real_photos(root, classes, holdout=0, draw=None):
    """The Street Fashion Style photos the network trains on: their vectors, and what their loss needs.

    holdout: the poster group that is never trained on (0, as in the runs that chose the settings, unless a
    cross-validation fold says otherwise; see real_photo_folds.py).
    draw: "tag" (SETTINGS' default): family -> one of its tags -> one of the tag's photos. "photo": family -> photo.

    Returns x (photos x 2048) and the "slots" a training step draws from: train() picks a slot at random, then a
    photo at random among the slot's `sizes` photos starting at `starts`; `covered` is the slot's family. With
    "photo" there is one slot per family. With "tag" there is a slot per (family, tag), repeated so that every
    family has the same number of slots, split evenly over its tags. Also mask (families x labels: 0 where the
    label belongs to the family, -1e9 elsewhere) and photo_family (each photo's family, in x's order).
    """
    draw = draw or SETTINGS["real_draw"]
    assert draw in ("tag", "photo"), draw
    rows, vec, lab_fam, families = real_photo_table(root, classes)
    with open(Path(root) / "emb" / "street-fashion-style" / "manifest.csv", encoding="utf-8", newline="") as f:
        tag_of = {r["id"]: r["style"] for r in csv.DictReader(f)}
    keep = sorted((f, tag_of[pid] if draw == "tag" else "", r) for f, r, group, _, pid in rows if group != holdout)
    if not keep:
        raise SystemExit("No real photo is left to train on once that poster group is held out.")
    fam = np.asarray([f for f, _, _ in keep])
    slots, start = [], 0                                         # (family, first photo, how many): one per family and tag
    for k in range(1, len(keep) + 1):
        if k == len(keep) or keep[k][:2] != keep[start][:2]:
            slots.append((keep[start][0], start, k - start))
            start = k
    per_family = {f: sum(1 for s in slots if s[0] == f) for f in {s[0] for s in slots}}
    most = math.lcm(*per_family.values())
    slots = [s for s in slots for _ in range(most // per_family[s[0]])]       # every family: `most` slots, shared evenly by its tags
    mask = torch.full((len(families), len(classes)), -1e9)
    mask[torch.from_numpy(lab_fam), torch.arange(len(classes))] = 0.0
    return {"x": np.ascontiguousarray(np.load(vec / "embeddings.npy", mmap_mode="r")[[r for _, _, r in keep]]),
            "starts": torch.tensor([s[1] for s in slots]), "sizes": torch.tensor([s[2] for s in slots]),
            "covered": torch.tensor([s[0] for s in slots]), "mask": mask, "photo_family": fam,
            "photos": len(keep), "families": len(per_family), "tags": len({(f, t) for f, t, _ in keep}), "holdout": holdout, "draw": draw}


def input_scaling(fit_x, how):
    """Measure the scaling on the fit images; return a function that applies it to any images."""
    if how == "none":
        return lambda x: x.astype(np.float32)
    mean = fit_x.mean(0, keepdims=True)
    if how == "center":                                          # subtract the average image, back to length 1
        return lambda x: ((x - mean) / (np.linalg.norm(x - mean, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    sd = fit_x.std(0, keepdims=True) + 1e-6                      # "standardize": z-score each input
    return lambda x: ((x - mean) / sd).astype(np.float32)


def image_weights(fit_y, balance):
    """Loss weight of each fit image: (images of its label) ** -balance, averaged over its labels; mean 1."""
    per_label = np.maximum(fit_y.sum(0), 1.0) ** -float(balance)
    w = fit_y @ per_label
    return (w * len(w) / w.sum()).astype(np.float32)


def build(n_in, n_labels, s):
    parts, width = [], n_in
    if s["input_dropout"] > 0:
        parts.append(nn.Dropout(s["input_dropout"]))
    for _ in range(s["layers"]):
        parts += [nn.Linear(width, s["hidden"]), nn.ReLU(), nn.Dropout(s["dropout"])]
        width = s["hidden"]
    parts.append(nn.Linear(width, n_labels))
    return nn.Sequential(*parts)


def predict(net, x):
    net.eval()
    with torch.no_grad():
        return net(x).cpu().numpy()


def train(fit_x, fit_y, eval_x, s, seed, device="cpu", each_epoch=None, real=None):
    """Train one network on the fit images and return its scores for the eval images (rows x labels).

    Only fit_x and fit_y are learned from, plus the real outfit photos when `real` (from real_photos) is given
    and s["real_weight"] is above 0. Without `real` this is the wiki-only network, whatever s says.
    eval_x is scaled with the fit images' statistics and scored.
    each_epoch(epoch, mean training loss, eval scores) is called after every epoch if given (for --curve).
    """
    weight = float(s.get("real_weight", 0)) if real is not None else 0.0
    torch.manual_seed(seed)
    scale = input_scaling(fit_x, s["prep"])
    X = torch.from_numpy(scale(fit_x)).to(device)
    Y = torch.from_numpy(fit_y).to(device)
    E = torch.from_numpy(scale(eval_x)).to(device)
    W = torch.from_numpy(image_weights(fit_y, s["balance"])).to(device)
    net = build(X.shape[1], Y.shape[1], s).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    steps = s["epochs"] * math.ceil(len(X) / s["batch"])
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / warm if i < warm else
                                              0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, steps - warm))))
    shuffle = torch.Generator().manual_seed(seed)                # drawn on the CPU: same order on any device
    pick = torch.Generator().manual_seed(seed + 100_000)         # the real photos' own draw: the wiki order is untouched
    if weight > 0:
        mask = real["mask"].to(device)
    for epoch in range(1, s["epochs"] + 1):
        net.train()
        order, total = torch.randperm(len(X), generator=shuffle).to(device), 0.0
        for i in range(0, len(X), s["batch"]):
            idx = order[i:i + s["batch"]]
            if weight > 0:
                f = torch.randint(len(real["covered"]), (REAL_BATCH,), generator=pick)       # a slot (see real_photos), then one of its photos
                rows = real["starts"][f] + (torch.rand(REAL_BATCH, generator=pick) * real["sizes"][f]).long()
                xr = torch.from_numpy(scale(real["x"][rows.numpy()])).to(device)
                logp = F.log_softmax(net(torch.cat([X[idx], xr])), dim=1)
                loss_wiki = (W[idx] * -(Y[idx] * logp[:len(idx)]).sum(1)).mean()
                loss_real = -torch.logsumexp(logp[len(idx):] + mask[real["covered"][f]], dim=1).mean()   # log of the family's total share
                loss = loss_wiki + weight * loss_real
            else:
                loss = (W[idx] * -(Y[idx] * F.log_softmax(net(X[idx]), dim=1)).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if each_epoch:
            each_epoch(epoch, total / len(X), predict(net, E))
    return predict(net, E)


def model_name(s):
    real = " + real outfit photos" if s.get("real_weight", 0) > 0 else ""
    if s["layers"] == 0:
        return "NN (softmax regression, no hidden layer)" + real
    return f"NN (MLP, {s['layers']} hidden layer{'s' if s['layers'] > 1 else ''} of {s['hidden']})" + real


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, nargs="+", default=[422], help="one run per seed")
    ap.add_argument("--layers", type=int, default=SETTINGS["layers"], help="hidden layers; 0 = softmax regression")
    ap.add_argument("--hidden", type=int, default=SETTINGS["hidden"], help="units in each hidden layer")
    ap.add_argument("--dropout", type=float, default=SETTINGS["dropout"], help="share of hidden units switched off in training")
    ap.add_argument("--input-dropout", type=float, default=SETTINGS["input_dropout"], help="share of inputs switched off in training")
    ap.add_argument("--weight-decay", type=float, default=SETTINGS["weight_decay"])
    ap.add_argument("--lr", type=float, default=SETTINGS["lr"], help="peak learning rate")
    ap.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    ap.add_argument("--batch", type=int, default=SETTINGS["batch"])
    ap.add_argument("--balance", type=float, default=SETTINGS["balance"], help="0 = every image counts alike, 1 = every label counts alike")
    ap.add_argument("--prep", default=SETTINGS["prep"], choices=("standardize", "center", "none"), help="input scaling, measured on the fit images")
    ap.add_argument("--real-weight", type=float, default=SETTINGS["real_weight"],
                    help="weight of the real outfit photos' loss; 0 = wiki images only (the NN as it was until 21 Nov)")
    ap.add_argument("--real-draw", default=SETTINGS["real_draw"], choices=("tag", "photo"),
                    help="how a real photo is drawn: family -> tag -> photo (default) or family -> photo (as until 21 Nov)")
    ap.add_argument("--real-holdout", type=int, default=0, choices=range(REAL_GROUPS),
                    help="the poster group never trained on; 0 (default) is the group held out when the settings were chosen")
    ap.add_argument("--final", action="store_true", help="score the TEST split (only once everything is frozen)")
    ap.add_argument("--curve", action="store_true", help="write the per-epoch learning curve (validation runs only)")
    ap.add_argument("--labels", default="merged", choices=("merged", "wearable"))
    ap.add_argument("--name", default=None, help="model name for the table; default describes the network")
    ap.add_argument("--root", default=None, help="default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--out", default=None, help="default <root>/runs/comparison")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda; the network is small enough for a CPU")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads; 1 (default) repeats exactly, 0 = all cores: faster, not exactly repeatable")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    s = {k: getattr(a, k) for k in SETTINGS}
    name = a.name or model_name(s)
    root = Path(a.root or vega_data.ROOT)
    out = Path(a.out) if a.out else root / "runs" / "comparison"
    (out / "preds").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    d = vega_data.load(final=a.final, labels=a.labels, root=root)
    fit, ev = d["fit"], d["eval"]
    real = real_photos(root, d["classes"], a.real_holdout, s["real_draw"]) if s["real_weight"] > 0 else None
    print(f"{name}: fit {len(fit['ids'])} images, eval {len(ev['ids'])} ({d['eval_split']}), {len(d['classes'])} labels, "
          f"device {a.device}, {torch.get_num_threads()} thread(s)\nsettings: {json.dumps(s)}", flush=True)
    if real is not None:
        print(f"real outfit photos: {real['photos']} Street Fashion Style photos in {real['families']} families"
              + (f" and {real['tags']} tags, drawn family -> tag -> photo" if real["draw"] == "tag" else ", drawn family -> photo")
              + f", {REAL_BATCH} per step, weight {s['real_weight']}, poster group {real['holdout']} held out", flush=True)
    fit_count = (fit["y"] > 0).sum(0)
    for seed in a.seed:
        t1 = time.perf_counter()
        curve = []

        def each_epoch(epoch, train_loss, scores):
            m = vega_score.metrics(scores, ev["truth"], fit_count)
            log_p = scores - scores.max(1, keepdims=True)
            log_p = log_p - np.log(np.exp(log_p).sum(1, keepdims=True))
            curve.append((epoch, train_loss, float(-(ev["y"] * log_p).sum(1).mean()), m["top5"], m["top1"]))

        scores = train(fit["x"], fit["y"], ev["x"], s, seed, a.device, each_epoch if a.curve and not a.final else None, real=real)
        slug = ("linear" if s["layers"] == 0 else f"mlp{s['layers']}x{s['hidden']}") + ("real" if real is not None else "")
        pred = out / "preds" / f"nn-{slug}-{d['eval_split']}-{a.labels}-seed{seed}.npz"
        np.savez_compressed(pred, ids=ev["ids"], scores=scores.astype(np.float32), classes=np.asarray(d["classes"]))
        vega_score.score(pred, name, OWNER, seed=seed, final=a.final, labels=a.labels, root=root, out=out,
                         note=f"settings {json.dumps(s)}; device {a.device}; threads {torch.get_num_threads()}; torch {torch.__version__}"
                              + (f"; real photos {real['photos']} in {real['families']} families, drawn by {real['draw']}, {REAL_BATCH} per step, "
                                 f"poster group {real['holdout']} held out" if real is not None else ""))
        if curve:
            (out / "curves").mkdir(exist_ok=True)
            path = out / "curves" / f"nn-{slug}-{a.labels}-seed{seed}.csv"
            path.write_text("epoch,train_loss,validation_loss,validation_top5,validation_top1\n"
                            + "".join(f"{e},{tl:.4f},{vl:.4f},{t5:.4f},{t1_:.4f}\n" for e, tl, vl, t5, t1_ in curve), encoding="utf-8")
        print(f"  seed {seed}: trained and scored in {time.perf_counter() - t1:.0f}s", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
