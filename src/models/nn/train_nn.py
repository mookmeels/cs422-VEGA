"""The NN: a small neural network on the Qwen image vectors. Owner: Andrew Nguyen.

    python train_nn.py --seed 422 423 424 425 426     # fit on train, score on validation, one run per seed
    python train_nn.py --seed 422 --curve             # also write the per-epoch learning curve
    python train_nn.py --seed 422 423 424 425 426 --final    # the one final run: fit on train + validation, score on test

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
import argparse, json, math, os, sys, time
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
SETTINGS = {"layers": 1, "hidden": 1024, "dropout": 0.7, "input_dropout": 0.0, "weight_decay": 1.0, "lr": 1e-3,
            "epochs": 40, "batch": 512, "balance": 1.0, "prep": "standardize"}


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


def train(fit_x, fit_y, eval_x, s, seed, device="cpu", each_epoch=None):
    """Train one network on the fit images and return its scores for the eval images (rows x labels).

    Only fit_x and fit_y are learned from. eval_x is scaled with the fit images' statistics and scored.
    each_epoch(epoch, mean training loss, eval scores) is called after every epoch if given (for --curve).
    """
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
    for epoch in range(1, s["epochs"] + 1):
        net.train()
        order, total = torch.randperm(len(X), generator=shuffle).to(device), 0.0
        for i in range(0, len(X), s["batch"]):
            idx = order[i:i + s["batch"]]
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
    if s["layers"] == 0:
        return "NN (softmax regression, no hidden layer)"
    return f"NN (MLP, {s['layers']} hidden layer{'s' if s['layers'] > 1 else ''} of {s['hidden']})"


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
    print(f"{name}: fit {len(fit['ids'])} images, eval {len(ev['ids'])} ({d['eval_split']}), {len(d['classes'])} labels, "
          f"device {a.device}, {torch.get_num_threads()} thread(s)\nsettings: {json.dumps(s)}", flush=True)
    fit_count = (fit["y"] > 0).sum(0)
    for seed in a.seed:
        t1 = time.perf_counter()
        curve = []

        def each_epoch(epoch, train_loss, scores):
            m = vega_score.metrics(scores, ev["truth"], fit_count)
            log_p = scores - scores.max(1, keepdims=True)
            log_p = log_p - np.log(np.exp(log_p).sum(1, keepdims=True))
            curve.append((epoch, train_loss, float(-(ev["y"] * log_p).sum(1).mean()), m["top5"], m["top1"]))

        scores = train(fit["x"], fit["y"], ev["x"], s, seed, a.device, each_epoch if a.curve and not a.final else None)
        slug = "linear" if s["layers"] == 0 else f"mlp{s['layers']}x{s['hidden']}"
        pred = out / "preds" / f"nn-{slug}-{d['eval_split']}-{a.labels}-seed{seed}.npz"
        np.savez_compressed(pred, ids=ev["ids"], scores=scores.astype(np.float32), classes=np.asarray(d["classes"]))
        vega_score.score(pred, name, OWNER, seed=seed, final=a.final, labels=a.labels, root=root, out=out,
                         note=f"settings {json.dumps(s)}; device {a.device}; threads {torch.get_num_threads()}; torch {torch.__version__}")
        if curve:
            (out / "curves").mkdir(exist_ok=True)
            path = out / "curves" / f"nn-{slug}-{a.labels}-seed{seed}.csv"
            path.write_text("epoch,train_loss,validation_loss,validation_top5,validation_top1\n"
                            + "".join(f"{e},{tl:.4f},{vl:.4f},{t5:.4f},{t1_:.4f}\n" for e, tl, vl, t5, t1_ in curve), encoding="utf-8")
        print(f"  seed {seed}: trained and scored in {time.perf_counter() - t1:.0f}s", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
