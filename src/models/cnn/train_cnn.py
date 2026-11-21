"""The CNN: a ResNet trained from scratch on the wiki pictures' pixels. Owner: Giancarlo Guevarra.

    python prepare_images.py                                         # once: the pictures, at 128 and 224 px
    python train_cnn.py --width 32 --size 128 --seed 422 --curve     # one tuning run: half-width ResNet-18, 128 px
    python train_cnn.py --width 64 --size 224 --seed 422 --curve     # the full ResNet-18 at 224 px
    python train_cnn.py --width W --size S --seed 422 423 424 425 426 --curve    # five seeds of the chosen settings
    python train_cnn.py ... --final                                  # final run: fit on train + validation (see below)

What it is. A ResNet-18 layout (He, Zhang, Ren and Sun 2016), built and trained here from random starting
weights; nothing pretrained. Every setting below was decided by Gianni step by step (19 Nov 2026, CNN plan):

    picture (size x size, colours scaled by the fit pictures' channel mean and spread)
    -> start: 7 x 7 filters, stride 2, batch norm, ReLU, 3 x 3 max-pool stride 2      (picture now 1/4 wide)
    -> 4 stages of 2 residual blocks; filters W, 2W, 4W, 8W; each new stage halves the picture (stride 2)
       block: 3 x 3 filters, batch norm, ReLU, 3 x 3 filters, batch norm, + shortcut, ReLU
    -> global average pooling -> dropout -> one linear layer -> 503 scores
    W = 32 is "half width" (about 2.9 M weights); W = 64 is the standard ResNet-18 (about 11.4 M).

How it learns.
  * Starting weights: He (Kaiming) normal for the filters; the last batch norm of every block starts at zero, so
    each block begins as a plain pass-through (Goyal et al. 2017).
  * Augmentation, training only, a fresh draw every time a picture is used: left-right flip (half the time), a
    shift of up to --shift of the picture's width and height, and a zoom in or out of up to --zoom. Gaps are
    filled with zero after colour scaling, which is the average colour, the same as the padding bars.
    Scoring always sees the plain picture.
  * Loss: the NN's (Andrew's): cross-entropy between the softmax and the picture's label row (a picture in two
    galleries has half its weight on each), each picture weighted by (pictures of its label) ** -balance, where a
    picture in two galleries counts half to each label (the same count as Andrew's NN).
  * Optimiser: AdamW. The learning rate rises linearly for the first --warmup share of steps, then falls to zero
    along a cosine. Weight decay applies to filter and linear weights, not to batch-norm scales or biases.
  * A fixed number of epochs, chosen on validation; nothing stops early by looking at the scored pictures.
  * --precision bf16 or fp16 runs most arithmetic in half precision (faster on the graphics card); fp32 is the
    safe default until a short test shows half precision gives the same quality.

Where it runs. Training runs on Gianni's PC (Radeon RX 7900 XTX); --device auto picks the CUDA/ROCm device
with the most memory, which skips the processor's built-in graphics. --no-miopen switches off AMD's MIOpen
library (torch.backends.cudnn) so PyTorch uses its own GPU kernels: on Gianni's PC (20 Nov 2026) MIOpen could not
compile its batch-norm kernel at run time and the run failed, while PyTorch's own kernels train fine. The choice
is recorded in runs.jsonl, the scorer's note and the weights file. A run whose predictions, weights or curve
file already exists in --out stops before training (repeats go in a new --out folder, or use --overwrite).
Repeats: on a graphics card the same seed
gives nearly but not exactly the same scores (the card adds numbers in varying order); the README reports the
measured gap.

Outputs, in --out (default <root>/runs/cnn/<validation or final>):
    preds/<tag>-<split>-<labels>-seed<seed>.npz    ids, scores, classes: the team's prediction format
    results.jsonl                                  this machine's scores (validation only; the team table is
                                                   made on ml-db from the same .npz files)
    curves/<tag>-<labels>-seed<seed>.csv           per epoch: train loss, validation loss, top-5, top-1
    weights/<tag>-<split>-seed<seed>.pt            the trained weights and every setting (for the filter pictures,
                                                   and for scoring other pictures later)
    runs.jsonl                                     one line per run: settings, timings, memory, device, versions
--final fits on train + validation and saves predictions and weights but never scores them here: the test split
is scored once, on ml-db, with vega_score.py --final, after the team freezes every setting.
--final --fit-only trains on train + validation and saves only the weights, so the PC needs no test pictures;
predict_cnn.py then makes the test predictions on ml-db from those weights (Andrew's option A, 20 Nov 2026).
--street <folder> (Gianni, 20 Nov 2026) also trains a second output on Street Fashion Style photos' own style
tags (prepare_street.py): each step adds --street-batch photos drawn tag-evenly, their tag cross-entropy times
--street-weight (0.25, Andrew's NN weight, not tuned). The 503 outputs, their loss and every other setting are
as for the plain CNN, but one forward pass carries both the wiki pictures and the street photos, so every batch norm layer is fitted on that mix (about half street photos), and the saved running averages used at test time are the mix too (scan, 21 Nov 2026). Its "train loss" includes 0.25 x the tag loss, so it
is not comparable with the plain runs' train loss. Posters of the held-out group are never trained on and give the
tag accuracy on unseen posters.
The model is named "... + street photos" and its files carry "-street" in the tag.
"""
import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_shared import shared_code  # noqa: E402  (puts the shared code on the path; check_cnn.py uses it too)
import vega_data   # noqa: E402
import vega_score  # noqa: E402

OWNER = "Giancarlo Guevarra"
# Decided by Gianni, 19 Nov 2026 ("1a 2a 3a 4a 5a 6a 7a 8a 9a"); width and size are what tuning compares.
SETTINGS = {"width": 32, "size": 128, "dropout": 0.3, "epochs": 60, "batch": 128, "lr": 1e-3, "weight_decay": 0.05,
            "warmup": 0.1, "balance": 1.0, "shift": 0.1, "zoom": 0.1, "flip": True, "precision": "fp32"}
BLOCKS = (2, 2, 2, 2)                                            # ResNet-18: two residual blocks per stage


class Block(nn.Module):
    """Two 3 x 3 filter layers with a shortcut that adds the block's input to its output."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = (nn.Identity() if stride == 1 and cin == cout else
                         nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, width, n_labels, dropout, blocks=BLOCKS, n_tags=0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 7, 2, 3, bias=False), nn.BatchNorm2d(width),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        layers, cin = [], width
        for stage, n in enumerate(blocks):
            cout = width * 2 ** stage
            for j in range(n):
                layers.append(Block(cin, cout, 2 if stage > 0 and j == 0 else 1))
                cin = cout
        self.stages = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_labels))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for b in self.stages:
            nn.init.zeros_(b.bn2.weight)                         # each block starts as a pass-through
        if n_tags:                                               # street photos' own style tags (a second output);
            self.tag_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                          nn.Linear(cin, n_tags))  # made last: the shared part starts as without it

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))

    def both(self, x, n_main):
        """One pass over a joined batch: the first n_main pictures get the 503 scores, the rest the tag scores."""
        f = self.stages(self.stem(x))
        return self.head(f[:n_main]), self.tag_head(f[n_main:])


def build(ck):
    """The network saved in a weights file (with its street-tag output if it has one), weights loaded."""
    s = ck["settings"]
    net = ResNet(s["width"], len(ck["classes"]), s["dropout"], n_tags=len(ck.get("street_tags") or []))
    net.load_state_dict(ck["state_dict"])
    return net


def run_tag(s):
    return f"cnn-r18w{s['width']}-{s['size']}" + ("-street" if s.get("street_weight") else "")


class Street:
    """Street Fashion Style photos prepared by prepare_street.py, read from disk a batch at a time.

    part "train" = posters outside the held-out group (the only photos ever trained on); "held_out" = group 0.
    The bars around each photo are painted with the run's own average colour, then colours are scaled exactly
    as for the wiki pictures (the wiki training pictures' mean and spread).
    """

    def __init__(self, folder, size, part, pad_rgb, mean, std, device):
        folder = Path(folder)
        info = json.loads((folder / "street.json").read_text(encoding="utf-8"))
        if info["size"] != size:
            raise SystemExit(f"{folder} holds {info['size']} px street photos, the run needs {size} px")
        self.tags = (folder / "tags.txt").read_text(encoding="utf-8").split("\n")[:-1]
        rows, tag = [], []
        with open(folder / "photos.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (int(r["group"]) == info["held_out_group"]) == (part == "held_out"):
                    rows.append(int(r["row"]))
                    tag.append(int(r["tag_index"]))
        self.rows, self.tag = np.asarray(rows, np.int64), np.asarray(tag, np.int64)
        self.by_tag = [np.flatnonzero(self.tag == k) for k in range(len(self.tags))]
        self.drawable = np.asarray([k for k, idx in enumerate(self.by_tag) if len(idx)], np.int64)   # tags with photos
        self.arr = np.load(folder / f"street-{size}.npy", mmap_mode="r")
        self.boxes = torch.from_numpy(np.load(folder / f"boxes-{size}.npy")).to(device)
        self.size, self.device = size, device
        self.pad = torch.tensor(pad_rgb, dtype=torch.uint8, device=device)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def draw(self, n, rng):
        """n photos, tag-evenly: a tag uniformly at random, then one of its photos (so big tags cannot dominate)."""
        tags = self.drawable[rng.integers(0, len(self.drawable), n)]
        return np.asarray([self.by_tag[k][rng.integers(len(self.by_tag[k]))] for k in tags], np.int64)

    def batch(self, pos):
        """Photos at these positions: colour-scaled pictures on the device, and their tag numbers."""
        rows = self.rows[pos]
        order = np.argsort(rows)                                  # read the file in order
        u8 = np.empty((len(rows), self.size, self.size, 3), np.uint8)
        u8[order] = self.arr[rows[order]]
        x = torch.from_numpy(u8).to(self.device)
        b = self.boxes[torch.from_numpy(rows).to(self.device)]
        ar = torch.arange(self.size, device=self.device)
        ys = (ar[None, :] >= b[:, 1:2]) & (ar[None, :] < b[:, 1:2] + b[:, 3:4])
        xs = (ar[None, :] >= b[:, 0:1]) & (ar[None, :] < b[:, 0:1] + b[:, 2:3])
        x[~(ys[:, :, None] & xs[:, None, :])] = self.pad          # the bars: the run's average colour
        x = x.permute(0, 3, 1, 2).float().div_(255)
        return (x - self.mean) / self.std, torch.from_numpy(self.tag[pos]).to(self.device)


def street_accuracy(net, street, device, precision, batch=256):
    """Tag top-1 on these photos, overall and averaged over tags (every tag counts equally)."""
    net.eval()
    hits = []
    with torch.no_grad(), autocast(device, precision):
        for i in range(0, len(street), batch):
            pos = np.arange(i, min(i + batch, len(street)))
            x, y = street.batch(pos)
            f = net.stages(net.stem(x))
            hits.append((net.tag_head(f).float().argmax(1) == y).cpu().numpy())
    hits = np.concatenate(hits)
    per_tag = [hits[idx].mean() for idx in street.by_tag if len(idx)]
    return {"photos": int(len(hits)), "tag_top1": round(float(hits.mean()), 4),
            "tag_top1_per_tag_mean": round(float(np.mean(per_tag)), 4), "tags": len(street.tags)}


def augment(x, gen, shift, zoom, flip):
    """Random flip, shift and zoom of a batch (already colour-scaled), one fresh draw per picture."""
    n = x.shape[0]
    r = torch.rand(n, 4, generator=gen, device=x.device)
    scale = 1 + zoom * (2 * r[:, 0] - 1)                         # above 1 shows more (zoom out), below 1 zooms in
    mirror = torch.where(r[:, 3] < 0.5, -1.0, 1.0) if flip else torch.ones(n, device=x.device)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = scale * mirror
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = 2 * shift * (2 * r[:, 1] - 1)               # the picture spans 2 units, so 2 * shift
    theta[:, 1, 2] = 2 * shift * (2 * r[:, 2] - 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def image_weights(fit_y, balance):
    """Loss weight of each fit picture: (pictures of its label) ** -balance, averaged over its labels; mean 1.

    A label's picture count sums the loader's label rows, so a picture in two galleries counts half to each (the
    same as Andrew's NN, train_nn.py image_weights).
    """
    per_label = np.maximum(fit_y.sum(0), 1.0) ** -float(balance)
    w = fit_y @ per_label
    return (w * len(w) / w.sum()).astype(np.float32)


def pick_device(want):
    if want != "auto":
        return torch.device(want)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    return torch.device(f"cuda:{best}")


def model_name(s):
    w = {64: "", 32: ", half width"}.get(s["width"], f", width {s['width']}")
    return f"CNN (ResNet-18{w}, {s['size']} px)"


class Pictures:
    """Prepared pictures of one part (fit or eval), held on the device as 8-bit pixels, scaled on the way out."""

    def __init__(self, folder, part, size, want_ids, mean, std, device):
        ids = (folder / f"{part}-ids.txt").read_text(encoding="utf-8").split()
        pos = {sha: i for i, sha in enumerate(ids)}
        missing = [sha for sha in want_ids if sha not in pos]
        if missing:
            raise SystemExit(f"{part}: {len(missing)} pictures the loader expects are not prepared (first: {missing[0]}); "
                             "rerun prepare_images.py")
        rows = np.asarray([pos[sha] for sha in want_ids])
        arr = np.load(folder / f"{part}-{size}.npy", mmap_mode="r")
        self.u8 = torch.from_numpy(np.ascontiguousarray(arr[rows])).to(device)    # n x S x S x 3, uint8
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.u8)

    def batch(self, idx):
        x = self.u8[idx].permute(0, 3, 1, 2).float().div_(255)
        return (x - self.mean) / self.std


def autocast(device, precision):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    return torch.autocast(device_type=device.type, dtype=dtype or torch.bfloat16, enabled=dtype is not None)


def predict(net, pics, device, precision, batch=256):
    net.eval()
    out = []
    with torch.no_grad(), autocast(device, precision):
        for i in range(0, len(pics), batch):
            out.append(net(pics.batch(torch.arange(i, min(i + batch, len(pics)), device=device))).float().cpu())
    return torch.cat(out).numpy()


def train(fit, fit_y, ev, s, seed, device, each_epoch=None, street=None):
    """Train one network on the fit pictures; return (eval scores, network, timing). Only fit is learned from."""
    torch.manual_seed(seed)
    street = street if street is not None and s.get("street_weight", 0) > 0 else None   # weight 0 = the plain CNN
    net = ResNet(s["width"], fit_y.shape[1], s["dropout"], n_tags=len(street.tags) if street else 0).to(device)
    decay = [p for p in net.parameters() if p.ndim > 1]
    no_decay = [p for p in net.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": s["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=s["lr"])
    per_epoch = math.ceil(len(fit) / s["batch"])
    steps = s["epochs"] * per_epoch
    warm = max(1, int(steps * s["warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / warm if i < warm else
                                              0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler(device.type, enabled=s["precision"] == "fp16")
    Y = torch.from_numpy(fit_y).to(device)
    W = torch.from_numpy(image_weights(fit_y, s["balance"])).to(device)
    shuffle = torch.Generator().manual_seed(seed)                # picture order drawn on the CPU: same on any device
    aug = torch.Generator(device=device).manual_seed(seed + 1)
    if street:                                                   # its own generators: the wiki draws stay the same
        street_rng = np.random.default_rng(seed + 2)
        street_aug = torch.Generator(device=device).manual_seed(seed + 3)
    train_seconds = 0.0
    for epoch in range(1, s["epochs"] + 1):
        net.train()
        t0 = time.perf_counter()
        order, total = torch.randperm(len(fit), generator=shuffle).to(device), 0.0
        for i in range(0, len(fit), s["batch"]):
            idx = order[i:i + s["batch"]]
            x = augment(fit.batch(idx), aug, s["shift"], s["zoom"], s["flip"])
            if street:
                xs, ys = street.batch(street.draw(s["street_batch"], street_rng))
                xs = augment(xs, street_aug, s["shift"], s["zoom"], s["flip"])
                with autocast(device, s["precision"]):
                    logits, tag_logits = net.both(torch.cat([x, xs]), len(idx))
            else:
                with autocast(device, s["precision"]):
                    logits = net(x)
            loss = (W[idx] * -(Y[idx] * F.log_softmax(logits.float(), dim=1)).sum(1)).mean()
            if street:
                loss = loss + s["street_weight"] * F.cross_entropy(tag_logits.float(), ys)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - t0
        if not math.isfinite(total):
            raise SystemExit(f"training loss is not a number at epoch {epoch}; try --precision fp32")
        if each_epoch:
            each_epoch(epoch, total / len(fit), predict(net, ev, device, s["precision"]))
        else:
            print(f"    epoch {epoch}/{s['epochs']}: train loss {total / len(fit):.4f}", flush=True)
    timing = {"train_seconds": round(train_seconds, 1),
              "train_pictures_per_second": round(s["epochs"] * len(fit) / max(train_seconds, 1e-9), 1)}
    return (None if ev is None else predict(net, ev, device, s["precision"])), net, timing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, nargs="+", default=[422], help="one run per seed")
    ap.add_argument("--width", type=int, default=SETTINGS["width"], help="filters in the first stage: 32 half width, 64 standard")
    ap.add_argument("--size", type=int, default=SETTINGS["size"], help="picture size in pixels (128 or 224, as prepared)")
    ap.add_argument("--dropout", type=float, default=SETTINGS["dropout"], help="share of features switched off before the last layer")
    ap.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    ap.add_argument("--batch", type=int, default=SETTINGS["batch"])
    ap.add_argument("--lr", type=float, default=SETTINGS["lr"], help="peak learning rate")
    ap.add_argument("--weight-decay", type=float, default=SETTINGS["weight_decay"])
    ap.add_argument("--warmup", type=float, default=SETTINGS["warmup"], help="share of steps with a rising learning rate")
    ap.add_argument("--balance", type=float, default=SETTINGS["balance"], help="0 = every picture alike, 1 = every label alike")
    ap.add_argument("--shift", type=float, default=SETTINGS["shift"], help="largest random shift, as a share of the picture")
    ap.add_argument("--zoom", type=float, default=SETTINGS["zoom"], help="largest random zoom in or out, as a share")
    ap.add_argument("--no-flip", dest="flip", action="store_false", help="turn off the random left-right flip")
    ap.add_argument("--precision", default=SETTINGS["precision"], choices=("fp32", "bf16", "fp16"))
    ap.add_argument("--final", action="store_true", help="fit on train + validation; predict test; never scored here")
    ap.add_argument("--curve", action="store_true", help="score validation after every epoch and save the learning curve")
    ap.add_argument("--labels", default="merged", choices=("merged", "wearable"))
    ap.add_argument("--name", default=None, help="model name for the table; default describes the network")
    ap.add_argument("--root", default=None, help="project folder; default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--prepared", default=None, help="prepared pictures; default <root>/runs/cnn/prepared/<mode>")
    ap.add_argument("--subset", action="store_true", help="allow a partly prepared folder (a quick check; not scored)")
    ap.add_argument("--out", default=None, help="default <root>/runs/cnn/<validation or final>")
    ap.add_argument("--device", default="auto", help="auto (the largest graphics card, else the CPU), cpu, cuda:0, ...")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads when running on the CPU (0 = torch's default)")
    ap.add_argument("--no-miopen", dest="miopen", action="store_false",
                    help="switch off MIOpen/cuDNN (torch.backends.cudnn) and use PyTorch's own GPU kernels")
    ap.add_argument("--street", default=None,
                    help="folder made by prepare_street.py: also train a second output on the street photos' own style "
                         "tags (Gianni, 20 Nov 2026); the 503 wiki outputs and their loss are as before, but the "
                         "network's batch norm now sees wiki pictures and street photos together (see README)")
    ap.add_argument("--street-weight", type=float, default=None, help="weight of the street photos' loss (with --street; "
                                                                      "default 0.25)")
    ap.add_argument("--street-batch", type=int, default=None, help="street photos per step, drawn tag-evenly (with --street; "
                                                                   "default 128)")
    ap.add_argument("--fit-only", action="store_true",
                    help="with --final: train on train + validation and save the weights only; needs no test pictures "
                         "(the test split is then predicted on ml-db with predict_cnn.py)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an earlier run's predictions, weights and curve (same setting and seed); "
                         "without it the run stops before training. Better: a new --out folder")
    a = ap.parse_args()
    if a.fit_only and not a.final:
        raise SystemExit("--fit-only is for the final run (--final): validation runs are always scored")
    if a.final and a.curve:
        raise SystemExit("--final never scores the test split during training, so there is no learning curve (drop --curve)")
    if len(set(a.seed)) != len(a.seed):
        raise SystemExit(f"a seed is listed twice ({' '.join(map(str, a.seed))}); its second run would replace the first's files")
    if not a.street and (a.street_weight is not None or a.street_batch is not None):
        raise SystemExit("--street-weight and --street-batch need --street <folder>; without it the plain network is trained")
    a.street_weight = 0.25 if a.street_weight is None else a.street_weight
    a.street_batch = 128 if a.street_batch is None else a.street_batch
    if a.street_weight < 0 or a.street_batch < 1:
        raise SystemExit("--street-weight must be 0 or more and --street-batch at least 1")
    s = {k: getattr(a, k) for k in SETTINGS}
    if not a.miopen:
        torch.backends.cudnn.enabled = False                     # before any work on the graphics card
    if a.threads:
        torch.set_num_threads(a.threads)
    device = pick_device(a.device)
    root = Path(a.root) if a.root else vega_data.ROOT
    mode = "final" if a.final else "validation"
    prepared = Path(a.prepared) if a.prepared else root / "runs" / "cnn" / "prepared" / mode
    out = Path(a.out) if a.out else root / "runs" / "cnn" / mode
    for sub in ("preds", "curves", "weights"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    stats = json.loads((prepared / "stats.json").read_text(encoding="utf-8"))
    if str(a.size) not in stats["per_size"]:
        raise SystemExit(f"size {a.size} is not prepared in {prepared} (prepared: {stats['sizes']})")
    if stats["split_sha256"] != vega_data.split_sha256(root):
        raise SystemExit("the prepared pictures come from another split file; rerun prepare_images.py")
    if not str(stats.get("mode", mode)).startswith(mode):
        raise SystemExit(f"{prepared} was prepared for a {stats.get('mode')} run, not a {mode} run; "
                         f"point --prepared at the {mode} preparation" + (" or drop --final" if a.final else " or add --final"))
    st = stats["per_size"][str(a.size)]

    d = vega_data.load(final=a.final, labels=a.labels, root=root, vectors=False)
    fit_d, ev_d = d["fit"], d["eval"]
    if a.subset:                                                 # a quick check on a partly prepared folder
        keep = {p: set((prepared / f"{p}-ids.txt").read_text(encoding="utf-8").split()) for p in ("fit", "eval")}
        for p, part in (("fit", fit_d), ("eval", ev_d)):
            m = np.isin(part["ids"], list(keep[p]))
            for k in ("ids", "split", "y"):
                if k in part:
                    part[k] = part[k][m]
            for k in ("paths", "truth"):
                if k in part:
                    part[k] = [v for v, ok in zip(part[k], m) if ok]
    street = held = None
    if a.street:
        s.update(street_weight=a.street_weight, street_batch=a.street_batch)
        street = Street(a.street, a.size, "train", st["pad_rgb"], st["mean"], st["std"], device)
        held = Street(a.street, a.size, "held_out", st["pad_rgb"], st["mean"], st["std"], device)
        s["street_tags"] = len(street.tags)
        print(f"street photos: {len(street)} for training, {len(held)} held out (unseen posters), {len(street.tags)} tags, "
              f"{a.street_batch} per step drawn tag-evenly, weight {a.street_weight}", flush=True)
    name = a.name or (model_name(s) + (" + street photos" if s.get("street_weight") else ""))
    tag = run_tag(s)
    taken = [p for seed in a.seed for p in (out / "preds" / f"{tag}-{d['eval_split']}-{a.labels}-seed{seed}.npz",
                                            out / "weights" / f"{tag}-{d['eval_split']}-seed{seed}.pt",
                                            out / "curves" / f"{tag}-{a.labels}-seed{seed}.csv") if p.exists()]
    if taken and not a.overwrite:                                # 20 Nov 2026: a repeat once replaced a tuning run's files
        raise SystemExit("an earlier run's files would be replaced:\n  " + "\n  ".join(map(str, taken))
                         + "\nuse a new --out folder, or --overwrite to replace them on purpose")
    stale = [p for p in taken if a.fit_only and p.parent.name == "preds"]
    if stale:                                                    # a fit-only run writes no predictions to replace them
        raise SystemExit("--fit-only would leave an older run's test predictions beside the new weights:\n  "
                         + "\n  ".join(map(str, stale)) + "\nuse a new --out folder, or move those files away first")
    t_load = time.perf_counter()
    fit = Pictures(prepared, "fit", a.size, list(fit_d["ids"]), st["mean"], st["std"], device)
    ev = None if a.fit_only else Pictures(prepared, "eval", a.size, list(ev_d["ids"]), st["mean"], st["std"], device)
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu"
    print(f"{name}: fit {len(fit)} pictures, eval {'none (--fit-only)' if ev is None else len(ev)} ({d['eval_split']}), {len(d['classes'])} labels, "
          f"device {device} ({dev_name}), pictures loaded in {time.perf_counter() - t_load:.1f}s\n"
          f"settings: {json.dumps(s)}; MIOpen/cuDNN {'on' if a.miopen else 'off'}", flush=True)
    fit_count = (fit_d["y"] > 0).sum(0)

    for seed in a.seed:
        t1 = time.perf_counter()
        curve = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def each_epoch(epoch, train_loss, scores):
            m = vega_score.metrics(scores, ev_d["truth"], fit_count)
            log_p = scores - scores.max(1, keepdims=True)
            log_p = log_p - np.log(np.exp(log_p).sum(1, keepdims=True))
            val_loss = float(-(ev_d["y"] * log_p).sum(1).mean())
            curve.append((epoch, train_loss, val_loss, m["top5"], m["top1"]))
            print(f"    epoch {epoch}/{s['epochs']}: train loss {train_loss:.4f}, validation loss {val_loss:.4f}, "
                  f"top-5 {100 * m['top5']:.1f}, top-1 {100 * m['top1']:.1f}", flush=True)

        scores, net, timing = train(fit, fit_d["y"], ev, s, seed, device,
                                    each_epoch if a.curve and not a.final else None, street=street)
        pred = out / "preds" / f"{tag}-{d['eval_split']}-{a.labels}-seed{seed}.npz"
        if scores is not None:
            np.savez_compressed(pred, ids=ev_d["ids"], scores=scores.astype(np.float32), classes=np.asarray(d["classes"]))
        torch.save({"state_dict": net.state_dict(), "settings": s, "seed": seed, "labels": a.labels,
                    "classes": list(d["classes"]), "colour_mean": st["mean"], "colour_std": st["std"],
                    "pad_rgb": st["pad_rgb"], "split_sha256": stats["split_sha256"], "final": a.final,
                    "miopen": a.miopen, "fit_only": a.fit_only,
                    "street_tags": street.tags if street is not None and s.get("street_weight") else None},
                   out / "weights" / f"{tag}-{d['eval_split']}-seed{seed}.pt")
        street_result = (street_accuracy(net, held, device, s["precision"])
                         if street is not None and s.get("street_weight") else None)
        if street_result:
            print(f"  seed {seed}: street tags on held-out posters: top-1 {100 * street_result['tag_top1']:.1f}%, "
                  f"per-tag mean {100 * street_result['tag_top1_per_tag_mean']:.1f}% "
                  f"({street_result['tags']} tags, chance {100 / street_result['tags']:.1f}%)", flush=True)
        note = (f"settings {json.dumps(s)}; device {dev_name}; torch {torch.__version__}; "
                f"miopen {'on' if a.miopen else 'off'}")
        result = None
        if a.fit_only:
            print(f"  seed {seed}: weights saved, no predictions (--fit-only); predict the test split on ml-db "
                  "with predict_cnn.py", flush=True)
        elif a.final:
            print(f"  seed {seed}: test predictions saved, NOT scored ({pred.name}); score once on ml-db with "
                  "vega_score.py --final after the team freezes settings", flush=True)
        elif a.subset:
            m = vega_score.metrics(scores, ev_d["truth"], fit_count)
            result = {k: round(float(v), 4) for k, v in m.items() if isinstance(v, (int, float))}
            print(f"  seed {seed} (subset, not scored by the team scorer): top-5 {100 * m['top5']:.1f}", flush=True)
        else:
            vega_score.score(pred, name, OWNER, seed=seed, final=False, labels=a.labels, root=root, out=out, note=note)
        if curve:
            (out / "curves" / f"{tag}-{a.labels}-seed{seed}.csv").write_text(
                "epoch,train_loss,validation_loss,validation_top5,validation_top1\n"
                + "".join(f"{e},{tl:.4f},{vl:.4f},{t5:.4f},{t1_:.4f}\n" for e, tl, vl, t5, t1_ in curve),
                encoding="utf-8")
        record = {"made": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "model": name, "tag": tag, "seed": seed,
                  "split": d["eval_split"], "labels": a.labels, "settings": s, "subset": a.subset,
                  "miopen": a.miopen, "fit_only": a.fit_only, "street_held_out": street_result, **timing,
                  "total_seconds": round(time.perf_counter() - t1, 1),
                  "weights": sum(p.numel() for p in net.parameters()),
                  "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if device.type == "cuda" else None,
                  "device": dev_name, "torch": torch.__version__, "python": platform.python_version(),
                  "machine": platform.node(), "result": result, "last_epoch": curve[-1] if curve else None}
        with open(out / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  seed {seed}: {record['total_seconds']:.0f}s in all, {timing['train_pictures_per_second']:.0f} "
              f"training pictures per second", flush=True)


if __name__ == "__main__":
    main()
