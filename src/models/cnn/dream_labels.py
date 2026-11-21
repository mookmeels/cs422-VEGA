"""Dreamed-up pictures of what the trained CNN thinks each aesthetic looks like. Owner: Giancarlo Guevarra.

    python dream_labels.py --weights <...>/weights/cnn-r18w64-224-validation-seed422.pt --no-miopen

For every label (503 merged aesthetics), start from faint random noise and nudge the picture, step by step, to make
the network's score for that label as high as possible, the network's weights fixed. The score is the label's raw
output before softmax (Simonyan, Vedaldi and Zisserman 2013, arXiv 1312.6034, section 2: "an image, which maximises
the class score"). Same method and settings as dream_filters.py (Olah, Mordvintsev and Schubert 2017, Distill):
a Fourier-basis picture with low frequencies weighted up, sigmoid colours, a random shift and zoom every step,
Adam, a fixed seed, the network in evaluation mode. Gianni, 20 Nov 2026: every label, split across pages.

Writes, in --out (default <root>/runs/cnn/filters/<weights file name>-label-dreams):
    labels_page01.png ...  32 labels per page (default), in label order; each tile captioned with the label's name and
                           its number of training pictures (half the labels have 14 or fewer, so expect those to
                           be weaker)
    label_dreams.json      settings, and per label its training-picture count and its final score
A network trained on 9,359 pictures will likely show palettes and textures rather than recognisable outfits.
"""
import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_cnn                                                  # noqa: E402  (also puts the shared code on the path)
import vega_data                                                  # noqa: E402


def dream(net, labels, size, mean, std, a, device):
    """Optimise one picture per label to maximise that label's score (before softmax)."""
    n = len(labels)
    gen_cpu = torch.Generator().manual_seed(a.seed)
    gen = torch.Generator(device=device).manual_seed(a.seed)
    fy = torch.fft.fftfreq(size, device=device)[:, None]
    fx = torch.fft.rfftfreq(size, device=device)[None, :]
    scale = size / torch.sqrt(fx ** 2 + fy ** 2).clamp(min=1.0 / size)      # low frequencies weighted up
    spec = (0.01 * torch.randn(n, 3, size, size // 2 + 1, 2, generator=gen_cpu)).to(device).requires_grad_(True)
    m = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    pick = torch.tensor(labels, device=device)
    rows = torch.arange(n, device=device)

    def render():
        img = torch.fft.irfft2(torch.view_as_complex(spec) * scale, s=(size, size)) / 4
        return torch.sigmoid(img)                                 # n x 3 x S x S, colours in 0..1

    def score(rgb, jitter):
        x = (rgb - m) / s
        if jitter:
            x = train_cnn.augment(x, gen, a.shift, a.zoom, flip=False)
        return net(x)[rows, pick]                                 # each picture's own label

    opt = torch.optim.Adam([spec], lr=a.lr)
    for _ in range(a.steps):
        opt.zero_grad()
        (-score(render(), True).sum()).backward()
        opt.step()
    with torch.no_grad():
        rgb = render()
        final = score(rgb, False)
    return (rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8), final.cpu().numpy()


def page(pictures, captions, title, path, tile=160, cols=8):
    rows = (len(captions) + cols - 1) // cols
    cap_h = 44
    img = Image.new("RGB", (cols * (tile + 8) + 8, rows * (tile + cap_h) + 34), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((8, 8), title, fill="black")
    for i, (cap, p) in enumerate(zip(captions, pictures)):
        x, y = 8 + (i % cols) * (tile + 8), 34 + (i // cols) * (tile + cap_h)
        img.paste(Image.fromarray(p).resize((tile, tile), Image.BILINEAR), (x, y))
        d.multiline_text((x, y + tile + 3), cap, fill="black", spacing=1)
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="a .pt file saved by train_cnn.py")
    ap.add_argument("--root", default=None, help="project folder; default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--out", default=None, help="default <root>/runs/cnn/filters/<weights file name>-label-dreams")
    ap.add_argument("--per-page", type=int, default=32)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--shift", type=float, default=0.05, help="largest random shift per step, as a share of the picture")
    ap.add_argument("--zoom", type=float, default=0.05, help="largest random zoom in or out per step")
    ap.add_argument("--size", type=int, default=None, help="picture size; default the size the network was trained at")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N labels (a quick check)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-miopen", dest="miopen", action="store_false",
                    help="switch off MIOpen/cuDNN, as for training on Gianni's PC (see train_cnn.py)")
    a = ap.parse_args()
    if not a.miopen:
        torch.backends.cudnn.enabled = False
    root = Path(a.root) if a.root else vega_data.ROOT
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    st = ck["settings"]
    size = a.size or st["size"]
    classes = list(ck["classes"])
    d = vega_data.load(labels=ck["labels"], root=root, vectors=False)
    if list(d["classes"]) != classes:
        raise SystemExit("the loader's label list differs from the one saved with the weights")
    counts = (d["fit"]["y"] > 0).sum(0)                           # training pictures per label
    device = train_cnn.pick_device(a.device)
    net = train_cnn.build(ck)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    out = Path(a.out) if a.out else root / "runs" / "cnn" / "filters" / (Path(a.weights).stem + "-label-dreams")
    out.mkdir(parents=True, exist_ok=True)

    todo = list(range(len(classes)))[: a.limit]
    pages = [todo[i:i + a.per_page] for i in range(0, len(todo), a.per_page)]
    record = {"weights": str(a.weights), "settings": st, "seed_of_weights": ck["seed"], "labels": ck["labels"],
              "dream": {"steps": a.steps, "lr": a.lr, "shift": a.shift, "zoom": a.zoom, "size": size, "seed": a.seed,
                        "miopen": a.miopen, "per_page": a.per_page,
                        "method": "class-score maximisation (score before softmax), Fourier parametrisation, sigmoid "
                                  "colours, random shift and zoom each step, Adam"},
              "labels_dreamed": []}
    t0 = time.perf_counter()
    for p, labels in enumerate(pages, 1):
        t1 = time.perf_counter()
        pictures, final = dream(net, labels, size, ck["colour_mean"], ck["colour_std"], a, device)
        captions = ["\n".join(textwrap.wrap(classes[k], 26)[:2]) + f"\n({int(counts[k])} training pictures)"
                    for k in labels]
        title = (f"What the CNN thinks each aesthetic looks like: labels {labels[0] + 1} to {labels[-1] + 1} of "
                 f"{len(classes)}, page {p} of {len(pages)} (dreamed up by maximising each label's score, {a.steps} "
                 f"steps; {Path(a.weights).stem})")
        page(pictures, captions, title, out / f"labels_page{p:02d}.png")
        record["labels_dreamed"] += [{"label": classes[k], "index": k, "page": p, "training_pictures": int(counts[k]),
                                      "final_score": round(float(v), 4)} for k, v in zip(labels, final)]
        print(f"page {p}/{len(pages)}: {len(labels)} labels in {time.perf_counter() - t1:.0f}s", flush=True)
    record["seconds"] = round(time.perf_counter() - t0, 1)
    (out / "label_dreams.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
