"""Dreamed-up pictures of what a trained CNN's filters look for (activation maximisation). Owner: Giancarlo Guevarra.

    python dream_filters.py --weights <...>/weights/cnn-r18w64-224-validation-seed422.pt \
        --chosen <...>/filters/cnn-r18w64-224-validation-seed422-32-per-stage/chosen.json --no-miopen

Only the first layer's filters can be drawn directly (show_filters.py, first_layer.png): every later filter works on
the previous layer's output, not on colours. To show a later filter, this script starts from faint random noise and
nudges the picture, step by step, to make that one filter fire as strongly as possible (gradient ascent on the
picture, the network's weights fixed). The result is a picture of the pattern the filter looks for.

Which filters: the same sample as show_filters.py, read from its chosen.json (by default 32 per stage, the most
selective), so each dreamed-up picture can be set beside the real training pictures that fire that filter most.

How (Olah, Mordvintsev and Schubert 2017, "Feature Visualization", Distill; Simonyan, Vedaldi and Zisserman 2013,
arXiv 1312.6034, section 2):
  * objective: the filter's mean over the picture at the end of its stage (Olah et al.'s "channel" objective),
    taken just before the stage's last ReLU so that a filter which starts switched off still gets a signal;
    dreams.json reports the real (after-ReLU) activation; all filters of a stage are optimised together, one
    picture each, as one batch;
  * the picture is built from smooth patterns (a Fourier basis with low frequencies weighted up, Olah et al.'s
    "preconditioning"), squashed into valid colours with a sigmoid, then colour-scaled exactly as training pictures;
  * transformation robustness (Olah et al.): every step sees the picture randomly shifted (up to --shift) and zoomed
    (up to --zoom), using the same function as training augmentation, so the picture cannot rely on single pixels;
  * Adam, --steps steps at learning rate --lr, a fixed --seed, network in evaluation mode (batch norm uses its
    stored averages, dropout off).
Not used: Olah et al.'s colour decorrelation (it needs a colour-correlation matrix from the training set) and
rotation. A network trained on 9,359 pictures may give noisier, less recognisable pictures than the classic examples
from networks trained on millions; that is a finding, not a fault of the method.

Writes, in --out (default: next to chosen.json's folder, with "-dreams" added):
    dream_stage1.png ... dream_stage4.png   one grid per stage: each sampled filter's dreamed-up picture, labelled
    dreams.json                              settings, and per filter its mean activation on the final picture
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_cnn                                                  # noqa: E402  (also puts the shared code on the path)


def sampled_filters(chosen):
    """{stage label: (filter numbers, filters in the stage or None)} from show_filters.py's chosen.json (either format)."""
    out = {}
    for label, v in chosen["stages"].items():
        if isinstance(v, dict):                                   # 32-per-stage format
            out[label] = ([s["filter"] for s in v["sample"]], v.get("filters_in_stage"))
        else:                                                     # the first, 6-filter format
            out[label] = ([s["filter"] for s in v], None)
    return out


def dream(net, end, filters, size, mean, std, a, device):
    """Optimise one picture per filter to maximise that filter's mean activation at the end of stage block `end`."""
    n = len(filters)
    gen_cpu = torch.Generator().manual_seed(a.seed)
    gen = torch.Generator(device=device).manual_seed(a.seed)
    fy = torch.fft.fftfreq(size, device=device)[:, None]
    fx = torch.fft.rfftfreq(size, device=device)[None, :]
    scale = size / torch.sqrt(fx ** 2 + fy ** 2).clamp(min=1.0 / size)      # low frequencies weighted up
    spec = (0.01 * torch.randn(n, 3, size, size // 2 + 1, 2, generator=gen_cpu)).to(device).requires_grad_(True)
    m = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    body = torch.nn.Sequential(net.stem, *list(net.stages)[:end])
    last = net.stages[end]
    pick = torch.tensor(filters, device=device)
    rows = torch.arange(n, device=device)

    def render():
        img = torch.fft.irfft2(torch.view_as_complex(spec) * scale, s=(size, size)) / 4
        return torch.sigmoid(img)                                 # n x 3 x S x S, colours in 0..1

    def pre_relu(rgb, jitter):
        """The stage's output just before its last ReLU (Block.forward without the final F.relu)."""
        x = (rgb - m) / s
        if jitter:
            x = train_cnn.augment(x, gen, a.shift, a.zoom, flip=False)
        h = body(x)
        y = last.bn2(last.conv2(torch.relu(last.bn1(last.conv1(h)))))
        return (y + last.shortcut(h))[rows, pick]                 # each picture's own filter: n x H x W

    opt = torch.optim.Adam([spec], lr=a.lr)
    for _ in range(a.steps):
        opt.zero_grad()
        (-pre_relu(render(), True).mean(dim=(1, 2)).sum()).backward()   # before the ReLU, so a filter that starts
        opt.step()                                                      # switched off still gets a signal
    with torch.no_grad():
        rgb = render()
        final = torch.relu(pre_relu(rgb, False)).mean(dim=(1, 2))      # the filter's real (after-ReLU) activation
    return (rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8), final.cpu().numpy()


def grid(pictures, filters, title, path, tile=160, cols=8):
    rows = (len(filters) + cols - 1) // cols
    img = Image.new("RGB", (cols * (tile + 6) + 6, rows * (tile + 22) + 34), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((6, 8), title, fill="black")
    for i, (f, p) in enumerate(zip(filters, pictures)):
        x, y = 6 + (i % cols) * (tile + 6), 34 + (i // cols) * (tile + 22)
        img.paste(Image.fromarray(p).resize((tile, tile), Image.BILINEAR), (x, y))
        d.text((x, y + tile + 4), f"filter {f}", fill="black")
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="a .pt file saved by train_cnn.py")
    ap.add_argument("--chosen", required=True, help="chosen.json written by show_filters.py for the same weights")
    ap.add_argument("--out", default=None, help="default: <chosen.json's folder>-dreams")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--shift", type=float, default=0.05, help="largest random shift per step, as a share of the picture")
    ap.add_argument("--zoom", type=float, default=0.05, help="largest random zoom in or out per step")
    ap.add_argument("--size", type=int, default=None, help="picture size; default the size the network was trained at")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-miopen", dest="miopen", action="store_false",
                    help="switch off MIOpen/cuDNN, as for training on Gianni's PC (see train_cnn.py)")
    a = ap.parse_args()
    if not a.miopen:
        torch.backends.cudnn.enabled = False
    ck = torch.load(a.weights, map_location="cpu", weights_only=True)   # never run code hidden in a file (scan, 21 Nov 2026)
    chosen_path = Path(a.chosen)
    chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
    made_for = str(chosen["weights"]).replace("\\", "/").rsplit("/", 1)[-1]   # Windows or Linux path
    if "weights_sha256" in chosen:                                # the exact file, not just its name (scan, 21 Nov 2026)
        if chosen["weights_sha256"] != hashlib.sha256(Path(a.weights).read_bytes()).hexdigest():
            raise SystemExit(f"chosen.json was made for another weights file ({made_for}, a different file "
                             f"from {a.weights} even if the name matches); rerun show_filters.py on these weights")
    elif made_for != Path(a.weights).name:
        raise SystemExit(f"chosen.json was made for {made_for}, not {Path(a.weights).name}")
    st = ck["settings"]
    size = a.size or st["size"]
    device = train_cnn.pick_device(a.device)
    net = train_cnn.build(ck)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    out = Path(a.out) if a.out else chosen_path.parent.with_name(chosen_path.parent.name + "-dreams")
    out.mkdir(parents=True, exist_ok=True)

    blocks = len(net.stages)
    record = {"weights": str(a.weights), "chosen": str(chosen_path), "settings": st, "seed_of_weights": ck["seed"],
              "dream": {"steps": a.steps, "lr": a.lr, "shift": a.shift, "zoom": a.zoom, "size": size, "seed": a.seed,
                        "miopen": a.miopen, "method": "activation maximisation, channel mean, Fourier parametrisation, "
                                                      "sigmoid colours, random shift and zoom each step, Adam"},
              "stages": {}}
    t0 = time.perf_counter()
    for label, (filters, total) in sampled_filters(chosen).items():
        k = int(label.replace("stage", ""))
        end = k * blocks // 4 - 1
        total = total or st["width"] * 2 ** (k - 1)              # stages have W, 2W, 4W, 8W filters
        t1 = time.perf_counter()
        pictures, final = dream(net, end, filters, size, ck["colour_mean"], ck["colour_std"], a, device)
        what = (f"stage {k}: a sample of {len(filters)} of its {total} filters (the most selective): dreamed-up "
                f"pictures that make each fire most (activation maximisation, {a.steps} steps)")
        grid(pictures, filters, f"{what} ({Path(a.weights).stem})", out / f"dream_{label}.png")
        record["stages"][label] = {"description": what, "filters_in_stage": total, "sample_size": len(filters),
                                   "seconds": round(time.perf_counter() - t1, 1),
                                   "filters": [{"filter": int(f), "mean_activation": round(float(v), 4)}
                                               for f, v in zip(filters, final)]}
        print(f"{label}: {len(filters)} filters in {time.perf_counter() - t1:.0f}s", flush=True)
    record["seconds"] = round(time.perf_counter() - t0, 1)
    (out / "dreams.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
