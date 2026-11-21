"""Draw what a trained CNN's filters respond to (done check 10, added by Gianni on 19 Nov 2026).

    python show_filters.py --weights <root>/runs/cnn/validation/weights/cnn-r18w32-128-validation-seed422.pt

Writes, in <root>/runs/cnn/filters/<weights file name>/:
    first_layer.png     every first-layer filter (7 x 7 x 3 colour numbers), drawn as a small colour square.
                        Each square is stretched on its own from its smallest to its largest number, so mid-grey
                        means zero; edges, stripes and colour blobs show up as learned patterns.
    stage1_top.png ...  for each of the 4 stages, a SAMPLE of its filters (default 32; the stages have W, 2W, 4W
    stage4_top.png      and 8W filters, so 64, 128, 256 and 512 in the standard ResNet-18), each with the training
                        pictures that make it fire most. Early stages tend to show colours, edges and textures;
                        stage 4 (just before the final layer) whole outfits, settings and styles.
    chosen.json         which filters (the sample, and each stage's total), which pictures (ids and labels) and
                        how strongly each fired.
The sample is the filters whose response varies most between pictures (the most selective), not chosen by hand.
Gianni, 20 Nov 2026: all four stages, 32 filters each, labelled as a sample.
Only training (fit) pictures are shown; validation and test pictures are never used here.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_cnn                                                  # noqa: E402  (also puts the shared code on the path)
import vega_data                                                  # noqa: E402


def first_layer(net, path, scale=6):
    w = net.stem[0].weight.detach().float().cpu().numpy()        # filters x 3 x 7 x 7
    n, k = w.shape[0], w.shape[2]
    cols = 8 if n <= 64 else 16
    rows = (n + cols - 1) // cols
    cell = k * scale + 6
    img = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i in range(n):
        f = w[i].transpose(1, 2, 0)
        lo, hi = f.min(), f.max()
        f = (f - lo) / (hi - lo if hi > lo else 1.0)
        tile = Image.fromarray((f * 255).round().astype(np.uint8)).resize((k * scale, k * scale), Image.NEAREST)
        img.paste(tile, ((i % cols) * cell + 3, (i // cols) * cell + 3))
    img.save(path)


def top_pictures(net, pics, stage_end, n_filters, n_pictures, device):
    """Per filter of the stage, its strongest response anywhere in each fit picture; pick the most selective filters."""
    grabbed = {}
    hook = net.stages[stage_end].register_forward_hook(lambda m, i, o: grabbed.__setitem__("x", o))
    peaks = []
    net.eval()
    with torch.no_grad():
        for s in range(0, len(pics), 256):
            net(pics.batch(torch.arange(s, min(s + 256, len(pics)), device=device)))
            peaks.append(grabbed["x"].float().amax(dim=(2, 3)).cpu())
    hook.remove()
    peaks = torch.cat(peaks).numpy()                              # pictures x filters
    chosen = np.argsort(-peaks.std(0))[:n_filters]
    return [(int(f), np.argsort(-peaks[:, f])[:n_pictures], peaks[:, f]) for f in chosen], peaks.shape[1]


def sheet(pics, picks, title, path, size=112):
    cols = len(picks[0][1])
    img = Image.new("RGB", (cols * (size + 4) + 150, len(picks) * (size + 4) + 30), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((6, 6), title, fill="black")
    for r, (f, rows, _) in enumerate(picks):
        y = 30 + r * (size + 4)
        d.text((6, y + size // 2 - 6), f"filter {f}", fill="black")
        for c, i in enumerate(rows):
            tile = Image.fromarray(pics.u8[int(i)].cpu().numpy()).resize((size, size), Image.BILINEAR)
            img.paste(tile, (150 + c * (size + 4), y))
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="a .pt file saved by train_cnn.py")
    ap.add_argument("--root", default=None, help="project folder; default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--prepared", default=None, help="prepared pictures; default <root>/runs/cnn/prepared/validation")
    ap.add_argument("--out", default=None, help="default <root>/runs/cnn/filters/<weights file name>")
    ap.add_argument("--filters", type=int, default=32, help="filters shown per stage (a sample: the most selective)")
    ap.add_argument("--pictures", type=int, default=8, help="pictures shown per filter")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-miopen", dest="miopen", action="store_false",
                    help="switch off MIOpen/cuDNN, as for training on Gianni's PC (see train_cnn.py)")
    a = ap.parse_args()
    if not a.miopen:
        torch.backends.cudnn.enabled = False
    root = Path(a.root) if a.root else vega_data.ROOT
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    s = ck["settings"]
    device = train_cnn.pick_device(a.device)
    net = train_cnn.build(ck)
    net.to(device)
    prepared = Path(a.prepared) if a.prepared else root / "runs" / "cnn" / "prepared" / "validation"
    out = Path(a.out) if a.out else root / "runs" / "cnn" / "filters" / Path(a.weights).stem
    out.mkdir(parents=True, exist_ok=True)

    ids = (prepared / "fit-ids.txt").read_text(encoding="utf-8").split()
    pics = train_cnn.Pictures(prepared, "fit", s["size"], ids, ck["colour_mean"], ck["colour_std"], device)
    d = vega_data.load(labels=ck["labels"], root=root, vectors=False)
    names = {sha: [d["classes"][k] for k in t] for sha, t in zip(d["fit"]["ids"], d["fit"]["truth"])}

    first_layer(net, out / "first_layer.png")
    record = {"weights": str(a.weights), "settings": s, "seed": ck["seed"], "stages": {}}
    blocks = len(net.stages)                                      # 4 stages of equal length, in order
    for k in range(1, 5):
        label, stage_end = f"stage{k}", k * blocks // 4 - 1
        picks, total = top_pictures(net, pics, stage_end, a.filters, a.pictures, device)
        what = (f"stage {k}: a sample of {len(picks)} of its {total} filters (the most selective), with the training "
                f"pictures that make each fire most")
        sheet(pics, picks, f"{what} ({Path(a.weights).stem})", out / f"{label}_top.png")
        record["stages"][label] = {"description": what, "filters_in_stage": total, "sample_size": len(picks),
                                   "sample": [{"filter": f, "pictures": [{"id": ids[int(i)],
                                                                          "labels": names.get(ids[int(i)], []),
                                                                          "peak": round(float(v[int(i)]), 4)}
                                                                         for i in rows]} for f, rows, v in picks]}
    (out / "chosen.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
