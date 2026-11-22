r"""Find the people in each image, so the image can later be cropped to them before it is embedded.   (Runs on the PC's CPU.)

Detector: torchvision's Faster R-CNN with a MobileNet-V3-Large FPN backbone, trained on COCO (class 1 = person),
weights fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth. Each image is rotated by its EXIF tag and converted to RGB
exactly as the embedder does, so the boxes are in the same frame as the pixels that will be cropped.

    python detect_people.py --list E:\vega\emb\eos-check-images.tsv --wiki ml-db-lan:/srv/ml/vega/wiki --out E:\vega\runs\person-crops\wiki-trainval.csv
    python detect_people.py --list E:\vega\emb\fashionstyle14\images.tsv --wiki ml-db-lan:/srv/ml/datasets/fashionstyle14/FashionStyle14_v1 --out E:\vega\runs\person-crops\fs14.csv

Writes one row per image: id, width, height, n_people (score >= 0.5), people (JSON list of [x0, y0, x1, y1, score],
original pixels, rounded), and a first crop (rows written before 19 Nov ~19:20 Pacific also skipped crops covering
>= 85% of the photo; that cut-off was dropped). The crops the experiments use are rebuilt from the `people` column by
square_crops.py (tight and square shapes), so this column is a convenience only:
  crop  = the box around every person scoring >= 0.7, widened by 10% of its size on each side, clipped to the image;
  no crop (crop_box empty) when no person scores >= 0.7.
Rows are appended as they finish, so a stopped run resumes where it left off.
"""
import argparse, csv, io, json, sys, time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms.functional import pil_to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import vega_embed as ve                                         # noqa: E402  (streams images from ml-db)

KEEP, CROP_SCORE, MARGIN, WHOLE = 0.5, 0.7, 0.10, 1.0


def to_rgb(data):
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            return bg
        return im.convert("RGB")


def crop_box(people, w, h):
    strong = [p for p in people if p[4] >= CROP_SCORE]
    if not strong:
        return ""
    x0, y0 = min(p[0] for p in strong), min(p[1] for p in strong)
    x1, y1 = max(p[2] for p in strong), max(p[3] for p in strong)
    mx, my = MARGIN * (x1 - x0), MARGIN * (y1 - y0)
    x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    if (x1 - x0) * (y1 - y0) >= WHOLE * w * h:                 # WHOLE = 1.0: skip only a crop equal to the whole image
        return ""
    return f"{int(x0)},{int(y0)},{int(round(x1))},{int(round(y1))}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, help="tsv of id and path")
    ap.add_argument("--wiki", required=True, help="folder the paths are relative to (host:path for ml-db)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--size", type=int, default=512, help="shorter side the detector works at (default 512)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.set_num_threads(a.threads)
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, box_score_thresh=KEEP)
    model.transform.min_size, model.transform.max_size = (a.size,), int(a.size * 1.6)
    model.eval().to(a.device)
    print(f"detector on {a.device}", flush=True)
    prep = weights.transforms()

    items = [tuple(l.split("\t")) for l in Path(a.list).read_text(encoding="utf-8").split("\n") if l.strip()]
    if a.limit:
        items = items[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {r["id"] for r in csv.DictReader(open(out, encoding="utf-8", newline=""))}
    todo = [it for it in items if it[0] not in done]
    print(f"{len(items)} listed, {len(done)} done before, {len(todo)} to do", flush=True)
    new = not out.exists()
    fh = open(out, "a", encoding="utf-8", newline="")
    w = csv.writer(fh, lineterminator="\n")
    if new:
        w.writerow(["id", "width", "height", "n_people", "people", "crop_box"])
    n, start = 0, time.perf_counter()
    with torch.inference_mode():
        for pid, data in ve.open_wiki(a.wiki).stream(todo):
            n += 1
            if isinstance(data, Exception):
                w.writerow([pid, "", "", "", "unreadable", ""])
                continue
            try:
                im = to_rgb(data)
            except Exception:
                w.writerow([pid, "", "", "", "unreadable", ""])
                continue
            pred = model([prep(pil_to_tensor(im)).to(a.device)])[0]
            people = [[round(float(b[0])), round(float(b[1])), round(float(b[2])), round(float(b[3])), round(float(s), 3)]
                      for b, l, s in zip(pred["boxes"], pred["labels"], pred["scores"]) if int(l) == 1 and float(s) >= KEEP]
            w.writerow([pid, im.width, im.height, len(people), json.dumps(people), crop_box(people, im.width, im.height)])
            if n % 200 == 0 or n == len(todo):
                fh.flush()
                rate = n / (time.perf_counter() - start)
                print(f"\r  {n}/{len(todo)}  {rate:.1f} img/s  about {(len(todo) - n) / rate / 60:.0f} min left   ", end="", flush=True)
    fh.close()
    print(f"\ntime elapsed: {time.perf_counter() - t0:.0f}s. saved in {out}")


if __name__ == "__main__":
    main()
