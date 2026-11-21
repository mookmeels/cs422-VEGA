"""Prepare the wiki pictures for Giancarlo's CNN, once per picture size.

Each picture is decoded once, turned into a standard sRGB colour picture, fitted whole inside a square at every
size asked for, and the bars around it are filled with the average colour of the fit pictures. The pixels are
saved exactly (8-bit, no re-compression), so training never decodes a JPEG again.

Decisions (Gianni, 19 and 20 Nov 2026, recorded in the CNN plan):
  * every picture becomes 3-channel colour: the first frame of animated pictures, camera rotation tags applied,
    embedded colour profiles converted to standard sRGB, white behind transparent areas, CMYK converted;
  * greyscale pictures stay grey (the same value in all three channels; nothing is colourised);
  * the whole picture fits inside the square, nothing is cut; the bars are the fit pictures' average colour;
  * prepared once per tested size (128 and 224 px), exact pixels kept.
The channel scaling (mean and spread of each colour channel) is measured here, on fit pictures only, and saved in
stats.json; train_cnn.py applies it. Because the bars are the average colour, they become zero after scaling.

    python prepare_images.py                  # fit = training split, eval = validation split
    python prepare_images.py --limit 50       # quick check on the first 50 pictures of each part
    python prepare_images.py --final          # fit = train + validation, eval = test (final run only)

Output, in <root>/runs/cnn/prepared/<mode>/ (mode: validation, final, or validation-limit<N>):
    fit-<S>.npy, eval-<S>.npy    uint8 pictures, n x S x S x 3, rows in the shared loader's order
    fit-ids.txt, eval-ids.txt    the picture ids (sha1) in that order
    boxes-<S>.npz                where each picture sits in its square: x0, y0, width, height (fit and eval)
    stats.json                   padding colour, channel mean and spread, split hash, counts, timings, versions
    pictures.csv                 per picture: format, mode, size, frames, rotation, transparency, colour profile
A rerun with the same split, ids, sizes and label set does nothing (add --force to redo it).
A picture that cannot be opened or converted stops the run with its path; nothing is skipped silently.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageCms, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
import vega_data  # noqa: E402  (the team's shared loader)

ORIENTATION_TAG = 0x0112
_SRGB = None


def _srgb_profile():
    """The standard sRGB colour profile, built once per worker process."""
    global _SRGB
    if _SRGB is None:
        _SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    return _SRGB


def to_srgb(path):
    """Open one picture and return it as an 8-bit RGB picture, plus a record of what was done to it."""
    with Image.open(path) as src:
        info = {"format": src.format or "", "mode": src.mode, "frames": getattr(src, "n_frames", 1)}
        src.seek(0)                                               # first frame of animated GIF, WEBP, PNG, MPO
        info["rotation_tag"] = int(src.getexif().get(ORIENTATION_TAG, 1) or 1)
        im = ImageOps.exif_transpose(src)                         # applies the camera rotation tag (a copy)
    icc = im.info.get("icc_profile")
    info["width"], info["height"] = im.size

    mode = im.mode
    if mode == "1":
        im, mode = im.convert("L"), "L"
    if mode == "P":                                               # palette pictures: to colour, keeping see-through
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
        mode = im.mode
    if mode in ("L", "RGB") and "transparency" in im.info:        # one see-through colour given in the file
        im = im.convert("LA" if mode == "L" else "RGBA")
        mode = im.mode
    if mode not in ("RGB", "RGBA", "L", "LA", "CMYK"):
        raise ValueError(f"unhandled picture mode {mode}")
    grey = mode in ("L", "LA")
    alpha = None
    if mode in ("RGBA", "LA"):
        alpha = im.getchannel("A")
        im = im.convert("L" if grey else "RGB")
    info["grey"] = grey
    info["transparency"] = alpha is not None and alpha.getextrema()[0] < 255

    info["profile"] = "none"
    if icc and not grey:                                          # a grey picture keeps its grey values as they are
        try:
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)), _srgb_profile(),
                                           outputMode="RGB")
            info["profile"] = "converted"
        except (ImageCms.PyCMSError, OSError, ValueError) as e:   # a broken profile: plain conversion, recorded
            info["profile"] = "broken: " + str(e).replace("\n", " ")[:80]
    if im.mode == "CMYK":
        im = im.convert("RGB")
    if alpha is not None:                                         # white behind see-through areas
        white = 255 if im.mode == "L" else (255, 255, 255)        # a bare 255 in RGB would mean red
        im = Image.composite(im, Image.new(im.mode, im.size, white), alpha)
    return im.convert("RGB"), info                                # grey: the same value copied to all 3 channels


def fit_square(im, size):
    """Shrink (or enlarge) the whole picture to fit a size x size square, centred. Returns pixels and its box."""
    w, h = im.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = im.resize((nw, nh), Image.Resampling.LANCZOS, reducing_gap=3.0)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    out = np.zeros((size, size, 3), np.uint8)                     # bars filled with the average colour later
    out[y0:y0 + nh, x0:x0 + nw] = np.asarray(small, np.uint8)
    return out, (x0, y0, nw, nh)


def prepare_one(path, sizes):
    """Worker: one picture at every size. Any failure is re-raised with the picture's path."""
    try:
        im, info = to_srgb(path)
        pixels, boxes = {}, {}
        for s in sizes:
            pixels[s], boxes[s] = fit_square(im, s)
            info[f"upscaled_{s}"] = max(im.size) < s
        return pixels, boxes, info
    except Exception as e:                                        # noqa: BLE001  (stop the run, naming the file)
        raise RuntimeError(f"{path}: {type(e).__name__}: {e}") from None


def ids_sha256(fit_ids, eval_ids):
    return hashlib.sha256(("\n".join(fit_ids) + "\n--\n" + "\n".join(eval_ids)).encode()).hexdigest()


def content_stats(arr, boxes):
    """Mean and spread of each colour channel over picture pixels only (not the bars), in 0 to 1 units."""
    total, total_sq, count = np.zeros(3), np.zeros(3), 0
    for i, (x0, y0, w, h) in enumerate(boxes):
        px = arr[i, y0:y0 + h, x0:x0 + w].reshape(-1, 3).astype(np.float64) / 255.0
        total += px.sum(0)
        total_sq += (px * px).sum(0)
        count += len(px)
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 0.0))
    return mean, std


def fill_bars(arr, boxes, colour):
    """Paint everything outside each picture's box with the given colour."""
    for i, (x0, y0, w, h) in enumerate(boxes):
        arr[i, :y0] = colour
        arr[i, y0 + h:] = colour
        arr[i, y0:y0 + h, :x0] = colour
        arr[i, y0:y0 + h, x0 + w:] = colour


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true", help="fit = train + validation, eval = test (final run only)")
    ap.add_argument("--sizes", default="128,224", help="square sizes in pixels, comma-separated (default 128,224)")
    ap.add_argument("--labels", default="merged", choices=("merged", "wearable"))
    ap.add_argument("--root", default=None, help="project folder; default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--out", default=None, help="default <root>/runs/cnn/prepared/<mode>")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="processes decoding pictures (default: all cores but two)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pictures of each part (a quick check)")
    ap.add_argument("--force", action="store_true", help="redo even if a matching preparation exists")
    args = ap.parse_args()

    sizes = sorted({int(s) for s in args.sizes.split(",") if s.strip()})
    root = Path(args.root) if args.root else vega_data.ROOT
    d = vega_data.load(final=args.final, labels=args.labels, root=root, vectors=False)
    parts = {"fit": d["fit"], "eval": d["eval"]}
    if args.limit:
        parts = {k: {"ids": v["ids"][:args.limit], "paths": v["paths"][:args.limit]} for k, v in parts.items()}
    mode = ("final" if args.final else "validation") + (f"-limit{args.limit}" if args.limit else "")
    out = Path(args.out) if args.out else root / "runs" / "cnn" / "prepared" / mode
    out.mkdir(parents=True, exist_ok=True)

    split_hash = vega_data.split_sha256(root)
    id_hash = ids_sha256(list(parts["fit"]["ids"]), list(parts["eval"]["ids"]))
    stats_file = out / "stats.json"
    if stats_file.exists() and not args.force:
        old = json.loads(stats_file.read_text(encoding="utf-8"))
        if (old.get("split_sha256"), old.get("ids_sha256"), old.get("sizes"), old.get("labels")) == \
                (split_hash, id_hash, sizes, args.labels):
            print(f"already prepared, nothing to do: {out}  (add --force to redo)")
            return
    stats_file.unlink(missing_ok=True)                            # stats.json is written last: present = complete

    for part, v in parts.items():
        if len(set(v["ids"])) != len(v["ids"]):
            raise SystemExit(f"{part}: the loader returned a picture id twice; stopping")
    print(f"{mode}: fit {len(parts['fit']['ids'])} pictures, eval {len(parts['eval']['ids'])} pictures, "
          f"sizes {sizes}, {args.workers} workers -> {out}", flush=True)

    records, boxes, timing = [], {s: {} for s in sizes}, {}
    work = partial(prepare_one, sizes=sizes)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part, v in parts.items():
            n = len(v["ids"])
            arrays = {s: np.lib.format.open_memmap(out / f"{part}-{s}.partial.npy", mode="w+", dtype=np.uint8,
                                                   shape=(n, s, s, 3)) for s in sizes}
            part_boxes = {s: np.zeros((n, 4), np.int32) for s in sizes}
            t0 = time.perf_counter()
            try:
                for i, (pixels, bx, info) in enumerate(pool.map(work, v["paths"], chunksize=8)):
                    for s in sizes:
                        arrays[s][i] = pixels[s]
                        part_boxes[s][i] = bx[s]
                    records.append({"part": part, "id": v["ids"][i], **info})
                    if (i + 1) % 1000 == 0 or i + 1 == n:
                        print(f"  {part}: {i + 1}/{n} pictures, {time.perf_counter() - t0:.0f} s", flush=True)
            except RuntimeError as e:
                raise SystemExit(f"stopped: could not prepare {e}") from None
            timing[f"{part}_seconds"] = round(time.perf_counter() - t0, 1)
            for s in sizes:
                boxes[s][part] = part_boxes[s]
                arrays[s].flush()
            del arrays

    size_stats = {}
    for s in sizes:
        fit = np.load(out / f"fit-{s}.partial.npy", mmap_mode="r+")
        mean, std = content_stats(fit, boxes[s]["fit"])          # fit pictures only
        pad = np.clip(np.round(mean * 255), 0, 255).astype(np.uint8)
        fill_bars(fit, boxes[s]["fit"], pad)
        fit.flush()
        del fit
        ev = np.load(out / f"eval-{s}.partial.npy", mmap_mode="r+")
        fill_bars(ev, boxes[s]["eval"], pad)                      # eval gets the fit colour, never its own
        ev.flush()
        del ev
        for part in parts:
            os.replace(out / f"{part}-{s}.partial.npy", out / f"{part}-{s}.npy")
        np.savez(out / f"boxes-{s}.npz", fit=boxes[s]["fit"], eval=boxes[s]["eval"])
        size_stats[str(s)] = {"pad_rgb": pad.tolist(), "mean": mean.round(6).tolist(), "std": std.round(6).tolist(),
                              "upscaled_fit": sum(r[f"upscaled_{s}"] for r in records if r["part"] == "fit"),
                              "upscaled_eval": sum(r[f"upscaled_{s}"] for r in records if r["part"] == "eval")}

    for part, v in parts.items():
        (out / f"{part}-ids.txt").write_text("\n".join(v["ids"]) + "\n", encoding="utf-8")
    fields = ["part", "id", "format", "mode", "width", "height", "frames", "rotation_tag", "grey", "transparency",
              "profile"] + [f"upscaled_{s}" for s in sizes]
    with open(out / "pictures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    def count(key, test=bool):
        return sum(1 for r in records if test(r[key]))

    summary = {
        "made": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "mode": mode, "labels": args.labels, "sizes": sizes,
        "split_sha256": split_hash, "ids_sha256": id_hash,
        "counts": {p: len(v["ids"]) for p, v in parts.items()},
        "per_size": size_stats,
        "pictures": {"grey": count("grey"), "transparency": count("transparency"),
                     "animated": count("frames", lambda n: n > 1),
                     "rotated": count("rotation_tag", lambda t: t not in (0, 1)),
                     "profile_converted": count("profile", lambda p: p == "converted"),
                     "profile_broken": count("profile", lambda p: p.startswith("broken")),
                     "formats": {f: count("format", lambda x, f=f: x == f) for f in sorted({r["format"] for r in records})},
                     "modes": {m: count("mode", lambda x, m=m: x == m) for m in sorted({r["mode"] for r in records})}},
        "timing": {**timing, "workers": args.workers},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__,
                     "machine": platform.node(), "system": platform.platform()},
        "resize": "Lanczos, reducing_gap 3.0, whole picture fitted inside the square, centred",
        "stats_note": "mean and std are over picture pixels only (not the bars), fit part only, 0 to 1 units",
    }
    stats_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("counts", "per_size", "pictures", "timing")}, indent=2))
    print(f"done: {out}")


if __name__ == "__main__":                                        # needed on Windows for the worker processes
    main()
