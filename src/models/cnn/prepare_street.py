"""Prepare the Street Fashion Style photos for the CNN's second output (their posters' own style tags).
Owner: Giancarlo Guevarra.

    python prepare_street.py --photos C:\\vega\\street --manifest C:\\vega\\street\\manifest.csv

Same picture handling as prepare_images.py (sRGB, first frame, camera rotation, white behind see-through areas,
grey stays grey, the whole photo fitted inside a square, Lanczos), at one size (default 224). The bars are left at
zero here and painted at training time with the run's own average colour (validation and final runs have
slightly different averages, both measured on wiki training pictures only).

Which photos (Gianni, 20 Nov 2026; no style families, no mapping to wiki aesthetics): every photo with a style
tag. Posters with sha1(name) % 10 == 0 form the held-out group, as for Andrew's NN: never trained on, kept to
measure tag accuracy on posters the network has never seen. Tags kept: those with at least --min-photos (500)
photos outside the held-out group; photos of other tags are left out. A photo that cannot be opened is skipped
and listed (three broken files are known). If more than --max-failed (default 1%) of the photos fail, the photos
folder is probably wrong or half-copied: the run stops with an error and does not write street.json, so a
corrected rerun does the work instead of skipping it.

Writes in --out (default <root>/runs/cnn/prepared/street):
    street-<S>.npy    uint8, rows x S x S x 3, bars zero (written as street-<S>.partial.npy, renamed when complete)
    boxes-<S>.npy     rows x 4 (x0, y0, width, height) of the photo inside the square
    photos.csv        row, id, poster, tag, tag_index, group (group 0 = held out); failed photos are not listed
    tags.txt          the kept tags, one per line, in index order (most training photos first)
    failed.txt        photos that could not be opened, with the reason
    street.json       counts, settings, timing; written last, so its presence means the preparation is complete
A rerun with the same manifest, size, threshold and photos folder does nothing, as long as street-<S>.npy is still
there (add --force to redo it).
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_images                                             # noqa: E402  (the same picture handling)

GROUPS = 10                                                       # sha1(poster) % 10; group 0 is held out
HELD_OUT = 0


def poster_group(name):
    return int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % GROUPS


def select(manifest, min_photos):
    """The photos to prepare, in manifest order, and the kept tags (most training photos first)."""
    rows = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tag = (r.get("style") or "").strip()
            if tag:
                rows.append({"id": r["id"], "poster": r["user"], "path": r["path"], "tag": tag,
                             "group": poster_group(r["user"])})
    counts = Counter(r["tag"] for r in rows if r["group"] != HELD_OUT)
    tags = sorted((t for t, n in counts.items() if n >= min_photos), key=lambda t: (-counts[t], t))
    index = {t: i for i, t in enumerate(tags)}
    keep = [dict(r, tag_index=index[r["tag"]]) for r in rows if r["tag"] in index]
    return keep, tags, counts


def work(job):
    """Worker: (row, path, size) -> (row, pixels, box, error)."""
    row, path, size = job
    try:
        pixels, boxes, _ = prepare_images.prepare_one(path, [size])
        return row, pixels[size], boxes[size], None
    except Exception as e:                                        # noqa: BLE001  (skip and list it)
        return row, None, None, str(e).replace("\n", " ")[:200]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True, help="folder holding chictopia/<poster>/<photo>.jpg")
    ap.add_argument("--manifest", required=True, help="Street Fashion Style manifest.csv (id, user, path, style, ...)")
    ap.add_argument("--root", default=None, help="project folder; default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--out", default=None, help="default <root>/runs/cnn/prepared/street")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--min-photos", type=int, default=500, help="a tag needs this many training photos to be kept")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None, help="only the first N selected photos (a quick check)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-failed", type=float, default=0.01, help="largest share of photos allowed to fail (default 0.01)")
    a = ap.parse_args()
    if a.limit is not None and a.limit < 1:
        raise SystemExit("--limit must be at least 1")
    root = Path(a.root) if a.root else prepare_images.vega_data.ROOT
    out = Path(a.out) if a.out else root / "runs" / "cnn" / "prepared" / ("street" + (f"-limit{a.limit}" if a.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    manifest_sha = hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
    done = out / "street.json"
    if done.exists() and not a.force:
        old = json.loads(done.read_text(encoding="utf-8"))
        if (old.get("manifest_sha256"), old.get("size"), old.get("min_photos"), old.get("limit"),
                old.get("photos_folder", str(Path(a.photos).resolve()))) == \
                (manifest_sha, a.size, a.min_photos, a.limit, str(Path(a.photos).resolve())) \
                and (out / f"street-{a.size}.npy").exists():
            print(f"already prepared, nothing to do: {out}  (add --force to redo)")
            return
    done.unlink(missing_ok=True)

    keep, tags, counts = select(a.manifest, a.min_photos)
    keep = keep[: a.limit]
    n, s = len(keep), a.size
    print(f"{n} photos, {len(tags)} tags (at least {a.min_photos} training photos each), size {s}, "
          f"{a.workers} workers -> {out}", flush=True)
    t0 = time.perf_counter()
    partial = out / f"street-{s}.partial.npy"
    arr = open_memmap(partial, mode="w+", dtype=np.uint8, shape=(n, s, s, 3))
    boxes = np.zeros((n, 4), np.int32)
    ok = np.zeros(n, bool)
    failed = []
    photos = Path(a.photos)
    jobs = ((i, str(photos / r["path"]), s) for i, r in enumerate(keep))
    with Pool(a.workers) as pool:
        for k, (i, px, box, err) in enumerate(pool.imap_unordered(work, jobs, chunksize=64), 1):
            if err is None:
                arr[i], boxes[i], ok[i] = px, box, True
            else:
                failed.append((keep[i]["id"], err))
            if k % 20000 == 0:
                print(f"  {k}/{n} photos, {time.perf_counter() - t0:.0f}s", flush=True)
    arr.flush()
    del arr
    np.save(out / f"boxes-{s}.npy", boxes)
    with open(out / "photos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "id", "poster", "tag", "tag_index", "group"])
        for i, r in enumerate(keep):
            if ok[i]:
                w.writerow([i, r["id"], r["poster"], r["tag"], r["tag_index"], r["group"]])
    (out / "tags.txt").write_text("\n".join(tags) + "\n", encoding="utf-8")
    (out / "failed.txt").write_text("".join(f"{i}\t{e}\n" for i, e in sorted(failed)), encoding="utf-8")
    if len(failed) > a.max_failed * max(n, 1):
        raise SystemExit(f"{len(failed)} of {n} photos could not be opened (more than {a.max_failed:.1%}); is --photos "
                         f"right and fully copied? See {out / 'failed.txt'}. Nothing was marked complete.")
    os.replace(partial, out / f"street-{s}.npy")
    good = [r for i, r in enumerate(keep) if ok[i]]
    summary = {"made": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "manifest": str(a.manifest), "manifest_sha256": manifest_sha,
               "size": s, "min_photos": a.min_photos, "limit": a.limit, "photos_folder": str(Path(a.photos).resolve()),
               "held_out_group": HELD_OUT, "groups": GROUPS,
               "photos": len(good), "failed": len(failed),
               "training_photos": sum(r["group"] != HELD_OUT for r in good),
               "held_out_photos": sum(r["group"] == HELD_OUT for r in good),
               "tags": {t: {"training": sum(r["group"] != HELD_OUT and r["tag"] == t for r in good),
                            "held_out": sum(r["group"] == HELD_OUT and r["tag"] == t for r in good)} for t in tags},
               "tags_left_out": {t: c for t, c in counts.items() if t not in set(tags)},
               "seconds": round(time.perf_counter() - t0, 1), "workers": a.workers}
    done.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"done: {summary['photos']} photos ({summary['training_photos']} training, {summary['held_out_photos']} "
          f"held out), {len(failed)} failed, {summary['seconds']:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
