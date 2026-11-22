"""One report figure of what the CNN learned, from pictures the other scripts already made. Owner: Giancarlo Guevarra.

    python make_cnn_figure.py --filters <show_filters.py folder> --label-dreams <dream_labels.py folder> \
        [--filter-dreams <dream_filters.py folder>] [--picked <label> ...] [--out <png>]

The filter-dream folder is <--filters>-dreams (dream_filters.py's default) unless --filter-dreams names it. Gianni, 22 Nov 2026: one
landscape figure, not a long strip, with filters from every stage and the pictures they fire on most, and dreams
of aesthetics beside real pictures of them. The report's figure uses the final network (trained on training plus
validation pictures), so it has no held-out pictures to score; the pictures beside each aesthetic are its own
training pictures, to compare the dream with what the aesthetic looks like. Panels:

  A  every first-layer filter (first_layer.png from show_filters.py, unchanged);
  B  per stage, the two most selective filters (the first two of show_filters.py's sample, which is sorted by
     how much the filter's response varies between pictures) that still have 4 pictures to show once sensitive
     ones are left out, each as its dream (cut from dream_filters.py's sheet) followed by the 4 training pictures
     that make it fire most (from chosen.json);
  C  the 3 labels with the most training pictures, by a fixed rule;
  D  optional (--picked): labels picked by hand, marked as such. Gianni, 22 Nov 2026, picked Lumberjack, Nu-Goth,
     Tomato Girl Summer and Naturecore because their dreams look close to the aesthetic.

Each label in C and D is its dream (cut from dream_labels.py's page) followed by 3 of the network's own training
pictures of that label, drawn by a fixed rule (a random order with seed 0, skipping pictures with a sensitive
label). A sensitive label picked by hand shows its dream only.

Pictures whose labels the wiki tags as sensitive (runs/catalog-audit/aesthetics.csv) are skipped, as in
nn/analyse_errors.py, and sensitive labels are not chosen for C. Test pictures are never read. Pictures come from a
prepared folder that holds training and validation pictures (by default the final run's fit pictures,
runs/cnn/prepared/final, 224 px, exact 8-bit pixels).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from find_shared import shared_code                               # noqa: E402

sys.path.insert(0, str(shared_code()))
import vega_data                                                  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Sheet geometry, copied from the scripts that drew the sheets (dream_filters.grid, dream_labels.page).
FILTER_SHEET = dict(tile=160, cols=8, x0=6, y0=34, dx=166, dy=182)
LABEL_SHEET = dict(tile=160, cols=8, x0=8, y0=34, dx=168, dy=204, per_page=32)


def font(size, bold=False):
    path = FONT_BOLD if bold else FONT
    return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default()


def cut(sheet, i, g):
    x, y = g["x0"] + (i % g["cols"]) * g["dx"], g["y0"] + (i // g["cols"]) * g["dy"]
    return sheet.crop((x, y, x + g["tile"], y + g["tile"]))


def sensitive_labels(root, classes):
    """The labels the wiki tags as sensitive, as in nn/analyse_errors.py."""
    path = root / "runs" / "catalog-audit" / "aesthetics.csv"
    if not path.exists():
        raise SystemExit(f"{path} is missing: cannot leave out sensitive pictures")
    m, _ = vega_data.label_map("merged", root)
    cat = pl.read_csv(path, infer_schema=False)
    names = {m[a] for a, s in zip(cat["aesthetic"].to_list(), cat["sensitive"].to_list()) if s == "true" and a in m}
    unknown = names - set(classes)
    if unknown:
        raise SystemExit(f"sensitive labels not in the label set: {sorted(unknown)[:5]}")
    return names


def framed(img, size, colour=None, width=5):
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    if colour is None:
        return img
    out = Image.new("RGB", (size, size), colour)
    out.paste(img.resize((size - 2 * width, size - 2 * width), Image.BILINEAR), (width, width))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filters", required=True, help="show_filters.py's output folder (first_layer.png, chosen.json)")
    ap.add_argument("--label-dreams", required=True, help="dream_labels.py's output folder")
    ap.add_argument("--filter-dreams", default=None, help="dream_filters.py's output folder; default <--filters>-dreams")
    ap.add_argument("--prepared", default=None, help="folder with fit-224.npy and fit-ids.txt holding train and "
                                                     "validation pictures; default <root>/runs/cnn/prepared/final")
    ap.add_argument("--root", default=None, help="default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--out", default=None, help="default <root>/runs/cnn/figure/<network>.png")
    ap.add_argument("--tile", type=int, default=150, help="picture size in panel B, pixels")
    ap.add_argument("--picked", nargs="+", default=[], help="labels picked by hand, shown as panel D (at most 4)")
    a = ap.parse_args()
    if len(a.picked) > 4 or len(set(a.picked)) != len(a.picked):
        raise SystemExit("--picked takes at most 4 different labels")

    root = Path(a.root) if a.root else vega_data.ROOT
    fdir, ldir = Path(a.filters), Path(a.label_dreams)
    ddir = Path(a.filter_dreams) if a.filter_dreams else fdir.parent / (fdir.name + "-dreams")
    chosen = json.loads((fdir / "chosen.json").read_text(encoding="utf-8"))
    dreams = json.loads((ddir / "dreams.json").read_text(encoding="utf-8"))
    ldreams = json.loads((ldir / "label_dreams.json").read_text(encoding="utf-8"))
    network = Path(chosen["weights"].replace("\\", "/")).stem

    # The three folders must all come from the same network.
    for name, rec in (("dreams.json", dreams), ("label_dreams.json", ldreams)):
        if Path(str(rec["weights"]).replace("\\", "/")).stem != network:
            raise SystemExit(f"{name} is from {rec['weights']}, not {network}")
    final = bool(chosen.get("final"))
    if final != ("-test-seed" in network):
        raise SystemExit(f"chosen.json says final={final}, but the network is named {network}")

    # The network's own training pictures: train + validation for a final network, train only otherwise.
    d = vega_data.load(final=False, labels="merged", root=root, vectors=False)
    classes = list(d["classes"])
    hide = sensitive_labels(root, classes)
    parts = ("fit", "eval") if final else ("fit",)
    train_ids = [str(x) for p in parts for x in d[p]["ids"]]
    train_truth = [set(t) for p in parts for t in d[p]["truth"]]
    train_count = np.zeros(len(classes), int)
    for t in train_truth:
        for k in t:
            train_count[k] += 1
    for e in ldreams["labels_dreamed"]:                           # dream_labels.py counted the same pictures
        if "training_pictures" in e and e["training_pictures"] != train_count[classes.index(e["label"])]:
            raise SystemExit(f"{e['label']}: {e['training_pictures']} training pictures in label_dreams.json, "
                             f"{train_count[classes.index(e['label'])]} here")

    prepared = Path(a.prepared) if a.prepared else root / "runs" / "cnn" / "prepared" / "final"
    pic_ids = (prepared / "fit-ids.txt").read_text(encoding="utf-8").split()
    pic_row = {x: i for i, x in enumerate(pic_ids)}
    pix = np.load(prepared / "fit-224.npy", mmap_mode="r")

    def picture(pid):
        return Image.fromarray(np.asarray(pix[pic_row[pid]]))

    def is_hidden(label_names):
        return any(n in hide for n in label_names)

    # ---- choose -------------------------------------------------------------------------------------------
    stages = []
    for k in range(1, 5):
        st = chosen["stages"][f"stage{k}"]
        order = [e["filter"] for e in dreams["stages"][f"stage{k}"]["filters"]] if "stages" in dreams else None
        rows = []
        for i, e in enumerate(st["sample"]):
            if order is not None and order[i] != e["filter"]:
                raise SystemExit(f"stage {k}: dream sheet order differs from chosen.json")
            pics = [p["id"] for p in e["pictures"] if not is_hidden(p["labels"])][:4]
            if len(pics) == 4:                                    # else too many of its pictures are sensitive
                rows.append((e["filter"], i, pics))
            if len(rows) == 2:
                break
        stages.append((k, st["filters_in_stage"], rows))

    by_name = {e["label"]: e for e in ldreams["labels_dreamed"]}
    ok = [k for k in range(len(classes)) if classes[k] not in hide]
    popular = sorted(ok, key=lambda k: (-train_count[k], classes[k]))[:3]
    for name in a.picked:
        if name not in classes:
            raise SystemExit(f"--picked {name!r} is not one of the {len(classes)} labels")

    def label_row(k):
        e = by_name[classes[k]]
        if e["index"] != k:
            raise SystemExit(f"{classes[k]}: index {e['index']} in label_dreams.json, {k} here")
        page = Image.open(ldir / f"labels_page{e['page']:02d}.png")
        dream = cut(page, e["index"] % LABEL_SHEET["per_page"], LABEL_SHEET)
        mine = sorted(train_ids[i] for i, t in enumerate(train_truth) if k in t)
        truth = dict(zip(train_ids, train_truth))
        sensitive = classes[k] in hide
        pics = []
        if not sensitive:
            for pid in np.random.default_rng(0).permutation(mine):
                if not is_hidden([classes[j] for j in truth[pid]]):
                    pics.append(str(pid))
                if len(pics) == 3:
                    break
        return dict(name=classes[k], train=int(train_count[k]), sensitive=sensitive, dream=dream, pics=pics)

    pop_rows = [label_row(k) for k in popular]
    picked_rows = [label_row(classes.index(n)) for n in a.picked]

    # ---- draw ---------------------------------------------------------------------------------------------
    T, g, pad = a.tile, 6, 36
    T2 = int(T * 1.25)                                            # tiles in panels C and D
    h1, h2, h3 = font(30, True), font(22, True), font(19)
    row_b = 5 * (T + g) + 14                                      # dream, gap, 4 pictures
    block_w, block_h = row_b + 120, 34 + 2 * (T + g + 24)
    b_w, b_h = 2 * block_w + 30, 2 * block_h + 20
    a_side = b_h - 60
    text_w = 300
    row_cd = text_w + 4 * (T2 + g) + 14
    width = max(pad + a_side + 50 + b_w + pad, pad + 2 * row_cd + 60 + pad)
    cd_h = 84 + max(3, len(picked_rows)) * (T2 + 18)
    height = pad + 44 + b_h + 40 + cd_h + pad
    img = Image.new("RGB", (width, height), "white")
    dr = ImageDraw.Draw(img)

    # A: first-layer filters
    x0, y0 = pad, pad
    dr.text((x0, y0), "A  First-layer filters (all 64)", font=h1, fill="black")
    fl = Image.open(fdir / "first_layer.png").convert("RGB").resize((a_side, a_side), Image.NEAREST)
    img.paste(fl, (x0, y0 + 50))
    dr.text((x0, y0 + 56 + a_side), "Each 7 x 7 colour filter, stretched to its own range", font=h3, fill=(90, 90, 90))

    # B: two filters per stage
    bx, by = pad + a_side + 50, pad
    dr.text((bx, by), "B  The two most selective filters of each stage: dream, then the 4 training pictures "
                      "it fires on most", font=h1, fill="black")
    fsheets = {k: Image.open(ddir / f"dream_stage{k}.png") for k in range(1, 5)}
    for n, (k, total, rows) in enumerate(stages):
        sx = bx + (n % 2) * (block_w + 30)
        sy = by + 50 + (n // 2) * (block_h + 20)
        dr.text((sx, sy), f"Stage {k} ({total} filters)", font=h2, fill="black")
        for r, (f, i, pics) in enumerate(rows):
            ry = sy + 34 + r * (T + g + 24)
            dr.text((sx, ry), f"filter {f}", font=h3, fill=(90, 90, 90))
            ry += 24
            img.paste(framed(cut(fsheets[k], i, FILTER_SHEET), T), (sx, ry))
            for c, pid in enumerate(pics):
                img.paste(framed(picture(pid), T), (sx + T + g + 14 + c * (T + g), ry))

    # C and D: labels, each as its dream and 3 of its own training pictures
    cy = pad + 44 + b_h + 40
    panels = [("C  Popular aesthetics (most training pictures)", pop_rows)]
    if picked_rows:
        panels.append(("D  Aesthetics picked by hand", picked_rows))
    for col, (title, rows) in enumerate(panels):
        cx = pad + col * (row_cd + 60)
        dr.text((cx, cy), title, font=h1, fill="black")
        dr.text((cx, cy + 42), "dream, then 3 of its training pictures", font=h3, fill=(90, 90, 90))
        for r, row in enumerate(rows):
            ry = cy + 84 + r * (T2 + 18)
            dr.text((cx, ry + 4), row["name"], font=h2, fill="black")
            note = f"{row['train']} training pictures"
            if row["sensitive"]:
                note += "\nreal pictures not shown:\nthe wiki tags it as sensitive"
            dr.multiline_text((cx, ry + 38), note, font=h3, fill=(90, 90, 90), spacing=4)
            x = cx + text_w
            img.paste(framed(row["dream"], T2), (x, ry))
            for c, pid in enumerate(row["pics"]):
                img.paste(framed(picture(pid), T2), (x + T2 + g + 14 + c * (T2 + g), ry))

    out = Path(a.out) if a.out else root / "runs" / "cnn" / "figure" / f"{network}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} exists; choose another --out (nothing is overwritten)")
    img.save(out, optimize=True)
    record = {"network": network, "final": final, "weights_sha256": chosen.get("weights_sha256"),
              "training_pictures": "train + validation" if final else "train",
              "stages": [{"stage": k, "filters_in_stage": t, "filters": [{"filter": f, "pictures": p}
                                                                         for f, _, p in rows]}
                         for k, t, rows in stages],
              "popular": [{k: v for k, v in r.items() if k != "dream"} for r in pop_rows],
              "picked_by_hand": [{k: v for k, v in r.items() if k != "dream"} for r in picked_rows],
              "sensitive_labels_left_out": len(hide)}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"written: {out} ({img.size[0]} x {img.size[1]} px) and {out.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
