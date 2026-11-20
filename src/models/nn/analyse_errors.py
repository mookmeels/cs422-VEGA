"""Error analysis of the NN on the VALIDATION split: where it is wrong, and what the mistakes look like.

    python analyse_errors.py              # reads the NN's saved validation predictions, writes runs/nn/errors/
    python analyse_errors.py --sheets     # also draws contact sheets of example mistakes (needs the images)

Reads the predictions that train_nn.py saved in <root>/runs/comparison/preds/ (one file per seed) and the
nearest-centroid baseline's file next to them. Never reads the test split: it loads the validation labels
through vega_data.load(), which does not hand out test labels.

Writes to <out> (default <root>/runs/nn/errors):
  summary.md        the tables below, in the order a report would use them
  per_label.csv     every label: training images, validation images, how often it is found, what it is mistaken for
  confusions.csv    every (true label -> first guess) mistake, with how the wiki links the two labels
  images.csv        every validation image: true labels, how many seeds got it, the guesses, the file
  examples-*.png    with --sheets: example images the network gets wrong (labels the wiki tags as sensitive left out)

"Linked by the wiki" uses runs/aesthetic-hierarchy/pairs.csv (parent/sub-style, same family box, "related",
"overlaps with", preceded/succeeded by) and runs/aesthetic-hierarchy-audit/families.csv. A merged label is
linked to another if any of its member aesthetics is. Numbers are averages over the seeds unless stated.
"""
import argparse, sys, time, unicodedata
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_nn                                                 # noqa: E402,F401  (puts the shared code on the path)
import vega_data                                                # noqa: E402
import vega_score                                               # noqa: E402

BINS = [(1, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 10 ** 9)]      # training images of a label
CONF = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001]


def ranking(scores):
    """Labels best first for every image, with the scorer's rule for equal scores; and each label's position."""
    n, k = scores.shape
    tie = np.random.default_rng(vega_score.TIE_SEED).permutation(k)
    order = np.lexsort((np.broadcast_to(tie, (n, k)), -scores), axis=1)
    pos = np.empty_like(order)
    pos[np.arange(n)[:, None], order] = np.arange(k)[None, :]
    return order, pos


def read_preds(path, ids, classes):
    z = np.load(path, allow_pickle=False)
    if [str(c) for c in z["classes"]] != classes:
        raise SystemExit(f"{path}: labels differ from the loader's")
    where = {str(a): i for i, a in enumerate(z["ids"])}
    return np.asarray(z["scores"], np.float64)[[where[a] for a in ids]]


def wiki_links(root, labels, classes):
    """k x k: 2 = the wiki puts the two labels in one family or calls one a sub-style of the other,
    1 = a looser wiki link (related, overlaps, preceded/succeeded by), 0 = no link. None if the files are absent."""
    pairs, fams = root / "runs" / "aesthetic-hierarchy" / "pairs.csv", root / "runs" / "aesthetic-hierarchy-audit" / "families.csv"
    if not pairs.exists() or not fams.exists():
        return None
    m, _ = vega_data.label_map(labels, root)
    at = {c: i for i, c in enumerate(classes)}
    link = np.zeros((len(classes), len(classes)), np.int8)

    def mark(a, b, level):
        if a in m and b in m and m[a] != m[b]:
            i, j = at[m[a]], at[m[b]]
            link[i, j] = link[j, i] = max(link[i, j], level)

    cols = ["a", "b", "wiki_parent", "wiki_section", "wiki_family", "wiki_overlaps", "wiki_related", "wiki_lineage"]
    for a, b, par, sec, fam, ov, rel, lin in pl.read_csv(pairs, infer_schema=False).select(cols).iter_rows():
        if par or sec or fam:
            mark(a, b, 2)
        elif ov == "true" or rel or lin == "true":
            mark(a, b, 1)
    for parent, children in pl.read_csv(fams, infer_schema=False).select("parent", "children").iter_rows():
        members = [parent] + [c.strip() for c in (children or "").split(";") if c.strip()]
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                mark(a, b, 2)
    return link


def sensitive_labels(root, labels, classes):
    path = root / "runs" / "catalog-audit" / "aesthetics.csv"
    if not path.exists():
        return set()
    m, _ = vega_data.label_map(labels, root)
    cat = pl.read_csv(path, infer_schema=False)
    return {m[a] for a, s in zip(cat["aesthetic"].to_list(), cat["sensitive"].to_list()) if s == "true" and a in m}


def pct(x):
    return f"{100 * x:.1f}"


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def sheet(path, items, title, cols=4, cell=(300, 300), cap=74):
    """items: (image file, [caption lines]). One PNG grid."""
    from PIL import Image, ImageDraw, ImageFont
    font, big = ImageFont.load_default(size=13), ImageFont.load_default(size=18)
    rows = (len(items) + cols - 1) // cols
    w, h = cell[0] + 16, cell[1] + cap + 12
    im = Image.new("RGB", (cols * w, rows * h + 40), "white")
    draw = ImageDraw.Draw(im)
    draw.text((8, 8), title, fill="black", font=big)
    for n, (file, lines) in enumerate(items):
        x, y = (n % cols) * w + 8, (n // cols) * h + 40
        try:
            t = Image.open(file).convert("RGB")
            t.thumbnail(cell)
            im.paste(t, (x + (cell[0] - t.width) // 2, y + (cell[1] - t.height) // 2))
        except Exception as e:                                   # an unreadable file should not stop the sheet
            draw.text((x, y + 20), f"(cannot open: {type(e).__name__})", fill="red", font=font)
        for i, line in enumerate(lines):
            line = unicodedata.normalize("NFKD", line).encode("ascii", "ignore").decode()    # the built-in font has no accents
            while draw.textlength(line, font=font) > cell[0] and len(line) > 4:
                line = line[:-4].rstrip() + "..."
            draw.text((x, y + cell[1] + 4 + 17 * i), line, fill=("black", "#8a1c1c", "#444444", "#444444")[min(i, 3)], font=font)
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="merged", choices=("merged", "wearable"))
    ap.add_argument("--pattern", default=None, help="prediction files; default nn-*-validation-<labels>-seed*.npz")
    ap.add_argument("--preds", default=None, help="folder with the prediction files; default <root>/runs/comparison/preds")
    ap.add_argument("--root", default=None, help="default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--out", default=None, help="default <root>/runs/nn/errors")
    ap.add_argument("--min-val", type=int, default=3, help="a label needs this many validation images to be listed among the worst")
    ap.add_argument("--sheets", action="store_true", help="draw contact sheets of example mistakes")
    a = ap.parse_args()
    t0 = time.perf_counter()
    root = Path(a.root or vega_data.ROOT)
    preds = Path(a.preds) if a.preds else root / "runs" / "comparison" / "preds"
    out = Path(a.out) if a.out else root / "runs" / "nn" / "errors"
    out.mkdir(parents=True, exist_ok=True)

    d = vega_data.load(labels=a.labels, root=root, vectors=False)          # validation; no vectors needed
    classes, ev = d["classes"], d["eval"]
    ids, truth, n, k = ev["ids"], ev["truth"], len(ev["ids"]), len(d["classes"])
    fit_count = (d["fit"]["y"] > 0).sum(0)
    val_count = np.zeros(k, int)
    for t in truth:
        val_count[t] += 1
    files = sorted(preds.glob(a.pattern or f"nn-*-validation-{a.labels}-seed*.npz"))
    if not files:
        raise SystemExit(f"no NN prediction files in {preds}")
    kinds = {f.name.split("-validation-")[0] for f in files}
    if len(kinds) > 1:
        raise SystemExit(f"{preds} holds predictions of several networks ({', '.join(sorted(kinds))}); choose one with --pattern")
    seeds = [f.stem.split("seed")[-1] for f in files]
    S = len(files)
    scores = [read_preds(f, ids, classes) for f in files]
    link = wiki_links(root, a.labels, classes)
    is_true = np.zeros((n, k), bool)
    for i, t in enumerate(truth):
        is_true[i, t] = True

    # ---- per seed ----------------------------------------------------------------------------------------
    rank = np.zeros((S, n), int)             # position of the best true label, 1 = first
    first = np.zeros((S, n), int)            # the first guess
    conf = np.zeros((S, n))                  # softmax share given to the first guess
    lab5, lab1 = np.zeros((S, k)), np.zeros((S, k))          # per label: found in the top 5 / as first guess
    for s, sc in enumerate(scores):
        order, pos = ranking(sc)
        rank[s] = np.where(is_true, pos, k).min(1) + 1
        first[s] = order[:, 0]
        p = np.exp(sc - sc.max(1, keepdims=True))
        conf[s] = (p / p.sum(1, keepdims=True))[np.arange(n), first[s]]
        lab5[s] = (is_true & (pos < 5)).sum(0)
        lab1[s] = (is_true & (pos < 1)).sum(0)
    hit5, hit1 = rank <= 5, rank == 1
    mean_scores = np.mean(scores, 0)
    m_order, _ = ranking(mean_scores)

    base_file = preds / f"baseline-nearest-centroid-validation-{a.labels}.npz"
    base = None
    if base_file.exists():
        b_order, b_pos = ranking(read_preds(base_file, ids, classes))
        b_rank = np.where(is_true, b_pos, k).min(1) + 1
        base = {"hit5": b_rank <= 5, "hit1": b_rank == 1, "lab5": (is_true & (b_pos < 5)).sum(0), "lab1": (is_true & (b_pos < 1)).sum(0)}

    text = [f"# NN error analysis, validation split ({n} images, {k} labels, label set \"{a.labels}\")", "",
            f"Network: {files[0].name.split('-validation-')[0]}, seeds {', '.join(seeds)}; image vectors emb/{getattr(vega_data, 'VECTORS', 'qwen2b')}-448. "
            "Figures are averages over the seeds unless stated. The test split is not read.", ""]

    # ---- 1. how many, and how stable ---------------------------------------------------------------------
    got = hit5.sum(0)
    text += ["## 1. How many images are missed, and is it the same images every time?", ""]
    text += table(["", "top-5", "top-1"], [["right, average over seeds", pct(hit5.mean()), pct(hit1.mean())],
                                             [f"right in all {S} seeds", pct((got == S).mean()), pct((hit1.sum(0) == S).mean())],
                                             ["right in some seeds, wrong in others", pct(((got > 0) & (got < S)).mean()), pct(((hit1.sum(0) > 0) & (hit1.sum(0) < S)).mean())],
                                             [f"wrong in all {S} seeds", pct((got == 0).mean()), pct((hit1.sum(0) == 0).mean())]])
    text += [f"{int((got == 0).sum())} images never have a true label in the top 5; {int(((got > 0) & (got < S)).sum())} depend on the seed.", ""]

    # ---- 2. how far off --------------------------------------------------------------------------------
    miss = ~hit5
    text += ["## 2. When the top 5 misses, where is the true label?", ""]
    cuts = [(6, 10), (11, 20), (21, 50), (51, k)]
    text += table(["position of the best true label", "share of the missed images"],
                  [[f"{lo} to {hi}", pct(((rank >= lo) & (rank <= hi) & miss).sum() / miss.sum())] for lo, hi in cuts])
    text += [f"Top-10 accuracy {pct((rank <= 10).mean())}, top-20 {pct((rank <= 20).mean())}.", ""]

    # ---- 3. by label size ------------------------------------------------------------------------------
    text += ["## 3. Does it depend on how many training images a label has?", "",
             "Per label: the share of its validation images where that label is in the top 5 (or is the first guess), averaged over the labels in the row.", ""]
    rows = []
    seen = val_count > 0
    for lo, hi in BINS:
        sel = seen & (fit_count >= lo) & (fit_count <= hi)
        if not sel.any():
            continue
        r5, r1 = (lab5[:, sel] / val_count[sel]).mean(), (lab1[:, sel] / val_count[sel]).mean()
        row = [f"{lo} to {hi}" if hi < 10 ** 9 else f"{lo} or more", int(sel.sum()), int(val_count[sel].sum()), pct(r5), pct(r1)]
        if base:
            row += [pct((base["lab5"][sel] / val_count[sel]).mean()), pct((base["lab1"][sel] / val_count[sel]).mean())]
        rows.append(row)
    text += table(["training images of the label", "labels", "validation images", "NN top-5", "NN first guess"]
                  + (["nearest centroid top-5", "nearest centroid first guess"] if base else []), rows)
    multi = np.asarray([len(t) > 1 for t in truth])
    text += [f"Images with one label ({int((~multi).sum())}): top-5 {pct(hit5[:, ~multi].mean())}. "
             f"Images with two or more labels ({int(multi.sum())}): top-5 {pct(hit5[:, multi].mean())} (any one of their labels counts).", ""]

    # ---- 4. what the wrong first guesses are -------------------------------------------------------------
    wrong = ~hit1
    conf_rows, pair = [], {}
    for s in range(S):
        for i in np.flatnonzero(wrong[s]):
            g = first[s, i]
            for t in truth[i]:
                pair[(t, g)] = pair.get((t, g), 0.0) + 1 / len(truth[i])
    text += ["## 4. What are the wrong first guesses?", ""]
    if link is not None:
        level = np.asarray([[max(link[t, first[s, i]] for t in truth[i]) for i in range(n)] for s in range(S)])
        chance = [(link[np.triu_indices(k, 1)] == v).mean() for v in (2, 1, 0)]
        text += table(["the first guess is...", "share of the wrong first guesses", "share of all label pairs (chance)"],
                      [["in the same wiki family as a true label, or its parent / sub-style", pct((level[wrong] == 2).mean()), pct(chance[0])],
                       ["linked more loosely by the wiki (related, overlaps, came before / after)", pct((level[wrong] == 1).mean()), pct(chance[1])],
                       ["not linked by the wiki", pct((level[wrong] == 0).mean()), pct(chance[2])]])
    text += [f"When the first guess is wrong, a true label is still in the top 5 for {pct((hit5 & wrong).sum() / wrong.sum())} of those images.", ""]

    def kind(t, g):
        return "" if link is None else ("family / sub-style", "related", "no wiki link")[2 - int(link[t, g])]

    top = sorted(pair.items(), key=lambda kv: -kv[1])
    text += ["Most frequent mistakes (true label -> first guess). \"Images per seed\" is the average number of validation images per seed.", ""]
    text += table(["true label", "first guess", "images per seed", "validation images of the true label", "wiki link"],
                  [[classes[t], classes[g], f"{c / S:.1f}", int(val_count[t]), kind(t, g)] for (t, g), c in top[:25]])
    pl.DataFrame({"true_label": [classes[t] for (t, g), _ in top], "first_guess": [classes[g] for (t, g), _ in top],
                  "images_per_seed": [round(c / S, 3) for _, c in top], "validation_images_of_true_label": [int(val_count[t]) for (t, g), _ in top],
                  "training_images_of_true_label": [int(fit_count[t]) for (t, g), _ in top], "training_images_of_guess": [int(fit_count[g]) for (t, g), _ in top],
                  "wiki_link": [kind(t, g) for (t, g), _ in top]}).write_csv(out / "confusions.csv")

    # ---- 5. labels ---------------------------------------------------------------------------------------
    r5 = np.where(seen, lab5.mean(0) / np.maximum(val_count, 1), np.nan)
    r1 = np.where(seen, lab1.mean(0) / np.maximum(val_count, 1), np.nan)
    worst_guess = {}
    for (t, g), c in top:
        worst_guess.setdefault(t, (g, c))
    wrongly = np.zeros(k)
    for s in range(S):
        np.add.at(wrongly, first[s][wrong[s]], 1 / S)
    text += ["## 5. Which labels does it find worst?", "",
             f"{int((seen & (r5 == 0)).sum())} of the {int(seen.sum())} labels with validation images are never in the top 5 for any of their images in any seed "
             f"({int((seen & (r5 == 0) & (val_count == 1)).sum())} of them have a single validation image); {int((seen & (r5 == 1)).sum())} are always found.", "",
             f"Lowest top-5 rate among labels with at least {a.min_val} validation images:", ""]
    cand = [j for j in np.argsort(np.where(seen & (val_count >= a.min_val), r5, np.inf), kind="stable")[:20] if seen[j] and val_count[j] >= a.min_val]
    text += table(["label", "training images", "validation images", "NN top-5", "NN first guess"] + (["nearest centroid top-5"] if base else []) + ["most often mistaken for"],
                  [[classes[j], int(fit_count[j]), int(val_count[j]), pct(r5[j]), pct(r1[j])] + ([pct(base["lab5"][j] / val_count[j])] if base else [])
                   + [classes[worst_guess[j][0]] if j in worst_guess else ""] for j in cand])
    text += ["Labels most often given as a WRONG first guess (the labels that soak up mistakes):", ""]
    text += table(["label", "wrong first guesses per seed", "training images", "its own top-5 rate"],
                  [[classes[j], f"{wrongly[j]:.1f}", int(fit_count[j]), pct(r5[j]) if seen[j] else "no validation image"] for j in np.argsort(-wrongly)[:15]])
    big = np.argsort(-fit_count)[:k // 10]
    text += [f"The {len(big)} labels with the most training images (a tenth of the labels) hold {pct(fit_count[big].sum() / fit_count.sum())} of the training label "
             f"assignments and receive {pct(wrongly[big].sum() / wrongly.sum())} of the wrong first guesses.", ""]
    pl.DataFrame({"label": classes, "training_images": fit_count.tolist(), "validation_images": val_count.tolist(),
                  "nn_top5": [None if np.isnan(v) else round(float(v), 4) for v in r5], "nn_first_guess": [None if np.isnan(v) else round(float(v), 4) for v in r1],
                  "centroid_top5": [round(float(base["lab5"][j] / val_count[j]), 4) if base and seen[j] else None for j in range(k)],
                  "wrong_first_guesses_received_per_seed": [round(float(v), 2) for v in wrongly],
                  "most_often_mistaken_for": [classes[worst_guess[j][0]] if j in worst_guess else "" for j in range(k)]}).write_csv(out / "per_label.csv")

    # ---- 6. confidence -----------------------------------------------------------------------------------
    text += ["## 6. Can the network's confidence be trusted?", "",
             "Confidence = the softmax share the network gives its first guess.", ""]
    rows = []
    for lo, hi in zip(CONF[:-1], CONF[1:]):
        sel = (conf >= lo) & (conf < hi)
        if sel.any():
            rows.append([f"{lo:.1f} to {min(hi, 1.0):.1f}", pct(sel.mean()), f"{conf[sel].mean():.2f}", f"{hit1[sel].mean():.2f}", f"{hit5[sel].mean():.2f}"])
    text += table(["confidence", "share of images", "average confidence", "first guess right", "true label in top 5"], rows)
    ece = 0.0                                                    # expected calibration error, ten equal-width bins
    for b in range(10):
        sel = (conf >= b / 10) & ((conf < (b + 1) / 10) | (b == 9))
        if sel.any():
            ece += sel.mean() * abs(conf[sel].mean() - hit1[sel].mean())
    text += [f"Average confidence {conf.mean():.2f} against {hit1.mean():.2f} of first guesses right (gap {conf.mean() - hit1.mean():+.2f}; expected calibration error {ece:.3f}). "
             f"Average confidence when right {conf[hit1].mean():.2f}, when wrong {conf[wrong].mean():.2f}.", ""]

    # ---- 7. against the baseline, and the seeds together ---------------------------------------------------
    if base:
        text += ["## 7. Image by image against nearest centroid", ""]
        both = lambda x, y: f"{(x & y).sum() / S:.0f}"
        text += table(["images (average per seed)", "top-5", "first guess"],
                      [["both right", both(hit5, base["hit5"][None]), both(hit1, base["hit1"][None])],
                       ["only the NN right", both(hit5, ~base["hit5"][None]), both(hit1, ~base["hit1"][None])],
                       ["only nearest centroid right", both(~hit5, base["hit5"][None]), both(~hit1, base["hit1"][None])],
                       ["both wrong", both(~hit5, ~base["hit5"][None]), both(~hit1, ~base["hit1"][None])]])
    m = vega_score.metrics(mean_scores, truth, fit_count)
    text += [f"## 8. The {S} networks averaged", "",
             f"Averaging the {S} networks' scores per image gives top-5 {pct(m['top5'])} and top-1 {pct(m['top1'])} "
             f"(single networks: {pct(hit5.mean())} and {pct(hit1.mean())}). This is not a row in the comparison table.", ""]

    # ---- images.csv and the example sheets -----------------------------------------------------------------
    modal = [int(np.bincount(first[:, i]).argmax()) for i in range(n)]
    pl.DataFrame({"sha1": [str(x) for x in ids], "true_labels": ["; ".join(classes[j] for j in t) for t in truth],
                  "seeds_with_true_label_in_top5": got.tolist(), "seeds_with_first_guess_right": hit1.sum(0).tolist(),
                  "average_position_of_true_label": [round(float(v), 1) for v in rank.mean(0)],
                  "most_common_first_guess": [classes[j] for j in modal], "average_confidence": [round(float(v), 3) for v in conf.mean(0)],
                  "top5_of_the_averaged_networks": ["; ".join(classes[j] for j in m_order[i, :5]) for i in range(n)],
                  "file": ev["paths"]}).write_csv(out / "images.csv")

    if a.sheets:
        hide = sensitive_labels(root, a.labels, classes)
        ok = lambda i: not ({classes[j] for j in truth[i]} | {classes[j] for j in m_order[i, :3]}) & hide

        def caption(i):
            g = m_order[i, :3]
            return ["true: " + ", ".join(classes[j] for j in truth[i]), f"guess: {classes[modal[i]]} ({conf[:, i].mean():.0%})",
                    "then: " + ", ".join(classes[j] for j in g if j != modal[i])[:80], f"true label at position {rank[:, i].mean():.0f}"]

        sure = [i for i in np.argsort(-conf.mean(0)) if hit1[:, i].sum() == 0 and ok(i)][:12]
        far = [i for i in np.argsort(-rank.mean(0)) if got[i] == 0 and ok(i)][:12]
        near = [i for i in np.argsort(rank.mean(0)) if hit1[:, i].sum() == 0 and got[i] == S and link is not None
                and max(link[t, modal[i]] for t in truth[i]) == 2 and ok(i)][:12]
        for name, pick, title in (("examples-confident-and-wrong.png", sure, f"Wrong in all {S} seeds with the highest confidence"),
                                  ("examples-far-off.png", far, f"True label furthest from the top (never in the top 5 in {S} seeds)"),
                                  ("examples-near-miss.png", near, "Near misses: first guess is in the same wiki family as the true label")):
            if pick:
                sheet(out / name, [(ev["paths"][i], caption(i)) for i in pick], title + " (validation images)")
        text += ["## Example images", "", "`examples-confident-and-wrong.png`, `examples-far-off.png`, `examples-near-miss.png`: 12 validation images each. "
                 f"Images whose true or guessed labels the wiki tags as sensitive are left out ({len(hide)} labels).", ""]

    (out / "summary.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print(f"written to {out} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
