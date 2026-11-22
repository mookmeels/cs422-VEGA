r"""The map of aesthetics as a picture: 503 aesthetics laid out so that ones that look alike sit close together.

    python aesthetic_map.py coords        (a Python with scikit-learn)   -> runs/aesthetic-map/coords.csv
    python aesthetic_map.py plot          (a Python with matplotlib)     -> runs/aesthetic-map/map.png
(On the PC these were two different environments, which is why the script has two steps.)

How alike two aesthetics look is code/app/vega_profile.aesthetic_map: computed from the wiki TRAINING images' vectors only
(the same numbers the app uses for "nearby" and "surprising"). The layout is t-SNE on 1 - similarity (perplexity 30,
seed 422). Colours are the 20 style families of runs/family-v1 (made by hand, not by this layout), so the picture also
shows how far the hand-made families agree with what the vectors see. Neither validation nor test images are read.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path("E:/vega")
OUT = ROOT / "runs" / "aesthetic-map"
OUT.mkdir(parents=True, exist_ok=True)
read = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))

if sys.argv[1] == "coords":
    from sklearn.manifold import TSNE
    sys.path.insert(0, str(ROOT / "code" / "shared"))
    sys.path.insert(0, str(ROOT / "code" / "app"))
    import vega_data
    import vega_profile as vp
    d = vega_data.load(root=ROOT)
    classes = [str(c) for c in d["classes"]]
    fx = np.asarray(d["fit"]["x"], np.float64)
    y = np.asarray(d["fit"]["y"]) > 0
    sim = vp.aesthetic_map(fx / np.linalg.norm(fx, axis=1, keepdims=True), y)
    dist = np.clip(1 - (sim + sim.T) / 2, 0, None)
    np.fill_diagonal(dist, 0)
    xy = TSNE(n_components=2, metric="precomputed", init="random", perplexity=30, random_state=422).fit_transform(dist)
    fam = {r["wiki_label"]: r["family"] for r in read(ROOT / "runs" / "family-v1" / "label_family.csv")}
    name = {r["family"]: r["name"] for r in read(ROOT / "runs" / "family-v1" / "families.csv")}
    lab_fam = np.asarray([fam[c] for c in classes])
    # how often an aesthetic's nearest aesthetic on the map's own similarity is of its family
    s2 = sim.copy()
    np.fill_diagonal(s2, -np.inf)
    same = float((lab_fam[s2.argmax(1)] == lab_fam).mean())
    with open(OUT / "coords.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aesthetic", "family", "family_name", "x", "y", "training_images"])
        for c, (a, b), n in zip(classes, xy, y.sum(0)):
            w.writerow([c, fam[c], name[fam[c]], f"{a:.3f}", f"{b:.3f}", int(n)])
    (OUT / "note.txt").write_text(f"nearest aesthetic is of the same family for {same:.1%} of the 503 aesthetics (20 families; chance is about "
                                  f"{float(np.mean([(lab_fam == f).mean() for f in lab_fam])):.1%})\n", encoding="utf-8")
    print(open(OUT / "note.txt").read())
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read(OUT / "coords.csv")
    names = sorted({r["family_name"] for r in rows})
    cmap = plt.get_cmap("tab20")
    colour = {n: cmap(i % 20) for i, n in enumerate(names)}
    fig, ax = plt.subplots(figsize=(15, 10.5), dpi=170)
    for n in names:
        sel = [r for r in rows if r["family_name"] == n]
        ax.scatter([float(r["x"]) for r in sel], [float(r["y"]) for r in sel], s=[14 + 1.2 * min(int(r["training_images"]), 120) for r in sel],
                   color=colour[n], alpha=0.85, linewidths=0.3, edgecolors="white", label=f"{n} ({len(sel)})")
    big = sorted(rows, key=lambda r: -int(r["training_images"]))[:45]
    for r in big:
        ax.annotate(r["aesthetic"], (float(r["x"]), float(r["y"])), fontsize=7.5, xytext=(3, 3), textcoords="offset points", color="#222222")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("The 503 aesthetics, placed by how alike their training images look (t-SNE); colour = hand-made style family; "
                 "the 45 largest are named", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5, frameon=False, title="Style family (aesthetics)")
    fig.tight_layout()
    fig.savefig(OUT / "map.png")
    print("written", OUT / "map.png")
