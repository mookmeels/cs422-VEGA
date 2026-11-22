"""CLIP against the team's Qwen vectors. The wiki test split and FashionStyle14's transfer-test styles are never read.

    python clip_compare.py --embed --weights <ViT-*.pt> --name clip-vitl14        # wiki training + validation images
    python clip_compare.py --score clip-vitb32 clip-vitl14                        # wiki validation -> runs/clip-compare/results.md
    python clip_compare.py --embed-fs14 --weights <ViT-*.pt> --name clip-vitl14   # FashionStyle14, the 11 non-transfer styles
    python clip_compare.py --fs14 clip-vitb32 clip-vitl14                         # real outfit photos -> runs/clip-compare/fs14.md
    (all: [--root E:/vega] [--images E:/pack-test/wiki] [--fs14-images E:/vega/datasets/fashionstyle14] [--seeds 5])

Why: the proposal named CLIP ViT-B/32 and the project uses Qwen3-VL-Embedding-2B. This measures the difference instead of
asserting it.

On the wiki (--score), same 9,359 training and 1,351 validation images and the same code for every encoder:
  nearest centroid   each label is the average of its training vectors; no setting to tune, so it favours no encoder
  NN (wiki only)     train_nn.train with the frozen settings and no street photos. NOTE: those settings were tuned on Qwen
                     vectors, so this row leans toward Qwen; the nearest-centroid row does not.
  speed              images per second on this GPU, end to end (decode, resize, encode), for CLIP as run here

On real outfit photos (--fs14), FashionStyle14's 11 non-transfer styles (lolita, fairy and ethnic are not read):
  its own styles     5 folds within the photos (seed 422), no wiki image involved: nearest centroid and 10 nearest
                     neighbours. Nothing to tune. This asks how well the encoder separates expert-labelled outfit styles.
  from the wiki      the wiki-only NN (frozen settings) scores each photo; the family with the largest summed probability
                     against the style's family (runs/family-v1). This asks how well wiki training carries over.

CLIP is run from OpenAI's published TorchScript weights with torch.jit.load: no CLIP library is installed. Preprocessing as
in OpenAI's code: resize the short side to 224 (bicubic), centre crop 224, RGB, normalise with CLIP's mean and deviation.
"""
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

T0 = time.time()
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROOT = Path(arg("--root", "E:/vega"))
IMAGES = Path(arg("--images", "E:/pack-test/wiki"))
FS_IMAGES = Path(arg("--fs14-images", "E:/vega/datasets/fashionstyle14"))
SEEDS = [422 + i for i in range(int(arg("--seeds", 5)))]
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "nn"))
import train_nn  # noqa: E402
import vega_data  # noqa: E402
import vega_score  # noqa: E402

log = lambda *a: print(*a, flush=True)
read_csv = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
TRANSFER_STYLES = {"lolita", "fairy", "ethnic"}
QWEN = "Qwen3-VL-Embedding-2B (the team's vectors)"


def names_after(flag):
    out = []
    for a in sys.argv[sys.argv.index(flag) + 1:]:                    # the names up to the next option
        if a.startswith("--"):
            break
        out.append(a)
    return out


def clip_vectors(weights, paths):
    """CLIP image vectors (unit length) for these files, and the seconds it took end to end."""
    from PIL import Image
    model = torch.jit.load(str(weights), map_location="cuda").eval()
    half = next(model.parameters()).dtype == torch.float16
    mean, std = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1), torch.tensor(STD, device="cuda").view(1, 3, 1, 1)

    def read(p):
        im = Image.open(p)
        im.seek(0)
        im = im.convert("RGB")
        w, h = im.size
        s = 224 / min(w, h)
        im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
        w, h = im.size
        left, top = (w - 224) // 2, (h - 224) // 2
        return np.asarray(im.crop((left, top, left + 224, top + 224)), np.uint8)

    out, t1 = [], time.time()
    with ThreadPoolExecutor(8) as pool, torch.no_grad():
        for i in range(0, len(paths), 128):
            batch = np.stack(list(pool.map(read, paths[i:i + 128])))
            x = (torch.from_numpy(batch).cuda().permute(0, 3, 1, 2).float() / 255 - mean) / std
            z = model.encode_image(x.half() if half else x).float()
            out.append(torch.nn.functional.normalize(z, dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32), time.time() - t1


def save(folder, vec, ids, secs, weights, what):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "embeddings.npy", vec)
    (folder / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (folder / "meta.json").write_text(json.dumps({"weights": str(weights), "images": len(ids), "dim": int(vec.shape[1]), "seconds": round(secs, 1),
                                                  "images_per_second": round(len(ids) / secs, 1), "gpu": torch.cuda.get_device_name(0), "what": what}, indent=1), encoding="utf-8")
    log(f"{folder.name}: {vec.shape[0]:,} x {vec.shape[1]} vectors in {secs:.0f}s = {len(ids) / secs:.0f} images per second, end to end on {torch.cuda.get_device_name(0)}")


def fs14_photos():
    """(sha1, style, file) of every FashionStyle14 photo of the 11 non-transfer styles that has one style and a Qwen vector."""
    fdir = ROOT / "emb" / "fashionstyle14"
    have = set((fdir / f"{vega_data.VECTORS}-448" / "ids.txt").read_text(encoding="utf-8").split())
    styles, path = {}, {}
    for r in read_csv(fdir / "manifest.csv"):
        styles.setdefault(r["sha1"], set()).add(r["style"])
        path[r["sha1"]] = r["path"]
    return sorted((s, next(iter(st)), FS_IMAGES / path[s]) for s, st in styles.items()
                  if len(st) == 1 and s in have and next(iter(st)) not in TRANSFER_STYLES)


def paired(a, b, draws):
    d = (np.asarray(a, np.float64) - np.asarray(b, np.float64)) * 100
    boot = np.asarray([d[i].mean() for i in draws])
    return f"{d.mean():+.1f} ({np.percentile(boot, 2.5):+.1f} to {np.percentile(boot, 97.5):+.1f})"


if "--embed" in sys.argv:
    name, weights = arg("--name", None), Path(arg("--weights", ""))
    assert name and weights.exists(), "--name and --weights are needed"
    d = vega_data.load(root=ROOT, vectors=False)                      # training and validation only
    ids = [str(a) for a in list(d["fit"]["ids"]) + list(d["eval"]["ids"])]
    paths = [IMAGES / Path(str(p)).relative_to(ROOT / "wiki") for p in list(d["fit"]["paths"]) + list(d["eval"]["paths"])]
    assert all(p.exists() for p in paths), "an image is missing"
    vec, secs = clip_vectors(weights, paths)
    save(ROOT / "emb" / f"{name}-224", vec, ids, secs, weights, "wiki training and validation images only; test split not read")

if "--embed-fs14" in sys.argv:
    name, weights = arg("--name", None), Path(arg("--weights", ""))
    assert name and weights.exists(), "--name and --weights are needed"
    photos = fs14_photos()
    assert all(p.exists() for _, _, p in photos), "a FashionStyle14 photo is missing"
    vec, secs = clip_vectors(weights, [p for _, _, p in photos])
    save(ROOT / "emb" / "fashionstyle14" / f"{name}-224", vec, [s for s, _, _ in photos], secs, weights,
         "FashionStyle14, the 11 non-transfer styles; lolita, fairy and ethnic not read")

if "--score" in sys.argv:
    names = names_after("--score")
    d = vega_data.load(root=ROOT)
    fit, ev = d["fit"], d["eval"]
    classes = [str(c) for c in d["classes"]]
    _ids, truth, _c, FIT_COUNT = vega_data._eval_truth(False, "merged", ROOT)
    assert [str(a) for a in _ids] == [str(a) for a in ev["ids"]]
    lab_fam, _ = vega_score.family_of(classes, ROOT)
    y = np.asarray(fit["y"], np.float64)
    sets = {QWEN: (np.asarray(fit["x"], np.float32), np.asarray(ev["x"], np.float32), None)}
    for n in names:
        folder = ROOT / "emb" / f"{n}-224"
        row = {s: i for i, s in enumerate((folder / "ids.txt").read_text(encoding="utf-8").split())}
        v = np.load(folder / "embeddings.npy")
        sets[n] = (v[[row[str(a)] for a in fit["ids"]]], v[[row[str(a)] for a in ev["ids"]]], json.loads((folder / "meta.json").read_text(encoding="utf-8")))
    s = dict(train_nn.SETTINGS, real_weight=0)
    L = ["# CLIP against Qwen on the wiki images (validation; the test split was not read)", "",
         f"Made by `code/experiments/clip_compare.py --score`. 9,359 training and 1,351 validation images, 503 labels. NN: train_nn.train, frozen settings, wiki only, seeds {SEEDS}, on the GPU.", "",
         "| encoder | numbers per image | nearest centroid: top-5 | top-1 | family top-1 | NN (wiki only): top-5 | top-1 | family top-1 | images per second |",
         "|---|---|---|---|---|---|---|---|---|"]
    PER = {}
    for n, (fx, ex, meta) in sets.items():
        fxn = fx / np.linalg.norm(fx, axis=1, keepdims=True)
        cen = (y.T @ fxn) / np.maximum(y.sum(0), 1e-12)[:, None]              # each label: the weighted average of its training vectors
        cen /= np.linalg.norm(cen, axis=1, keepdims=True) + 1e-12
        z = (ex / np.linalg.norm(ex, axis=1, keepdims=True)) @ cen.T
        m, fm = vega_score.metrics(z.astype(np.float64), truth, FIT_COUNT), vega_score.family_metrics(z.astype(np.float64), truth, lab_fam)
        rows, hit5, hit1 = [], [], []
        for seed in SEEDS:
            zz = train_nn.train(fx, fit["y"], ex, s, seed, "cuda").astype(np.float64)
            a, b = vega_score.metrics(zz, truth, FIT_COUNT), vega_score.family_metrics(zz, truth, lab_fam)
            rows.append([a["top5"], a["top1"], b["family_top1"]])
            order = vega_score.ranking(zz)[:, :5]
            hit5.append([bool(set(o.tolist()) & set(np.asarray(t).tolist())) for o, t in zip(order, truth)])
            hit1.append([int(o[0]) in set(np.asarray(t).tolist()) for o, t in zip(order, truth)])
        PER[n] = (np.mean(hit5, axis=0), np.mean(hit1, axis=0))               # per image: share of seeds right
        r = np.asarray(rows) * 100
        L.append(f"| {n} | {fx.shape[1]:,} | {m['top5']:.1%} | {m['top1']:.1%} | {fm['family_top1']:.1%} | "
                 + " | ".join(f"{r[:, k].mean():.1f} ± {r[:, k].std():.1f}" for k in range(3)) + f" | {meta['images_per_second'] if meta else ''} |")
        log(L[-1])
    rng = np.random.default_rng(422)
    draws = [rng.integers(0, len(truth), len(truth)) for _ in range(2000)]
    L += ["", f"## NN: each CLIP encoder minus Qwen, on the same validation images (per image, the share of the {len(SEEDS)} seeds that are right; paired bootstrap over images, 2,000 draws, seed 422)", "",
          "| encoder | top-5 difference (95% interval) | top-1 difference (95% interval) |", "|---|---|---|"]
    for n in names:
        L.append(f"| {n} | {paired(PER[n][0], PER[QWEN][0], draws)} | {paired(PER[n][1], PER[QWEN][1], draws)} |")
        log(L[-1])
    L += ["", "The NN's settings were tuned on Qwen vectors, so its columns lean toward Qwen. Nearest centroid has nothing to tune.", f"time elapsed {time.time() - T0:.0f}s"]
    out = ROOT / "runs" / "clip-compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"wrote {out / 'results.md'}; time elapsed {time.time() - T0:.0f}s")

if "--fs14" in sys.argv:
    names = names_after("--fs14")
    photos = fs14_photos()
    sha = [s for s, _, _ in photos]
    style = np.asarray([st for _, st, _ in photos])
    kinds = sorted(set(style.tolist()))
    lab = np.asarray([kinds.index(st) for st in style])
    fdir = ROOT / "emb" / "fashionstyle14"

    def vectors(folder):
        row = {s: i for i, s in enumerate((folder / "ids.txt").read_text(encoding="utf-8").split())}
        v = np.load(folder / "embeddings.npy", mmap_mode="r")
        v = np.ascontiguousarray(v[[row[s] for s in sha]]).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    sets = {QWEN: vectors(fdir / f"{vega_data.VECTORS}-448")}
    for n in names:
        sets[n] = vectors(fdir / f"{n}-224")
    rng = np.random.default_rng(422)
    fold = np.empty(len(lab), int)
    for k in range(len(kinds)):                                              # 5 folds, each style spread evenly
        idx = rng.permutation(np.flatnonzero(lab == k))
        fold[idx] = np.arange(len(idx)) % 5
    draws = [rng.integers(0, len(lab), len(lab)) for _ in range(2000)]

    # the wiki-only NN and the families, for the "from the wiki" column
    d = vega_data.load(root=ROOT)
    fit, classes = d["fit"], [str(c) for c in d["classes"]]
    fams = [r["family"] for r in read_csv(ROOT / "runs" / "family-v1" / "families.csv")]
    of_label = {r["wiki_label"]: r["family"] for r in read_csv(ROOT / "runs" / "family-v1" / "label_family.csv")}
    lab_fam = np.asarray([fams.index(of_label[c]) for c in classes])        # numbered as in families.csv, like the styles' families below
    fs_map ={r["style_or_tag"]: fams.index(r["family"]) for r in read_csv(ROOT / "runs" / "family-v1" / "outside_family.csv")
              if r["source"] == "FashionStyle14" and r["family"] in fams}
    want = np.asarray([fs_map[st] for st in style])
    wiki = {QWEN: np.asarray(fit["x"], np.float32)}
    for n in names:
        folder = ROOT / "emb" / f"{n}-224"
        row = {s: i for i, s in enumerate((folder / "ids.txt").read_text(encoding="utf-8").split())}
        wiki[n] = np.load(folder / "embeddings.npy")[[row[str(a)] for a in fit["ids"]]]
    s = dict(train_nn.SETTINGS, real_weight=0)

    PER, L = {}, ["# CLIP against Qwen on real outfit photos (FashionStyle14, 11 styles; lolita, fairy and ethnic not read)", "",
                  f"Made by `code/experiments/clip_compare.py --fs14`. {len(lab):,} photos. 'Its own styles': 5 folds within the photos, no wiki image. "
                  f"'From the wiki': the wiki-only NN (frozen settings, seeds {SEEDS}) and the style's family.", "",
                  "| encoder | its own styles: nearest centroid | 10 nearest neighbours | from the wiki: family right |", "|---|---|---|---|"]
    for n, v in sets.items():
        cen_hit, knn_hit = np.zeros(len(lab)), np.zeros(len(lab))
        for f in range(5):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            cen = np.stack([v[tr][lab[tr] == k].mean(0) for k in range(len(kinds))])
            cen /= np.linalg.norm(cen, axis=1, keepdims=True)
            cen_hit[te] = (v[te] @ cen.T).argmax(1) == lab[te]
            sim = torch.from_numpy(v[te]).cuda() @ torch.from_numpy(v[tr]).cuda().T
            near = lab[tr][sim.topk(10, dim=1).indices.cpu().numpy()]
            votes = np.stack([(near == k).sum(1) for k in range(len(kinds))], 1)
            knn_hit[te] = votes.argmax(1) == lab[te]
        fam_hit = []
        for seed in SEEDS:
            z = train_nn.train(wiki[n], fit["y"], v, s, seed, "cuda").astype(np.float64)
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            pf = np.zeros((len(p), len(fams)))
            np.add.at(pf.T, lab_fam, (p / p.sum(1, keepdims=True)).T)
            fam_hit.append(pf.argmax(1) == want)
        fam_hit = np.asarray(fam_hit, np.float64)
        PER[n] = (cen_hit, knn_hit, fam_hit.mean(0))
        L.append(f"| {n} | {cen_hit.mean():.1%} | {knn_hit.mean():.1%} | {100 * fam_hit.mean(1).mean():.1f} ± {100 * fam_hit.mean(1).std():.1f} |")
        log(L[-1])
    L += ["", "## Each CLIP encoder minus Qwen, on the same photos (paired bootstrap over photos, 2,000 draws, seed 422)", "",
          "| encoder | nearest centroid | 10 nearest neighbours | from the wiki: family right |", "|---|---|---|---|"]
    for n in names:
        L.append(f"| {n} | " + " | ".join(paired(PER[n][k], PER[QWEN][k], draws) for k in range(3)) + " |")
        log(L[-1])
    L += ["", "## By style, 10 nearest neighbours within FashionStyle14", "", "| style | photos | " + " | ".join(sets) + " |", "|---|---|" + "---|" * len(sets)]
    for k, st in enumerate(kinds):
        L.append(f"| {st} | {(lab == k).sum():,} | " + " | ".join(f"{PER[n][1][lab == k].mean():.0%}" for n in sets) + " |")
    L += ["", "Qwen's vectors were made with the instruction \"Represent the fashion aesthetic of this image.\" at 448 px; CLIP takes no instruction and sees 224 px.",
          "FashionStyle14's 11 styles were already used to choose and check the NN's street-photo settings (21 Nov); here they only compare encoders.",
          f"time elapsed {time.time() - T0:.0f}s"]
    out = ROOT / "runs" / "clip-compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fs14.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"wrote {out / 'fs14.md'}; time elapsed {time.time() - T0:.0f}s")
