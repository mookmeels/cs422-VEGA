r"""The transfer test: do models trained on the wiki work on ordinary outfit photos from outside it?

Photos: FashionStyle14 (Takagi et al., ICCVW 2017): 13,186 outfit photos, each filed by experts under one of 14
styles (5 photos filed under two styles are left out). Their Qwen vectors are in emb/fashionstyle14/, made exactly
like the wiki's. None of these photos is ever trained on.

Which styles are scored: runs/transfer/rows.csv. A style counts only where use = yes, i.e. where the team accepted
that it corresponds to particular wiki labels. Each row gives
  labels   the wiki labels that count as a right answer for the style's photos
  family   a wider set of related wiki labels (a secondary figure; taken from the draft table, by name only)
The rows were checked with data on 19 Nov (runs/outside-label-map-audit/): lolita, fairy and ethnic held in both
directions, and those three are the default. Changing rows.csv changes the test; results record which rows file
they were scored with, and the table only uses results scored with the current one.

How a model takes part, AFTER it is frozen (same rule as the test split: never change a model because of this score):

    import sys; sys.path.insert(0, "/srv/ml/vega/code/shared")
    from vega_transfer import load
    t = load()                      # t["fit"]:  the wiki's training + validation images, as in the final run
                                    # t["eval"]: ids, x (Qwen vectors), paths of ALL 13,186 photos; no styles given
    ... train on t["fit"] only; score every photo in t["eval"] for all 503 labels ...
    np.savez("my-transfer.npz", ids=t["eval"]["ids"], scores=scores, classes=np.asarray(t["classes"]))

    python vega_transfer.py --pred my-transfer.npz --model "CNN (ResNet-18)" --owner "Name" --seed 422
    python vega_transfer.py --table

Metrics, for each scored style, then averaged over the scored styles (each style counts equally):
  top5        share of the style's photos with one of its labels in the top 5
  top1        share whose first guess is one of its labels
  family5     top5, counting the wider family as right
  precision1  of ALL photos (any of the 14 styles) whose first guess is one of the style's labels, the share that
              really are that style. A model that sends every outfit photo to one label gets a high top5 and a
              low precision1, so read the two together.
Ties are broken exactly as in vega_score.py. Results: runs/transfer/results.jsonl, table.md, table.csv.
"""
import argparse, csv, datetime, getpass, hashlib, json, os, socket, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vega_data                                                # noqa: E402
from vega_score import TIE_SEED                                 # noqa: E402

SOURCE = "fashionstyle14"


def _out(root):
    return Path(root) / "runs" / "transfer"


def rows_file(root=None):
    return _out(root or vega_data.ROOT) / "rows.csv"


def rows(root=None):
    """The styles scored (use = yes), with their labels and families, checked against the 503 labels."""
    root = Path(root or vega_data.ROOT)
    f = rows_file(root)
    if not f.exists():
        raise SystemExit(f"{f} not found: it lists which outside styles count as which wiki labels.")
    _, classes = vega_data.label_map("merged", root)
    known, out = set(classes), []
    with open(f, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["use"].strip().lower() != "yes":
                continue
            labels = [x.strip() for x in r["labels"].split(";") if x.strip()]
            family = [x.strip() for x in r["family"].split(";") if x.strip()]
            bad = [x for x in labels + family if x not in known]
            if not labels or bad:
                raise SystemExit(f"{f}, style {r['style']!r}: " + (f"not among the 503 labels: {bad}" if bad else "no labels"))
            out.append({"style": r["style"], "labels": labels, "family": sorted(set(family) | set(labels))})
    if not out:
        raise SystemExit(f"{f}: no row has use = yes")
    return out


def _photos(root):
    """sha1 -> path (relative to the dataset folder), and sha1 -> style for photos filed under exactly one style."""
    folder = Path(root) / "emb" / SOURCE
    path = {}
    for line in (folder / "images.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, p = line.split("\t")
            path[sha] = p
    styles = defaultdict(set)
    with open(folder / "manifest.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            styles[r["sha1"]].add(r["style"])
    return path, {s: next(iter(v)) for s, v in styles.items() if len(v) == 1}


def load(root=None):
    """Fit = the wiki's training + validation images with labels. Eval = every transfer photo, without its style."""
    root = Path(root or vega_data.ROOT)
    d = vega_data.load(root=root)
    f, v = d["fit"], d["eval"]
    fit = {"ids": np.concatenate([f["ids"], v["ids"]]), "x": np.concatenate([f["x"], v["x"]]),
           "y": np.concatenate([f["y"], v["y"]]), "paths": f["paths"] + v["paths"],
           "split": np.concatenate([f["split"], v["split"]])}
    path, style = _photos(root)
    vec = root / "emb" / SOURCE / f"{vega_data.VECTORS}-448"
    ids = (vec / "ids.txt").read_text(encoding="utf-8").split()
    mat = np.load(vec / "embeddings.npy", mmap_mode="r")
    keep = [i for i, s in enumerate(ids) if s in style]
    base = Path(os.environ.get("VEGA_DATASETS", root.parent / "datasets")) / "fashionstyle14" / "FashionStyle14_v1"
    ev = {"ids": np.asarray([ids[i] for i in keep]), "x": np.ascontiguousarray(mat[keep], np.float32),
          "paths": [str(base / path[ids[i]]) for i in keep]}
    return {"classes": d["classes"], "labels": "merged", "fit": fit, "eval": ev, "source": SOURCE}


def metrics(scores, ids, root=None):
    root = Path(root or vega_data.ROOT)
    _, style = _photos(root)
    _, classes = vega_data.label_map("merged", root)
    cidx = {c: i for i, c in enumerate(classes)}
    n, k = scores.shape
    tie = np.random.default_rng(TIE_SEED).permutation(k)
    order = np.lexsort((np.broadcast_to(tie, (n, k)), -scores), axis=1)
    top5, first = order[:, :5], order[:, 0]
    st = np.asarray([style[i] for i in ids])
    per = {}
    for r in rows(root):
        right, fam = [cidx[x] for x in r["labels"]], [cidx[x] for x in r["family"]]
        sel = st == r["style"]
        claimed = np.isin(first, right)
        per[r["style"]] = {"photos": int(sel.sum()), "top5": float(np.isin(top5[sel], right).any(1).mean()),
                           "top1": float(np.isin(first[sel], right).mean()), "family5": float(np.isin(top5[sel], fam).any(1).mean()),
                           "precision1": float((claimed & sel).sum() / claimed.sum()) if claimed.any() else None,
                           "first_guesses_on_its_labels": int(claimed.sum())}
    pooled = {m: float(np.mean([p[m] for p in per.values()])) for m in ("top5", "top1", "family5")}
    prec = [p["precision1"] for p in per.values() if p["precision1"] is not None]
    pooled["precision1"] = float(np.mean(prec)) if prec else None
    return pooled, per


def score(pred, model, owner, seed=None, root=None, note="", quiet=False):
    root = Path(root or vega_data.ROOT)
    out = _out(root)
    out.mkdir(parents=True, exist_ok=True)
    _, classes = vega_data.label_map("merged", root)
    _, style = _photos(root)
    z = np.load(pred, allow_pickle=False)
    for key in ("ids", "scores", "classes"):
        if key not in z:
            raise SystemExit(f"{pred}: missing '{key}'. Save with np.savez(path, ids=..., scores=..., classes=...).")
    if [str(c) for c in z["classes"]] != classes:
        raise SystemExit(f"{pred}: label names or their order differ from the 503 merged labels.")
    s = np.asarray(z["scores"], np.float64)
    where = {str(a): i for i, a in enumerate(z["ids"])}
    if len(where) != len(z["ids"]) or s.shape != (len(z["ids"]), len(classes)):
        raise SystemExit(f"{pred}: scores must be (photos x labels) with no repeated id; got {s.shape}.")
    ids = sorted(style)
    missing = [a for a in ids if a not in where]
    if missing:
        raise SystemExit(f"{pred}: {len(missing)} of the {len(ids)} transfer photos have no prediction; score every photo in load()['eval'].")
    s = s[[where[a] for a in ids]]
    if not np.isfinite(s).all():
        raise SystemExit(f"{pred}: scores contain NaN or infinity.")
    pooled, per = metrics(s, ids, root)
    row = {"model": model, "owner": owner, "seed": seed, "source": SOURCE, "fit": "wiki training + validation",
           "styles": list(per), "photos_scored": int(sum(p["photos"] for p in per.values())), "photos_total": len(ids),
           **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in pooled.items()}, "per_style": per, "note": note,
           "pred_file": str(pred), "pred_sha256": hashlib.sha256(Path(pred).read_bytes()).hexdigest(),
           "rows_sha256": hashlib.sha256(rows_file(root).read_bytes()).hexdigest(), "vectors": vega_data.VECTORS,
           "split_sha256": vega_data.split_sha256(root),
           "when_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
           "host": socket.gethostname(), "user": getpass.getuser()}
    with open(out / "results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    if not quiet:
        p = "n/a" if pooled["precision1"] is None else f"{pooled['precision1']:.1%}"
        print(f"{model:<34} seed {str(seed):<5} transfer top-5 {pooled['top5']:.1%}  top-1 {pooled['top1']:.1%}  "
              f"family top-5 {pooled['family5']:.1%}  precision {p}  [" + ", ".join(f"{k} {v['top5']:.0%}" for k, v in per.items()) + "]", flush=True)
    return row


def table(root=None):
    """runs/transfer/table.md and table.csv: each model's latest run per seed, scored with the current rows file."""
    root = Path(root or vega_data.ROOT)
    out = _out(root)
    current = hashlib.sha256(rows_file(root).read_bytes()).hexdigest()
    latest = {}
    for line in open(out / "results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["rows_sha256"] == current and r["vectors"] == vega_data.VECTORS:
            latest[(r["model"], r["seed"])] = r
    by_model = defaultdict(list)
    for (m, _), r in latest.items():
        by_model[m].append(r)
    styles = [r["style"] for r in rows(root)]

    def ms(rs, key, sub=None):
        v = [(r["per_style"][sub][key] if sub else r[key]) for r in rs]
        v = [x for x in v if x is not None]
        if not v:
            return "n/a"
        return f"{100 * np.mean(v):.1f}" + (f" ± {100 * np.std(v, ddof=1):.1f}" if len(v) > 1 else "")

    order = sorted(by_model, key=lambda m: -np.mean([r["top5"] for r in by_model[m]]))
    head = ["model", "owner", "runs", "top-5", "top-1", "family top-5", "precision of first guess"] + [f"{s} top-5" for s in styles]
    body = []
    for m in order:
        rs = by_model[m]
        seeds = sorted(str(r["seed"]) for r in rs if r["seed"] is not None)
        body.append([m, rs[0]["owner"], f"{len(rs)} ({', '.join(seeds)})" if seeds else "1 (no randomness)",
                     ms(rs, "top5"), ms(rs, "top1"), ms(rs, "family5"), ms(rs, "precision1")] + [ms(rs, "top5", s) for s in styles])
    n = latest[next(iter(latest))]["photos_scored"] if latest else 0
    cap = (f"Transfer test: models fitted on the wiki's training + validation images, scored on FashionStyle14 outfit photos they never saw. "
           f"Styles scored: {', '.join(styles)} ({n} photos); each style counts equally. Percentages. Rows file sha256 {current[:12]}. "
           f"Image vectors: emb/{vega_data.VECTORS}-448.")
    md = [cap, "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in body]
    (out / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(out / "table.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(head)
        w.writerows(body)
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help=".npz with ids, scores, classes")
    ap.add_argument("--model")
    ap.add_argument("--owner")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--root", default=None)
    ap.add_argument("--table", action="store_true", help="rebuild runs/transfer/table.md from results.jsonl")
    a = ap.parse_args()
    if a.pred:
        if not (a.model and a.owner):
            raise SystemExit("--model and --owner are required with --pred")
        score(a.pred, a.model, a.owner, a.seed, a.root, a.note)
    if a.table or not a.pred:
        table(a.root)


if __name__ == "__main__":
    main()
