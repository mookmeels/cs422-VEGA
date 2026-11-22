#!/usr/bin/env python3
"""
make_split.py - the one train/validation/test split every VEGA model reads.

Starts from the seeded, stratified split Codex built in project-data-v1 (seed 422,
70/10/20, rarest aesthetic first, byte-identical files kept together) and closes its
one gap: near-duplicate images (re-encodes, crops, recolours, reposts) that landed on
both sides of the split.

Near-duplicate rule (chosen by looking at sample pairs; see WORKLOG.md 2026-11-16):
two images are duplicates when
    cosine >= 0.97                                   (qwen2b-448 embeddings), or
    cosine >= 0.93 and pixel correlation >= 0.70, or
    cosine >= 0.85 and pixel correlation >= 0.90
where pixel correlation is the Pearson correlation of 32x32 grayscale thumbnails.
Duplicates are joined transitively into groups. A group whose members sit in
different splits moves as a whole:
    any member in the development pilot (evaluation-v1)  -> train (pilot images were looked at)
    otherwise                                            -> the split most members are in;
                                                            ties broken by a seeded hash
Images Codex left out of the catalog stay "excluded".

Writes  <out>/split.csv        sha1, sha256, split, codex_split, group, group_size
        <out>/dup_pairs.csv    every duplicate pair with its cosine and pixel correlation
        <out>/report.json      rule, counts, moves, per-aesthetic coverage, checks, versions
        <out>/thumbs32.npy     thumbnail cache, about 90 MB (rebuilt if missing, about 4 minutes; safe to delete)

Usage (on ml-db):
    /srv/ml/vega/.venv/bin/python make_split.py --out /srv/ml/vega/splits/wiki-split-v2
"""
import argparse, collections, hashlib, json, platform, sys, time, warnings
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

SEED = 422
RULE = [(0.97, -1.0), (0.93, 0.70), (0.85, 0.90)]   # (min cosine, min pixel correlation)
SPLITS = ("train", "validation", "test")


def thumbnails(ids, image_dir, cache):
    if cache.exists():
        t = np.load(cache)
        if t.shape == (len(ids), 1024):
            return t
    path = {q.stem: q for q in image_dir.rglob("*.*")}
    t = np.zeros((len(ids), 1024), np.float32)
    warnings.simplefilter("ignore")
    for k, s in enumerate(ids):
        try:
            im = Image.open(path[s]); im.seek(0); im.draft("L", (64, 64))
            v = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.BILINEAR), np.float32).ravel()
            t[k] = (v - v.mean()) / (v.std() + 1e-6) / 32      # unit length: dot product = correlation
        except Exception as e:
            print("  could not read", s, e)
    np.save(cache, t)
    return t


def duplicate_pairs(x, t):
    lo = min(c for c, _ in RULE)
    out = []
    for i in range(0, len(x), 2048):
        m = x[i:i + 2048] @ x.T
        for j in range(m.shape[0]):
            m[j, :i + j + 1] = -1                                # each pair once
        a, b = np.nonzero(m >= lo)
        cos = m[a, b]; a = a + i
        pix = (t[a] * t[b]).sum(1)
        keep = np.zeros(len(a), bool)
        for c, p in RULE:
            keep |= (cos >= c) & (pix >= p)
        out += list(zip(a[keep].tolist(), b[keep].tolist(), cos[keep].tolist(), pix[keep].tolist()))
    return out


def groups_of(n, pairs):
    parent = list(range(n))
    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]; v = parent[v]
        return v
    for a, b, _, _ in pairs:
        parent[find(a)] = find(b)
    return [find(v) for v in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--emb", default="/srv/ml/vega/emb/qwen2b-448")
    ap.add_argument("--wiki", default="/srv/ml/vega/wiki")
    ap.add_argument("--codex", default="/srv/ml/vega/project-data-v1")
    ap.add_argument("--pilot", default="/srv/ml/datasets/evaluation-v1/manifest.json")
    a = ap.parse_args()
    start = time.perf_counter()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    x = np.load(Path(a.emb) / "embeddings.npy")
    ids = (Path(a.emb) / "ids.txt").read_text(encoding="utf-8").split()
    codex, sha256, names = {}, {}, {}
    for line in open(Path(a.codex) / "images.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["source"] != "aesthetics-wiki":
            continue
        for sp in r["source_paths"]:
            s = Path(sp).stem
            codex[s], sha256[s], names[s] = r["split"], r["sha256"], r["aesthetic_names"]
    pilot = {r["id"] for r in json.loads(Path(a.pilot).read_text())["images"]}
    before = [codex.get(s, "excluded") for s in ids]
    print(f"{len(ids)} embedded wiki images; Codex split: " +
          ", ".join(f"{k} {before.count(k)}" for k in (*SPLITS, "excluded")))

    print("  thumbnails")
    t = thumbnails(ids, Path(a.wiki) / "images", out / "thumbs32.npy")
    print("  near-duplicate pairs")
    pairs = duplicate_pairs(x, t)
    root = groups_of(len(ids), pairs)
    members = collections.defaultdict(list)
    for v, r in enumerate(root):
        members[r].append(v)

    after = list(before)
    moves, straddling, by_pilot = 0, 0, 0
    for r, vs in members.items():
        in_split = [v for v in vs if before[v] in SPLITS]
        held = {before[v] for v in in_split}
        if len(held) <= 1:
            continue
        straddling += 1
        if any(sha256.get(ids[v]) in pilot for v in vs):
            target = "train"; by_pilot += 1
        else:
            votes = collections.Counter(before[v] for v in in_split)
            top = max(votes.values())
            tied = sorted(s for s in SPLITS if votes.get(s) == top)
            key = hashlib.sha256(f"{SEED}:{min(ids[v] for v in vs)}".encode()).digest()[0]
            target = tied[key % len(tied)]
        for v in in_split:
            if after[v] != target:
                after[v] = target; moves += 1

    # checks
    gsplits = collections.defaultdict(set)
    for v in range(len(ids)):
        if after[v] in SPLITS:
            gsplits[root[v]].add(after[v])
    assert all(len(s) <= 1 for s in gsplits.values()), "a duplicate group still straddles splits"
    crossing_after = sum(1 for a_, b_, _, _ in pairs
                         if after[a_] in SPLITS and after[b_] in SPLITS and after[a_] != after[b_])
    pilot_held_out = sum(1 for v in range(len(ids)) if sha256.get(ids[v]) in pilot and after[v] != "train")
    crossing_before = sum(1 for a_, b_, _, _ in pairs
                          if before[a_] in SPLITS and before[b_] in SPLITS and before[a_] != before[b_])

    per = collections.defaultdict(lambda: {"before": collections.Counter(), "after": collections.Counter()})
    for v, s in enumerate(ids):
        for n in names.get(s, []):
            per[n]["before"][before[v]] += 1; per[n]["after"][after[v]] += 1
    lost = sorted(n for n, c in per.items()
                  if (c["before"]["validation"] and not c["after"]["validation"]) or
                     (c["before"]["test"] and not c["after"]["test"]))
    mixed_labels = sum(1 for vs in members.values() if len(vs) > 1 and
                       len({tuple(sorted(names.get(ids[v], []))) for v in vs}) > 1)

    size = collections.Counter(root)
    # "\r\n" keeps the files byte-identical to the first wiki-split-v2 (reports record split.csv's sha256)
    pl.DataFrame({"sha1": ids, "sha256": [sha256.get(s) for s in ids], "split": after, "codex_split": before,
                  "group": [ids[r] for r in root], "group_size": [size[r] for r in root]},
                 schema_overrides={"sha256": pl.String}).write_csv(out / "split.csv", line_terminator="\r\n")
    pl.DataFrame({"sha1_a": [ids[p[0]] for p in pairs], "sha1_b": [ids[p[1]] for p in pairs],
                  "cosine": [p[2] for p in pairs], "pixel_corr": [p[3] for p in pairs]},
                 schema={"sha1_a": pl.String, "sha1_b": pl.String, "cosine": pl.Float64, "pixel_corr": pl.Float64}
                 ).write_csv(out / "dup_pairs.csv", float_precision=4, line_terminator="\r\n")

    multi = [n for n in size.values() if n > 1]
    report = {
        "version": "wiki-split-v2", "seed": SEED, "base": "project-data-v1 split (Codex, seed 422)",
        "rule": {"pairs": [{"min_cosine": c, "min_pixel_corr": p} for c, p in RULE],
                 "embedding": a.emb, "pixel_corr": "Pearson correlation of 32x32 grayscale thumbnails",
                 "grouping": "transitive (union-find)",
                 "move": "pilot member -> train; else majority split, ties by seeded hash"},
        "images": len(ids),
        "counts_before": dict(collections.Counter(before)), "counts_after": dict(collections.Counter(after)),
        "duplicate_pairs": len(pairs), "duplicate_groups": len(multi), "images_in_groups": sum(multi),
        "largest_groups": sorted(multi, reverse=True)[:5],
        "groups_with_different_gallery_labels": mixed_labels,
        "groups_straddling_before": straddling, "moved_to_pilot_rule": by_pilot, "images_moved": moves,
        "checks": {"pairs_crossing_splits_before": crossing_before, "pairs_crossing_splits_after": crossing_after,
                   "pilot_images_held_out": pilot_held_out},
        "aesthetics_that_lost_validation_or_test": lost,
        "per_aesthetic": {n: {"before": dict(c["before"]), "after": dict(c["after"])} for n, c in sorted(per.items())},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pillow": Image.__version__, "polars": pl.__version__},
        "seconds": round(time.perf_counter() - start, 1),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(pairs)} duplicate pairs in {len(multi)} groups ({sum(multi)} images); largest {report['largest_groups']}")
    print(f"{straddling} groups straddled splits; {moves} images moved ({by_pilot} groups sent to train by the pilot rule)")
    print("after: " + ", ".join(f"{k} {after.count(k)}" for k in (*SPLITS, "excluded")))
    print(f"pairs crossing splits: {crossing_before} before, {crossing_after} after; pilot images held out: {pilot_held_out}")
    print(f"aesthetics that lost all validation or test images: {len(lost)} {lost[:10]}")
    print(f"time elapsed: {report['seconds']:.0f}s\nsaved in {out}")


if __name__ == "__main__":
    main()
