"""Summarise the CNN's runs: the tuning comparison, the rule that picks the winner, and the learning curves.

    python summarise_tuning.py                      # every run in <root>/runs/cnn/validation
    python summarise_tuning.py --out <folder>       # runs saved somewhere else

Reads what train_cnn.py writes: runs.jsonl (settings, timings, memory), results.jsonl (validation scores from the
team's scorer) and curves/*.csv (per epoch). Writes, in <out>/summary/:
    summary.md      every run, the tuning table, the rule and its verdict, and seed means for any setting run
                    with more than one seed
    curves.png      training and validation loss, and validation top-5, per epoch, one line per setting (seed 422)

The rule (Gianni, 19 Nov 2026, decision 9, written before any tuning run): among the tuning runs (seed 422, full
length), keep the best validation top-5; if another run is within 0.5 point of it, keep whichever trained faster.
The tuning runs are the FIRST seed-422 run of each plain setting at full length and the tuning precision
(fp32); a later repeat never replaces it, and an earlier shorter or other-precision run (a smoke test, a speed test)
cannot take its place. Street-photo runs ("-street" tags) and runs with the 21 Nov 2026 test options (tags such as "-crop60-st15",
"-mix", "-ls10", "-os") are different recipes, not tuning settings: they are
listed and get their own seed means, but never enter the verdict or the tuning curves (scan, 21 Nov 2026). Only the
merged label set is read; runs and scores of any other label set are left out. Seed means follow the team table (vega_table.py): the latest run of each seed counts. A setting and
seed run more than once gets a "repeats" table showing how far the runs differ.
Only validation pictures are used; the test split plays no part.
"""
import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import find_shared  # noqa: E402,F401  (puts the shared code on the path)
import vega_data  # noqa: E402

TUNING_SEED = 422
MARGIN = 0.5                                                      # points of top-5
TUNING_PRECISION = "fp32"                                         # the precision the tuning runs used (decision, 20 Nov)
COLOURS = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14), (23, 190, 207)]


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect(out):
    """Every scored validation run, in the order it was made, each with its own score.

    train_cnn.py appends a score (results.jsonl) and then a run record (runs.jsonl) for every run, so the k-th
    run of a (setting, seed) belongs with the k-th score of that (setting, seed). A repeat therefore keeps its own
    score instead of replacing the first run's (20 Nov 2026: a seed-422 repeat once took a tuning run's place).
    Each row gets "run": 1 for the first run of its (setting, seed), 2 for a repeat, and so on.
    """
    scores = {}
    for s in read_jsonl(out / "results.jsonl"):
        name = s.get("pred_file", "").replace("\\", "/").rsplit("/", 1)[-1]   # Windows or Linux path; <tag>-validation-<labels>-seed<seed>.npz
        if s.get("split") == "validation" and s.get("labels", "merged") == "merged" and "-validation-merged-" in name:
            scores.setdefault((name.split("-validation-")[0], s["seed"]), []).append(s)
    seen, rows = {}, []
    for r in read_jsonl(out / "runs.jsonl"):
        if r.get("split") != "validation" or r.get("subset") or r.get("labels", "merged") != "merged":
            continue
        key = (r["tag"], r["seed"])
        k = seen[key] = seen.get(key, 0) + 1
        if k <= len(scores.get(key, [])):
            s = scores[key][k - 1]
            pct = {f: None if s.get(f) is None else 100 * s[f] for f in ("top5", "top1", "label_top5", "thin_top5")}
            rows.append({**r, **pct, "mrr": s["mrr"], "run": k})
    return rows


def fmt(v):
    return "-" if v is None else f"{v:.1f}"


def decide(tuning):
    """The pre-written rule. Returns (winner, explanation)."""
    best = max(tuning, key=lambda r: r["top5"])
    close = [r for r in tuning if r["top5"] >= best["top5"] - MARGIN]
    winner = min(close, key=lambda r: r["train_seconds"])
    if winner is best:
        why = (f"{best['tag']} has the best top-5 ({best['top5']:.1f})"
               + (f"; no other run is within {MARGIN} point" if len(close) == 1 else
                  f"; it is also the fastest of the runs within {MARGIN} point ({', '.join(r['tag'] for r in close)})"))
    else:
        why = (f"{best['tag']} has the best top-5 ({best['top5']:.1f}), but {winner['tag']} is within {MARGIN} point "
               f"({winner['top5']:.1f}) and trains faster ({winner['train_seconds'] / 60:.1f} min against "
               f"{best['train_seconds'] / 60:.1f} min), so the rule keeps {winner['tag']}")
    return winner, why


def read_curve(out, tag, seed):
    path = out / "curves" / f"{tag}-merged-seed{seed}.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def draw_curves(curves, path):
    """Two panels: losses (solid = training, dashed = validation) and validation top-5, one colour per setting."""
    W, H, pad = 1200, 520, 60
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    panels = [("loss (solid: training, dashed: validation)", ("train_loss", "validation_loss")),
              ("validation top-5 (%)", ("validation_top5",))]
    pw = (W - 3 * pad) // 2
    for p, (title, keys) in enumerate(panels):
        x0, y0, x1, y1 = pad + p * (pw + pad), pad, pad + p * (pw + pad) + pw, H - pad
        vals = [100 * v[k] if k == "validation_top5" else v[k] for c in curves.values() for v in c for k in keys]
        lo, hi = min(vals), max(vals)
        hi = hi if hi > lo else lo + 1
        n = max(len(c) for c in curves.values())
        d.rectangle([x0, y0, x1, y1], outline=(120, 120, 120))
        d.text((x0, y0 - 22), title, fill="black")
        for frac in (0, 0.5, 1):
            yv = lo + frac * (hi - lo)
            yy = y1 - frac * (y1 - y0)
            d.line([x0 - 4, yy, x0, yy], fill=(120, 120, 120))
            d.text((x0 - 52, yy - 6), f"{yv:.2f}" if hi - lo < 10 else f"{yv:.0f}", fill=(80, 80, 80))
        d.text((x0, y1 + 8), "epoch 1", fill=(80, 80, 80))
        d.text((x1 - 50, y1 + 8), f"epoch {n}", fill=(80, 80, 80))
        for i, (tag, c) in enumerate(curves.items()):
            col = COLOURS[i % len(COLOURS)]
            for k in keys:
                pts = [(x0 + (row["epoch"] - 1) / max(n - 1, 1) * (x1 - x0),
                        y1 - ((100 * row[k] if k == "validation_top5" else row[k]) - lo) / (hi - lo) * (y1 - y0))
                       for row in c]
                if k == "validation_loss":                         # dashed
                    for j in range(0, len(pts) - 1, 2):
                        d.line([pts[j], pts[j + 1]], fill=col, width=2)
                else:
                    d.line(pts, fill=col, width=2)
    for i, tag in enumerate(curves):
        d.rectangle([pad + i * 200, H - 26, pad + i * 200 + 14, H - 12], fill=COLOURS[i % len(COLOURS)])
        d.text((pad + i * 200 + 20, H - 26), tag, fill="black")
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="project folder; default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--out", default=None, help="where train_cnn.py wrote its runs; default <root>/runs/cnn/validation")
    a = ap.parse_args()
    root = Path(a.root) if a.root else vega_data.ROOT
    out = Path(a.out) if a.out else root / "runs" / "cnn" / "validation"
    rows = collect(out)
    if not rows:
        raise SystemExit(f"no scored validation runs found in {out}")
    (out / "summary").mkdir(exist_ok=True)

    full = max(r["settings"]["epochs"] for r in rows)
    tuning, taken = [], set()
    for r in rows:                                                # in the order made: the first qualifying run counts
        if (r["seed"] == TUNING_SEED and r["settings"]["epochs"] == full and r["settings"]["precision"] == TUNING_PRECISION
                and re.fullmatch(r"cnn-r18w\d+-\d+", r["tag"]) and r["tag"] not in taken):
            tuning.append(r)
            taken.add(r["tag"])
    lines = ["# CNN runs: tuning and seeds", "",
             f"Folder: `{out}`. Validation split only ({rows[0]['split']}), label set merged. "
             f"Top-5, top-1 and per-label top-5 are percentages.", "",
             "## Every run (in the order made; run 2 = a repeat of the same setting and seed)", "",
             "| setting | seed | run | made | epochs | precision | weights | top-5 | top-1 | MRR | per-label top-5 "
             "| thin top-5 | train minutes | pictures/s | peak GPU GB | device |", "|" + "---|" * 16]
    for r in rows:
        s = r["settings"]
        lines.append(f"| {r['tag']} | {r['seed']} | {r['run']} | {r.get('made', '-')} | {s['epochs']} | {s['precision']} "
                     f"| {r['weights'] / 1e6:.2f} M "
                     f"| {r['top5']:.1f} | {r['top1']:.1f} | {r['mrr']:.3f} | {fmt(r['label_top5'])} | {fmt(r['thin_top5'])} "
                     f"| {r['train_seconds'] / 60:.1f} | {r['train_pictures_per_second']:.0f} "
                     f"| {r['peak_gpu_memory_gb'] if r['peak_gpu_memory_gb'] is not None else '-'} | {r['device']} |")
    lines += ["", "## Tuning (seed 422, full length, fp32, the first run of each plain setting; street runs excluded)", ""]
    if len(tuning) >= 2:
        winner, why = decide(tuning)
        lines += [f"Rule, written before the runs: keep the best validation top-5; if another run is within {MARGIN} "
                  "point, keep whichever trained faster.", "", f"**Verdict: {winner['tag']}.** {why}.", ""]
    else:
        lines += [f"Only {len(tuning)} tuning run so far; the rule needs at least two.", ""]
    latest = {}
    for r in rows:                                                # later runs replace earlier ones, as vega_table.py
        latest[(r["tag"], r["seed"])] = r
    groups = {}
    for (t, _), r in sorted(latest.items()):
        groups.setdefault(t, []).append(r)
    multi = {t: g for t, g in groups.items() if len(g) > 1}
    if multi:
        lines += ["## Settings run with several seeds (mean ± standard deviation, as the team table; "
                  "the latest run of each seed)", "",
                  "| setting | seeds | top-5 | top-1 |", "|---|---|---|---|"]
        for t, g in multi.items():
            t5, t1 = [r["top5"] for r in g], [r["top1"] for r in g]
            lines.append(f"| {t} | {', '.join(str(r['seed']) for r in g)} | {statistics.mean(t5):.1f} ± "
                         f"{statistics.stdev(t5):.1f} | {statistics.mean(t1):.1f} ± {statistics.stdev(t1):.1f} |")
        lines.append("")
    repeats = [(key, [r for r in rows if (r["tag"], r["seed"]) == key]) for key in latest]
    repeats = [(key, g) for key, g in repeats if len(g) > 1]
    if repeats:
        lines += ["## Repeats (the same setting and seed run more than once)", "",
                  "On a graphics card the same seed gives nearly, not exactly, the same result. The gap below is how "
                  "far the repeat landed from the first run.", "",
                  "| setting | seed | runs: top-5 | runs: top-1 | top-5 gap | top-1 gap |", "|---|---|---|---|---|---|"]
        for (t, seed), g in repeats:
            t5 = ", ".join(f"{r['top5']:.1f}" for r in g)
            t1 = ", ".join(f"{r['top1']:.1f}" for r in g)
            lines.append(f"| {t} | {seed} | {t5} | {t1} | {g[-1]['top5'] - g[0]['top5']:+.1f} "
                         f"| {g[-1]['top1'] - g[0]['top1']:+.1f} |")
        lines.append("")
    curves = {r["tag"]: c for r in tuning if (c := read_curve(out, r["tag"], TUNING_SEED))}
    if curves:
        draw_curves(curves, out / "summary" / "curves.png")
        replaced = [t for (t, seed), _ in repeats if seed == TUNING_SEED and t in curves]
        lines += ["## Learning curves", "", "`curves.png`: losses and validation top-5 per epoch, seed 422."
                  + (f" The curve file of {', '.join(replaced)} is from its latest seed-422 run (a repeat replaced "
                     "the tuning run's file)." if replaced else ""), "",
                  "| setting | last epoch train loss | last epoch validation loss | best epoch (top-5) | last epoch top-5 |",
                  "|---|---|---|---|---|"]
        for t, c in curves.items():
            best = max(c, key=lambda row: row["validation_top5"])
            lines.append(f"| {t} | {c[-1]['train_loss']:.3f} | {c[-1]['validation_loss']:.3f} | "
                         f"{int(best['epoch'])} ({100 * best['validation_top5']:.1f}) | {100 * c[-1]['validation_top5']:.1f} |")
        lines += ["", "The epoch count is fixed in advance; the best epoch is shown only to describe the curve, "
                  "never to pick a stopping point.", ""]
    (out / "summary" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
