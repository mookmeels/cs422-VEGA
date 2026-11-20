"""Build the consolidated comparison table from runs/comparison/results.jsonl.

    python vega_table.py                 # prints the table, writes table.md and table.csv next to results.jsonl

One row per model and split. Runs with different seeds are averaged and shown as mean ± standard deviation
(the spread between seeds); a model scored once, or with no seed, shows a single value. If the same model,
split and seed was scored more than once, only the latest run counts. Validation and test are separate rows.
"""
import argparse, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vega_data                                                # noqa: E402

COLS = [("top5", "top-5"), ("top1", "top-1"), ("mrr", "MRR"), ("label_top5", "per-label top-5"),
        ("thin_top5", "thin labels top-5"), ("rest_top5", "other labels top-5"), ("coverage5", "label coverage")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None, help="folder holding results.jsonl; default <root>/runs/comparison")
    ap.add_argument("--labels", default="merged")
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.root or vega_data.ROOT) / "runs" / "comparison"
    rows = [json.loads(l) for l in open(out / "results.jsonl", encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["labels"] == a.labels]
    latest = {}
    for r in rows:                                               # later lines replace earlier ones
        latest[(r["model"], r["split"], r["seed"])] = r
    groups = {}
    for (model, split, _), r in latest.items():
        groups.setdefault((split, model), []).append(r)

    def cell(rs, key, pct=True):
        v = [r[key] for r in rs if r.get(key) is not None]
        if not v:
            return "n/a"
        f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
        return f(v[0]) if len(v) == 1 else f"{f(statistics.mean(v))} ± {f(statistics.stdev(v))}"

    head = ["split", "model", "owner", "runs (seeds)"] + [c[1] for c in COLS]
    table = []
    for split in ("validation", "test"):
        part = [(m, rs) for (s, m), rs in groups.items() if s == split]
        part.sort(key=lambda mr: -statistics.mean(r["top5"] for r in mr[1]))
        for model, rs in part:
            seeds = sorted(str(r["seed"]) for r in rs if r["seed"] is not None)
            used = sorted({r.get("vectors", "qwen2b") for r in rs})     # rows from before 17 Nov have no "vectors" entry
            if used != [vega_data.VECTORS]:
                model += f" [OLD VECTORS: {', '.join(used)}; re-run]"
            table.append([split, model, rs[0]["owner"], f"{len(rs)} ({', '.join(seeds)})" if seeds else "1 (no randomness)"]
                         + [cell(rs, k, pct=(k != "mrr")) for k, _ in COLS])
    if not table:
        raise SystemExit("no results yet")
    any_row = next(iter(latest.values()))
    cap = (f"Label set: {a.labels} ({any_row['n_labels']} labels). Percentages except MRR. "
           f"Validation: models fitted on the training split. Test: fitted on train + validation, scored once. "
           f"Split file sha256 {any_row['split_sha256'][:12]}. Image vectors: emb/{vega_data.VECTORS}-448.")
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in table]
    (out / "table.md").write_text(cap + "\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    (out / "table.csv").write_text("\n".join(",".join('"' + c.replace('"', "'") + '"' for c in r) for r in [head] + table) + "\n", encoding="utf-8")
    print(cap + "\n")
    width = [max(len(r[i]) for r in [head] + table) for i in range(len(head))]
    for r in [head] + table:
        print("  ".join(c.ljust(w) for c, w in zip(r, width)))


if __name__ == "__main__":
    main()
