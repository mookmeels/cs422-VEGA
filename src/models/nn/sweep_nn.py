"""Choose the NN's settings on the VALIDATION split. Never reads the test split.

    python sweep_nn.py --stage 1          # one setting changed at a time from the starting point
    python sweep_nn.py --stage 2          # the changes that helped in stage 1, combined
    python sweep_nn.py --summary          # rebuild summary.md from sweep.jsonl without training

Every configuration is trained on the training split and scored on validation with the shared metrics
(vega_score.metrics), once per seed. Rows go to <out>/sweep.jsonl (default <root>/runs/nn/sweep), not to the
team's comparison file; summary.md shows mean ± standard deviation over the seeds. A (stage, name, seed)
that is already in sweep.jsonl is skipped, so an interrupted sweep can be restarted.

Each set of image vectors has its own folder, and a folder never mixes vectors: runs/nn/sweep is the first
sweep (17 Nov, on emb/qwen2b-448); runs/nn/sweep-qwen2beos is the sweep repeated after the team switched to
the corrected vectors (emb/qwen2beos-448). The vectors come from vega_data.VECTORS.

START is where the search began: the labeller's best network (lab_experiments.py, "mlp-1024").
The lists below are the complete record of what was tried.
"""
import argparse, json, socket, statistics, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_nn                                                 # noqa: E402  (also puts the shared code on the path)
import vega_data, vega_score                                    # noqa: E402

VECTORS = getattr(vega_data, "VECTORS", "qwen2b")
START = {"layers": 1, "hidden": 1024, "dropout": 0.5, "input_dropout": 0.0, "weight_decay": 1e-2, "lr": 1e-3,
         "epochs": 40, "batch": 512, "balance": 0.5, "prep": "standardize"}

STAGES = {
    1: [("start", {}),
        ("no hidden layer (softmax regression)", {"layers": 0}),
        ("no hidden layer, lr 3e-4", {"layers": 0, "lr": 3e-4}),
        ("no hidden layer, lr 3e-3", {"layers": 0, "lr": 3e-3}),
        ("2 hidden layers", {"layers": 2}),
        ("hidden 256", {"hidden": 256}),
        ("hidden 512", {"hidden": 512}),
        ("hidden 2048", {"hidden": 2048}),
        ("hidden 4096", {"hidden": 4096}),
        ("dropout 0", {"dropout": 0.0}),
        ("dropout 0.2", {"dropout": 0.2}),
        ("dropout 0.7", {"dropout": 0.7}),
        ("input dropout 0.2", {"input_dropout": 0.2}),
        ("weight decay 0", {"weight_decay": 0.0}),
        ("weight decay 1e-3", {"weight_decay": 1e-3}),
        ("weight decay 1e-1", {"weight_decay": 1e-1}),
        ("weight decay 1", {"weight_decay": 1.0}),
        ("lr 3e-4", {"lr": 3e-4}),
        ("lr 3e-3", {"lr": 3e-3}),
        ("epochs 10", {"epochs": 10}),
        ("epochs 20", {"epochs": 20}),
        ("epochs 80", {"epochs": 80}),
        ("balance 0 (every image alike)", {"balance": 0.0}),
        ("balance 1 (every label alike)", {"balance": 1.0}),
        ("inputs centred, not standardized", {"prep": "center"}),
        ("inputs not scaled", {"prep": "none"}),
        ("batch 128", {"batch": 128})],
    # Stage 1 found no single change that beat the start by more than the spread between seeds. Stage 2 asks,
    # with 5 seeds: do the changes with the (slightly) highest stage-1 averages add up when combined, and how
    # good can the model without a hidden layer get when it is given its own learning rate?
    # (Stage 2 was designed from the first sweep's stage 1. It is run unchanged on the corrected vectors too,
    # so the two sweeps can be compared row by row.)
    2: [("start", {}),
        ("weight decay 1 + balance 1", {"weight_decay": 1.0, "balance": 1.0}),
        ("weight decay 1 + balance 1 + input dropout 0.2", {"weight_decay": 1.0, "balance": 1.0, "input_dropout": 0.2}),
        ("weight decay 1 + balance 1 + dropout 0.7", {"weight_decay": 1.0, "balance": 1.0, "dropout": 0.7}),
        ("no hidden layer, lr 3e-4", {"layers": 0, "lr": 3e-4}),
        ("no hidden layer, lr 1e-4", {"layers": 0, "lr": 1e-4}),
        ("no hidden layer, lr 3e-4, weight decay 1", {"layers": 0, "lr": 3e-4, "weight_decay": 1.0})],
    # Stage 3 (corrected vectors only): on those vectors the balance weighting had the highest stage-1 averages,
    # in both directions, and balance 1 looked better on thin labels. Five seeds each, to compare with "start"
    # and "weight decay 1 + balance 1" from stage 2.
    3: [("balance 1 (every label alike)", {"balance": 1.0}),
        ("balance 0 (every image alike)", {"balance": 0.0}),
        ("weight decay 1", {"weight_decay": 1.0})],
    # Stage 4 (corrected vectors only): a confirmation on FRESH seeds (run with --seeds 427 428 429 430 431).
    # In stage 2 the combination "weight decay 1 + balance 1 + dropout 0.7" was 0.5 points of top-5 above
    # "start", but it was the best of many, which flatters it. Rule written before running this stage: adopt
    # a combination as the NN's settings only if, on the five fresh seeds, its mean top-5 is above start's by
    # more than the larger of the two standard deviations across seeds. Otherwise the start settings stay.
    4: [("start", {}),
        ("weight decay 1 + balance 1 + dropout 0.7", {"weight_decay": 1.0, "balance": 1.0, "dropout": 0.7}),
        ("weight decay 1 + balance 1", {"weight_decay": 1.0, "balance": 1.0})],
}

COLS = [("top5", "top-5"), ("top1", "top-1"), ("mrr", "MRR"), ("label_top5", "per-label top-5"),
        ("thin_top5", "thin labels top-5"), ("rest_top5", "other labels top-5")]


def summary(out):
    rows = [json.loads(l) for l in open(out / "sweep.jsonl", encoding="utf-8") if l.strip()]
    text = ["NN settings sweep on the validation split (trained on the training split; the test split is not read).",
            f"Image vectors: emb/{rows[0].get('vectors', 'qwen2b')}-448.",
            "Mean ± standard deviation over seeds; percentages except MRR. Sorted by top-5 within each stage.",
            f"Starting point: {json.dumps(START)}", ""]
    for stage in sorted({r["stage"] for r in rows}):
        groups = {}
        for r in rows:
            if r["stage"] == stage:
                groups.setdefault(r["name"], []).append(r)
        text += [f"Stage {stage}", "", "| configuration | changed from start | seeds | " + " | ".join(c[1] for c in COLS) + " | seconds per run |",
                 "|" + "---|" * (len(COLS) + 4)]
        for name, rs in sorted(groups.items(), key=lambda g: -statistics.mean(r["top5"] for r in g[1])):
            def cell(key):
                v = [r[key] for r in rs]
                f = (lambda x: f"{x:.3f}") if key == "mrr" else (lambda x: f"{100 * x:.1f}")
                return f(v[0]) if len(v) == 1 else f"{f(statistics.mean(v))} ± {f(statistics.stdev(v))}"
            changed = ", ".join(f"{k}={v}" for k, v in rs[0]["settings"].items() if START[k] != v) or "nothing"
            text.append(f"| {name} | {changed} | {len(rs)} | " + " | ".join(cell(k) for k, _ in COLS)
                        + f" | {statistics.mean(r['seconds'] for r in rs):.0f} |")
        text.append("")
    (out / "summary.md").write_text("\n".join(text), encoding="utf-8")
    return "\n".join(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, default=1, choices=sorted(STAGES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[422, 423, 424])
    ap.add_argument("--configs", default=":", help="slice of the stage's list, e.g. 0:14, to share a stage between machines")
    ap.add_argument("--summary", action="store_true", help="only rebuild summary.md")
    ap.add_argument("--root", default=None, help="default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--out", default=None, help="default <root>/runs/nn/sweep-<vectors> (runs/nn/sweep for the old qwen2b vectors)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads; 0 = torch's default")
    a = ap.parse_args()
    root = Path(a.root or vega_data.ROOT)
    out = Path(a.out) if a.out else root / "runs" / "nn" / ("sweep" if VECTORS == "qwen2b" else f"sweep-{VECTORS}")
    out.mkdir(parents=True, exist_ok=True)
    if a.summary:
        print(summary(out))
        return
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.perf_counter()
    done = set()
    if (out / "sweep.jsonl").exists():
        old = [json.loads(l) for l in open(out / "sweep.jsonl", encoding="utf-8") if l.strip()]
        other = {r.get("vectors", "qwen2b") for r in old} - {VECTORS}
        if other:
            raise SystemExit(f"{out} holds runs on other vectors ({', '.join(sorted(other))}); this run uses {VECTORS}. Use another --out.")
        done = {(r["stage"], r["name"], r["seed"]) for r in old}
    d = vega_data.load(root=root)                                # fit = train, eval = validation
    fit, ev = d["fit"], d["eval"]
    fit_count = (fit["y"] > 0).sum(0)
    lo, hi = (int(v) if v else None for v in a.configs.split(":"))
    todo = STAGES[a.stage][lo:hi]
    print(f"stage {a.stage}: {len(todo)} configurations x seeds {a.seeds}; fit {len(fit['ids'])}, "
          f"validation {len(ev['ids'])}, {len(d['classes'])} labels, vectors {VECTORS}, output {out}", flush=True)
    for name, change in todo:
        s = {**START, **change}
        for seed in a.seeds:
            if (a.stage, name, seed) in done:
                continue
            t1 = time.perf_counter()
            m = vega_score.metrics(train_nn.train(fit["x"], fit["y"], ev["x"], s, seed), ev["truth"], fit_count)
            row = {"stage": a.stage, "name": name, "seed": seed, "settings": s, **{k: round(v, 6) for k, v in m.items() if isinstance(v, float)},
                   "seconds": round(time.perf_counter() - t1, 1), "split": "validation", "labels": d["labels"], "vectors": VECTORS,
                   "host": socket.gethostname(), "threads": torch.get_num_threads(), "split_sha256": vega_data.split_sha256(root)}
            with open(out / "sweep.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {name:<40} seed {seed}  top-5 {m['top5']:.1%}  top-1 {m['top1']:.1%}  per-label top-5 {m['label_top5']:.1%}  "
                  f"thin {m['thin_top5']:.1%}  {row['seconds']:.0f}s", flush=True)
    print(summary(out))
    print(f"\ndone in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
