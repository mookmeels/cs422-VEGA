"""Test-split predictions from CNN weights trained elsewhere (the final run, option A). Owner: Giancarlo Guevarra.

    python prepare_images.py --final --sizes 224                    # once, on ml-db: fit = train + validation, eval = test
    python predict_cnn.py --weights <final weights>/*.pt            # on ml-db, after the weights come back from the PC

The final run (Andrew, 20 Nov 2026, option A): the PC trains on train + validation with train_cnn.py --final
--fit-only, from the fit part of ml-db's final preparation (no test pictures leave ml-db). The weights come back to
ml-db, and this script predicts the test pictures here, saving them in the team's format:
    <out>/preds/<tag>-test-<labels>-seed<seed>.npz        ids, scores, classes (as train_cnn.py would have saved)
It never scores them: the test split is scored once, with vega_score.py --final, after the team freezes settings.

Before predicting, it checks that the weights were made for the final run, from the same split and label set, and
with exactly the colour scaling of this preparation (so the PC trained on the same fit pictures this folder holds).
An existing prediction file is never replaced unless --overwrite is given, and two weights files that would write
the same prediction file (the same setting and seed, for example from two run folders) are refused. Every check runs
on every weights file before the first prediction is written, so a refusal leaves no half-finished set. Scores that
are not finite numbers are refused rather than saved.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_cnn                                                  # noqa: E402  (also puts the shared code on the path)
import vega_data                                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, nargs="+", help="weights files saved by train_cnn.py --final")
    ap.add_argument("--root", default=None, help="project folder; default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--prepared", default=None, help="default <root>/runs/cnn/prepared/final")
    ap.add_argument("--out", default=None, help="default <root>/runs/cnn/final")
    ap.add_argument("--device", default="auto", help="auto (the largest graphics card, else the CPU), cpu, cuda:0, ...")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch's default)")
    ap.add_argument("--no-miopen", dest="miopen", action="store_false", help="switch off MIOpen/cuDNN (see train_cnn.py)")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing prediction file")
    a = ap.parse_args()
    if not a.miopen:
        torch.backends.cudnn.enabled = False
    if a.threads:
        torch.set_num_threads(a.threads)
    root = Path(a.root) if a.root else vega_data.ROOT
    prepared = Path(a.prepared) if a.prepared else root / "runs" / "cnn" / "prepared" / "final"
    out = Path(a.out) if a.out else root / "runs" / "cnn" / "final"
    stats = json.loads((prepared / "stats.json").read_text(encoding="utf-8"))
    if stats.get("mode") != "final":
        raise SystemExit(f"{prepared} is not a final-run preparation (mode {stats.get('mode')!r})")
    if stats["split_sha256"] != vega_data.split_sha256(root):
        raise SystemExit("the prepared pictures come from another split file; rerun prepare_images.py --final")
    device = train_cnn.pick_device(a.device)

    checked, loaded = [], {}
    for w in a.weights:
        ck = torch.load(w, map_location="cpu", weights_only=True)   # never run code hidden in a file (scan, 21 Nov 2026)
        s = ck["settings"]
        st = stats["per_size"].get(str(s["size"]))
        problems = [why for bad, why in (
            (not ck.get("final"), "was not made by a final run (--final)"),
            (ck["split_sha256"] != stats["split_sha256"], "comes from another split file"),
            (st is None, f"needs {s['size']} px pictures, which are not prepared here"),
            (st is not None and (ck["colour_mean"] != st["mean"] or ck["colour_std"] != st["std"]
                                 or ck["pad_rgb"] != st["pad_rgb"]),
             "was trained with a different colour scaling than this preparation")) if bad]
        if problems:
            raise SystemExit(f"{w}: " + "; ".join(problems))
        tag = train_cnn.run_tag(s)
        pred = out / "preds" / f"{tag}-test-{ck['labels']}-seed{ck['seed']}.npz"
        if pred.exists() and not a.overwrite:
            raise SystemExit(f"{pred} exists already; use --overwrite to replace it on purpose")
        clash = [c[0] for c in checked if c[5] == pred]
        if clash:
            raise SystemExit(f"{w} and {clash[0]} would both write {pred.name} (same setting and seed); "
                             "predict them in separate commands with separate --out folders")
        d = loaded.get(ck["labels"]) or loaded.setdefault(ck["labels"], vega_data.load(final=True, labels=ck["labels"],
                                                                                        root=root, vectors=False))
        if list(d["classes"]) != list(ck["classes"]):
            raise SystemExit(f"{w}: the label list differs from the loader's")
        prepared_ids = set((prepared / "eval-ids.txt").read_text(encoding="utf-8").split())
        if not set(d["eval"]["ids"]) <= prepared_ids:
            raise SystemExit(f"{w}: some test pictures the loader expects are not in {prepared}; rerun prepare_images.py --final")
        train_cnn.build(ck)                                       # the weights fit the network they describe
        checked.append((w, ck, s, st, tag, pred))
        print(f"{Path(w).name}: width {s['width']}, {s['size']} px, {s['epochs']} epochs, "
              f"{'street photos' if s.get('street_weight') else 'no street photos'}, seed {ck['seed']} -> {pred.name}")

    (out / "preds").mkdir(parents=True, exist_ok=True)
    for w, ck, s, st, tag, pred in checked:
        t0 = time.perf_counter()
        d = loaded[ck["labels"]]
        ev = train_cnn.Pictures(prepared, "eval", s["size"], list(d["eval"]["ids"]), st["mean"], st["std"], device)
        net = train_cnn.build(ck)
        net.to(device)
        scores = train_cnn.predict(net, ev, device, s["precision"])
        if not np.isfinite(scores).all():
            raise SystemExit(f"{w}: {int((~np.isfinite(scores)).sum())} scores are not finite numbers; nothing saved for it")
        np.savez_compressed(pred, ids=d["eval"]["ids"], scores=scores.astype(np.float32), classes=np.asarray(d["classes"]))
        print(f"seed {ck['seed']}: {len(ev)} test pictures predicted in {time.perf_counter() - t0:.0f}s -> {pred.name} "
              "(NOT scored; score once with vega_score.py --final after the team freezes settings)", flush=True)


if __name__ == "__main__":
    main()
