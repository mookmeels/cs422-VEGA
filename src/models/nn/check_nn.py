"""Check train_nn.py on made-up data. Touches no real image, label or split.

    python check_nn.py

Uses the fake project from the shared code's vega_selftest.py (240 made-up images, 5 labels) and checks that:
  * the network learns the easy made-up labels, with and without a hidden layer;
  * nothing is learned from the images being scored: their scores do not change when other images are
    scored alongside them, or when the eval set is replaced;
  * the same seed gives the same scores and another seed gives different ones, also when the same command
    is run again in a new process. (The made-up data is too small to show the multi-thread effect described
    in train_nn.py; on real data, repeat a run and compare the two prediction files.)
  * the command line runs end to end for validation and for --final, and the table shows mean ± spread.
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_nn                                                 # noqa: E402  (also puts the shared code on the path)
import vega_data, vega_score, vega_selftest                     # noqa: E402


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        vega_selftest.fake_project(root, np.random.default_rng(0))
        d = vega_data.load(root=root)
        fit, ev = d["fit"], d["eval"]
        count = (fit["y"] > 0).sum(0)
        s = {**train_nn.SETTINGS, "hidden": 32, "epochs": 30, "batch": 32, "lr": 1e-2}

        a = train_nn.train(fit["x"], fit["y"], ev["x"], s, seed=1)
        lin = train_nn.train(fit["x"], fit["y"], ev["x"], {**s, "layers": 0}, seed=1)
        for name, sc in (("one hidden layer", a), ("no hidden layer", lin)):
            m = vega_score.metrics(sc, ev["truth"], count)
            assert sc.shape == (len(ev["ids"]), len(d["classes"])) and m["top1"] > 0.6, (name, m)
        print("learning: both the MLP and the softmax regression learn the made-up labels")

        # score the first 20 eval images next to wildly different "images": their scores must not move
        # (beyond float rounding), which they would if the scaling or the weights looked at the eval set
        junk = 1000 * np.random.default_rng(5).normal(size=(28, fit["x"].shape[1])).astype(np.float32)
        mixed = train_nn.train(fit["x"], fit["y"], np.vstack([ev["x"][:20], junk]), s, seed=1)
        assert np.allclose(mixed[:20], a[:20], atol=1e-4), "an image's score depends on which other images are scored"
        w = train_nn.image_weights(fit["y"], 0.5)
        assert abs(w.mean() - 1) < 1e-5 and np.allclose(train_nn.image_weights(fit["y"], 0), 1)
        scale = train_nn.input_scaling(fit["x"], "standardize")
        z = scale(fit["x"])
        assert np.abs(z.mean(0)).max() < 1e-4 and np.abs(z.std(0) - 1).max() < 1e-3, "scaling is not the fit images' z-score"
        print("no leakage: scores do not depend on the other eval images; scaling comes from the fit images")

        assert np.array_equal(a, train_nn.train(fit["x"], fit["y"], ev["x"], s, seed=1)), "same seed, different scores"
        assert not np.array_equal(a, train_nn.train(fit["x"], fit["y"], ev["x"], s, seed=2)), "the seed changes nothing"

        env = {**os.environ, "VEGA_ROOT": str(root), "VEGA_SHARED": str(train_nn.SHARED)}
        base = [sys.executable, str(HERE / "train_nn.py"), "--hidden", "32", "--epochs", "30", "--batch", "32", "--lr", "0.01", "--seed", "1", "2", "3"]
        for extra in (["--curve"], ["--final", "--curve"]):
            out = subprocess.run(base + extra, env=env, capture_output=True, text=True)
            assert out.returncode == 0, out.stderr[-2000:]
        cmp = root / "runs" / "comparison"
        rows = [json.loads(l) for l in open(cmp / "results.jsonl", encoding="utf-8")]
        assert [r["split"] for r in rows] == ["validation"] * 3 + ["test"] * 3 and {r["owner"] for r in rows} == {"Andrew Nguyen"}
        assert all(r["top1"] > 0.6 for r in rows) and all('"epochs": 30' in r["note"] for r in rows), rows
        assert len((cmp / "FINAL-RUNS.log").read_text(encoding="utf-8").strip().split("\n")) == 3
        curves = sorted(p.name for p in (cmp / "curves").iterdir())
        assert len(curves) == 3, curves                          # written for validation only, never for --final
        assert len((cmp / "curves" / curves[0]).read_text(encoding="utf-8").strip().split("\n")) == 31
        assert subprocess.run([sys.executable, str(train_nn.SHARED / "vega_table.py")], env=env, capture_output=True, text=True).returncode == 0
        table = (cmp / "table.md").read_text(encoding="utf-8")
        assert "3 (1, 2, 3)" in table and "±" in table and "| test |" in table, table
        print("command line: validation and --final runs are scored, logged and tabulated with mean ± spread")

        again = subprocess.run(base + ["--out", str(root / "again")], env=env, capture_output=True, text=True)
        assert again.returncode == 0, again.stderr[-2000:]
        for seed in (1, 2, 3):
            one, two = (np.load(f / "preds" / f"nn-mlp1x32-validation-merged-seed{seed}.npz")["scores"] for f in (cmp, root / "again"))
            assert np.array_equal(one, two), f"seed {seed}: the same command gave different scores in a new process"
        print("seeds: the same seed repeats exactly, in one process and in a new one; another seed gives another network")
    print("all checks passed")


if __name__ == "__main__":
    main()
