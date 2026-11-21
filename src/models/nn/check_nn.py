"""Check train_nn.py on made-up data. Touches no real image, label or split.

    python check_nn.py

Uses the fake project from the shared code's vega_selftest.py (240 made-up images, 5 labels) and checks that:
  * the network learns the easy made-up labels, with and without a hidden layer;
  * nothing is learned from the images being scored: their scores do not change when other images are
    scored alongside them, or when the eval set is replaced;
  * the same seed gives the same scores and another seed gives different ones, also when the same command
    is run again in a new process. (The made-up data is too small to show the multi-thread effect described
    in train_nn.py; on real data, repeat a run and compare the two prediction files.)
  * the command line runs end to end for validation and for --final, and the table shows mean ± spread;
  * real outfit photos (made-up ones): the default command stops if they are missing; photos with a vague tag
    and photos of held-out posters are not trained on; the photos change the network; and with weight 0, or
    with no photos given, the network is exactly the wiki-only one;
  * folds by poster (real_photo_folds.py): a photo is scored only in the fold of its own poster's group, once,
    and a fold that is saved is never trained again.
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
        base = [sys.executable, str(HERE / "train_nn.py"), "--hidden", "32", "--epochs", "30", "--batch", "32", "--lr", "0.01",
                "--real-weight", "0", "--seed", "1", "2", "3"]   # the wiki-only network first; real photos are checked at the end
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

        # real outfit photos. Without them the default command must stop and say so, not quietly train on wiki images only.
        small = [sys.executable, str(HERE / "train_nn.py"), "--hidden", "32", "--epochs", "30", "--batch", "32", "--lr", "0.01", "--seed", "1",
                 "--out", str(root / "with-real")]
        out = subprocess.run(small, env=env, capture_output=True, text=True)
        assert out.returncode != 0 and "--real-weight 0" in out.stderr and not (root / "with-real" / "results.jsonl").exists(), out.stderr[-500:]

        # made-up real photos: copies of fit images with a little noise, tagged with the FAMILY of their label
        fam_of = {"Alpha": "one", "Delta": "one", "Epsilon": "two", "Gamma": "two", "Zeta": "two"}
        (root / "runs" / "family-v1").mkdir(parents=True)
        (root / "runs" / "family-v1" / "families.csv").write_text("family,name\none,One\ntwo,Two\n", encoding="utf-8")
        (root / "runs" / "family-v1" / "label_family.csv").write_text(
            "wiki_label,family,family_name\n" + "".join(f"{c},{f},{f}\n" for c, f in fam_of.items()), encoding="utf-8")
        (root / "runs" / "family-v1" / "outside_family.csv").write_text(
            "source,style_or_tag,family,photos,posters\nStreet Fashion Style,TagOne,one,0,\nStreet Fashion Style,TagTwo,two,0,\n"
            "Street Fashion Style,Vague,,0,\nFashionStyle14,TagOne,two,0,\n", encoding="utf-8")
        group = lambda u: int(__import__("hashlib").sha1(u.encode("utf-8")).hexdigest(), 16) % train_nn.REAL_GROUPS
        held = lambda u: group(u) == 0                           # poster group 0 is the one left out by default
        names = [f"poster{i}" for i in range(400)]
        users = [u for u in names if not held(u)][:8] + [u for u in names if held(u)][:2]      # 8 posters to train on, 2 held out
        rng = np.random.default_rng(7)
        vec_dir = root / "emb" / "street-fashion-style" / f"{vega_data.VECTORS}-448"
        vec_dir.mkdir(parents=True)
        man, ids, vecs, expect = ["id,user,pic,path,style"], [], [], 0
        for j in range(140):
            src = j % len(fit["ids"])
            family = fam_of[d["classes"][int(fit["truth"][src][0])]]
            style = "Vague" if j % 14 == 13 else {"one": "TagOne", "two": "TagTwo"}[family]
            user = users[j % len(users)]
            v = fit["x"][src] + 0.05 * rng.normal(size=fit["x"].shape[1])
            man.append(f"real-{j:04d},{user},{j}.jpg,x/{j}.jpg,{style}")
            ids.append(f"real-{j:04d}")
            vecs.append((v / np.linalg.norm(v)).astype(np.float32))
            expect += style != "Vague" and not held(user)
        (root / "emb" / "street-fashion-style" / "manifest.csv").write_text("\n".join(man) + "\n", encoding="utf-8")
        (vec_dir / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        np.save(vec_dir / "embeddings.npy", np.asarray(vecs, np.float32))

        real = train_nn.real_photos(root, d["classes"])
        assert real["photos"] == expect and 0 < expect < 130 and real["families"] == 2, (real["photos"], expect)   # no vague tag, no held-out poster
        assert tuple(real["mask"].shape) == (2, 5) and (real["mask"] == 0).sum(1).tolist() == [2, 3], real["mask"]
        sr = {**s, "real_weight": 0.25}
        b = train_nn.train(fit["x"], fit["y"], ev["x"], sr, seed=1, real=real)
        assert vega_score.metrics(b, ev["truth"], count)["top1"] > 0.6, "the network with real photos does not learn the made-up labels"
        assert np.array_equal(b, train_nn.train(fit["x"], fit["y"], ev["x"], sr, seed=1, real=real)), "real photos: same seed, different scores"
        assert not np.array_equal(b, a), "the real photos change nothing"
        assert np.array_equal(a, train_nn.train(fit["x"], fit["y"], ev["x"], {**s, "real_weight": 0}, seed=1, real=real)), "weight 0 is not the wiki-only network"
        assert np.array_equal(a, train_nn.train(fit["x"], fit["y"], ev["x"], sr, seed=1)), "without photos the network must be the wiki-only one"
        out = subprocess.run(small, env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
        row = json.loads(open(root / "with-real" / "results.jsonl", encoding="utf-8").readline())
        assert row["model"].endswith("+ real outfit photos") and row["top1"] > 0.6 and '"real_weight": 0.25' in row["note"], row
        assert (root / "with-real" / "preds" / "nn-mlp1x32real-validation-merged-seed1.npz").exists()
        print("real photos: refused when missing; vague tags and held-out posters stay out; they change the network; weight 0 is the wiki-only network exactly")

        # folds by poster (real_photo_folds.py): leaving out another group changes who is trained on, and a photo is
        # only ever scored in the fold of its own poster's group
        table = train_nn.real_photo_table(root, d["classes"])[0]
        other = group(users[0])                                  # a group that is not 0 and has a poster
        assert other != 0 and len(table) == sum(1 for l in man[1:] if not l.endswith(",Vague"))
        alt = train_nn.real_photos(root, d["classes"], holdout=other)
        assert alt["photos"] == sum(1 for _, _, g, _, _ in table if g != other) and alt["photos"] != real["photos"]
        folds = [sys.executable, str(HERE / "real_photo_folds.py"), "--folds", "0", str(other), "--out", str(root / "folds")]
        assert subprocess.run(folds, env=env, capture_output=True, text=True).returncode != 0, "the fold script ran without being told which mode"
        out = subprocess.run(folds + ["--final"], env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
        seen = []
        for k in (0, other):
            z = np.load(root / "folds" / f"fold{k}-seed422.npz")
            assert len(z["ids"]) > 0 and all(group(str(u)) == k for u in z["poster"]), f"fold {k} scored a photo of another poster group"
            assert z["nn"].shape == z["wiki_only"].shape == (len(z["ids"]), 2) and not np.array_equal(z["nn"], z["wiki_only"])
            seen += z["ids"].tolist()
        assert len(seen) == len(set(seen)) == sum(1 for _, _, g, _, _ in table if g in (0, other))
        assert "INCOMPLETE" in (root / "folds" / "results.md").read_text(encoding="utf-8")
        assert len((root / "folds" / "FOLD-RUNS.log").read_text(encoding="utf-8").strip().split("\n")) == 2
        before = (root / "folds" / "fold0-seed422.npz").read_bytes()
        again = subprocess.run(folds + ["--final"], env=env, capture_output=True, text=True)
        assert again.returncode == 0 and again.stdout.count("already saved, not trained again") == 2, again.stdout[-800:]
        assert (root / "folds" / "fold0-seed422.npz").read_bytes() == before
        print("folds by poster: each photo is scored only in its own poster's fold, once; a saved fold is never trained again")
    print("all checks passed")


if __name__ == "__main__":
    main()
