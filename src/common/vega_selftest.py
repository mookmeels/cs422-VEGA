"""Check the shared loader and scorer on made-up data. Touches no real image, label or split.

    python vega_selftest.py

Builds a tiny fake project in a temporary folder (6 aesthetics, 2 of them merged, 240 images) and checks:
  * the metrics against values worked out by hand, including tied scores;
  * that a model never receives test labels, and that test is only scored with --final (and logged);
  * that the scorer refuses predictions with missing images, repeated ids or labels in the wrong order.
"""
import subprocess, sys, tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vega_data, vega_score                                    # noqa: E402


def fake_project(root, rng):
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
    merged = {"Alpha": "Alpha", "Beta": "Alpha", "Gamma": "Gamma", "Delta": "Delta", "Epsilon": "Epsilon", "Zeta": "Zeta"}
    img, txt = f"emb/{vega_data.VECTORS}-448", f"emb/{vega_data.VECTORS}-text"          # the folders the loader reads
    for d in ("wiki", "splits/wiki-split-v2", img, txt, "runs/labeler-wearable", "runs/labeler-wearable-merged"):
        (root / d).mkdir(parents=True)
    centres = rng.normal(size=(6, 16))
    man, split, ids, vecs = ["aesthetic,file,caption,sha1,path,width,height,mime,bytes"], ["sha1,sha256,split,codex_split,group,group_size"], [], []
    for i in range(240):
        sha = f"{i:040x}"
        a = i % 6
        man.append(f"{names[a]},f{i}.jpg,,{sha},images/{sha[:2]}/{sha}.jpg,1,1,image/jpeg,1")
        if i % 40 == 0:                                          # a few images sit in two galleries
            man.append(f"{names[(a + 2) % 6]},f{i}.jpg,,{sha},images/{sha[:2]}/{sha}.jpg,1,1,image/jpeg,1")
        split.append(f"{sha},x,{('train', 'train', 'train', 'validation', 'test')[(i // 6) % 5]},train,{sha},1")
        ids.append(sha)
        v = centres[a] + 0.8 * rng.normal(size=16)
        vecs.append(v / np.linalg.norm(v))
    (root / "wiki/manifest.csv").write_text("\n".join(man) + "\n", encoding="utf-8")
    (root / "splits/wiki-split-v2/split.csv").write_text("\n".join(split) + "\n", encoding="utf-8")
    (root / img / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    np.save(root / img / "embeddings.npy", np.asarray(vecs, np.float32))
    (root / txt / "titles.txt").write_text("\n".join(names + ["Dropped"]) + "\n", encoding="utf-8")
    t = np.vstack([centres, rng.normal(size=(1, 16))])
    np.save(root / txt / "text-intro.npy", (t / np.linalg.norm(t, axis=1, keepdims=True)).astype(np.float32))
    (root / "runs/labeler-wearable/catalog.csv").write_text(
        "aesthetic,catalog,images,top1,categories\n" + "".join(f"{n},wearable,40,0.5,x\n" for n in names) + "Dropped,dropped: not wearable,5,0.1,x\n", encoding="utf-8")
    (root / "runs/labeler-wearable-merged/label_map.csv").write_text(
        "aesthetic,images,merged_label,merged_images,how,tie_to,merged_label_without_images_only\n"
        + "".join(f"{n},40,{merged[n]},40,,,{merged[n]}\n" for n in names), encoding="utf-8")


def expect_exit(fn, what):
    try:
        fn()
    except SystemExit as e:
        assert isinstance(e.code, str), e.code
        return
    raise AssertionError(f"scorer accepted {what}")


def main():
    # 1. metrics by hand. 3 images, 4 labels. Tie order: a fixed permutation, so work out the expected ranks from it.
    tie = np.random.default_rng(vega_score.TIE_SEED).permutation(4)
    scores = np.asarray([[0.9, 0.1, 0.0, 0.0],      # truth {0}: rank 1
                         [0.2, 0.7, 0.1, 0.0],      # truth {0, 2}: best is label 0, rank 2
                         [0.5, 0.5, 0.5, 0.5]])     # truth {3}: all tied, rank decided by the fixed order
    truth = [np.asarray([0]), np.asarray([0, 2]), np.asarray([3])]
    r3 = 1 + int((tie < tie[3]).sum())
    m = vega_score.metrics(scores, truth, fit_count=np.asarray([20, 20, 3, 3]))
    assert abs(m["top1"] - (1 + (r3 == 1)) / 3) < 1e-12, m
    assert abs(m["mrr"] - (1 + 1 / 2 + 1 / r3) / 3) < 1e-12, m
    assert m["top5"] == 1.0 and m["coverage5"] == 1.0, m           # only 4 labels, so everything is in the top 5
    assert m["labels_scored"] == 3 and m["thin_labels_scored"] == 2, m
    flat = vega_score.metrics(np.zeros((400, 50)), [np.asarray([i % 50]) for i in range(400)], np.full(50, 20))
    assert abs(flat["top5"] - 0.10) < 1e-12 and abs(flat["top1"] - 0.02) < 1e-12, flat   # all tied = chance, not 100%
    print("metrics: match the hand-worked values; fully tied scores score at chance")

    # 1b. the family level by hand: labels 0 and 1 are family 0, labels 2 and 3 are family 1
    fam = np.asarray([0, 0, 1, 1])
    fm = vega_score.family_metrics(scores, truth, fam)
    first_of_tied = int(np.argmin(tie))                            # image 3: every label tied, so the fixed order decides
    # image 1: first guess label 0, family 0, right. image 2: first guess label 1 (a wrong label) but family 0, and the
    # image's labels 0 and 2 make its families {0, 1}: right. image 3: truth label 3, family 1.
    assert abs(fm["family_top1"] - (2 + (fam[first_of_tied] == 1)) / 3) < 1e-12, fm
    assert fm["family_top3"] == 1.0 and fm["n_families"] == 2, fm  # only two families, so the right one is always in the first 3
    assert abs(vega_score.family_metrics(scores, truth, np.asarray([0, 1, 2, 3]))["family_top1"] - m["top1"]) < 1e-12   # one family per label = top-1
    print("family level: matches the hand-worked values; with one family per label it equals top-1")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        fake_project(root, np.random.default_rng(0))
        # 2. what a model is given
        d = vega_data.load(root=root)
        assert d["classes"] == ["Alpha", "Delta", "Epsilon", "Gamma", "Zeta"], d["classes"]
        assert set(d["fit"]["split"]) == {"train"} and "y" in d["eval"] and len(d["eval"]["ids"]) == 48
        assert np.allclose(d["fit"]["y"].sum(1), 1) and (d["fit"]["y"] > 0).sum(1).max() == 2
        f = vega_data.load(final=True, root=root)
        assert set(f["fit"]["split"]) == {"train", "validation"} and len(f["eval"]["ids"]) == 48
        assert "y" not in f["eval"] and "truth" not in f["eval"], "test labels were handed to a model"
        assert not set(f["eval"]["ids"]) & set(f["fit"]["ids"]) and not set(d["eval"]["ids"]) & set(d["fit"]["ids"])
        print("loader: merged labels, multi-label rows, no fit/eval overlap, and no test labels given out")

        # 3. baselines end to end through the command line, validation then final
        env = {**__import__("os").environ, "VEGA_ROOT": str(root)}
        base = [sys.executable, str(HERE.parent / "baselines" / "baselines.py")]
        for extra in ([], ["--final"]):
            out = subprocess.run(base + extra, env=env, capture_output=True, text=True)
            assert out.returncode == 0, out.stderr[-2000:]
        cmp = root / "runs" / "comparison"
        rows = [__import__("json").loads(l) for l in open(cmp / "results.jsonl", encoding="utf-8")]
        assert [r["split"] for r in rows] == ["validation"] * 3 + ["test"] * 3
        cen = {r["split"]: r for r in rows if "centroid" in r["model"]}
        assert cen["validation"]["top1"] > 0.6 and cen["test"]["top1"] > 0.6, cen     # the fake classes are easy
        assert len((cmp / "FINAL-RUNS.log").read_text(encoding="utf-8").strip().split("\n")) == 3
        assert subprocess.run([sys.executable, str(HERE / "vega_table.py")], env=env, capture_output=True, text=True).returncode == 0
        assert "| test |" in (cmp / "table.md").read_text(encoding="utf-8")
        assert all("family_top1" not in r and r["family_table_sha256"] is None for r in rows), "family scores without a family table"
        print("baselines + table: run for validation and for --final; each test scoring is logged")

        # 3b. with a family table in place the same scoring pass adds the family level, for validation and for test
        (root / "runs" / "family-v1").mkdir(parents=True)
        (root / "runs" / "family-v1" / "label_family.csv").write_text(
            'wiki_label,family,family_name\nAlpha,one,"One, the first"\nDelta,one,"One, the first"\nEpsilon,two,Two\nGamma,two,Two\nZeta,two,Two\n', encoding="utf-8")
        for extra in ([], ["--final"]):
            out = subprocess.run(base + extra, env=env, capture_output=True, text=True)
            assert out.returncode == 0, out.stderr[-2000:]
        rows = [__import__("json").loads(l) for l in open(cmp / "results.jsonl", encoding="utf-8")][6:]
        assert [r["split"] for r in rows] == ["validation"] * 3 + ["test"] * 3
        assert all(r["family_top1"] >= r["top1"] and r["family_top3"] == 1.0 and r["n_families"] == 2 and r["family_table_sha256"] for r in rows), rows
        assert subprocess.run([sys.executable, str(HERE / "vega_table.py")], env=env, capture_output=True, text=True).returncode == 0
        assert "family top-1" in (cmp / "table.md").read_text(encoding="utf-8")
        (root / "runs" / "family-v1" / "label_family.csv").write_text("wiki_label,family,family_name\nAlpha,one,One\n", encoding="utf-8")
        assert vega_score.family_of(d["classes"], root) == (None, None), "a table that misses labels must be ignored, not half-used"
        (root / "runs" / "family-v1" / "label_family.csv").unlink()
        print("family table: scored in the same pass for validation and test; left out when the table is missing or incomplete")

        # 4. the scorer's refusals
        e, k = d["eval"], np.asarray(d["classes"])
        s = np.random.default_rng(1).normal(size=(len(e["ids"]), len(k)))
        p = root / "p.npz"
        perm = np.random.default_rng(2).permutation(len(e["ids"]))
        np.savez(p, ids=e["ids"][perm], scores=s[perm], classes=k)
        a = vega_score.score(p, "shuffled", "test", root=root, quiet=True)
        np.savez(p, ids=e["ids"], scores=s, classes=k)
        b = vega_score.score(p, "ordered", "test", root=root, quiet=True)
        assert a["top5"] == b["top5"] and a["mrr"] == b["mrr"], "row order changed the score"
        np.savez(p, ids=e["ids"][:-1], scores=s[:-1], classes=k)
        expect_exit(lambda: vega_score.score(p, "x", "x", root=root, quiet=True), "a file with a missing image")
        np.savez(p, ids=e["ids"], scores=s[:, ::-1], classes=k[::-1])
        expect_exit(lambda: vega_score.score(p, "x", "x", root=root, quiet=True), "labels in the wrong order")
        np.savez(p, ids=np.concatenate([e["ids"], e["ids"][:1]]), scores=np.vstack([s, s[:1]]), classes=k)
        expect_exit(lambda: vega_score.score(p, "x", "x", root=root, quiet=True), "a repeated image id")
        np.savez(p, ids=e["ids"], scores=s, classes=k)
        expect_exit(lambda: vega_score.score(p, "x", "x", final=True, root=root, quiet=True), "validation predictions as a test run")
        print("scorer: row order does not matter; missing images, repeated ids, wrong label order and wrong split are refused")
    print("all checks passed")


if __name__ == "__main__":
    main()
