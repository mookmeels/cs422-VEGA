"""Check prepare_images.py and train_cnn.py on made-up pictures. Touches no real picture, label or split.

    python check_cnn.py

Builds the team's fake project (vega_selftest.py: 240 made-up images, 5 labels) in a temporary folder and draws a
picture for every image: each label has its own colour and stripe direction, with noise, at assorted sizes and
shapes, plus one see-through PNG and one greyscale picture. Then checks that:
  1. picture preparation keeps the loader's order, makes every bar (fit and eval) the fit average colour, keeps grey grey,
     puts white behind see-through areas, does nothing on a rerun, and stops naming the file when a picture
     cannot be opened;
  2. the colour averages come from fit pictures only: making every eval picture bright red changes nothing
     in the fit pictures, the padding colour or the channel scaling;
  3. the CNN learns the made-up labels (top-1 far above the 20% of guessing);
  4. an eval picture's scores do not depend on which other pictures are scored with it;
  5. the same seed gives the same scores on the CPU, and another seed gives different ones;
  6. the command line runs end to end: validation runs are scored by the team's scorer and get a learning curve;
     a --final run saves test predictions but never scores them and never touches FINAL-RUNS.log; whether
     MIOpen was switched off (--no-miopen) is recorded in the run log, the scorer's note and the weights;
     a repeat of the same setting and seed into the same folder stops before training (no file replaced);
     the final preparation's test pictures have fit-coloured bars; train_cnn.py refuses a seed listed twice,
     --final --curve, street options without --street, and a preparation made for the other mode.
  7. the final run, option A: --final --fit-only trains with no test pictures present and saves weights only;
     predict_cnn.py gives the same test scores as train_cnn.py's own final prediction, and refuses
     validation weights and existing files; it also refuses (writing nothing) weights with another colour
     scaling or split, a validation preparation, and two weights files that would write one prediction file.
  8. street photos: prepare_street.py keeps tags with enough photos, skips a broken photo, marks held-out
     posters, and stops without marking anything complete when most photos fail or --limit is 0; photos are
     drawn tag-evenly and never from held-out posters; weight 0 gives exactly the plain
     CNN; --street learns the tags (held-out posters well above chance); street models load everywhere.
Runs on the CPU in about a minute or two.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_cnn                                                  # noqa: E402  (also puts the shared code on the path)
import vega_data, vega_score, vega_selftest                       # noqa: E402

COLOURS = [(200, 40, 40), (40, 160, 60), (50, 70, 210), (220, 200, 40), (150, 60, 170)]
PASSED = []


def check(ok, what):
    if not ok:
        raise SystemExit(f"FAIL {what}")
    PASSED.append(what)
    print(f"PASS {what}", flush=True)


def draw_pictures(root, rng):
    """One made-up picture per image: its first label sets the colour and the stripe direction."""
    d = vega_data.load(root=root, vectors=False)
    fin = vega_data.load(final=True, root=root, vectors=False)
    label = {sha: int(t[0]) for part in (d["fit"], d["eval"]) for sha, t in zip(part["ids"], part["truth"])}
    paths = {sha: p for part in (d["fit"], d["eval"], fin["eval"]) for sha, p in zip(part["ids"], part["paths"])}
    for sha in fin["eval"]["ids"]:
        label.setdefault(sha, int(sha, 16) % 5)                   # test images: any label, never read by the checks
    special = sorted(d["fit"]["ids"])[:2]                         # one see-through PNG, one greyscale JPEG
    for n, (sha, path) in enumerate(sorted(paths.items())):
        w, h = int(rng.integers(40, 140)), int(rng.integers(40, 140))
        k = label[sha]
        yy, xx = np.mgrid[0:h, 0:w]
        stripes = ((xx if k % 2 else yy) // 6 % 2).astype(np.float32)
        img = np.asarray(COLOURS[k], np.float32) * (0.6 + 0.4 * stripes[..., None])
        img = np.clip(img + rng.normal(0, 20, img.shape), 0, 255).astype(np.uint8)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if sha == special[0]:
            a = np.full((h, w), 255, np.uint8)
            a[:, : w // 2] = 0                                    # left half see-through
            Image.fromarray(np.dstack([img, a]), "RGBA").save(p, "PNG")   # .jpg name, PNG inside, as on the wiki
        elif sha == special[1]:
            Image.fromarray(img).convert("L").save(p, "JPEG", quality=95)
        else:
            Image.fromarray(img).save(p, "JPEG", quality=95)
    return special


def bars_wrong(folder, part, size, pad):
    """How many pictures of this prepared part have a bar pixel that is not the padding colour."""
    arr = np.load(folder / f"{part}-{size}.npy")
    bad = 0
    for i, (x0, y0, w, h) in enumerate(np.load(folder / f"boxes-{size}.npz")[part]):
        m = np.ones((size, size), bool)
        m[y0:y0 + h, x0:x0 + w] = False
        bad += int((arr[i][m] != pad).any())
    return bad


def run(args, env):
    return subprocess.run([sys.executable] + [str(a) for a in args], env=env, capture_output=True, text=True)


def main():
    torch.set_num_threads(min(2, torch.get_num_threads()))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        root = tmp / "project"
        root.mkdir()
        vega_selftest.fake_project(root, np.random.default_rng(0))
        see_through, grey = draw_pictures(root, np.random.default_rng(1))
        env = {**os.environ, "VEGA_ROOT": str(root), "VEGA_SHARED": str(train_cnn.shared_code())}
        prep = [HERE / "prepare_images.py", "--sizes", "64", "--workers", "2"]

        # 1. picture preparation
        out = run(prep, env)
        check(out.returncode == 0, "prepare_images.py runs on the fake project" + ("" if out.returncode == 0 else "\n" + out.stderr[-1500:]))
        folder = root / "runs" / "cnn" / "prepared" / "validation"
        d = vega_data.load(root=root, vectors=False)
        for part in ("fit", "eval"):
            ids = (folder / f"{part}-ids.txt").read_text(encoding="utf-8").split()
            check(ids == list(d[part]["ids"]), f"{part}: every picture once, in the loader's order")
        stats = json.loads((folder / "stats.json").read_text(encoding="utf-8"))
        pad = np.asarray(stats["per_size"]["64"]["pad_rgb"], np.uint8)
        fit = np.load(folder / "fit-64.npy")
        boxes = np.load(folder / "boxes-64.npz")["fit"]
        bad = 0
        for i, (x0, y0, w, h) in enumerate(boxes):
            m = np.ones((64, 64), bool)
            m[y0:y0 + h, x0:x0 + w] = False
            bad += int((fit[i][m] != pad).any()) + int(not (w == 64 or h == 64))
        check(bad == 0, f"every bar is the fit average colour {pad.tolist()}, and each picture's long side fills the square")
        check(bars_wrong(folder, "eval", 64, pad) == 0, "every eval picture's bar is the fit average colour too (scan, 21 Nov)")
        ids = list(d["fit"]["ids"])
        gx, gy, gw, gh = boxes[ids.index(grey)]
        g = fit[ids.index(grey)][gy:gy + gh, gx:gx + gw]              # the picture itself; the bars are the average colour
        check((g[..., 0] == g[..., 1]).all() and (g[..., 1] == g[..., 2]).all(), "a greyscale picture stays grey")
        x0, y0, w, h = boxes[ids.index(see_through)]
        left = fit[ids.index(see_through)][y0:y0 + h, x0:x0 + w // 2 - 2]
        check(left.min() >= 250, "see-through areas become white")
        out = run(prep, env)
        check(out.returncode == 0 and "nothing to do" in out.stdout, "a rerun with nothing changed does nothing")

        # 2. colour averages from fit pictures only
        red_root = tmp / "red"
        shutil.copytree(root, red_root, ignore=shutil.ignore_patterns("cnn"))   # keep the label files under runs/
        for p in d["eval"]["paths"]:
            q = red_root / Path(p).relative_to(root)
            Image.new("RGB", (90, 50), (255, 0, 0)).save(q, "JPEG")
        out = run(prep + ["--root", red_root], {**env, "VEGA_ROOT": str(red_root)})
        check(out.returncode == 0, "prepare_images.py runs with every eval picture made red" + ("" if out.returncode == 0 else "\n" + out.stderr[-1500:]))
        red = red_root / "runs" / "cnn" / "prepared" / "validation"
        red_stats = json.loads((red / "stats.json").read_text(encoding="utf-8"))
        same = all(red_stats["per_size"]["64"][k] == stats["per_size"]["64"][k] for k in ("pad_rgb", "mean", "std"))
        check(same and np.array_equal(np.load(red / "fit-64.npy"), fit),
              "red eval pictures change nothing in the fit pictures, the padding colour or the channel scaling")

        broken_root = tmp / "broken"
        shutil.copytree(root, broken_root, ignore=shutil.ignore_patterns("cnn"))   # keep the label files under runs/
        victim = broken_root / Path(d["fit"]["paths"][3]).relative_to(root)
        victim.write_bytes(b"this is not a picture")
        out = run(prep + ["--root", broken_root], {**env, "VEGA_ROOT": str(broken_root)})
        check(out.returncode != 0 and victim.name in (out.stdout + out.stderr), "a picture that cannot be opened stops the run, naming the file")

        # 3 to 5. learning, no leakage between scored pictures, seeds
        st = stats["per_size"]["64"]
        cpu = torch.device("cpu")
        fitp = train_cnn.Pictures(folder, "fit", 64, list(d["fit"]["ids"]), st["mean"], st["std"], cpu)
        evp = train_cnn.Pictures(folder, "eval", 64, list(d["eval"]["ids"]), st["mean"], st["std"], cpu)
        s = {**train_cnn.SETTINGS, "width": 8, "size": 64, "epochs": 12, "batch": 16, "lr": 3e-3}
        count = (d["fit"]["y"] > 0).sum(0)
        scores, net, _ = train_cnn.train(fitp, d["fit"]["y"], evp, s, 1, cpu)
        m = vega_score.metrics(scores, d["eval"]["truth"], count)
        check(scores.shape == (len(d["eval"]["ids"]), len(d["classes"])) and m["top1"] > 0.6,
              f"the CNN learns the made-up labels (top-1 {100 * m['top1']:.0f}%, guessing would give 20%)")
        few = train_cnn.Pictures(folder, "eval", 64, list(d["eval"]["ids"][:7]), st["mean"], st["std"], cpu)
        alone = train_cnn.predict(net, few, cpu, "fp32")
        check(np.allclose(alone, scores[:7], atol=1e-4), "an eval picture's scores do not depend on the other pictures scored with it")
        again, _, _ = train_cnn.train(fitp, d["fit"]["y"], evp, s, 1, cpu)
        other, _, _ = train_cnn.train(fitp, d["fit"]["y"], evp, s, 2, cpu)
        check(np.array_equal(again, scores), "the same seed gives the same scores (CPU)")
        check(not np.array_equal(other, scores), "another seed gives different scores")

        # 6. the command line, validation and final
        cmd = [HERE / "train_cnn.py", "--width", "8", "--size", "64", "--epochs", "12", "--batch", "16", "--lr", "0.003",
               "--device", "cpu", "--threads", "2", "--seed", "1", "2"]
        out = run(cmd + ["--curve", "--no-miopen"], env)
        check(out.returncode == 0, "train_cnn.py runs for validation" + ("" if out.returncode == 0 else "\n" + out.stderr[-1500:]))
        vout = root / "runs" / "cnn" / "validation"
        rows = [json.loads(line) for line in open(vout / "results.jsonl", encoding="utf-8")]
        check(len(rows) == 2 and all(r["owner"] == train_cnn.OWNER and r["split"] == "validation" for r in rows),
              "each validation seed is scored by the team's scorer, under Giancarlo's name")
        curve = (vout / "curves" / "cnn-r18w8-64-merged-seed1.csv").read_text(encoding="utf-8").strip().split("\n")
        check(len(curve) == 13, "the learning curve has a row per epoch")
        n_runs = len(open(vout / "runs.jsonl", encoding="utf-8").read().splitlines())
        out = run(cmd + ["--curve", "--no-miopen"], env)
        check(out.returncode != 0 and "would be replaced" in out.stderr and "seed1.npz" in out.stderr
              and len(open(vout / "runs.jsonl", encoding="utf-8").read().splitlines()) == n_runs,
              "a repeat into the same folder stops before training and replaces nothing")
        out = run(prep + ["--final"], env)
        check(out.returncode == 0, "prepare_images.py --final runs")
        fpad = np.asarray(json.loads((root / "runs/cnn/prepared/final/stats.json").read_text(encoding="utf-8"))
                          ["per_size"]["64"]["pad_rgb"], np.uint8)
        check(bars_wrong(root / "runs/cnn/prepared/final", "eval", 64, fpad) == 0,
              "in the final preparation every test picture's bar is the fit average colour")
        refusals = {
            "a seed is listed twice": run(cmd + ["1"], env),
            "no learning curve": run(cmd + ["--final", "--curve"], env),
            "need --street": run(cmd + ["--street-weight", "0.5"], env),
            "was prepared for a final run": run(cmd + ["--prepared", root / "runs/cnn/prepared/final"], env)}
        check(all(o.returncode != 0 and why in o.stderr for why, o in refusals.items()),
              "train_cnn.py refuses a repeated seed, --final --curve, street options without --street, and a "
              "preparation made for the other mode" + "".join(f"\n{w}: {o.stderr[-300:]}" for w, o in refusals.items()
                                                            if w not in o.stderr))
        out = run(cmd + ["--final"], env)
        fout = root / "runs" / "cnn" / "final"
        check(out.returncode == 0 and (fout / "preds" / "cnn-r18w8-64-test-merged-seed2.npz").exists()
              and not (fout / "results.jsonl").exists() and not list(root.rglob("FINAL-RUNS.log")),
              "--final saves test predictions but never scores them or writes FINAL-RUNS.log")
        vruns = [json.loads(line) for line in open(vout / "runs.jsonl", encoding="utf-8")]
        fruns = [json.loads(line) for line in open(fout / "runs.jsonl", encoding="utf-8")]
        vnote = [json.loads(line)["note"] for line in open(vout / "results.jsonl", encoding="utf-8")]
        vweights = torch.load(vout / "weights" / "cnn-r18w8-64-validation-seed1.pt", map_location="cpu", weights_only=True)   # never run code hidden in a file (scan, 21 Nov 2026)
        check(all(r["miopen"] is False for r in vruns) and all(r["miopen"] is True for r in fruns)
              and all("miopen off" in n for n in vnote) and vweights["miopen"] is False,
              "the MIOpen choice is recorded: off with --no-miopen, on by default (run log, scorer's note, weights)")
        p = np.load(fout / "preds" / "cnn-r18w8-64-test-merged-seed2.npz")
        fin = vega_data.load(final=True, root=root, vectors=False)
        check(list(p["ids"]) == list(fin["eval"]["ids"]) and list(p["classes"]) == fin["classes"],
              "test predictions are in the team's format, ready for vega_score.py --final")

        # 7. the final run, option A: train on the PC without test pictures, predict the test split on ml-db
        prep_final = root / "runs" / "cnn" / "prepared" / "final"
        no_test = tmp / "prepared-fit-only"                       # what the PC gets: the fit part only
        no_test.mkdir()
        for f in ("fit-64.npy", "fit-ids.txt", "stats.json"):
            shutil.copy(prep_final / f, no_test / f)
        fo = root / "runs" / "cnn" / "final-fit-only"
        out = run(cmd + ["--final", "--fit-only", "--prepared", no_test, "--out", fo], env)
        check(out.returncode == 0 and all((fo / "weights" / f"cnn-r18w8-64-test-seed{k}.pt").exists() for k in (1, 2))
              and not list((fo / "preds").glob("*.npz")),
              "--final --fit-only trains without any test pictures and saves weights only" +
              ("" if out.returncode == 0 else "\n" + out.stderr[-1500:]))
        pc = root / "runs" / "cnn" / "final-predicted"
        out = run([HERE / "predict_cnn.py", "--weights", fout / "weights" / "cnn-r18w8-64-test-seed2.pt",
                   "--device", "cpu", "--threads", "2", "--out", pc], env)
        mine = np.load(pc / "preds" / "cnn-r18w8-64-test-merged-seed2.npz") if out.returncode == 0 else None
        check(mine is not None and list(mine["ids"]) == list(p["ids"]) and np.allclose(mine["scores"], p["scores"], atol=1e-5),
              "predict_cnn.py gives the same test scores as train_cnn.py's own final prediction" +
              ("" if out.returncode == 0 else "\n" + out.stderr[-1500:]))
        out = run([HERE / "predict_cnn.py", "--weights", fo / "weights" / "cnn-r18w8-64-test-seed1.pt",
                   "--device", "cpu", "--threads", "2", "--out", pc], env)
        bad = run([HERE / "predict_cnn.py", "--weights", vout / "weights" / "cnn-r18w8-64-validation-seed1.pt",
                   "--device", "cpu", "--out", pc], env)
        again = run([HERE / "predict_cnn.py", "--weights", fout / "weights" / "cnn-r18w8-64-test-seed2.pt",
                     "--device", "cpu", "--out", pc], env)
        check(out.returncode == 0 and (pc / "preds" / "cnn-r18w8-64-test-merged-seed1.npz").exists()
              and bad.returncode != 0 and "not made by a final run" in bad.stderr
              and again.returncode != 0 and "exists already" in again.stderr
              and not list(root.rglob("FINAL-RUNS.log")),
              "predict_cnn.py predicts fit-only weights, refuses validation weights and existing files, never scores")
        good = torch.load(fout / "weights" / "cnn-r18w8-64-test-seed2.pt", map_location="cpu", weights_only=True)
        tampered = {"colour": dict(good, colour_mean=[v + 0.01 for v in good["colour_mean"]]),
                    "split": dict(good, split_sha256="0" * 64)}
        for name, ck in tampered.items():
            (tmp / f"bad-{name}").mkdir()
            torch.save(ck, tmp / f"bad-{name}" / "cnn-r18w8-64-test-seed2.pt")
        (tmp / "twin").mkdir()
        shutil.copy(fout / "weights" / "cnn-r18w8-64-test-seed2.pt", tmp / "twin" / "cnn-r18w8-64-test-seed2.pt")
        pred = [HERE / "predict_cnn.py", "--device", "cpu", "--weights"]
        refused = {
            "different colour scaling": run(pred + [tmp / "bad-colour" / "cnn-r18w8-64-test-seed2.pt", "--out", tmp / "p1"], env),
            "another split file": run(pred + [tmp / "bad-split" / "cnn-r18w8-64-test-seed2.pt", "--out", tmp / "p2"], env),
            "not a final-run preparation": run(pred + [fout / "weights" / "cnn-r18w8-64-test-seed2.pt", "--prepared", folder,
                                                       "--out", tmp / "p3"], env),
            "would both write": run(pred + [fout / "weights" / "cnn-r18w8-64-test-seed2.pt",
                                            tmp / "twin" / "cnn-r18w8-64-test-seed2.pt", "--out", tmp / "p4"], env)}
        check(all(o.returncode != 0 and why in o.stderr for why, o in refused.items())
              and not any((tmp / f"p{k}" / "preds").exists() and list((tmp / f"p{k}" / "preds").glob("*.npz"))
                          for k in range(1, 5)),
              "predict_cnn.py refuses other colour scaling, another split, a validation preparation, and two weights "
              "files that would write one prediction file, writing nothing" +
              "".join(f"\n{w}: {o.stderr[-300:]}" for w, o in refused.items() if w not in o.stderr))
        # 8. street photos (a second output for their own style tags; Gianni, 20 Nov 2026)
        sdir = tmp / "street"
        tags = {"Red": (200, 40, 40), "Green": (40, 160, 60), "Blue": (50, 70, 210), "Rare": (220, 200, 40)}
        rng = np.random.default_rng(3)
        lines = ["id,user,pic,path,style"]
        k = 0
        for poster in range(40):
            for tag, colour in tags.items():
                for _ in range((1 if poster < 8 else 0) if tag == "Rare" else 3):   # Rare: too few photos, left out
                    path = f"chictopia/p{poster}/{k}.jpg"
                    (sdir / path).parent.mkdir(parents=True, exist_ok=True)
                    w, h = int(rng.integers(40, 120)), int(rng.integers(40, 120))
                    yy, xx = np.mgrid[0:h, 0:w]
                    img = np.asarray(colour, np.float32) * (0.6 + 0.4 * (((xx + yy) // 5) % 2)[..., None])
                    Image.fromarray(np.clip(img + rng.normal(0, 20, img.shape), 0, 255).astype(np.uint8)).save(sdir / path, "JPEG")
                    lines.append(f"sfs-{k:06d},p{poster},{k}.jpg,{path},{tag}")
                    k += 1
        broken = sdir / "chictopia/p5/0broken.jpg"
        broken.write_bytes(b"not a picture")
        lines.append("sfs-999999,p5,0broken.jpg,chictopia/p5/0broken.jpg,Red")
        (sdir / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        sprep = [HERE / "prepare_street.py", "--photos", sdir, "--manifest", sdir / "manifest.csv", "--size", "64",
                 "--min-photos", "20", "--workers", "2", "--out", tmp / "street-prepared"]
        out = run(sprep, env)
        sp = tmp / "street-prepared"
        info = json.loads((sp / "street.json").read_text(encoding="utf-8")) if out.returncode == 0 else {}
        held_posters = {f"p{i}" for i in range(40) if int(__import__("hashlib").sha1(f"p{i}".encode()).hexdigest(), 16) % 10 == 0}
        rows = [r.split(",") for r in (sp / "photos.csv").read_text(encoding="utf-8").split()[1:]] if info else []
        check(out.returncode == 0 and sorted((sp / "tags.txt").read_text(encoding="utf-8").split()) == ["Blue", "Green", "Red"]
              and info["failed"] == 1 and "sfs-999999" in (sp / "failed.txt").read_text(encoding="utf-8")
              and all((r[2] in held_posters) == (r[5] == "0") for r in rows) and len(held_posters) > 0
              and run(sprep, env).stdout.startswith("already prepared"),
              "prepare_street.py keeps tags with enough photos, skips and lists a broken photo, marks held-out posters, "
              "does nothing on a rerun" + ("" if out.returncode == 0 else "\n" + out.stderr[-1500:]))
        wrong = run([HERE / "prepare_street.py", "--photos", tmp / "no-such-folder", "--manifest", sdir / "manifest.csv",
                     "--size", "64", "--min-photos", "20", "--workers", "2", "--out", tmp / "street-wrong"], env)
        zero = run(sprep[:-1] + [tmp / "street-zero", "--limit", "0"], env)
        check(wrong.returncode != 0 and "could not be opened" in wrong.stderr and not (tmp / "street-wrong" / "street.json").exists()
              and zero.returncode != 0 and "--limit must be at least 1" in zero.stderr,
              "prepare_street.py stops, marking nothing complete, when most photos fail (a wrong --photos), and refuses --limit 0")
        street = train_cnn.Street(sp, 64, "train", st["pad_rgb"], st["mean"], st["std"], cpu)
        held = train_cnn.Street(sp, 64, "held_out", st["pad_rgb"], st["mean"], st["std"], cpu)
        drawn = street.draw(6000, np.random.default_rng(0))
        per = np.bincount(street.tag[drawn], minlength=3)
        check(abs(per - 2000).max() < 300 and not set(street.rows[drawn]) & set(held.rows) and len(held) > 0,
              f"street photos are drawn tag-evenly ({per.tolist()} of 6000) and never from held-out posters")
        x, y = street.batch(np.arange(3))
        check(x.shape == (3, 3, 64, 64) and torch.isfinite(x).all() and y.tolist() == street.tag[:3].tolist(),
              "a street batch comes out colour-scaled, with its tags")
        plain0, _, _ = train_cnn.train(fitp, d["fit"]["y"], evp, {**s, "street_weight": 0.0, "street_batch": 16}, 1, cpu,
                                       street=street)
        check(np.array_equal(plain0, scores), "street weight 0 gives exactly the CNN without street photos")
        sv = root / "runs" / "cnn" / "street-validation"
        out = run(cmd + ["--street", sp, "--street-batch", "16", "--out", sv], env)
        sruns = [json.loads(line) for line in open(sv / "runs.jsonl", encoding="utf-8")] if out.returncode == 0 else []
        sres = [json.loads(line) for line in open(sv / "results.jsonl", encoding="utf-8")] if out.returncode == 0 else []
        check(out.returncode == 0 and len(sruns) == 2 and all(r["tag"] == "cnn-r18w8-64-street" for r in sruns)
              and all(r["street_held_out"]["tag_top1"] > 0.6 for r in sruns)
              and all(r["model"].endswith("+ street photos") for r in sres),
              "train_cnn.py --street trains both outputs: street tags on held-out posters well above chance (33%), "
              "scored under its own name" + ("" if out.returncode == 0 else "\n" + out.stderr[-1500:]))
        sf = root / "runs" / "cnn" / "street-final"
        out = run(cmd + ["--final", "--fit-only", "--street", sp, "--street-batch", "16", "--prepared", no_test, "--out", sf], env)
        out2 = run([HERE / "predict_cnn.py", "--weights", sf / "weights" / "cnn-r18w8-64-street-test-seed1.pt",
                    "--device", "cpu", "--out", sf], env) if out.returncode == 0 else out
        out3 = run([HERE / "show_filters.py", "--weights", sv / "weights" / "cnn-r18w8-64-street-validation-seed1.pt",
                    "--device", "cpu", "--filters", "4", "--out", tmp / "street-filters"], env)
        check(out.returncode == 0 and out2.returncode == 0 and out3.returncode == 0
              and (sf / "preds" / "cnn-r18w8-64-street-test-merged-seed1.npz").exists(),
              "a street model trains for the final run without test pictures, and predict_cnn.py and show_filters.py load it"
              + "".join("\n" + o.stderr[-800:] for o in (out, out2, out3) if o.returncode))
    print(f"all {len(PASSED)} checks passed")


if __name__ == "__main__":
    main()
