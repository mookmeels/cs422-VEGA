"""The NN on real outfit photos it never trained on: ten folds by poster. Part of the FINAL RUN. Owner: Andrew Nguyen.

    python real_photo_folds.py --final        # after the freeze, once: every fold, then the report
    python real_photo_folds.py --report       # only rebuild results.md from the folds already saved
    python real_photo_folds.py --check        # does it run? two folds, one epoch, nothing kept, no score shown

Why. The wiki test split holds wiki images, and on those the NN with real photos and the wiki-only NN are level.
What the real photos bought can only be seen on real photos, and the two sets used so far are not untouched:
FashionStyle14 chose the weight, and poster group 0 was scored in that experiment (runs/real-photos-v1).

How. Street Fashion Style posters fall into 10 groups by name (sha1 mod 10; all photos of one poster are in one
group). For fold k the NN is trained exactly as train_nn.py trains it (wiki images plus real photos, settings
unchanged), except that poster group k is the one left out; then the photos of group k are scored. Over the ten
folds every photo with a mapped style tag is scored once, by a network that never saw its poster.
  * A photo is right when the family that gets the most probability in total (summed over its aesthetics) is the
    family its poster's tag stands for (frozen tables in runs/family-v1).
  * The wiki-only NN is scored on the same photos in the same pass. It never trains on real photos, so it needs
    no folds; the difference between the two is what the real photos bought.
  * Differences: paired bootstrap that resamples POSTERS, 2,000 draws, seed 422.

What it is and is not.
  * It measures the training recipe, not one network: ten networks are trained. The NN in the team table is fold 0.
  * With --final the wiki part is fitted on train + validation, as in the final run. The wiki test split is not
    scored here and its labels are never read.
  * The tags are the posters' own, and the networks learn from the same kind of tag, so this is agreement with
    posters, not with an expert. 12 of the 20 families have photos.
  * Fold 0 is not untouched: its photos were scored once on 21 Nov by a network fitted on the training split.
  * A fold that is already saved is never trained again, so the result cannot be re-rolled.

Writes to <root>/runs/real-photos-v1/folds: fold<k>-seed<s>.npz, results.md, results.json, FOLD-RUNS.log.
"""
import argparse, datetime, getpass, json, socket, sys, tempfile, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_nn                                                 # noqa: E402  (also puts the shared code on the path)

vega_data = train_nn.vega_data


def softmax(z):
    z = z.astype(np.float64)
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def run_fold(k, seed, fit, s, table, mm, family_of_label, root, classes):
    """Train both networks with poster group k left out; return what was scored for that group's photos."""
    rows, _, _, families = table
    held = sorted((r, f, poster, pid) for f, r, group, poster, pid in rows if group == k)
    if not held:
        return None
    x = np.ascontiguousarray(mm[[r for r, _, _, _ in held]])
    real = train_nn.real_photos(root, classes, holdout=k)
    with_photos = softmax(train_nn.train(fit["x"], fit["y"], x, s, seed, real=real)) @ family_of_label
    photos = real["photos"]
    del real
    wiki_only = softmax(train_nn.train(fit["x"], fit["y"], x, s, seed)) @ family_of_label
    return {"ids": np.asarray([pid for _, _, _, pid in held]), "family": np.asarray([f for _, f, _, _ in held]),
            "poster": np.asarray([poster for _, _, poster, _ in held]), "nn": with_photos.astype(np.float16),
            "wiki_only": wiki_only.astype(np.float16), "families": np.asarray(families), "trained_on": photos}


def report(out, final):
    """Build results.md from the saved folds. Says so plainly when folds are missing."""
    files = sorted(out.glob("fold*-seed*.npz"))
    if not files:
        raise SystemExit(f"no saved fold in {out}")
    seeds = sorted({int(f.stem.split("seed")[1]) for f in files})
    folds = sorted({int(f.stem.split("-")[0][4:]) for f in files})
    complete = [s for s in seeds if all((out / f"fold{k}-seed{s}.npz").exists() for k in folds)]
    hit, fam, poster, fold_of, families = {"nn": {}, "wiki_only": {}}, {}, {}, {}, None
    for s in complete:
        for k in folds:
            z = np.load(out / f"fold{k}-seed{s}.npz", allow_pickle=False)
            families = [str(f) for f in z["families"]]
            for i, pid in enumerate(z["ids"].tolist()):
                fam[pid], poster[pid], fold_of[pid] = int(z["family"][i]), str(z["poster"][i]), k
            for m in hit:
                right = z[m].astype(np.float32).argmax(1) == z["family"]
                for pid, r in zip(z["ids"].tolist(), right):
                    hit[m].setdefault(pid, []).append(float(r))
    ids = sorted(fam)
    a = np.asarray([np.mean(hit["nn"][i]) for i in ids])
    b = np.asarray([np.mean(hit["wiki_only"][i]) for i in ids])
    f = np.asarray([fam[i] for i in ids])
    g = np.asarray([fold_of[i] for i in ids])
    names = sorted(set(poster.values()))
    where = {u: j for j, u in enumerate(names)}
    p = np.asarray([where[poster[i]] for i in ids])

    rng = np.random.default_rng(422)
    tot, cnt = np.bincount(p, weights=a - b), np.bincount(p).astype(np.float64)
    pick = rng.integers(0, len(tot), size=(2000, len(tot)))
    boot = 100 * tot[pick].sum(1) / cnt[pick].sum(1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    covered = sorted(set(f.tolist()))
    each = lambda v: float(np.mean([v[f == c].mean() for c in covered]))

    whole = folds == list(range(train_nn.REAL_GROUPS))
    L = ["# The NN on real outfit photos it never trained on: folds by poster" + ("" if final else " (NOT the final run)"), ""]
    if not whole:
        L += [f"**INCOMPLETE: folds {folds} of 0 to {train_nn.REAL_GROUPS - 1}. The figures below cover only those folds.**", ""]
    L += [f"{len(ids):,} Street Fashion Style photos of {len(names):,} posters, {len(covered)} families, seed{'s' if len(complete) > 1 else ''} "
          f"{', '.join(map(str, complete))}. Each photo is scored by a network that never trained on its poster. "
          "Right = the family given the most probability in total is the family the poster's tag stands for.", "",
          "| model | family top-1, all photos | the same, each family equal |", "|---|---|---|",
          f"| NN (wiki + real outfit photos), cross-validated | {100 * a.mean():.1f} | {100 * each(a):.1f} |",
          f"| wiki-only NN, same photos | {100 * b.mean():.1f} | {100 * each(b):.1f} |", "",
          f"Difference, NN minus wiki-only NN: {100 * (a - b).mean():+.1f} points (95% interval {lo:+.1f} to {hi:+.1f}; "
          "paired bootstrap over posters, 2,000 draws, seed 422).", "",
          "## By fold (the poster group left out)", "", "| fold | photos | NN | wiki-only NN |", "|---|---|---|---|"]
    for k in folds:
        note = " (scored once before, on 21 Nov)" if k == 0 else ""
        L.append(f"| {k}{note} | {int((g == k).sum()):,} | {100 * a[g == k].mean():.1f} | {100 * b[g == k].mean():.1f} |")
    fa = [100 * a[g == k].mean() for k in folds]
    L += ["", f"Across folds the NN ranges from {min(fa):.1f} to {max(fa):.1f} (standard deviation {np.std(fa, ddof=1) if len(fa) > 1 else 0:.1f}).", "",
          "## By family", "", "| family | photos | NN | wiki-only NN |", "|---|---|---|---|"]
    for c in sorted(covered, key=lambda c: -(f == c).sum()):
        L.append(f"| {families[c]} | {int((f == c).sum()):,} | {100 * a[f == c].mean():.1f} | {100 * b[f == c].mean():.1f} |")
    L += ["", "Read with: ten networks were trained, so this measures the recipe; the tags are the posters' own; 8 of the 20 families "
          "have no photos here. See the top of real_photo_folds.py."]
    (out / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps({
        "final": final, "seeds": complete, "folds": folds, "photos": len(ids), "posters": len(names),
        "nn": 100 * a.mean(), "wiki_only": 100 * b.mean(), "nn_each_family_equal": 100 * each(a), "wiki_only_each_family_equal": 100 * each(b),
        "difference": 100 * (a - b).mean(), "interval": [lo, hi]}, indent=1), encoding="utf-8")
    return L, ids, g, poster


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true", help="the real run, after the freeze: wiki part fitted on train + validation")
    ap.add_argument("--report", action="store_true", help="only rebuild results.md from the saved folds")
    ap.add_argument("--check", action="store_true", help="does it run? two folds, one epoch, nothing kept, no score shown")
    ap.add_argument("--seed", type=int, nargs="+", default=[422])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(train_nn.REAL_GROUPS)), choices=range(train_nn.REAL_GROUPS))
    ap.add_argument("--root", default=None, help="default $VEGA_ROOT or /srv/ml/vega")
    ap.add_argument("--out", default=None, help="default <root>/runs/real-photos-v1/folds")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    if a.final + a.report + a.check != 1:
        raise SystemExit("Say which: --final (the real run, after the freeze), --report (rebuild the table), or --check (does it run?).")
    if a.threads:
        torch.set_num_threads(a.threads)
    root = Path(a.root or vega_data.ROOT)
    out = Path(a.out) if a.out else root / "runs" / "real-photos-v1" / "folds"
    if a.report:
        saved = json.loads((out / "results.json").read_text(encoding="utf-8"))["final"] if (out / "results.json").exists() else True
        print("\n".join(report(out, saved)[0]))
        return

    t0 = time.perf_counter()
    s = dict(train_nn.SETTINGS)
    keep = None
    if a.check:
        keep = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        out, s["epochs"], a.folds, a.seed = Path(keep.name), 1, a.folds[:2], a.seed[:1]
    out.mkdir(parents=True, exist_ok=True)
    d = vega_data.load(final=a.final, root=root)
    classes, fit = d["classes"], d["fit"]
    table = train_nn.real_photo_table(root, classes)
    rows, vec, lab_fam, families = table
    family_of_label = np.zeros((len(classes), len(families)))
    family_of_label[np.arange(len(classes)), lab_fam] = 1
    mm = np.load(vec / "embeddings.npy", mmap_mode="r")
    print(f"{'FINAL' if a.final else 'CHECK'}: wiki fit {len(fit['ids'])} images ({'train + validation' if a.final else 'train'}), "
          f"{len(rows)} real photos with a mapped tag, folds {a.folds}, seeds {a.seed}, {s['epochs']} epoch(s)", flush=True)
    for seed in a.seed:
        for k in a.folds:
            path = out / f"fold{k}-seed{seed}.npz"
            if path.exists():
                print(f"  fold {k} seed {seed}: already saved, not trained again", flush=True)
                continue
            t1 = time.perf_counter()
            r = run_fold(k, seed, fit, s, table, mm, family_of_label, root, classes)
            if r is None:
                print(f"  fold {k} seed {seed}: no poster in this group, skipped", flush=True)
                continue
            np.savez_compressed(path, **{key: v for key, v in r.items() if key != "trained_on"})
            if a.final:
                with open(out / "FOLD-RUNS.log", "a", encoding="utf-8") as f:
                    f.write(f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S} UTC  {getpass.getuser()}@{socket.gethostname()}  "
                            f"fold {k} seed {seed}: {len(r['ids'])} photos scored, trained on {r['trained_on']} others\n")
            print(f"  fold {k} seed {seed}: {len(r['ids'])} photos scored by networks that trained on {r['trained_on']} others; "
                  f"{time.perf_counter() - t1:.0f}s", flush=True)
    L, ids, g, poster = report(out, a.final)
    if a.check:
        groups = {u: int(__import__("hashlib").sha1(u.encode("utf-8")).hexdigest(), 16) % train_nn.REAL_GROUPS for u in set(poster.values())}
        assert all(groups[poster[i]] == k for i, k in zip(ids, g)), "a photo was scored in a fold that is not its poster's group"
        assert len(ids) == len(set(ids)), "a photo was scored twice"
        print(f"check passed: {len(ids)} photos scored in folds {sorted(set(g.tolist()))}, each in its own poster's fold, none twice; "
              f"the report was built; nothing was kept and no score is shown. {time.perf_counter() - t0:.0f}s")
        keep.cleanup()
    else:
        print("\n".join(L))
        print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
