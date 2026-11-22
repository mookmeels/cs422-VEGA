"""Copy what the course asks to be saved (configurations, split, result tables, the code that was run) into the team
repository's working copy. Nothing is committed; rerun it after a new result. Run on ml-db:

    /srv/ml/vega/.venv/bin/python -B /srv/ml/vega/code/experiments/repo_stage.py

Writes under /srv/ml/cs422-VEGA: configs/*.json, splits/wiki-split-v2/, results/, and the code that exists only in
/srv/ml/vega/code (src/app, src/experiments, the final-run scripts, src/embed). Files owned by another member's model
folder (src/models/cnn, src/models/hclustering) are not touched.
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

V, R = Path("/srv/ml/vega"), Path("/srv/ml/cs422-VEGA")
sys.path.insert(0, str(V / "code" / "shared"))
sys.path.insert(0, str(V / "code" / "nn"))
import train_nn  # noqa: E402
import vega_data  # noqa: E402
import vega_score  # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
done = []


# Paths under one member's Windows user folder are not copied into the repository: the repository's copies read the
# location from the environment instead. The scripts on the PC and in /srv/ml/vega/code keep their own defaults.
U = "Users"                                                              # (spelled apart so this file passes its own check)
SCRUB = [
    (rf'"C:/{U}/[^/"]+/miniconda3/python\.exe"', "sys.executable"),
    (rf'Path\(r"C:\\{U}\\[^\\"]+\\Documents\\[^"]*qwen-benchmark"\)',
     'Path(os.environ.get("VEGA_LLAMA_DIR", "llama-runtime"))             # the folder that holds runtime/llama-server.exe'),
    (rf'Path\(r"C:\\{U}\\[^\\"]+\\\.lmstudio"\)',
     'Path(os.environ.get("VEGA_LMSTUDIO_DIR", str(Path.home() / ".lmstudio")))   # the LM Studio folder: models and CUDA libraries'),
]
PRIVATE = re.compile(rf"C:[\\/]+{U}[\\/]".encode("ascii"))               # a file still holding a user-folder path is not copied
TEXT = {".py", ".sh", ".md", ".txt", ".json", ".csv", ".html", ".log", ".jsonl"}


def put(src, dst):
    src, dst = Path(src), R / dst
    if not src.exists():
        print("MISSING, skipped:", src)
        return
    data = src.read_bytes()
    if src.suffix in TEXT:
        for a, b in SCRUB:
            data = re.sub(a.encode("utf-8"), lambda m, b=b: b.encode("utf-8"), data)
        if PRIVATE.search(data):
            print("NOT COPIED, holds a private path:", src)
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.read_bytes() != data:
        dst.write_bytes(data)
        done.append(str(dst.relative_to(R)))


def put_json(obj, dst):
    dst = R / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=1, ensure_ascii=False) + "\n"
    if not dst.exists() or dst.read_text(encoding="utf-8") != text:
        dst.write_text(text, encoding="utf-8")
        done.append(str(dst.relative_to(R)))


# ---------------------------------------------------------------- configs
SEEDS = [422, 423, 424, 425, 426]
d = vega_data.load(root=V, vectors=False)
classes = [str(c) for c in d["classes"]]
_, family_sha = vega_score.family_of(classes, V)
counts = {"train": int(len(d["fit"]["ids"])), "validation": int(len(d["eval"]["ids"]))}
put_json({"what": "The shared protocol every model uses (src/common).",
          "split_file": "splits/wiki-split-v2/split.csv", "split_sha256": vega_data.split_sha256(V),
          "images": {**counts, "test": 2618}, "labels": len(classes), "label_set": "merged",
          "image_vectors": vega_data.VECTORS + "-448", "family_table": "runs/family-v1/label_family.csv", "family_table_sha256": family_sha,
          "primary_metric": "top5 (recall@5)", "supporting_metrics": ["top1", "mrr", "label_top5", "coverage5", "family_top1", "family_top3"],
          "ties": "equal scores are ordered by a fixed random order of the labels, seed 422 (vega_score.TIE_SEED)",
          "seeds_for_stochastic_models": SEEDS,
          "validation": "fit on train, score validation", "test": "fit on train + validation, score test once (results/comparison/FINAL-RUNS.log.txt)"},
         "configs/protocol.json")
meta = json.loads((V / "emb" / f"{vega_data.VECTORS}-448" / "meta.json").read_text(encoding="utf-8"))
put_json({"what": "How the image vectors were made (src/embed). The encoder is frozen; no model changes it.",
          "encoder": "Qwen3-VL-Embedding-2B, f16 GGUF (DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF), run with llama.cpp",
          "longest_edge_pixels": meta["edge"], "dimensions": meta["dimensions"], "pooling": meta["pooling"], "normalized": meta["normalized"],
          "add_eos_token": meta["add_eos_token"], "image_prompt": meta["image_prompt"], "images_embedded": meta["rows"],
          "images_listed": meta["images_listed"], "unreadable": meta["failed"], "seconds_on_rtx_4080": meta["seconds"], "finished": meta["finished"]},
         "configs/embedding.json")
put_json({"what": "Andrew Nguyen's model: an MLP on the frozen image vectors (src/models/nn/train_nn.py). These are train_nn.SETTINGS, the defaults of the script.",
          "settings": train_nn.SETTINGS, "real_photos_per_step": train_nn.REAL_BATCH, "poster_groups": train_nn.REAL_GROUPS,
          "held_out_poster_group": 0, "seeds": SEEDS, "device": "cpu, 1 thread (repeats exactly)",
          "rows": {"NN (MLP, 1 hidden layer of 1024)": "train_nn.py --seed 422 423 424 425 426 --real-weight 0",
                   "NN (MLP, 1 hidden layer of 1024) + real outfit photos": "train_nn.py --seed 422 423 424 425 426"},
          "final_run": "the same commands with --final (src/models/nn/nn_final_run.sh), run once on 22 Nov 2026 02:00 UTC",
          "search": "146 runs over 32 configurations on validation (sweep_nn.py); see src/models/nn/README.md",
          "train_nn_sha256": sha(V / "code" / "nn" / "train_nn.py")},
         "configs/nn.json")
cnn = {}
for line in open(V / "runs" / "cnn" / "final-from-pc" / "runs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    c = cnn.setdefault(r["tag"], {"model": r["model"], "settings": r["settings"], "seeds": [], "labels": r["labels"], "device": r["device"], "torch": r["torch"]})
    if r["seed"] not in c["seeds"]:
        c["seeds"].append(r["seed"])
put_json({"what": "Giancarlo Guevarra's model: a ResNet-18 trained from pixels (src/models/cnn). Settings as recorded by train_cnn.py for the final runs "
                  "(runs/cnn/final-from-pc/runs.jsonl). The adopted recipe is the row with crop, mixup/CutMix, label smoothing and oversampling.",
          "note_from_the_cnn_readme": "the records of the adopted runs still say shift 0.1 and zoom 0.1, which were not applied (random crop replaces them)",
          "runs": cnn, "commands": "see src/models/cnn/README.txt"},
         "configs/cnn.json")
hc = Path("/srv/ml/cs422-VEGA/src/models/hclustering/hclustering.py")
put_json({"what": "Edmond Luong's model: hierarchical (average-linkage) clustering on the image vectors (src/models/hclustering/hclustering.py).",
          "vectors": "standardized with the mean and standard deviation of the fit images", "similarity": "cosine", "linkage": "average",
          "cut": 0.16, "seeds": "none: the method has no randomness, so one run", "hclustering_sha256": sha(hc),
          "validation": "python hclustering.py, then vega_score.py --pred my-preds.npz --model 'Hclustering(standardized 0.16)'",
          "final_run": "src/experiments/hclustering_final_run.sh: the script unedited, with the loader switched to final by src/experiments/hclustering_run.py"},
         "configs/hclustering.json")
put_json({"what": "The three shared baselines (src/baselines/baselines.py). No settings and no randomness.",
          "popularity": "labels ranked by their number of fit images", "nearest_centroid": "cosine to the mean fit vector of each label",
          "zero_shot_text": "cosine to the vector of each label's wiki introduction (emb/" + vega_data.VECTORS + "-text, field intro)",
          "final_run": "src/baselines/baselines_final_run.sh, once on 22 Nov 2026 07:09 UTC"},
         "configs/baselines.json")

# ---------------------------------------------------------------- the split
for f in ("split.csv", "README.txt", "fix-2026-11-19.json", "dup_pairs.csv"):
    put(V / "splits" / "wiki-split-v2" / f, f"splits/wiki-split-v2/{f}")

# ---------------------------------------------------------------- result tables and logs
C = V / "runs" / "comparison"
for f in ("table.md", "table.csv", "results.jsonl", "FINAL-RUNS.log", "nn-final-2026-11-22.log", "baselines-final-2026-11-22.log",
          "cnn-final-2026-11-22.log", "hclustering-final-2026-11-22.log"):
    if f.endswith("-final-2026-11-22.log") and (not (C / f).exists() or "Finished" not in (C / f).read_text(encoding="utf-8")):
        print("not finished, left out:", f)
        continue
    put(C / f, f"results/comparison/{f}" + (".txt" if f.endswith(".log") else ""))      # the repository ignores *.log
for src, dst in (("runs/context-all-models/results.md", "context-all-models.md"), ("runs/context-all-models/results.json", "context-all-models.json"),
                 ("runs/clip-compare/results.md", "encoders-wiki.md"), ("runs/clip-compare/fs14.md", "encoders-outfit-photos.md"),
                 ("runs/real-photos-v1/results.md", "real-photos.md"), ("runs/real-photos-v1/RULES.md", "real-photos-rules.md"),
                 ("runs/personas-v1/results.md", "personas-wiki.md"), ("runs/personas-v1/posters.md", "personas-posters.md"),
                 ("runs/personas-v1/fs14-sets.md", "personas-outfit-sets.md"), ("runs/personas-v1/weight-variants.md", "personas-weighting.md"),
                 ("runs/personas-v1/many-photos.md", "personas-many-photos.md"), ("runs/personas-v1/surprise-variants.md", "personas-surprise.md"),
                 ("runs/ratings-v1/build.md", "rating-page-build.md"), ("runs/fairness-v1/spread.md", "fairness-spread.md"),
                 ("runs/fairness-v1/head.md", "fairness-head.md"), ("runs/ratings-v1/results.md", "ratings-page-1.md"),
                 ("runs/ratings-v1/answers.txt", "ratings-page-1-answers.txt"), 
                 ("runs/ratings-v2/build.md", "ratings-page-2-build.md"),
                 ("runs/aesthetic-map/map.png", "aesthetic-map.png"), ("runs/aesthetic-map/coords.csv", "aesthetic-map-coords.csv"), ("runs/family-v1/FROZEN.md", "family-table-frozen.md"),
                 ("runs/family-v1/label_family.csv", "family-table.csv"), ("runs/family-v1/families.csv", "families.csv"),
                 ("runs/family-v1/outside_family.csv", "outside-tags-to-families.csv")):
    put(V / src, f"results/{dst}")

# The rating pages' key files name the posters whose photos are shown; they stay on ml-db (runs/ratings-v1, runs/ratings-v2), out of git.
# ---------------------------------------------------------------- code that exists only in /srv/ml/vega/code
for f in sorted((V / "code" / "app").glob("*")):
    if f.suffix in (".py", ".html", ".json", ".md"):
        put(f, f"src/app/{f.name}")
put(V / "code" / "nn" / "nn_final_run.sh", "src/models/nn/nn_final_run.sh")
put(V / "code" / "baselines" / "baselines_final_run.sh", "src/baselines/baselines_final_run.sh")
for f in ("cnn_final_score.sh", "hclustering_run.py", "hclustering_final_run.sh", "context_all_models.py", "fairness_spread.py", "fairness_head.py",
          "clip_compare.py", "real_photos_nn.py", "family_v1.py", "family_v1_freeze.py", "tag_iter.py", "family_iter.py", "detect_people.py",
          "embed_eos_check.py", "repo_stage.py", "rate5_build.py", "rate5_score.py", "aesthetic_map.py", "make_submission.py", "report_facts.py", "nn_paired.py", "overfit_check.py", "recheck_near_duplicates.py"):
    put(V / "code" / "experiments" / f, f"src/experiments/{f}")
for f in ("vega_embed.py", "vega_text.py"):
    put(V / "scripts" / f, f"src/embed/{f}")
put(V / "code" / "shared" / "make_split.py", "src/common/make_split.py")

print(f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}: {len(done)} files written or updated in {R}")
for f in done:
    print("  ", f)
