"""Build the files to upload to WebCampus from the team repository: scripts as .py.txt, no ZIP, team name in every name.

    python make_submission.py [--repo /srv/ml/cs422-VEGA] [--out /srv/ml/vega/submission]

The course asks for Python scripts as .py.txt files uploaded separately (restore .py to run them), a README, and the
saved configurations and result tables. WebCampus takes loose files, so the folder a script belongs to is kept in its
name: src/models/nn/train_nn.py becomes VEGA__src__models__nn__train_nn.py.txt. To run the code again, rename each file
back: drop the VEGA__ prefix and the .txt, and turn every "__" into a folder. MANIFEST lists every file, its original
path and its sha256, and `--restore <folder>` does the renaming.
Shell scripts, READMEs, configs and result tables are copied the same way (.sh.txt, .md, .json, .csv, .log, .jsonl).
Notebooks are never copied. The split file is included; images, vectors and model weights are not (see the README).
"""
import hashlib
import shutil
import sys
from pathlib import Path

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
REPO, OUT = Path(arg("--repo", "/srv/ml/cs422-VEGA")), Path(arg("--out", "/srv/ml/vega/submission"))
TEAM = "VEGA"
CODE = {".py": ".py.txt", ".sh": ".sh.txt", ".html": ".html.txt"}
KEEP = {".md", ".txt", ".json", ".csv", ".log", ".jsonl"}
SKIP_DIRS = {".git", "notebooks", "__pycache__", "labels"}

if "--restore" in sys.argv:
    src, dst = Path(arg("--restore", ".")), Path(arg("--to", "restored"))
    n = 0
    for f in sorted(src.glob(f"{TEAM}__*")):
        name = f.name[len(TEAM) + 2:]
        for txt, orig in ((".py.txt", ".py"), (".sh.txt", ".sh"), (".html.txt", ".html")):
            if name.endswith(txt):
                name = name[:-len(txt)] + orig
        target = dst.joinpath(*name.split("__"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target)
        n += 1
    print(f"{n} files restored under {dst}")
    raise SystemExit

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
rows = []
for f in sorted(REPO.rglob("*")):
    rel = f.relative_to(REPO)
    if not f.is_file() or any(p in SKIP_DIRS for p in rel.parts) or f.name.startswith(".") or f.suffix == ".ipynb" or f.suffix in (".npz", ".npy", ".pt", ".pyc"):
        continue
    if f.suffix in CODE:
        name = "__".join(rel.parts)[:-len(f.suffix)] + CODE[f.suffix]
    elif f.suffix in KEEP or f.name == "LICENSE":
        name = "__".join(rel.parts)
    else:
        print("left out (not a script, table or note):", rel)
        continue
    shutil.copyfile(f, OUT / f"{TEAM}__{name}")
    rows.append((f"{TEAM}__{name}", str(rel).replace("\\", "/"), hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_size))
with open(OUT / f"{TEAM}__MANIFEST.txt", "w", encoding="utf-8") as m:
    m.write(f"{TEAM}: files for the course submission. To run the code, restore each file to its original path (or: python make_submission.py --restore <this folder> --to <new folder>).\n")
    m.write("uploaded name\toriginal path in the repository\tsha256\tbytes\n")
    for r in rows:
        m.write("\t".join(str(x) for x in r) + "\n")
n_py = sum(r[0].endswith(".py.txt") for r in rows)
print(f"{len(rows)} files in {OUT} ({n_py} Python scripts as .py.txt), {sum(r[3] for r in rows) / 1e6:.1f} MB; manifest: {TEAM}__MANIFEST.txt")
