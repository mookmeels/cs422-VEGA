"""Run Edmond's hclustering.py exactly as it is, in a folder of its own, and time it. It never scores anything.

    python hclustering_run.py --out /srv/ml/vega/runs/hclustering-check-validation            # validation, as Edmond runs it
    python hclustering_run.py --out /srv/ml/vega/runs/hclustering-final --final               # fit = train + validation, eval = test

The script itself is not edited. It calls the team loader as load(); with --final this runner makes that call
load(final=True), which is the loader's own switch for the final run (test images come back without labels).
The script writes my-preds.npz into the current folder, so the runner changes into --out first and Edmond's own
my-preds.npz is left alone. Scoring is a separate step (vega_score.py, with --final for the test split, once).
"""
import argparse
import functools
import hashlib
import os
import runpy
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--script", default="/srv/ml/cs422-VEGA/src/models/hclustering/hclustering.py")
ap.add_argument("--out", required=True)
ap.add_argument("--final", action="store_true", help="fit on train + validation and predict the TEST split")
a = ap.parse_args()

script = Path(a.script).resolve()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
if (out / "my-preds.npz").exists():
    raise SystemExit(f"{out / 'my-preds.npz'} exists already; choose another --out")

sys.path.insert(0, "/srv/ml/vega/code/shared")
import vega_data  # noqa: E402

if a.final:
    vega_data.load = functools.partial(vega_data.load, final=True)

print(f"script: {script}", flush=True)
print(f"script sha256: {hashlib.sha256(script.read_bytes()).hexdigest()}", flush=True)
print(f"mode: {'FINAL (test split)' if a.final else 'validation'}; started {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
os.chdir(out)
t0 = time.time()
runpy.run_path(str(script), run_name="__main__")
print(f"predictions: {out / 'my-preds.npz'}", flush=True)
print(f"Time elapsed: {time.time() - t0:.0f} s", flush=True)
