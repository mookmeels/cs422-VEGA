"""Find the team's shared code (vega_data.py, vega_score.py) and put it on the import path.

The CNN's scripts run from two layouts: the team repo (src/models/cnn, shared code in src/common) and the project
folder on ml-db (code/cnn, shared code in code/shared). VEGA_SHARED, if set, wins.
"""
import os
import sys
from pathlib import Path


def shared_code():
    """The folder holding the team's loader and scorer."""
    here = Path(__file__).resolve().parent
    places = [here.parent / "shared", here.parent.parent / "common",
              Path(os.environ.get("VEGA_ROOT", "/srv/ml/vega")) / "code" / "shared"]
    if os.environ.get("VEGA_SHARED"):
        places.insert(0, Path(os.environ["VEGA_SHARED"]))
    for p in places:
        if (p / "vega_data.py").exists():
            return p
    raise SystemExit("vega_data.py not found; set VEGA_SHARED to its folder. Looked in: " + ", ".join(map(str, places)))


if str(shared_code()) not in sys.path:
    sys.path.insert(0, str(shared_code()))
