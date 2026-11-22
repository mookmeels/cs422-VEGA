"""Freeze family table v1. Run once on ml-db, after Andrew accepted the proposal (21 Nov: no changes).

    python family_v1_freeze.py      -> runs/family-v1/label_family.csv, outside_family.csv, FROZEN.md

From here on the table does not change: label_family.csv is what every family score and the family NN read.
The script refuses to overwrite a frozen table that differs.
"""
import hashlib
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else Path("/srv/ml/vega")
sys.path.insert(0, str(ROOT / "code" / "shared"))
import vega_data  # noqa: E402

V1, DRAFT = ROOT / "runs" / "family-v1", ROOT / "runs" / "family-draft"
p = pl.read_csv(V1 / "proposal.csv")
classes = list(vega_data.load(root=ROOT, vectors=False)["classes"])
assert sorted(p["wiki_label"]) == sorted(classes) and p["wiki_label"].n_unique() == 503
fams = pl.read_csv(V1 / "families.csv")
assert set(p["family"]) == set(fams["family"]) and len(fams) == 20

table = p.select("wiki_label", "family", "family_name", "draft_family", "status", "training_images", "validation_images")
target = V1 / "label_family.csv"
text = table.write_csv()
if target.exists():
    assert target.read_text(encoding="utf-8") == text, "a different frozen table is already there; not overwriting"
target.write_text(text, encoding="utf-8", newline="")
sha = hashlib.sha256(target.read_bytes()).hexdigest()

out = pl.read_csv(DRAFT / "outside_family.csv", infer_schema=False)
assert set(out["family"].drop_nulls()) - {"", "vague"} <= set(fams["family"])
out.write_csv(V1 / "outside_family.csv")

(V1 / "FROZEN.md").write_text(f"""# Family table v1: FROZEN

Accepted by Andrew on 2026-11-21 (about 00:45 UTC) with no changes to the proposal ("i actually have zero suggestions").
Frozen {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}.

- `label_family.csv`: 503 wiki labels, 20 families. sha256 `{sha}`.
- `outside_family.csv`: which FashionStyle14 styles and Street Fashion Style tags stand for which family (unchanged from the draft;
  the two new families, seasons and holidays, and design, graphics and screens, have no outside photos).
- `proposal.csv`, `review.html`, `summary.md`: how the table was made and what Andrew was shown.

One family level for the overall aesthetic. Outfit against vibe is a property of the photo, not of the label.

## Rules for the family NN (fixed before any family model was trained)

1. This table is frozen. No label moves once a family model has been scored.
2. Same network and settings as the 503-label NN (`code/nn/train_nn.py`, SETTINGS), no re-tuning on families. Seeds 422 to 426.
   Trained on the training split, scored on validation. One CPU thread.
3. The comparison is the family NN against the 503-label NN with its guesses grouped by family, on the same validation images,
   paired bootstrap (2,000 draws, seed 422), each model measured per image as the share of its 5 seeds that are right.
   Grouping is done two ways, both reported: by SUMMING the 503-label NN's probabilities within each family (the main
   comparison, because it is the stronger way to group) and by taking the family of its first guess.
4. Family top-1 on all validation images is the family figure (the first family guessed is one of the image's families).
   Right family in the first 3, and outfit images against the other images, are reported beside it.
5. First run on wiki images only. Adding Street Fashion Style or FashionStyle14 photos to training is a separate, later
   experiment with its own decision. The transfer-test photos stay out of everything.
6. The test split is not read until the one final run.

## To be stated in the report

The image vectors chose which labels were looked at again and decided the placement of the aesthetics Claude Code did not
know, so the table may favour models that use those vectors (the NN and the baselines) over the CNN and the clustering model.
""", encoding="utf-8")

s = (V1 / "summary.md").read_text(encoding="utf-8").split("\n")
if "ACCEPTED" not in s[0]:
    s[0] = "# Family table v1: ACCEPTED by Andrew on 2026-11-21 with no changes (see FROZEN.md); below, the proposal as it was shown"
    (V1 / "summary.md").write_text("\n".join(s), encoding="utf-8")
print(f"frozen: {len(table)} labels, {len(fams)} families, sha256 {sha[:12]}")
