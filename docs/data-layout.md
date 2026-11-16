# Data layout on ml-db

Snapshot taken 2026-11-16. Data lives outside git, under `/srv/ml` on the team machine `ml-db` (`~/shared` is a shortcut to `/srv/ml`). Group `mlteam` (all three members) has access; outsiders have none.

| Path | Owner | What it holds |
|---|---|---|
| `/srv/ml/datasets/` | erika | 2 TB data disk. Fashionpedia (48,833 images), DeepFashion (some parts still locked), a small evaluation pilot, download logs and data scripts (`tools/`) |
| `/srv/ml/datasets/catalog/` | erika | image audit: SQLite databases, duplicate list |
| `/srv/ml/vega/wiki/` | erika | Aesthetics Wiki scrape: 1,161 pages, 23,056 images, `manifest.csv` (image to aesthetic) |
| `/srv/ml/vega/emb/` | erika | Qwen3-VL-Embedding-2B vectors (2048 numbers per image, 448 px) and the labeler (`qwen2b-labeler/`) |
| `/srv/ml/vega/project-data-v1/` | erika | cleaned dataset: 26,900 images, train / validation / test JSONL files with labels |
| `/srv/ml/vega-db/` | root | Qdrant vector database tools. Holds credentials: never copy anything from here into git |
| `/srv/ml/letsModel/` | edmond | edmond's work area |
| `/srv/ml/cs422-VEGA/` | shared | this repo |

Not downloaded yet: Places365 and Open Images.

Note: the labeler in `vega/emb/qwen2b-labeler/` was trained on its own 70/10/20 split of the wiki images, not on the official `project-data-v1` splits, and that split's image list is not saved on ml-db. Do not compare its scores with models evaluated on the official splits.
