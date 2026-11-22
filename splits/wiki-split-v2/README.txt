wiki-split-v2 — the train/validation/test split every VEGA model should read
=========================================================================

split.csv     one row per embedded wiki image (same rows as emb/qwen2b-448/ids.txt)
              sha1        id used in ids.txt and image file names
              sha256      id used in project-data-v1 and the catalog
              split       train | validation | test | excluded   <- use this
              codex_split the project-data-v1 split it started from
              group       near-duplicate group (sha1 of one member); group_size 1 = no duplicate
dup_pairs.csv every near-duplicate pair, with cosine and pixel correlation
report.json   rule, counts before/after, per-aesthetic counts, checks, software versions
threshold-evidence/  contact sheets of sample pairs used to choose the rule, and the scripts that drew them

Made by /srv/ml/vega/code/shared/make_split.py (seed 422). Starts from Codex's stratified
project-data-v1 split and moves whole near-duplicate groups so no group crosses splits.

Counts: train 16,053 / validation 2,333 / test 4,515 / excluded 154 (aesthetics not in the catalog).
179 duplicate pairs, 146 groups, 75 images moved, 0 pairs crossing splits afterwards.
Decopunk, Realism and Showa Retro lost their only validation image (each keeps test images).

Rules: choose settings on validation. Score test once, with every setting frozen.
Fit anything learned (scalers, centroids, clusters, labellers) on train rows only.

UPDATE 2026-11-19 (see fix-2026-11-19.json and handoff/WORKLOG.md): the near-duplicate rule was re-run on the corrected
vectors (emb/qwen2beos-448). One pair crossed validation and test: the same film still, filed under Balletcore in
validation and under Coquette + Waif in test. The validation image was moved to test, so the pair sits on one side.
Counts for the 503 merged labels are now train 9,359 / validation 1,351 / test 2,618. Nothing else changed.
The previous file is split.before-2026-11-19-fix.csv (sha256 8c72fbc1eb59...); this one is 19f1816c1a75....
