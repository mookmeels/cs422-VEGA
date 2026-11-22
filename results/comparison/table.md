Label set: merged (503 labels). Percentages except MRR. Validation: models fitted on the training split. Test: fitted on train + validation, scored once. Split file sha256 19f1816c1a75. Image vectors: emb/qwen2beos-aesthetic-448. Family columns: the same predictions with each guessed label replaced by its style family (20 families, runs/family-v1); n/a = scored before 21 Nov and not re-scored.

| split | model | owner | runs (seeds) | top-5 | top-1 | MRR | per-label top-5 | thin labels top-5 | other labels top-5 | label coverage | family top-1 | family in top 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | NN (MLP, 1 hidden layer of 1024) | Andrew Nguyen | 5 (422, 423, 424, 425, 426) | 88.3 ± 0.4 | 62.7 ± 0.5 | 0.741 ± 0.002 | 86.6 ± 0.5 | 85.1 ± 0.8 | 88.4 ± 0.3 | 100.0 ± 0.0 | 75.6 ± 0.5 | 94.1 ± 0.2 |
| validation | NN (MLP, 1 hidden layer of 1024) + real outfit photos | Andrew Nguyen | 5 (422, 423, 424, 425, 426) | 87.9 ± 0.1 | 62.3 ± 0.6 | 0.736 ± 0.003 | 86.1 ± 0.1 | 84.5 ± 0.4 | 88.0 ± 0.3 | 99.9 ± 0.1 | 75.3 ± 0.4 | 93.9 ± 0.3 |
| validation | Baseline: nearest centroid | shared (baseline) | 1 (no randomness) | 82.0 | 53.5 | 0.660 | 82.6 | 83.5 | 81.6 | 99.8 | 69.6 | 91.5 |
| validation | Hclustering(standardized 0.16) | Si_mpl | 5 (422, 423, 424, 425, 426) | 44.4 ± 0.0 | 26.5 ± 0.0 | 0.348 ± 0.000 | 43.1 ± 0.0 | 40.6 ± 0.0 | 46.1 ± 0.0 | 80.3 ± 0.0 | 47.2 ± 0.0 | 69.4 ± 0.0 |
| validation | Baseline: zero-shot text | shared (baseline) | 1 (no randomness) | 41.7 | 21.0 | 0.315 | 41.1 | 38.2 | 44.5 | 82.3 | 42.4 | 71.4 |
| validation | Hclustering(standardized 0.25) | Si_mpl | 1 (426) | 39.0 | 26.4 | 0.326 | 41.5 | 43.4 | 39.3 | 76.5 | 45.9 | 61.5 |
| validation | CNN (ResNet-18, 224 px) + street photos | Giancarlo Guevarra | 5 (422, 423, 424, 425, 426) | 31.8 ± 0.8 | 11.8 ± 0.4 | 0.221 ± 0.004 | 26.7 ± 0.8 | 21.7 ± 1.3 | 32.6 ± 1.3 | 99.8 ± 0.2 | 26.6 ± 1.1 | 53.1 ± 1.0 |
| validation | CNN (ResNet-18, 224 px) | Giancarlo Guevarra | 5 (422, 423, 424, 425, 426) | 30.9 ± 0.6 | 12.0 ± 0.3 | 0.220 ± 0.003 | 26.6 ± 0.3 | 22.5 ± 0.8 | 31.5 ± 1.2 | 99.8 ± 0.3 | 26.8 ± 1.0 | 52.9 ± 0.7 |
| validation | CNN (ResNet-18, 224 px, random crop 0.6, stretch 0.15, mixup/CutMix, label smoothing 0.1, oversampling) + street photos | Giancarlo Guevarra | 5 (422, 423, 424, 425, 426) | 30.1 ± 0.9 | 11.9 ± 0.1 | 0.216 ± 0.002 | 29.0 ± 0.9 | 28.4 ± 0.9 | 29.8 ± 0.8 | 99.4 ± 0.2 | 27.0 ± 0.9 | 52.2 ± 0.9 |
| validation | Baseline: popularity | shared (baseline) | 1 (no randomness) | 5.8 | 1.4 | 0.047 | 1.0 | 0.0 | 2.2 | 1.0 | 8.1 | 22.9 |
| test | NN (MLP, 1 hidden layer of 1024) | Andrew Nguyen | 5 (422, 423, 424, 425, 426) | 87.9 ± 0.2 | 61.0 ± 0.1 | 0.726 ± 0.001 | 85.9 ± 0.3 | 85.3 ± 0.4 | 86.4 ± 0.3 | 100.0 ± 0.0 | 73.9 ± 0.2 | 93.1 ± 0.2 |
| test | NN (MLP, 1 hidden layer of 1024) + real outfit photos | Andrew Nguyen | 5 (422, 423, 424, 425, 426) | 87.5 ± 0.2 | 61.2 ± 0.5 | 0.727 ± 0.003 | 85.5 ± 0.2 | 84.4 ± 0.5 | 86.3 ± 0.1 | 99.9 ± 0.1 | 73.7 ± 0.5 | 93.0 ± 0.2 |
| test | Baseline: nearest centroid | shared (baseline) | 1 (no randomness) | 79.7 | 50.5 | 0.635 | 81.3 | 84.4 | 79.1 | 100.0 | 67.6 | 89.1 |
| test | Baseline: zero-shot text | shared (baseline) | 1 (no randomness) | 43.4 | 22.0 | 0.325 | 43.4 | 43.6 | 43.2 | 88.7 | 42.2 | 70.5 |
| test | Hclustering(standardized 0.16) | Si_mpl | 1 (no randomness) | 40.1 | 23.6 | 0.311 | 38.2 | 35.6 | 40.2 | 79.9 | 44.9 | 66.3 |
| test | CNN (ResNet-18, 224 px) | Giancarlo Guevarra | 5 (422, 423, 424, 425, 426) | 32.7 ± 0.5 | 13.6 ± 0.6 | 0.235 ± 0.004 | 27.6 ± 0.5 | 21.3 ± 1.1 | 32.2 ± 0.4 | 99.9 ± 0.2 | 27.6 ± 0.6 | 53.7 ± 0.6 |
| test | CNN (ResNet-18, 224 px) + street photos | Giancarlo Guevarra | 5 (422, 423, 424, 425, 426) | 32.6 ± 0.5 | 14.0 ± 0.5 | 0.236 ± 0.003 | 27.0 ± 0.4 | 20.4 ± 0.7 | 31.8 ± 1.0 | 99.9 ± 0.1 | 27.9 ± 0.3 | 53.5 ± 0.4 |
| test | CNN (ResNet-18, 224 px, random crop 0.6, stretch 0.15, mixup/CutMix, label smoothing 0.1, oversampling) + street photos | Giancarlo Guevarra | 5 (422, 423, 424, 425, 426) | 32.4 ± 0.2 | 13.3 ± 0.3 | 0.232 ± 0.002 | 31.3 ± 0.2 | 29.7 ± 0.7 | 32.6 ± 0.3 | 100.0 ± 0.0 | 28.0 ± 0.4 | 53.9 ± 0.5 |
| test | Baseline: popularity | shared (baseline) | 1 (no randomness) | 6.0 | 1.5 | 0.048 | 1.0 | 0.0 | 1.7 | 1.0 | 8.6 | 23.4 |
