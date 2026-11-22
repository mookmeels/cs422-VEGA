# CLIP against Qwen on the wiki images (validation; the test split was not read)

Made by `code/experiments/clip_compare.py --score`. 9,359 training and 1,351 validation images, 503 labels. NN: train_nn.train, frozen settings, wiki only, seeds [422, 423, 424, 425, 426], on the GPU.

| encoder | numbers per image | nearest centroid: top-5 | top-1 | family top-1 | NN (wiki only): top-5 | top-1 | family top-1 | images per second |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-Embedding-2B (the team's vectors) | 2,048 | 83.5% | 53.3% | 69.4% | 88.3 ± 0.4 | 62.7 ± 0.5 | 75.9 ± 0.5 |  |
| clip-vitb32 | 512 | 74.1% | 44.0% | 60.0% | 82.9 ± 0.5 | 53.9 ± 0.3 | 67.8 ± 0.5 | 244.1 |
| clip-vitl14 | 768 | 81.3% | 54.1% | 68.5% | 89.8 ± 0.2 | 65.9 ± 0.4 | 77.4 ± 0.3 | 71.6 |

## NN: each CLIP encoder minus Qwen, on the same validation images (per image, the share of the 5 seeds that are right; paired bootstrap over images, 2,000 draws, seed 422)

| encoder | top-5 difference (95% interval) | top-1 difference (95% interval) |
|---|---|---|
| clip-vitb32 | -5.4 (-7.1 to -3.9) | -8.8 (-11.1 to -6.6) |
| clip-vitl14 | +1.5 (+0.1 to +2.8) | +3.2 (+1.2 to +5.4) |

The NN's settings were tuned on Qwen vectors, so its columns lean toward Qwen. Nearest centroid has nothing to tune.
time elapsed 29s
