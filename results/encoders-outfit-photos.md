# CLIP against Qwen on real outfit photos (FashionStyle14, 11 styles; lolita, fairy and ethnic not read)

Made by `code/experiments/clip_compare.py --fs14`. 10,312 photos. 'Its own styles': 5 folds within the photos, no wiki image. 'From the wiki': the wiki-only NN (frozen settings, seeds [422, 423, 424, 425, 426]) and the style's family.

| encoder | its own styles: nearest centroid | 10 nearest neighbours | from the wiki: family right |
|---|---|---|---|
| Qwen3-VL-Embedding-2B (the team's vectors) | 79.5% | 80.4% | 33.9 ± 0.6 |
| clip-vitb32 | 72.5% | 73.8% | 29.7 ± 0.1 |
| clip-vitl14 | 76.9% | 78.6% | 34.7 ± 0.3 |

## Each CLIP encoder minus Qwen, on the same photos (paired bootstrap over photos, 2,000 draws, seed 422)

| encoder | nearest centroid | 10 nearest neighbours | from the wiki: family right |
|---|---|---|---|
| clip-vitb32 | -7.1 (-8.0 to -6.1) | -6.6 (-7.5 to -5.6) | -4.2 (-5.1 to -3.4) |
| clip-vitl14 | -2.6 (-3.5 to -1.6) | -1.8 (-2.6 to -0.9) | +0.7 (-0.2 to +1.6) |

## By style, 10 nearest neighbours within FashionStyle14

| style | photos | Qwen3-VL-Embedding-2B (the team's vectors) | clip-vitb32 | clip-vitl14 |
|---|---|---|---|---|
| conservative | 923 | 88% | 78% | 79% |
| dressy | 890 | 97% | 96% | 96% |
| feminine | 812 | 84% | 77% | 87% |
| gal | 953 | 68% | 76% | 74% |
| girlish | 1,103 | 59% | 57% | 61% |
| kireime-casual | 1,050 | 75% | 70% | 73% |
| mode | 1,061 | 83% | 82% | 85% |
| natural | 854 | 85% | 75% | 83% |
| retro | 841 | 76% | 53% | 66% |
| rock | 803 | 88% | 75% | 80% |
| street | 1,022 | 87% | 75% | 83% |

Qwen's vectors were made with the instruction "Represent the fashion aesthetic of this image." at 448 px; CLIP takes no instruction and sees 224 px.
FashionStyle14's 11 styles were already used to choose and check the NN's street-photo settings (21 Nov); here they only compare encoders.
time elapsed 28s
