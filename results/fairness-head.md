# Same outfit, wearer hidden: what changes when the head is covered

Made 2026-11-22 09:06 UTC by code/experiments/fairness_head.py. 1,980 FashionStyle14 photos (180 per style at most, 11 styles) with one clearly found person at least 0.6 of the photo tall. NN with real outfit photos, frozen settings, seeds [422, 423, 424]; it never trained on FashionStyle14. Intervals are 95%, resampling photos.

| version | family right | first family differs from the original's | first aesthetic differs | how far the family shares move (0 to 1) |
|---|---|---|---|---|
| the photo as it is | 55.4% (53.3 to 57.5) |  |  |  |
| head covered (grey block) | 53.5% (51.4 to 55.6) | 18.2% (16.9 to 19.6) | 26.7% (25.2 to 28.2) | 0.134 |
| head blurred | 53.3% (51.2 to 55.3) | 17.9% (16.6 to 19.3) | 26.2% (24.7 to 27.7) | 0.134 |
| waist and hips covered | 54.7% (52.6 to 56.8) | 15.6% (14.3 to 16.9) | 23.0% (21.6 to 24.4) | 0.118 |
| feet covered | 53.3% (51.2 to 55.4) | 16.8% (15.6 to 18.1) | 22.9% (21.5 to 24.3) | 0.119 |

- Covering the head changes the first family +2.6 (+1.0 to +4.1) points more often than covering the waist and hips, and +1.4 (-0.2 to +3.0) more often than covering the feet.
- Family right: head covered -1.9 (-3.1 to -0.7) points against the original; waist and hips covered -0.7 (-1.8 to +0.4).

## By style: how often the first family changes

| style | photos | head covered | head blurred | waist and hips covered | feet covered | family right, original | family right, head covered |
|---|---|---|---|---|---|---|---|
| natural | 180 | 28.9% | 28.0% | 19.4% | 21.3% | 36.1% | 31.5% |
| mode | 180 | 21.5% | 21.3% | 20.7% | 23.0% | 24.6% | 27.0% |
| street | 180 | 21.3% | 20.7% | 17.0% | 21.9% | 59.1% | 53.3% |
| girlish | 180 | 20.7% | 19.1% | 18.0% | 23.1% | 47.0% | 42.0% |
| kireime-casual | 180 | 20.4% | 19.6% | 15.7% | 24.3% | 45.9% | 43.7% |
| retro | 180 | 20.0% | 20.9% | 15.2% | 11.1% | 62.6% | 55.4% |
| gal | 180 | 19.6% | 18.9% | 22.0% | 16.1% | 25.2% | 25.7% |
| rock | 180 | 17.2% | 16.7% | 12.0% | 18.3% | 62.4% | 69.6% |
| feminine | 180 | 14.6% | 13.3% | 13.3% | 10.0% | 83.9% | 73.3% |
| conservative | 180 | 12.6% | 12.8% | 13.1% | 13.3% | 75.7% | 78.9% |
| dressy | 180 | 3.9% | 6.1% | 5.0% | 2.6% | 87.2% | 88.1% |

Check: the re-embedded originals match the stored vectors (cosine median 0.9966, lowest 0.9725).

Time elapsed: 666 s
