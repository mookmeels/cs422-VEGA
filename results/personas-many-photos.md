# Many photos: what changes from 4 photos to 50

Made by `code/app/many_photos.py` on cuda; the app's network (seed 422). Expert-labelled outfit photos unless it says posters; 1,000 sets of each kind and size.

## A. The profile: is the pooled top family the main style's family?

Differences are against the plain average on the same sets, in points, with 95% intervals.

| photos | sets | plain average | agreement, geometric | difference | agreement, ordinary average | difference |
|---|---|---|---|---|---|---|
| 2 | one style | 66.2% | 69.1% | +2.9 (+0.8 to +5.0) | 66.2% | +0.0 (+0.0 to +0.0) |
| 2 | a side style | 34.6% | 33.3% | -1.3 (-4.1 to +1.5) | 34.6% | +0.0 (+0.0 to +0.0) |
| 3 | one style | 70.7% | 71.5% | +0.8 (-1.1 to +2.7) | 71.5% | +0.8 (-0.6 to +2.2) |
| 3 | a side style | 56.0% | 61.8% | +5.8 (+3.1 to +8.5) | 61.1% | +5.1 (+3.2 to +7.0) |
| 4 | one style | 73.0% | 75.5% | +2.5 (+0.4 to +4.6) | 74.1% | +1.1 (-0.3 to +2.5) |
| 4 | a side style | 64.9% | 69.1% | +4.2 (+1.9 to +6.5) | 68.3% | +3.4 (+1.8 to +5.0) |
| 5 | one style | 76.7% | 79.3% | +2.6 (+0.7 to +4.5) | 76.8% | +0.1 (-1.0 to +1.2) |
| 5 | a side style | 70.9% | 75.0% | +4.1 (+1.9 to +6.3) | 72.9% | +2.0 (+0.7 to +3.3) |
| 8 | one style | 79.3% | 81.3% | +2.0 (+0.1 to +3.9) | 79.8% | +0.5 (-0.7 to +1.7) |
| 8 | a side style | 71.7% | 75.5% | +3.8 (+1.7 to +5.9) | 74.6% | +2.9 (+1.5 to +4.3) |
| 12 | one style | 84.7% | 85.9% | +1.2 (-0.6 to +3.0) | 85.4% | +0.7 (-0.4 to +1.8) |
| 12 | a side style | 78.0% | 79.7% | +1.7 (-0.3 to +3.7) | 79.7% | +1.7 (+0.5 to +2.9) |
| 16 | one style | 86.3% | 87.1% | +0.8 (-1.0 to +2.6) | 86.6% | +0.3 (-0.6 to +1.2) |
| 16 | a side style | 79.7% | 81.1% | +1.4 (-0.5 to +3.3) | 80.7% | +1.0 (-0.2 to +2.2) |
| 20 | one style | 87.9% | 85.6% | -2.3 (-4.1 to -0.5) | 87.9% | +0.0 (-0.9 to +0.9) |
| 20 | a side style | 80.3% | 81.3% | +1.0 (-1.1 to +3.1) | 81.5% | +1.2 (+0.2 to +2.2) |
| 30 | one style | 89.7% | 89.0% | -0.7 (-2.6 to +1.2) | 89.9% | +0.2 (-0.8 to +1.2) |
| 30 | a side style | 82.0% | 82.7% | +0.7 (-1.3 to +2.7) | 84.2% | +2.2 (+1.1 to +3.3) |
| 50 | one style | 92.8% | 90.2% | -2.6 (-4.3 to -0.9) | 92.0% | -0.8 (-1.6 to +0.0) |
| 50 | a side style | 83.8% | 83.4% | -0.4 (-2.3 to +1.5) | 86.0% | +2.2 (+1.0 to +3.4) |

## B. The surprise, as the app now works it out

Pooling: agreement weights with a geometric mean up to 16 photos, with an ordinary average beyond. Relevance of a family: its share in the photo a tenth of the way down (the strongest up to 10 photos, the 2nd strongest of 20, the 5th of 50). 'Old rule' is the strongest photo with the one cut for small sets.

| photos | old rule: one-style sets shown a surprise | cut now | one style: shown | one odd photo: shown | a side style: shown | of those, its family named | posters: shown |
|---|---|---|---|---|---|---|---|
| 2 | 3.2% | 0.41 | 2.7% | 17.0% | 15.9% | 37.1% | 3.7% |
| 3 | 8.1% | 0.41 | 7.8% | 25.8% | 27.7% | 62.1% | 8.1% |
| 4 | 10.5% | 0.41 | 10.5% | 28.8% | 27.5% | 50.5% | 15.6% |
| 5 | 12.7% | 0.41 | 11.6% | 32.3% | 32.3% | 51.1% | 18.4% |
| 8 | 21.0% | 0.49 | 8.3% | 21.4% | 27.8% | 53.2% | 12.1% |
| 12 | 25.8% | 0.47 | 11.3% | 16.4% | 24.7% | 54.7% | 10.5% |
| 16 | 26.7% | 0.45 | 9.9% | 7.6% | 24.5% | 59.6% | 10.3% |
| 20 | 35.1% | 0.43 | 11.4% | 9.9% | 29.7% | 60.6% | 7.3% |
| 30 | 39.0% | 0.39 | 13.1% | 13.2% | 33.2% | 65.1% | 6.0% |
| 50 | 53.2% | 0.35 | 11.2% | 9.9% | 30.1% | 65.8% | 6.7% |

The cut for every number of photos is in `code/app/surprise_cuts.json`. A side style is about a fifth of the photos; with 10 photos or fewer one photo is enough to raise a surprise, beyond that a single odd photo is deliberately no longer enough.

time elapsed 45s
