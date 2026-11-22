# Should some photos count more than others? Ways to weight a person's photos

Made by `code/app/weight_variants.py`, seeds [422, 423, 424] on cuda for the outfit photos; wiki personas use the held-out predictions of oof.py. No weight uses a label, except the ORACLE row, which shows what perfect weights would reach. Differences are against the plain average on the same sets, in points, with 95% intervals (wiki: home aesthetics resampled; outfit photos: sets resampled).

## Wiki personas, all photos of one aesthetic: the home aesthetic (1,044 sets)

| photos weighted by | is first | against equal | in the top five | against equal |
|---|---|---|---|---|
| equal (the app) | 80.2% | | 95.5% | |
| confidence ^1 | 76.6% | -3.5 (-5.0 to -2.2) | 95.0% | -0.5 (-1.3 to +0.4) |
| confidence ^2 | 75.3% | -4.9 (-6.7 to -3.3) | 95.2% | -0.3 (-1.4 to +1.1) |
| agreement on the map ^2 | 81.3% | +1.1 (+0.4 to +1.9) | 95.5% | +0.0 (-0.7 to +0.5) |
| agreement on the map ^8 | 81.3% | +1.1 (-0.1 to +2.5) | 96.1% | +0.6 (-0.5 to +1.9) |
| agreement on family ^1 | 81.4% | +1.2 (-0.1 to +2.8) | 95.7% | +0.2 (-0.6 to +1.1) |
| agreement on family ^2 | 80.3% | +0.1 (-1.5 to +1.7) | 95.8% | +0.3 (-0.7 to +1.2) |
| agreement on family ^4 | 78.6% | -1.5 (-3.6 to +0.8) | 94.4% | -1.1 (-3.0 to +0.9) |
| drop the odd one out | 78.2% | -2.0 (-4.0 to +0.1) | 94.6% | -0.9 (-2.7 to +0.8) |
| geometric mean | 89.0% | +8.8 (+6.0 to +12.0) | 98.9% | +3.4 (+1.5 to +5.4) |
| geometric, by agreement on family ^1 (vega_profile's "agree") | 88.4% | +8.2 (+5.6 to +11.1) | 98.4% | +2.9 (+1.1 to +5.0) |
| geometric, by agreement on family ^2 | 86.4% | +6.2 (+3.5 to +9.3) | 98.3% | +2.8 (+1.1 to +4.9) |

## Wiki personas, one photo of another family: the home aesthetic (1,044 sets)

| photos weighted by | is first | against equal | in the top five | against equal |
|---|---|---|---|---|
| equal (the app) | 70.1% | | 90.7% | |
| confidence ^1 | 64.8% | -5.3 (-6.8 to -3.9) | 91.4% | +0.7 (-0.3 to +1.6) |
| confidence ^2 | 61.0% | -9.1 (-11.0 to -7.3) | 90.8% | +0.1 (-1.1 to +1.4) |
| agreement on the map ^2 | 73.5% | +3.4 (+2.3 to +4.6) | 91.4% | +0.7 (-0.1 to +1.4) |
| agreement on the map ^8 | 74.4% | +4.3 (+2.7 to +5.9) | 92.4% | +1.7 (+0.5 to +3.2) |
| agreement on family ^1 | 74.6% | +4.5 (+2.5 to +6.6) | 91.9% | +1.1 (+0.2 to +2.2) |
| agreement on family ^2 | 73.2% | +3.1 (+0.8 to +5.3) | 92.2% | +1.5 (+0.4 to +2.8) |
| agreement on family ^4 | 70.3% | +0.2 (-2.2 to +2.8) | 91.2% | +0.5 (-1.0 to +2.1) |
| drop the odd one out | 72.3% | +2.2 (+0.7 to +3.7) | 91.7% | +1.0 (-0.3 to +2.3) |
| geometric mean | 69.3% | -0.9 (-4.5 to +2.4) | 92.7% | +2.0 (-0.1 to +4.1) |
| geometric, by agreement on family ^1 (vega_profile's "agree") | 79.5% | +9.4 (+6.6 to +12.3) | 94.9% | +4.2 (+1.9 to +6.7) |
| geometric, by agreement on family ^2 | 78.8% | +8.7 (+5.7 to +11.8) | 94.3% | +3.5 (+1.2 to +5.8) |
| ORACLE: only the photos of the main style | 74.3% | +4.2 (+3.0 to +5.5) | 93.4% | +2.7 (+1.5 to +3.9) |

## Expert-labelled outfit photos, four of one style: the style's family (3,000 sets)

| photos weighted by | is first | against equal | in the top three | against equal |
|---|---|---|---|---|
| equal (the app) | 73.1% | | 93.7% | |
| confidence ^1 | 68.4% | -4.8 (-5.5 to -4.0) | 93.0% | -0.7 (-1.0 to -0.3) |
| confidence ^2 | 65.7% | -7.5 (-8.4 to -6.5) | 92.2% | -1.5 (-1.9 to -1.0) |
| agreement on the map ^2 | 72.6% | -0.6 (-0.9 to -0.3) | 93.7% | +0.1 (-0.1 to +0.2) |
| agreement on the map ^8 | 71.2% | -1.9 (-2.4 to -1.3) | 93.4% | -0.3 (-0.6 to +0.1) |
| agreement on family ^1 | 73.5% | +0.4 (-0.2 to +1.0) | 93.7% | +0.0 (-0.3 to +0.3) |
| agreement on family ^2 | 72.9% | -0.2 (-0.9 to +0.5) | 93.4% | -0.2 (-0.7 to +0.2) |
| agreement on family ^4 | 72.0% | -1.1 (-2.0 to -0.2) | 92.7% | -1.0 (-1.6 to -0.4) |
| drop the odd one out | 70.8% | -2.3 (-3.4 to -1.4) | 90.9% | -2.8 (-3.6 to -1.9) |
| geometric mean | 76.4% | +3.3 (+2.4 to +4.3) | 93.3% | -0.4 (-1.0 to +0.4) |
| geometric, by agreement on family ^1 (vega_profile's "agree") | 75.9% | +2.8 (+1.8 to +3.7) | 92.8% | -0.8 (-1.6 to -0.1) |
| geometric, by agreement on family ^2 | 75.2% | +2.1 (+1.1 to +3.1) | 92.4% | -1.3 (-2.0 to -0.5) |

## Expert-labelled outfit photos, three of one style and one of another family: the main style's family (3,000 sets)

| photos weighted by | is first | against equal | in the top three | against equal |
|---|---|---|---|---|
| equal (the app) | 64.5% | | 88.2% | |
| confidence ^1 | 56.0% | -8.5 (-9.5 to -7.5) | 87.0% | -1.2 (-1.7 to -0.7) |
| confidence ^2 | 51.8% | -12.7 (-13.9 to -11.6) | 85.3% | -2.9 (-3.7 to -2.2) |
| agreement on the map ^2 | 65.6% | +1.1 (+0.6 to +1.5) | 88.9% | +0.7 (+0.4 to +1.0) |
| agreement on the map ^8 | 65.6% | +1.0 (+0.3 to +1.7) | 89.7% | +1.5 (+1.0 to +2.1) |
| agreement on family ^1 | 67.3% | +2.8 (+2.1 to +3.5) | 90.0% | +1.8 (+1.4 to +2.3) |
| agreement on family ^2 | 66.6% | +2.0 (+1.1 to +2.9) | 90.0% | +1.8 (+1.2 to +2.5) |
| agreement on family ^4 | 66.1% | +1.6 (+0.6 to +2.6) | 89.0% | +0.8 (-0.0 to +1.7) |
| drop the odd one out | 66.3% | +1.8 (+0.9 to +2.8) | 88.0% | -0.2 (-1.0 to +0.6) |
| geometric mean | 65.1% | +0.5 (-0.8 to +1.9) | 86.8% | -1.4 (-2.4 to -0.3) |
| geometric, by agreement on family ^1 (vega_profile's "agree") | 68.6% | +4.1 (+2.9 to +5.3) | 88.1% | -0.1 (-1.1 to +0.8) |
| geometric, by agreement on family ^2 | 68.4% | +3.8 (+2.8 to +5.0) | 87.9% | -0.3 (-1.3 to +0.7) |
| ORACLE: only the photos of the main style | 70.0% | +5.4 (+4.6 to +6.4) | 91.5% | +3.3 (+2.7 to +4.0) |

## Repeat on the other half of the wiki personas, all photos of one aesthetic (1,038 sets)

| photos weighted by | is first | against equal | in the top five | against equal |
|---|---|---|---|---|
| equal (the app) | 82.8% | | 94.4% | |
| confidence ^1 | 78.4% | -4.3 (-6.2 to -2.8) | 94.8% | +0.4 (+0.1 to +0.8) |
| confidence ^2 | 76.1% | -6.6 (-8.9 to -4.7) | 95.1% | +0.7 (+0.0 to +1.3) |
| agreement on the map ^2 | 83.8% | +1.1 (+0.3 to +2.0) | 94.6% | +0.2 (+0.0 to +0.5) |
| agreement on the map ^8 | 85.1% | +2.3 (+1.0 to +3.9) | 95.1% | +0.7 (+0.2 to +1.3) |
| agreement on family ^1 | 83.8% | +1.1 (+0.0 to +2.3) | 95.1% | +0.7 (+0.2 to +1.3) |
| agreement on family ^2 | 83.5% | +0.8 (-0.6 to +2.2) | 95.4% | +1.0 (+0.3 to +1.7) |
| agreement on family ^4 | 82.1% | -0.7 (-2.4 to +1.2) | 95.2% | +0.8 (+0.0 to +1.6) |
| drop the odd one out | 81.0% | -1.7 (-3.7 to +0.1) | 94.7% | +0.3 (-0.5 to +1.1) |
| geometric mean | 89.2% | +6.5 (+4.1 to +8.8) | 98.5% | +4.0 (+1.8 to +6.6) |
| geometric, by agreement on family ^1 (vega_profile's "agree") | 88.8% | +6.1 (+4.0 to +8.2) | 98.0% | +3.6 (+1.1 to +6.0) |
| geometric, by agreement on family ^2 | 88.0% | +5.2 (+3.2 to +7.3) | 97.2% | +2.8 (+0.6 to +5.3) |

## Repeat on the other half of the wiki personas, one photo of another family (1,038 sets)

| photos weighted by | is first | against equal | in the top five | against equal |
|---|---|---|---|---|
| equal (the app) | 74.5% | | 91.6% | |
| confidence ^1 | 67.5% | -6.9 (-9.0 to -5.2) | 91.9% | +0.3 (-0.4 to +1.0) |
| confidence ^2 | 64.0% | -10.5 (-12.8 to -8.4) | 91.7% | +0.1 (-0.7 to +0.9) |
| agreement on the map ^2 | 79.7% | +5.2 (+3.6 to +7.0) | 91.9% | +0.3 (-0.1 to +0.8) |
| agreement on the map ^8 | 79.6% | +5.1 (+3.2 to +7.1) | 93.3% | +1.6 (+0.9 to +2.4) |
| agreement on family ^1 | 78.2% | +3.8 (+1.8 to +5.8) | 92.3% | +0.7 (-0.1 to +1.4) |
| agreement on family ^2 | 76.8% | +2.3 (+0.3 to +4.3) | 92.3% | +0.7 (-0.2 to +1.6) |
| agreement on family ^4 | 74.5% | +0.0 (-2.3 to +2.4) | 91.0% | -0.6 (-2.1 to +0.9) |
| drop the odd one out | 77.6% | +3.1 (+1.3 to +4.9) | 91.0% | -0.6 (-1.7 to +0.5) |
| geometric mean | 70.7% | -3.8 (-7.2 to -0.7) | 92.6% | +1.0 (-1.0 to +3.0) |
| geometric, by agreement on family ^1 (vega_profile's "agree") | 82.2% | +7.7 (+5.1 to +10.5) | 95.1% | +3.5 (+1.3 to +5.5) |
| geometric, by agreement on family ^2 | 81.8% | +7.3 (+4.6 to +10.2) | 94.3% | +2.7 (+0.7 to +5.0) |
| ORACLE: only the photos of the main style | 79.1% | +4.6 (+3.3 to +6.2) | 93.0% | +1.3 (+0.7 to +2.0) |

## What it does to the surprise score (expert-labelled outfit photos; per family, shown from 0.38 as in the app)

| profile | sets of one style shown a surprise | sets with an off-style photo shown a surprise | of those, it names the off-style photo's family | names that family, all such sets | the cut that shows 1 set of one style in 10 | at that cut: sets with an off-style photo shown | of those, right family |
|---|---|---|---|---|---|---|---|
| equal (the app) | 10.5% | 29.9% | 52.8% | 35.6% | 0.385 | 29.0% | 53.0% |
| geometric, by agreement on family ^1 | 13.3% | 34.5% | 53.1% | 38.5% | 0.413 | 28.8% | 55.0% |

time elapsed 52s
