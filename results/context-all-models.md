# The three input settings for every model (validation images only)

Made 2026-11-22 08:46 UTC by code/experiments/context_all_models.py. Nothing trained; the test split is not read.
Personas from validation images: 516 aesthetic personas over 86 home labels (1 outfit + 2 or 3 context images of one label), 1900 family personas over 19 families (1 outfit + 3 context images of one family).
Each photo's scores become 1 / rank (the scorer's ranking), then code/app/vega_profile.pooled: outfit, equal, agree, agree_mean.
± is the spread over a model's seeds; intervals are 95%, resampling outfit images.

## Aesthetic personas: home aesthetic in the top 5

| model | runs | outfit photo only | all photos, equal | weighted by agreement (geometric mean, the app's rule) | weighted by agreement (ordinary average) | all equal - outfit only | weighted (app's rule) - equal |
|---|---|---|---|---|---|---|---|
| NN, wiki images only | 5 | 89.5 ± 0.8 | 99.7 ± 0.3 | 100.0 ± 0.0 | 99.7 ± 0.3 | +10.1 (+4.8 to +16.0) | +0.3 (+0.0 to +1.1) |
| NN + real outfit photos | 5 | 88.6 ± 0.8 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | +11.4 (+6.1 to +17.6) | +0.0 (+0.0 to +0.0) |
| Baseline: nearest centroid | 1 | 72.1 | 94.8 | 98.8 | 95.3 | +22.7 (+14.2 to +31.1) | +4.1 (+1.5 to +7.7) |
| Hierarchical clustering | 1 | 39.0 | 54.1 | 45.0 | 54.3 | +15.1 (+8.8 to +21.8) | -9.1 (-15.8 to -3.0) |
| Baseline: zero-shot text | 1 | 45.2 | 55.2 | 53.9 | 55.0 | +10.1 (+3.2 to +17.8) | -1.4 (-6.5 to +3.3) |
| CNN + street photos | 5 | 31.6 ± 1.7 | 55.3 ± 2.2 | 75.2 ± 4.0 | 54.3 ± 2.9 | +23.7 (+16.3 to +31.4) | +19.8 (+14.6 to +25.3) |
| CNN | 5 | 28.1 ± 2.3 | 54.1 ± 2.3 | 75.9 ± 2.9 | 52.9 ± 2.4 | +26.0 (+19.5 to +32.9) | +21.7 (+16.2 to +27.3) |
| CNN, adopted recipe + street photos | 5 | 24.4 ± 2.6 | 40.9 ± 1.2 | 68.6 ± 1.6 | 40.9 ± 1.5 | +16.5 (+9.7 to +23.2) | +27.8 (+22.1 to +33.7) |
| Baseline: popularity | 1 | 5.8 | 5.8 | 5.8 | 5.8 | +0.0 (+0.0 to +0.0) | +0.0 (+0.0 to +0.0) |

## Aesthetic personas: home aesthetic first

| model | runs | outfit photo only | all photos, equal | weighted by agreement (geometric mean, the app's rule) | weighted by agreement (ordinary average) | all equal - outfit only | weighted (app's rule) - equal |
|---|---|---|---|---|---|---|---|
| NN, wiki images only | 5 | 60.3 ± 1.3 | 94.3 ± 1.0 | 96.8 ± 0.4 | 94.3 ± 1.5 | +34.0 (+26.4 to +41.5) | +2.6 (+0.2 to +5.2) |
| NN + real outfit photos | 5 | 61.0 ± 1.9 | 91.9 ± 1.4 | 96.0 ± 1.2 | 91.8 ± 1.6 | +30.9 (+23.2 to +38.7) | +4.1 (+0.9 to +7.8) |
| Baseline: nearest centroid | 1 | 44.4 | 79.8 | 90.3 | 80.4 | +35.5 (+26.0 to +44.1) | +10.5 (+5.7 to +15.9) |
| Hierarchical clustering | 1 | 28.1 | 38.2 | 32.0 | 37.8 | +10.1 (+3.2 to +16.7) | -6.2 (-10.5 to -1.9) |
| Baseline: zero-shot text | 1 | 18.0 | 33.7 | 34.7 | 33.7 | +15.7 (+8.5 to +23.5) | +1.0 (-3.4 to +5.2) |
| CNN + street photos | 5 | 10.3 ± 0.8 | 31.9 ± 4.7 | 46.9 ± 2.2 | 30.0 ± 3.5 | +21.6 (+15.4 to +27.8) | +15.0 (+10.4 to +19.8) |
| CNN | 5 | 10.7 ± 1.7 | 29.8 ± 2.5 | 46.1 ± 3.9 | 28.1 ± 1.7 | +19.2 (+12.9 to +25.5) | +16.3 (+11.3 to +21.7) |
| CNN, adopted recipe + street photos | 5 | 9.7 ± 1.0 | 24.2 ± 1.6 | 35.2 ± 2.3 | 22.5 ± 1.2 | +14.5 (+9.9 to +19.3) | +11.0 (+7.1 to +15.0) |
| Baseline: popularity | 1 | 1.2 | 1.2 | 1.2 | 1.2 | +0.0 (+0.0 to +0.0) | +0.0 (+0.0 to +0.0) |

## Family personas: home family first

| model | runs | outfit photo only | all photos, equal | weighted by agreement (geometric mean, the app's rule) | weighted by agreement (ordinary average) | all equal - outfit only | weighted (app's rule) - equal |
|---|---|---|---|---|---|---|---|
| NN, wiki images only | 5 | 72.6 ± 0.6 | 84.9 ± 0.7 | 70.5 ± 1.3 | 89.0 ± 0.6 | +12.2 (+7.8 to +16.7) | -14.4 (-17.3 to -11.8) |
| NN + real outfit photos | 5 | 72.1 ± 1.3 | 85.3 ± 0.5 | 71.8 ± 1.3 | 89.2 ± 0.6 | +13.2 (+9.1 to +17.4) | -13.5 (-16.0 to -11.0) |
| Baseline: nearest centroid | 1 | 68.4 | 79.7 | 78.1 | 84.2 | +11.4 (+6.7 to +16.3) | -1.7 (-4.1 to +0.8) |
| Hierarchical clustering | 1 | 44.2 | 43.5 | 26.5 | 48.7 | -0.7 (-6.1 to +4.9) | -17.0 (-19.8 to -14.1) |
| Baseline: zero-shot text | 1 | 40.5 | 49.7 | 45.0 | 55.5 | +9.2 (+4.7 to +13.7) | -4.7 (-7.3 to -2.2) |
| CNN + street photos | 5 | 22.7 ± 1.3 | 28.5 ± 1.3 | 31.3 ± 1.2 | 31.2 ± 2.3 | +5.8 (+2.5 to +8.9) | +2.8 (+0.9 to +4.8) |
| CNN | 5 | 22.8 ± 1.6 | 30.3 ± 1.3 | 31.4 ± 1.3 | 33.2 ± 2.0 | +7.5 (+4.3 to +10.8) | +1.1 (-1.1 to +3.2) |
| CNN, adopted recipe + street photos | 5 | 23.4 ± 1.7 | 31.8 ± 1.2 | 31.8 ± 0.6 | 34.1 ± 1.7 | +8.4 (+4.9 to +11.8) | -0.1 (-1.8 to +1.4) |
| Baseline: popularity | 1 | 5.3 | 5.3 | 5.3 | 5.3 | +0.0 (+0.0 to +0.0) | +0.0 (+0.0 to +0.0) |

## Family personas: home family in the first 3

| model | runs | outfit photo only | all photos, equal | weighted by agreement (geometric mean, the app's rule) | weighted by agreement (ordinary average) | all equal - outfit only | weighted (app's rule) - equal |
|---|---|---|---|---|---|---|---|
| NN, wiki images only | 5 | 92.3 ± 0.8 | 99.0 ± 0.2 | 95.0 ± 0.3 | 99.3 ± 0.1 | +6.7 (+3.7 to +10.1) | -4.0 (-5.0 to -3.0) |
| NN + real outfit photos | 5 | 92.0 ± 0.7 | 99.0 ± 0.1 | 94.8 ± 0.7 | 99.1 ± 0.2 | +7.0 (+4.1 to +10.4) | -4.1 (-5.2 to -3.2) |
| Baseline: nearest centroid | 1 | 88.3 | 97.7 | 97.5 | 97.8 | +9.4 (+5.8 to +13.4) | -0.2 (-1.1 to +0.8) |
| Hierarchical clustering | 1 | 67.1 | 80.9 | 39.7 | 80.8 | +13.9 (+9.3 to +18.7) | -41.2 (-45.2 to -37.3) |
| Baseline: zero-shot text | 1 | 68.9 | 84.2 | 76.9 | 86.5 | +15.2 (+11.2 to +19.6) | -7.3 (-9.9 to -4.6) |
| CNN + street photos | 5 | 45.0 ± 2.5 | 55.3 ± 2.8 | 57.3 ± 1.5 | 56.6 ± 2.8 | +10.3 (+7.1 to +13.5) | +1.9 (-1.1 to +4.9) |
| CNN | 5 | 45.9 ± 1.1 | 57.7 ± 1.4 | 58.4 ± 2.2 | 59.6 ± 1.7 | +11.7 (+8.1 to +15.4) | +0.7 (-2.3 to +3.5) |
| CNN, adopted recipe + street photos | 5 | 45.4 ± 1.8 | 59.9 ± 1.8 | 60.1 ± 1.5 | 60.2 ± 2.1 | +14.5 (+10.8 to +18.2) | +0.2 (-2.7 to +2.7) |
| Baseline: popularity | 1 | 15.8 | 15.8 | 15.8 | 15.8 | +0.0 (+0.0 to +0.0) | +0.0 (+0.0 to +0.0) |

Time elapsed: 37 s
