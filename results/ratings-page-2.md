# Precision@5 per system by people's ratings, page version v2-187f9c

Made by `code/experiments/rate5_score.py` from 2 rater(s): Edmond, Giancarlo. 166 screens over 10 sets of four real outfit photos; 0 screens nobody could judge. An aesthetic fits by majority of the raters.

| system | precision@5 (95% interval) | nDCG@5 | expert sets, precision@5 | posters' sets, precision@5 |
|---|---|---|---|---|
| NN + real outfit photos | 0.20 (0.10 to 0.30) | 0.20 (0.10 to 0.30) | 0.32 | 0.08 |
| Baseline: nearest centroid | 0.20 (0.08 to 0.34) | 0.22 (0.08 to 0.39) | 0.32 | 0.08 |
| Baseline: zero-shot text | 0.14 (0.04 to 0.24) | 0.13 (0.03 to 0.23) | 0.16 | 0.12 |
| Baseline: popularity | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 | 0.00 |

Against the popularity baseline, precision@5, same sets:

- NN + real outfit photos: +0.20 (+0.10 to +0.30)
- Baseline: nearest centroid: +0.20 (+0.08 to +0.34)
- Baseline: zero-shot text: +0.14 (+0.04 to +0.24)
- NN minus nearest centroid: +0.00 (-0.10 to +0.10)
- NN minus zero-shot text: +0.06 (-0.04 to +0.16)

## Agreement between raters

- Fleiss' kappa, three answers (fits, doesn't fit, can't tell), 166 screens answered by all 2 raters: 0.16
- Fleiss' kappa, fits against doesn't fit, 150 screens where nobody said can't tell: 0.23
- Edmond alone: NN precision@5 0.62; said can't tell on 15 screens
- Giancarlo alone: NN precision@5 0.18; said can't tell on 1 screens

time elapsed 0s
