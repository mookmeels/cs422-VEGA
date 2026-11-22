# Which surprise score? Per aesthetic against per family

Made by `code/app/surprise_variants.py`, seeds [422, 423, 424] on cuda for the outfit photos; wiki personas use the held-out predictions of oof.py. Each mixed set has four photos, the last one off the set's main style. Intervals resample sets (1,000 draws).

## wiki personas: 1,044 mixed sets, 1,044 pure sets

| picked by | names the off-style photo's family | names its exact aesthetic | the reason given is that photo | tells mixed from pure | mixed sets shown when 1 pure set in 10 is |
|---|---|---|---|---|---|
| per aesthetic (the proposal's) | 55.7% (52.9% to 58.7%) | 40.8% (37.8% to 43.7%) | 75.7% (72.9% to 78.1%) | 86.8% | 61.6% (cut 0.29); of those, right family 66.3% |
| per family | 57.1% (54.0% to 60.1%) | 42.4% (39.4% to 45.4%) | 75.3% (72.7% to 78.0%) | 81.1% | 46.0% (cut 0.45); of those, right family 72.9% |
| family match only | 57.1% (54.0% to 60.0%) | 43.3% (40.4% to 46.2%) | 72.5% (69.8% to 75.2%) | 79.1% | 42.3% (cut 0.86); of those, right family 79.0% |
| second family, no score | 52.1% (49.0% to 55.0%) | 39.7% (36.6% to 42.7%) | 66.1% (63.1% to 68.8%) | 69.5% | 8.0% (cut 0.28); of those, right family 39.8% |

Per family against per aesthetic, same sets, family named: +1.3 points (-1.3 to +4.1).

## expert-labelled outfit photos: 3,000 mixed sets, 3,000 pure sets

| picked by | names the off-style photo's family | the reason given is that photo | tells mixed from pure | mixed sets shown when 1 pure set in 10 is |
|---|---|---|---|---|
| per aesthetic (the proposal's) | 29.2% (27.8% to 30.6%) | 49.0% (47.6% to 50.6%) | 63.8% | 22.2% (cut 0.24); of those, right family 39.5% |
| per family | 37.3% (35.8% to 38.9%) | 58.4% (56.9% to 60.1%) | 68.7% | 28.9% (cut 0.38); of those, right family 53.7% |
| family match only | 36.3% (34.8% to 37.9%) | 53.5% (52.0% to 55.1%) | 65.5% | 22.3% (cut 0.71); of those, right family 60.0% |
| second family, no score | 32.1% (30.7% to 33.6%) | 44.2% (42.7% to 45.7%) | 54.3% | 7.9% (cut 0.26); of those, right family 39.1% |

Per family against per aesthetic, same sets, family named: +8.1 points (+6.7 to +9.5).

## real posters: 1,540 mixed sets, 1,605 pure sets

| picked by | names the off-style photo's family | the reason given is that photo | tells mixed from pure | mixed sets shown when 1 pure set in 10 is |
|---|---|---|---|---|
| per aesthetic (the proposal's) | 15.7% (14.2% to 17.1%) | 31.7% (29.9% to 33.6%) | 54.6% | 14.5% (cut 0.18); of those, right family 17.7% |
| per family | 16.9% (15.3% to 18.6%) | 32.9% (30.7% to 35.1%) | 56.5% | 15.8% (cut 0.33); of those, right family 22.1% |
| family match only | 17.4% (15.6% to 18.9%) | 33.1% (31.1% to 35.2%) | 54.8% | 13.1% (cut 0.57); of those, right family 24.6% |
| second family, no score | 15.2% (13.7% to 16.8%) | 28.2% (26.4% to 29.8%) | 49.2% | 9.2% (cut 0.23); of those, right family 21.1% |

Per family against per aesthetic, same sets, family named: +1.2 points (-0.5 to +2.8).

'Tells mixed from pure': how often a mixed set's score is higher than a pure set's (50% is chance). The last column sets the cut where 10% of pure sets would be shown a surprise, and says how many mixed sets clear it.
time elapsed 28s
