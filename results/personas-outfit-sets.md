# A control on expert-labelled outfit photos (FashionStyle14, 11 styles)

Made by `code/app/fs14_persona_eval.py`, seeds [422, 423, 424] on cuda. 10,312 photos never trained on; per photo the network names the right family 54.1% of the time. 3,000 pure and 3,000 mixed sets, built by style.

## Question 1: more photos of one style

| photos pooled | names the style's family | the style's family among the first three |
|---|---|---|
| 1 | 56.9% (55.2% to 58.6%) | 83.5% (82.2% to 84.7%) |
| 2 | 65.3% (63.7% to 66.8%) | 89.4% (88.4% to 90.4%) |
| 3 | 70.3% (68.8% to 71.8%) | 92.0% (91.2% to 93.0%) |
| 4 | 73.1% (71.4% to 74.7%) | 93.7% (92.8% to 94.4%) |

## Question 2: three photos of one style and one of a style from another family

| | share of sets |
|---|---|
| the profile's first aesthetic is of the main style's family | 56.4% (54.8% to 57.9%) |
| **the surprise score's first suggestion is of the other style's family** | 28.2% (26.8% to 29.6%) |
| relevance alone picks that family | 25.9% (24.6% to 27.3%) |
| the profile's second family is that family (no surprise score) | 30.4% (29.1% to 31.9%) |
| a random aesthetic at least as unexpected is of that family | 5.5% |
| the photo given as the reason is the odd one out | 48.5% (47.0% to 50.0%) |

- On its own the network puts the odd photo in its style's family 54.0% of the time; in those cases the surprise score's first suggestion is of that family 47.8% of the time.
- The largest S of a mixed set is higher than that of a pure set 65.8% of the time (50% is chance); medians 0.17 and 0.13. At the app's cut of 0.25, 20.8% of mixed sets and 7.9% of pure sets would show a surprise.

Intervals resample sets (1,000 draws). Sets are by style, not by person.
time elapsed 35s
