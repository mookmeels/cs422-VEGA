# Personas from wiki images: the context question and the serendipity question

Made by `code/app/persona_eval.py`. 347 aesthetics have an outfit image and two context images with one label; 6 pure and 6 mixed personas each: 4,164 personas. Settings were chosen on 2,076 personas of 173 home aesthetics; **every figure below is from the other 2,088 personas of 174 home aesthetics.** Each image is scored by a network that never trained on it. The test split was not read.

The weighted profile's settings, chosen on the tune half: an outfit photo counts x1; a photo's strongest match is raised to the power 0.

## Question 1: does context help? Where the home aesthetic ranks

### Pure personas (1,044): 1 outfit and 2 or 3 context images of one aesthetic

| profile from | home is first | home in top 5 | MRR | top 5 against outfit only (95% interval) |
|---|---|---|---|---|
| outfit photo only | 50.3% | 76.6% | 0.622 | |
| one context image only | 53.8% | 82.6% | 0.664 | +5.9 (+0.8 to +10.8) |
| outfit + 1 context image | 65.4% | 89.1% | 0.759 | +12.5 (+8.7 to +16.2) |
| outfit + 2 context images | 74.5% | 94.6% | 0.829 | +18.0 (+13.7 to +21.9) |
| context images only, no outfit | 74.6% | 95.1% | 0.829 | +18.5 (+13.7 to +23.1) |
| all photos, equal | 80.2% | 95.5% | 0.867 | +18.9 (+14.3 to +23.2) |
| weighted profile (outfit x1, strength ^0) | 80.2% | 95.5% | 0.867 | +18.9 (+14.3 to +23.2) |

### Mixed personas (1,044): one of the context images is from another family's aesthetic

| profile from | home is first | home in top 5 | MRR | top 5 against outfit only (95% interval) |
|---|---|---|---|---|
| outfit photo only | 50.3% | 76.6% | 0.622 | |
| all photos, equal | 70.1% | 90.7% | 0.789 | +14.1 (+9.6 to +18.4) |
| weighted profile (outfit x1, strength ^0) | 70.1% | 90.7% | 0.789 | +14.1 (+9.6 to +18.4) |

Intervals resample home aesthetics with all their personas (2,000 draws, seed 422).

## Question 2: does the surprise score find the planted aesthetic?

On the 1,044 mixed personas. The planted aesthetic B is one photo of four, from another family than the home.

| how the suggestion is picked | B is the first suggestion | B is among the first three |
|---|---|---|
| **S = relevance x unexpectedness** (the proposal's score) | 40.8% (37.9% to 43.8%) | 52.9% (49.7% to 56.2%) |
| relevance alone (strongest single-photo match, top aesthetic excluded) | 34.9% (31.9% to 38.0%) | 49.8% (46.7% to 53.3%) |
| more of the same: ranks 2 to 4 of the profile | | 49.3% (46.3% to 52.8%) |
| a random aesthetic at least as unexpected as the one S picked | 0.69% | |

- **What unexpectedness adds over relevance alone:** +5.9 (+4.4 to +7.6) points on 'B is the first suggestion' (95% interval, same personas).
- **The ceiling is the network, not the score.** The network's own first guess for the planted photo is B in 53.3% (50.8% to 55.7%) of mixed personas. In those 556 personas S puts B first 75.2% of the time.
- B is already inside the profile's top 5 for 62.7% (60.2% to 65.5%) of mixed personas, so most planted aesthetics are visible before any surprise score.
- S's first suggestion is in B's family for 55.7% (52.3% to 59.3%) of mixed personas (that includes the cases where it is B itself).
- **Explanation:** when S picks B, the photo named as the reason is the planted photo in 100.0% of cases (426 personas). Over all mixed personas, the reason given for S's first suggestion is the planted photo in 75.7% (72.2% to 79.4%).
- **Can the score tell when there is something to find?** The largest S of a mixed persona is higher than that of a pure persona 86.8% of the time (50% would be chance; median 0.34 against 0.15).

## What this is and is not

- The personas are synthetic and their images come from wiki galleries, as the proposal planned for training and validation personas. Real people's photo sets are not tested here.
- 'Context' means non-outfit images of the same aesthetic (rooms, objects, scenery, collages), not a person's own room or car.
- The planted aesthetic is relevant by construction. Whether a surprising suggestion FEELS fitting to a person needs human ratings, which this does not have.
- Aesthetics without an outfit image and two context images cannot be a home, so the personas cover 347 of the 503 labels.

time elapsed 10s
