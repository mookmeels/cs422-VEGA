# Real people's photo sets: Street Fashion Style posters the network never trained on

Made by `code/app/poster_eval.py`, seeds [422, 423, 424] on cuda. 519 held-out posters (group 0), 18,610 photos with a tag in use. Judged at family level against the posters' own tags. The wiki test split was not read.

## Question 1: do more photos of a person give a better profile?

321 posters with 8 or more photos; 1,478 draws of five photos. The reference is the family the poster uses most in their OTHER photos.

**Read this first:** that family is 'Casual and everyday' for 72.9% of the draws, so always answering it would be right 72.9% of the time. The network is trained with every family drawn equally often, so it does not know that. Two columns correct for it: 'with the street prior' multiplies the network's family shares by how common each family is in the other nine groups' posts, and 'every family counted the same' averages the hit rate over the 10 main families.

| photos pooled | names the poster's main family | with the street prior | every family counted the same | main family among the first three | names a family the poster uses at all |
|---|---|---|---|---|---|
| 1 | 24.7% (22.3% to 27.3%) | 57.0% (53.5% to 60.4%) | 27.7% | 56.2% (53.5% to 59.2%) | 68.8% (65.8% to 71.8%) |
| 2 | 23.7% (21.2% to 26.3%) | 65.7% (61.9% to 69.4%) | 30.6% | 60.5% (57.4% to 63.6%) | 70.3% (67.2% to 73.2%) |
| 3 | 25.6% (22.8% to 28.5%) | 70.5% (66.4% to 74.4%) | 33.2% | 63.2% (60.1% to 66.6%) | 73.0% (70.0% to 75.8%) |
| 5 | 28.6% (25.3% to 32.0%) | 74.7% (70.5% to 78.5%) | 31.6% | 66.6% (63.2% to 69.9%) | 77.7% (74.6% to 80.6%) |

Five photos against one, same draws:
- names the main family: +3.9 points (+1.1 to +6.5)
- with the street prior: +17.7 points (+15.1 to +20.2)
- main family among the first three: +10.4 points (+7.6 to +13.0)
- names a family the poster uses: +8.8 points (+6.6 to +11.3)
- wiki-only NN, main family among the first three: 51.3% with one photo, 62.8% with five (the NN: 56.2% and 66.6%)

Intervals resample posters (2,000 draws, seed 422).

## Question 2: does the surprise score find the photo that is off the poster's usual style?

308 posters whose most-used family A has four or more photos and who tagged at least one photo with another family B. 1,540 sets: three photos tagged A and one tagged B.

| | share of sets |
|---|---|
| the profile's first aesthetic is of family A | 28.3% (25.4% to 31.3%) |
| **the surprise score's first suggestion is of family B** | 16.7% (15.0% to 18.4%) |
| relevance alone picks an aesthetic of family B | 16.3% (14.5% to 18.0%) |
| the profile's second family is B (no surprise score) | 15.2% (13.3% to 16.9%) |
| a random aesthetic at least as unexpected is of family B | 4.6% |
| the photo given as the reason is the one tagged B | 31.8% (29.5% to 34.2%) |

- Surprise score against relevance alone, same sets: +0.3 points (-0.6 to +1.4).
- **The ceiling is the network.** On its own it puts the B-tagged photo in family B only 36.9% of the time. In those cases the surprise score's first suggestion is of family B 32.7% of the time.
- The largest S of a set with a B photo is higher than that of a set of four A photos 55.0% of the time (50% is chance); medians 0.12 and 0.11. At the app's cut of 0.25, 3.3% of mixed sets and 1.9% of pure sets would show a surprise.

## What this is and is not

- The ground truth is the poster's own style word, mapped to one of 12 families by our table. It is coarse and noisy: a photo-by-photo check gives the same network about a third right.
- All photos are outfit posts. There is no room, car or object photo here, so this tests 'more photos of a person', not 'context'.
- A photo tagged with another family is off the poster's usual style by their own word. Whether the aesthetic the score names would feel right to them is not measured.

time elapsed 110s
