# The second rating page (precision@5 per system), version v2-187f9c

Made by `code/experiments/rate5_build.py` on cuda. 10 sets of four real outfit photos (5 of one FashionStyle14 style each, 5 of one held-out poster each), 166 screens: every aesthetic in the top five of any of the four systems, once per set.
Of the NN's 50 suggestions, also in the top five of: centroid 19, zeroshot 14, popularity 0.
`key.csv` says which systems suggested each aesthetic and at which rank. Paste returned answer lines into `answers.txt` and run `rate5_score.py`.
time elapsed 43s
