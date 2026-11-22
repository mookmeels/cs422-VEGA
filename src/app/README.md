# code/app: the application layer

From one person's photos to a profile, nearby aesthetics, a surprising suggestion and the photo behind each. Nothing here
changes or retrains the team's models: every script calls `code/nn/train_nn.py` as it is, with the frozen settings, and
none reads the test split. Results are in `runs/personas-v1/` and `runs/ratings-v1/`.

| file | what it does |
|---|---|
| `vega_profile.py` | The engine: pooled profile, the map of aesthetics, the surprise score (the proposal's per aesthetic, and per family), `suggest`. |
| `vega_app.py`, `vega_app.html` | The app on this PC (127.0.0.1:8770): 1 to 5 photos in, suggestions out. Trains its own copy of the network at start; saves nothing. |
| `oof.py` | Held-out predictions for every training and validation image, for the persona tests. |
| `persona_eval.py` | Personas made of wiki images: does pooling help, does the surprise score find a planted aesthetic. |
| `poster_eval.py` | The same on real posters (Street Fashion Style, held-out group). |
| `fs14_persona_eval.py` | The same on FashionStyle14's 11 non-transfer styles. |
| `surprise_variants.py` | Per aesthetic against per family for the surprise score, on the three kinds of sets. |
| `weight_variants.py` | Should some of a person's photos count more than others: ways to weight the photos of a set, against the plain average. |
| `many_photos.py`, `surprise_cuts.json` | What changes from 4 photos to 50: which pooling to use at each size, and the surprise cut per number of photos (the JSON, read by the app). |
| `rate_build.py`, `rate_page.html` | Builds the rating page `runs/ratings-v1/vega-rate.html`: people click whether a suggestion fits. |
| `rate_score.py` | Scores the answer lines people send back (`runs/ratings-v1/answers.txt`) into `results.md`. |
