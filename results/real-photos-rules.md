# Real photos in training: rules, fixed before anything was trained

Written 2026-11-21, before the first run. Experiment on validation data only; whether its model is adopted for the
final run is Andrew's and the team's decision, taken with every number below visible.

## The question

Does adding Street Fashion Style outfit photos to the NN's training make it better on real outfit photos, and what
does it cost on the wiki images?

## Why it is built this way

- A Street Fashion Style photo carries the poster's style tag, which stands for a FAMILY (frozen table
  `runs/family-v1/outside_family.csv`), not for one of the 503 aesthetics.
- The family NN experiment (21 Nov) showed that the 503 outputs should stay: a network trained on families alone
  was 7.5 points worse. So the network keeps its 503 outputs, and a real photo is scored on the probability its
  family receives in total (the sum over that family's aesthetics).
- Known risk (family pilot, 19 Nov): when only some families have real photos, photos of the other families get
  pulled into the covered ones. Here 12 of the 20 families have Street Fashion Style photos and 8 have none
  (cute, lolita, historical, fantasy, club, regional, seasons, design). Rule 6 measures that.

## Rules

1. **Labels.** Family table v1, frozen. Only photos whose tag is mapped to a family are used (187,577 listed);
   Chic, Trendy, Eclectic and the small tags are not used.
2. **Network and settings.** `code/nn/train_nn.py`, network and SETTINGS unchanged: 503 outputs, the same loss on
   the wiki training images in the same order, input scaling measured on the wiki training images only.
   Seeds 422 to 426, one CPU thread.
3. **The one addition.** Each training step also takes 512 real photos and adds `weight` x their loss, where a
   photo's loss is minus the log of the summed probability of its family. Real photos are drawn family by family
   (each covered family equally often, then a photo of it at random), so the 72,000 casual photos do not drown the
   914 goth ones.
4. **Settings tried: exactly three.** weight 0 (the NN as it is), 0.25 and 1.0. No other value, no other change,
   whatever the results.
5. **Held out.** Street Fashion Style is split by POSTER: posters whose name hashes to 0 mod 10 are never trained
   on. FashionStyle14 is never trained on at all. Its three transfer-test styles (lolita, fairy, ethnic) are not
   read or scored here. The wiki test split is not read.
6. **What is measured**, each against weight 0, paired bootstrap, 2,000 draws, seed 422, each model measured per
   image as the share of its 5 seeds that are right:
   - wiki validation: aesthetic top-5 and top-1; family top-1 (probabilities summed); outfit images and others;
   - FashionStyle14, the 11 other styles: family top-1 (a different source with expert labels: the real test);
   - Street Fashion Style, held-out posters: family top-1 (bootstrap over posters);
   - the shortcut: wiki validation outfit images whose families have no real photos: family top-1, and how often
     the first guess is a family that does have real photos.
7. **How the result is read**, decided now:
   - "helps on real photos" = the FashionStyle14 difference has its whole 95% interval above zero;
   - "costs on wiki" = wiki aesthetic top-5, or family top-1 on the outfit images of uncovered families, has its
     whole interval below zero;
   - helps without a cost: a candidate for adoption; helps with a cost: a trade-off, reported as such;
     no help: reported as a negative result. Nothing is re-run with other settings to rescue a result.
8. **Not touched.** The team table, the shared loader and scorer, `code/nn/`, the transfer test. A model trained
   with bohemian street photos is no longer a wiki-only model, so it would not enter the transfer table as one.
