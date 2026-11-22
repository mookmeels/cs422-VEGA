# The rating page, version v1-50dd61

Made by `code/app/rate_build.py` on cuda. 20 sets of four real outfit photos, 80 screens: for each set its top, nearby and surprising suggestion and a random control, one per screen, in blocks of eight screens holding two of each kind.
Only sets for which the app shows a surprise (score at or above 0.41) are used. A family shown as surprising is left out of the profile and the nearby list (as in the app):

- **posters**: 10 sets kept, the first 10 of 55 drawn for which the app shows a surprise.
- **expert**: 10 sets kept, the first 10 of 38 drawn for which the app shows a surprise.

## Over all the sets drawn

- **posters**: a surprise is shown for 60 of 364 sets (16.5%). Before the surprising family was left out of the profile, the surprising aesthetic was also the profile's FIRST aesthetic in 2 of them (3%) and among its five in 21 (35%).
- **expert**: a surprise is shown for 178 of 600 sets (29.7%). Before the surprising family was left out of the profile, the surprising aesthetic was also the profile's FIRST aesthetic in 1 of them (1%) and among its five in 36 (20%).
  - In those 178 sets the profile's first aesthetic is of the family of the three photos' style in 69.1% before, 68.5% after.
- **expert, four photos of ONE style** (no off-style photo): a surprise is shown for 64 of 600 sets (10.7%). The cut was set for 10%, with networks trained on the training images only; the app's network also saw the validation images.
- Photos are combined as in the app: `agree` (vega_profile.pooled).

`key.csv` says what each screen was. Paste returned answer lines into `answers.txt`, one per line, and run `rate_score.py`.
time elapsed 25s
