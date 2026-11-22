# Family table v1: FROZEN

Accepted by Andrew on 2026-11-21 (about 00:45 UTC) with no changes to the proposal ("i actually have zero suggestions").
Frozen 2026-11-21 00:48 UTC.

- `label_family.csv`: 503 wiki labels, 20 families. sha256 `7eea470aedf1b91c7531eb947d594649ad1abafaa3e8aa0c59e6c04d6bb40dde`.
- `outside_family.csv`: which FashionStyle14 styles and Street Fashion Style tags stand for which family (unchanged from the draft;
  the two new families, seasons and holidays, and design, graphics and screens, have no outside photos).
- `proposal.csv`, `review.html`, `summary.md`: how the table was made and what Andrew was shown.

One family level for the overall aesthetic. Outfit against vibe is a property of the photo, not of the label.

## Rules for the family NN (fixed before any family model was trained)

1. This table is frozen. No label moves once a family model has been scored.
2. Same network and settings as the 503-label NN (`code/nn/train_nn.py`, SETTINGS), no re-tuning on families. Seeds 422 to 426.
   Trained on the training split, scored on validation. One CPU thread.
3. The comparison is the family NN against the 503-label NN with its guesses grouped by family, on the same validation images,
   paired bootstrap (2,000 draws, seed 422), each model measured per image as the share of its 5 seeds that are right.
   Grouping is done two ways, both reported: by SUMMING the 503-label NN's probabilities within each family (the main
   comparison, because it is the stronger way to group) and by taking the family of its first guess.
4. Family top-1 on all validation images is the family figure (the first family guessed is one of the image's families).
   Right family in the first 3, and outfit images against the other images, are reported beside it.
5. First run on wiki images only. Adding Street Fashion Style or FashionStyle14 photos to training is a separate, later
   experiment with its own decision. The transfer-test photos stay out of everything.
6. The test split is not read until the one final run.

## To be stated in the report

The image vectors chose which labels were looked at again and decided the placement of the aesthetics Claude Code did not
know, so the table may favour models that use those vectors (the NN and the baselines) over the CNN and the clustering model.

## Amendment 1, 2026-11-21 04:59 UTC: FashionStyle14 "natural" stands for Casual and everyday, not Natural, country and outdoors

Decided by Andrew. Reason: the pairing was wrong by meaning. FashionStyle14's "natural" is a Japanese street style of plain,
loose, soft everyday clothes; the Natural, country and outdoors family is hiking, western, cottage and forest looks. The
pairing had been made from the shared word. Found when the real-photo check was looked at style by style
(runs/real-photos-v1/misses-before-natural-fix.md), so the figures before and after are both reported.

What it touches: only `outside_family.csv`, one row, and only the real-photo check on FashionStyle14. That dataset is never
trained on, so no model changes. `label_family.csv` (the 503 wiki labels) is untouched: sha256 as above.
The file before the change: `outside_family.before-natural-fix.csv`. After: sha256 `943ab36511d53a810432c0c187859b31853f6918a55b5afe2ad4299b6a3cf6ab`.

## Amendment 2, 2026-11-21 05:49 UTC: five labels moved because their own wiki pages contradict the placement

Decided by Andrew after seeing each page's content beside the table's placement (runs/family-v1/page_reading_1.md).
The table had been made from label names and general knowledge without the pages being read.

| label | was | now | what the page says |
|---|---|---|---|
| Poster Gal | Glam, sexy and party | Cute and kawaii | a 2000s chibi drawing style with no distinct fashion (page: other names Y2K Chibi, deviantARTcore; tags Anime, Art Styles) |
| Too Cool | Street, hip-hop and sporty | Regional and traditional dress | a Chinese kitsch movement built on traditional and rural Chinese dress; the qipao is its staple |
| New Partisans | Club, rave and dance scene | Rock, punk, grunge and emo | a mid-1980s movement on the Yugoslav rock scene |
| Corporate Grunge | Street, hip-hop and sporty | Rock, punk, grunge and emo | the commercial design style that followed grunge; tagged Grunge; its fashion is grunge made more polished |
| Nordic Twee | Romantic, feminine and girly | Natural, country and outdoors | described by its page as a westernised revival of Mori Kei, which is in this family |

`label_family.csv` sha256 is now `f6394f3b533fb7a2abab56b9f92b27391bd5d89f47014956809f3532bc1a35bd`. The table before this amendment: `label_family.before-amendment-2.csv`
(sha256 7eea470aedf1...). `proposal.csv` is kept as the record of what Andrew was shown on 21 Nov and is not changed.

What had to be redone because of it: the family figures of every validation row (they are computed from this table), and the
NN with real outfit photos (its real-photo loss sums probabilities within a family, so it is trained through this table).
The settings experiment in runs/real-photos-v1 was run on the table as first frozen; it was not repeated.

## Amendment 3, 2026-11-21 06:14 UTC: what a family means (the look rule), and one label moved under it

Decided by Andrew ("Yes, by look") after arguing the alternative. His argument: a wiki parent link is history, and a model that
sees pixels cannot recover history. Measured before deciding: `look_vs_lineage.md` (of the 17 sub-styles whose wiki parent is in
another family, the training photos sit with the table's family in 12, with the parent's in 4, tie in 1).

**The rule.**
1. A label's family is decided by its LOOK: what its own wiki page describes (fashion section, motifs, the page's own words for
   the style) and where its training photos sit among the other labels' training photos.
2. A wiki parent in another family is a flag that triggers that check. It does not move the label.
3. A label moves only when its page and its photos point to the SAME other family. Otherwise it stays where the table has it.
4. Placements are never taken from a model's errors, and no validation or test image is used for them.

**Applied to the labels in question** (page read with `read_pages.py`; photos with `look_vs_lineage.py --labels`: the share of each
training photo's 10 nearest neighbours outside the label that belong to a family, base = that family's share of all images):

| label | images | table has | the page | the photos | outcome |
|---|---|---|---|---|---|
| Farmer's Daughter | 13 | Natural, country and outdoors | a variation of Nymphet; other name Country Coquette; tags Coquette, Feminine, Sexy; lingerie corsets, milkmaid dresses, tied plaid shirts | Romantic 38% (base 8%), Natural 25% (base 5%); nearest labels Coquette, Trailer Park Princess | **moved to Romantic, feminine and girly** |
| Amekaji | 7 | Casual and everyday | its own words: an exaggerated take on American West Coast streetwear; tags Streetwear, Hip-Hop | Cute 40%, Glam 29%, Street 9%, Casual 0%; nearest labels are other Gyaru sub-styles | page and photos disagree: **left for Andrew** |
| New Beat | 11 | Club, rave and dance scene | a Belgian dance-music subculture; tags Rave Culture, Nightlife | Goth 25%, Rock 16%, Club 15% | stays (page supports the table) |
| Gloomcore | 33 | Goth and dark | foggy forests and shores; says it is an aesthetic of visuals, not fashion; values the mysterious and dark | Fantasy 33%, Natural 29%, Goth 27% | stays (no single family indicated) |
| Whimsicraft | 18 | Design, graphics and screens | 1980s-90s art-fair crafts; tagged CARI | no family above 13% | stays |
| Gyaru | 19 | Glam, sexy and party | tags Sexy, Nightlife and Parties; bold makeup, animal print, mini skirts | spread over the families of its own sub-styles | stays (page supports the table) |
| Lulicore | 11 | Classic, preppy and office | a colour trend (brown with pastel accents); names no style family | Romantic 25%, Casual 23%, Classic 15% | stays (no single family indicated) |
| Starrflesh | 19 | Fantasy, sci-fi and costume | surreal body imagery, tagged Creepy | Fantasy 33%, Goth 9% | stays (photos support the table) |
| LOLcore | 11 | Glam, sexy and party | cartoon and fast-food prints, humour | Glam 18%, Cute 17%, Club 14% | stays (no single family indicated) |
| Avantropop | 23 | Club, rave and dance scene | a TV-ident graphic style; fashion note lists neon electropop streetwear | Club 20% is the largest | stays |
| Tropical | 33 | Bohemian, hippie and folk | tropical regions, beachwear; its named kin Tropicabana and Tropicalcore are in the same family | Casual 38% (base 6%), Bohemian 15% (base 2%): about equally enriched | stays |

The other 12 parent-flagged sub-styles stay: their photos sit with the family the table gave them.

`label_family.csv` sha256 is now `9e77ed475033277c4494c5c5e0c09e32bcb0b4e71355f5edb14152bb66acd2d9`. The table before this amendment: `label_family.before-amendment-3.csv`
(sha256 f6394f3b533f...). Redone because of it: the family figures of every validation row and the NN with real outfit photos.

## Amendment 4, 2026-11-21 06:16 UTC: Amekaji moved from Casual and everyday to Street, hip-hop and sporty

Decided by Andrew ("Yes, Street"). This is the one case Amendment 3 left for him, because its page and its photos disagree, so
rule 3 could not settle it.
- The page's own words for the style are an exaggerated take on American West Coast streetwear drawn from surf, skate and
  hip-hop; its tags include Streetwear and Hip-Hop.
- Its 7 training photos have no neighbours in Casual and everyday (0%, base 6%). Their neighbours are other Gyaru sub-styles,
  which the table spreads over Cute, Glam and Rock by look, so the photos do not name one family; Street has 9% (base 6%).
- Moved on the page's wording. 7 training images.

`label_family.csv` sha256 is now `fc9d825bd99b6cf2bab8b9613051d6d2a5b4ff817681eca1aab3da353b7cce2d`. The table before this amendment: `label_family.before-amendment-4.csv`
(sha256 9e77ed475033...). The retraining started for Amendment 3 was stopped before it scored anything and was run once, on
this table.

## Amendment 5, 2026-11-21 07:35 UTC: three broad street tags are no longer used, and street photos are drawn tag by tag

Decided by Andrew ("Yes, adopt both") after the tests in `runs/tag-iter` (WORKLOG, 21 Nov 07:07 to 07:32 UTC).

1. **`outside_family.csv`:** the Street Fashion Style tags **Elegant** (was Elegant and dressy), **Urban** (was Street, hip-hop
   and sporty) and **Androgynous** (was Modern, minimal and avant-garde) now carry `broad` in the family column, which is
   not a family, so their photos are not trained on or scored (the same way `vague` works for Chic, Trendy and Eclectic).
   30 tags remain in use. Their families keep their narrower tags: Formal; Sports and Sneaker; Menswear and Runway.
   Before: `outside_family.before-amendment-5.csv`.
2. **`code/nn/train_nn.py`:** a street photo is drawn as family -> one of the family's tags -> one of the tag's photos
   (setting `real_draw: tag`). It was family -> photo, which let a family's largest tag supply most of its photos.
   `--real-draw photo` gives the earlier behaviour exactly.

**Why.** Retraining with each of the 33 tags left out in turn showed that a narrow tag teaches its family and a broad one
on the same family dilutes it: leaving out Formal lowers FashionStyle14's "dressy" by 5.6 points while leaving out Elegant
raises it by 13.7; Sports -9.2 against Urban +8.9 for "street". Drawing tags evenly is the same idea as one rule.

**Evidence.** Chosen on the even-sha1 half of FashionStyle14's 11 non-transfer styles; checked once on the odd-sha1 half,
which nothing had read, with the comparison written down first: family accuracy 51.2% -> 54.5%, **+3.3 points (95%
interval +2.6 to +3.9)**, 10 seeds, paired bootstrap over photos. Wiki top-5 87.9 -> 87.8, wiki family top-1 75.3 -> 75.5.

**To be stated in the report.** FashionStyle14 was a check when the real photos were first added (Amendment 1's figures).
From this amendment on it is also what the tag mapping and the draw were chosen on, so it is no longer an untouched
check; the transfer-test styles (lolita, fairy, ethnic) and the wiki test split still are. The three tags were picked from
33 leave-one-out results. GPU runs on the PC chose the change; the model of record is retrained on ml-db's CPU.

`outside_family.csv` sha256 is now `4270d503b4061225e47e6044c906d31acb2f503ee09e5bf0b23910a581ca594a`. `label_family.csv` is unchanged.
