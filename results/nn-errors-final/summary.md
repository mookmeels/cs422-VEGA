# NN error analysis, validation split (1351 images, 503 labels, label set "merged")

Network: nn-mlp1x1024real, seeds 422, 423, 424, 425, 426; image vectors emb/qwen2beos-aesthetic-448. Figures are averages over the seeds unless stated. The test split is not read.

## 1. How many images are missed, and is it the same images every time?

|  | top-5 | top-1 |
|---|---|---|
| right, average over seeds | 87.9 | 62.3 |
| right in all 5 seeds | 84.2 | 54.3 |
| right in some seeds, wrong in others | 6.7 | 16.8 |
| wrong in all 5 seeds | 9.0 | 28.9 |

122 images never have a true label in the top 5; 91 depend on the seed.

## 2. When the top 5 misses, where is the true label?

| position of the best true label | share of the missed images |
|---|---|
| 6 to 10 | 41.5 |
| 11 to 20 | 28.2 |
| 21 to 50 | 19.8 |
| 51 to 503 | 10.6 |

Top-10 accuracy 92.9, top-20 96.3.

## 3. Does it depend on how many training images a label has?

Per label: the share of its validation images where that label is in the top 5 (or is the first guess), averaged over the labels in the row.

| training images of the label | labels | validation images | NN top-5 | NN first guess | nearest centroid top-5 | nearest centroid first guess |
|---|---|---|---|---|---|---|
| 4 to 7 | 17 | 18 | 74.1 | 50.6 | 64.7 | 58.8 |
| 8 to 15 | 255 | 430 | 85.2 | 58.8 | 84.7 | 57.9 |
| 16 to 31 | 172 | 519 | 87.3 | 60.0 | 83.0 | 54.0 |
| 32 to 63 | 45 | 271 | 90.0 | 64.6 | 78.5 | 47.7 |
| 64 or more | 12 | 157 | 91.1 | 60.5 | 73.0 | 38.4 |

Images with one label (1310): top-5 87.6. Images with two or more labels (41): top-5 96.6 (any one of their labels counts).

## 4. What are the wrong first guesses?

| the first guess is... | share of the wrong first guesses | share of all label pairs (chance) |
|---|---|---|
| in the same wiki family as a true label, or its parent / sub-style | 14.8 | 1.6 |
| linked more loosely by the wiki (related, overlaps, came before / after) | 26.0 | 1.4 |
| not linked by the wiki | 59.2 | 97.0 |

When the first guess is wrong, a true label is still in the top 5 for 67.8 of those images.

Most frequent mistakes (true label -> first guess). "Images per seed" is the average number of validation images per seed.

| true label | first guess | images per seed | validation images of the true label | wiki link |
|---|---|---|---|---|
| Kawaii | Babycore | 2.2 | 12 | family / sub-style |
| Nymphet | Cherry Emoji Twitter | 2.0 | 3 | related |
| UrBling | Post-Grunge Maximalism | 2.0 | 5 | related |
| Cinnamon Girl | Vanilla Girl | 2.0 | 3 | related |
| Rococo | Spanish Golden Age | 2.0 | 16 | related |
| Dunhuang Feng | Shenxian-fu | 2.0 | 3 | no wiki link |
| Edwardian | Victorian | 1.8 | 3 | related |
| Dark Cabaret | Burlesque | 1.8 | 2 | related |
| Visual Kei | Hair Metal | 1.8 | 8 | related |
| Mafia Aesthetic | Hellenic | 1.6 | 4 | related |
| Rococo | Victorian | 1.6 | 16 | no wiki link |
| Nu-Metal | Corporate Grunge | 1.4 | 4 | related |
| Paty de SP | Milipili | 1.4 | 4 | related |
| Victorian | Rococo | 1.3 | 8 | no wiki link |
| Soft Grunge | Sadgirl | 1.2 | 12 | family / sub-style |
| Goth | Horror Punk | 1.2 | 6 | family / sub-style |
| Victorian | Royalcore | 1.2 | 8 | no wiki link |
| Mermaidcore | Ethereal | 1.2 | 3 | family / sub-style |
| Bubblegum Bitch | McBling | 1.2 | 6 | family / sub-style |
| Coquette | Morute | 1.2 | 15 | family / sub-style |
| Soft Grunge | Hipster | 1.2 | 12 | no wiki link |
| Dreamy Indie Girlhood | Cottagecore | 1.2 | 3 | no wiki link |
| Candycore | Kawaii | 1.2 | 3 | related |
| Space Age | Raygun Gothic | 1.0 | 2 | related |
| Indie Kid | Candycore | 1.0 | 2 | no wiki link |

## 5. Which labels does it find worst?

15 of the 501 labels with validation images are never in the top 5 for any of their images in any seed (10 of them have a single validation image); 317 are always found.

Lowest top-5 rate among labels with at least 3 validation images:

| label | training images | validation images | NN top-5 | NN first guess | nearest centroid top-5 | most often mistaken for |
|---|---|---|---|---|---|---|
| Craftcore | 19 | 3 | 33.3 | 20.0 | 66.7 | Clowncore |
| Cyber Ghetto | 23 | 3 | 33.3 | 20.0 | 0.0 | Neo-Y2K |
| Jirai Kei | 22 | 3 | 33.3 | 0.0 | 33.3 | Olderbrothercore |
| Pink Princess | 26 | 4 | 35.0 | 25.0 | 50.0 | Coquette |
| Mermaidcore | 18 | 3 | 40.0 | 33.3 | 66.7 | Ethereal |
| Bubblegum Dance | 19 | 3 | 46.7 | 0.0 | 0.0 | Millennium Orientalism |
| Wonderland | 24 | 4 | 50.0 | 50.0 | 50.0 | Rococo |
| Nymphet | 21 | 3 | 53.3 | 0.0 | 66.7 | Cherry Emoji Twitter |
| Uchuu Kei | 22 | 3 | 53.3 | 0.0 | 66.7 | Fairy Kei |
| Goth | 45 | 6 | 56.7 | 50.0 | 66.7 | Horror Punk |
| Cottagecore | 53 | 8 | 62.5 | 27.5 | 62.5 | Wartime Austerity |
| Gloomcore | 33 | 7 | 62.9 | 57.1 | 71.4 | Darkest Academia |
| Werewolf | 29 | 4 | 65.0 | 25.0 | 75.0 | Ghostcore |
| 2014 Girly | 18 | 3 | 66.7 | 66.7 | 66.7 | Soft Girl |
| Avantropop | 23 | 3 | 66.7 | 40.0 | 66.7 | Flat Design |
| Baddie | 10 | 3 | 66.7 | 33.3 | 66.7 | Blokette |
| Barbiecore | 19 | 3 | 66.7 | 33.3 | 66.7 | Granola |
| Bubbleglam | 18 | 3 | 66.7 | 33.3 | 100.0 | New Romantic |
| Draincore | 23 | 3 | 66.7 | 33.3 | 33.3 | Hyperpop |
| E-Boy | 13 | 3 | 66.7 | 33.3 | 66.7 | Roadman |

Labels most often given as a WRONG first guess (the labels that soak up mistakes):

| label | wrong first guesses per seed | training images | its own top-5 rate |
|---|---|---|---|
| Tropical | 6.4 | 33 | 100.0 |
| Coquette | 6.0 | 103 | 81.3 |
| Kawaii | 6.0 | 75 | 96.7 |
| Dark Academia | 5.6 | 55 | 75.0 |
| Groovival | 5.4 | 41 | 100.0 |
| Blokette | 5.0 | 10 | 90.0 |
| Trailer Park Princess | 5.0 | 52 | 100.0 |
| McBling | 5.0 | 41 | 100.0 |
| Babycore | 4.8 | 18 | 80.0 |
| Superflat Pop | 4.8 | 91 | 93.8 |
| Gothabilly | 4.6 | 38 | 80.0 |
| Victorian | 4.6 | 48 | 87.5 |
| Soft Girl | 4.4 | 17 | 50.0 |
| Post-Grunge Maximalism | 4.2 | 28 | 100.0 |
| Cleancore | 4.0 | 77 | 100.0 |

The 50 labels with the most training images (a tenth of the labels) hold 28.7 of the training label assignments and receive 27.8 of the wrong first guesses.

## 6. Can the network's confidence be trusted?

Confidence = the softmax share the network gives its first guess.

| confidence | share of images | average confidence | first guess right | true label in top 5 |
|---|---|---|---|---|
| 0.0 to 0.2 | 3.0 | 0.17 | 0.18 | 0.56 |
| 0.2 to 0.4 | 19.2 | 0.31 | 0.31 | 0.75 |
| 0.4 to 0.6 | 22.0 | 0.50 | 0.49 | 0.84 |
| 0.6 to 0.8 | 17.9 | 0.70 | 0.66 | 0.93 |
| 0.8 to 0.9 | 11.2 | 0.85 | 0.80 | 0.94 |
| 0.9 to 1.0 | 26.7 | 0.96 | 0.91 | 0.98 |

Average confidence 0.65 against 0.62 of first guesses right (gap +0.03; expected calibration error 0.033). Average confidence when right 0.75, when wrong 0.49.

## 7. Image by image against nearest centroid

| images (average per seed) | top-5 | first guess |
|---|---|---|
| both right | 1068 | 646 |
| only the NN right | 119 | 195 |
| only nearest centroid right | 40 | 77 |
| both wrong | 124 | 433 |

## 8. The 5 networks averaged

Averaging the 5 networks' scores per image gives top-5 88.4 and top-1 63.0 (single networks: 87.9 and 62.3). This is not a row in the comparison table.

## Example images

`examples-confident-and-wrong.png`, `examples-far-off.png`, `examples-near-miss.png`: 12 validation images each. Images whose true or guessed labels the wiki tags as sensitive are left out (116 labels).
