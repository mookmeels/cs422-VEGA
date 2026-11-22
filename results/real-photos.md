# Real photos in training: results (validation data only; rules in RULES.md)

Wiki: 9,359 training images, 1,351 validation images (721 show an outfit). Street Fashion Style: 166,551 training photos in 12 of the 20 families; 21,024 photos of 530 held-out posters. FashionStyle14: 10,312 photos of 11 styles, never trained on. Seeds 422 to 426, mean ± spread.

| model | wiki aesthetic top-1 | wiki aesthetic top-5 | wiki family top-1 | FashionStyle14 family top-1 | the same, each style equal | held-out posters family top-1 |
|---|---|---|---|---|---|---|
| weight 0 (the NN as it is) | 62.7 ± 0.4 | 88.3 ± 0.4 | 75.9 ± 0.6 | 33.1 ± 0.5 | 33.2 | 23.7 ± 0.4 |
| weight 0.25 | 62.3 ± 0.2 | 87.9 ± 0.4 | 75.5 ± 0.4 | 49.4 ± 0.8 | 50.6 | 31.6 ± 0.3 |
| weight 1.0 | 61.3 ± 0.4 | 87.0 ± 0.3 | 74.3 ± 0.6 | 52.1 ± 0.6 | 53.5 | 33.0 ± 0.2 |

## Differences from weight 0 (paired bootstrap, 2,000 draws, seed 422; 95% interval)

| model | wiki aesthetic top-5 | wiki aesthetic top-1 | wiki family top-1 | FashionStyle14 family top-1 | held-out posters family top-1 |
|---|---|---|---|---|---|
| weight 0.25 | -0.4 (-0.9 to +0.1) | -0.4 (-1.3 to +0.4) | -0.3 (-1.2 to +0.6) | +16.3 (+15.4 to +17.1) | +7.9 (+5.9 to +9.8) |
| weight 1.0 | -1.3 (-2.0 to -0.7) | -1.5 (-2.5 to -0.4) | -1.5 (-2.7 to -0.5) | +19.0 (+18.0 to +20.0) | +9.3 (+7.2 to +11.3) |

## Wiki validation by kind of image, and the shortcut check

189 validation images show an outfit and belong only to families that have no real photos (cute, lolita, historical, fantasy, club, regional, seasons, design). If real photos teach the network that "a photographed outfit means one of the 12 covered families", these images are where it shows.

| model | outfit images: family top-1 | other images: family top-1 | uncovered-family outfit images: family top-1 | their first guess is a covered family |
|---|---|---|---|---|
| weight 0 (the NN as it is) | 75.4 ± 0.4 | 76.4 ± 0.8 | 71.6 ± 1.5 | 21.9 ± 1.6 |
| weight 0.25 | 74.0 ± 0.6 | 77.3 ± 0.5 | 69.9 ± 1.4 | 24.1 ± 1.2 |
| weight 1.0 | 71.8 ± 0.7 | 77.2 ± 0.7 | 67.1 ± 0.5 | 28.0 ± 1.0 |
| weight 0.25 minus weight 0 | -1.4 (-2.8 to -0.0) | +1.0 (+0.0 to +1.9) | -1.7 (-3.8 to +0.1) | +2.2 (+0.5 to +4.2) |
| weight 1.0 minus weight 0 | -3.6 (-5.5 to -1.7) | +0.8 (-0.3 to +1.8) | -4.6 (-7.7 to -1.9) | +6.1 (+3.5 to +9.1) |

## How the rules read it (rule 7)

- **weight 0.25:** helps on real photos: yes; costs on wiki: no. helps on real photos without a cost on wiki: a candidate for adoption.
- **weight 1.0:** helps on real photos: yes; costs on wiki: yes. helps on real photos, with a cost on wiki: a trade-off.

## FashionStyle14 by style (family top-1, mean of 5 seeds)

| style | family | photos | weight 0 (the NN as it is) | weight 0.25 | weight 1.0 |
|---|---|---|---|---|---|
| conservative | Classic, preppy and office | 923 | 29.1 | 66.4 | 76.1 |
| dressy | Elegant and dressy | 890 | 5.9 | 71.5 | 84.5 |
| feminine | Romantic, feminine and girly | 812 | 82.6 | 84.6 | 80.3 |
| gal | Glam, sexy and party | 953 | 17.7 | 27.6 | 31.4 |
| girlish | Romantic, feminine and girly | 1103 | 57.3 | 47.4 | 40.6 |
| kireime-casual | Casual and everyday | 1050 | 32.4 | 31.6 | 31.4 |
| mode | Modern, minimal and avant-garde | 1061 | 29.8 | 22.8 | 22.5 |
| natural | Natural, country and outdoors | 854 | 11.0 | 16.3 | 15.0 |
| retro | Vintage and retro | 841 | 31.6 | 62.1 | 67.7 |
| rock | Rock, punk, grunge and emo | 803 | 36.8 | 67.7 | 72.8 |
| street | Street, hip-hop and sporty | 1022 | 30.5 | 58.1 | 65.9 |

## Held-out Street Fashion Style posters by family (family top-1, mean of 5 seeds)

| family | photos | weight 0 (the NN as it is) | weight 0.25 | weight 1.0 |
|---|---|---|---|---|
| Rock, punk, grunge and emo | 1007 | 14.5 | 49.7 | 52.5 |
| Classic, preppy and office | 2299 | 36.0 | 31.4 | 33.5 |
| Casual and everyday | 8651 | 31.3 | 25.7 | 26.3 |
| Romantic, feminine and girly | 1304 | 30.4 | 43.2 | 44.9 |
| Vintage and retro | 2457 | 7.5 | 29.0 | 31.9 |
| Street, hip-hop and sporty | 1346 | 6.6 | 23.4 | 23.7 |
| Natural, country and outdoors | 503 | 11.2 | 28.0 | 27.9 |
| Goth and dark | 72 | 36.4 | 46.4 | 40.8 |
| Modern, minimal and avant-garde | 478 | 21.7 | 41.2 | 44.1 |
| Glam, sexy and party | 404 | 13.5 | 51.4 | 55.1 |
| Bohemian, hippie and folk | 1322 | 22.5 | 46.7 | 48.6 |
| Elegant and dressy | 1181 | 7.5 | 33.6 | 35.9 |

Self-check (weight 0 against the team table's saved predictions): seed 422: largest difference from the team table's saved scores 1.91e-06; the same five first guesses for every image: True; seed 423: largest difference from the team table's saved scores 2.86e-06; the same five first guesses for every image: True; seed 424: largest difference from the team table's saved scores 2.86e-06; the same five first guesses for every image: True; seed 425: largest difference from the team table's saved scores 1.91e-06; the same five first guesses for every image: True; seed 426: largest difference from the team table's saved scores 1.91e-06; the same five first guesses for every image: True

Time elapsed: 1363 s. torch 2.15.0+cpu, one CPU thread.
