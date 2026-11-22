# Who is served worse: family accuracy on real outfit photos by style, poster and region

Made 2026-11-22 08:54 UTC by code/experiments/fairness_spread.py. NN with real outfit photos, frozen settings, seeds [422, 423, 424], trained on the wiki training images and the street photos of poster groups 1 to 9. Nothing tuned; test split not read.
Street Fashion Style, held-out posters (group 0): 18,610 photos by 519 posters, family right for 34.1%. FashionStyle14, 11 styles: 10,312 photos, family right for 54.1%.

## 1. By style (FashionStyle14, expert labels)

| style | family it maps to | photos | family right |
|---|---|---|---|
| feminine | Romantic, feminine and girly | 812 | 85.5% |
| dressy | Elegant and dressy | 890 | 83.9% |
| conservative | Classic, preppy and office | 923 | 76.5% |
| rock | Rock, punk, grunge and emo | 803 | 66.6% |
| street | Street, hip-hop and sporty | 1,022 | 65.7% |
| retro | Vintage and retro | 841 | 59.3% |
| girlish | Romantic, feminine and girly | 1,103 | 44.6% |
| kireime-casual | Casual and everyday | 1,050 | 39.9% |
| natural | Casual and everyday | 854 | 33.5% |
| gal | Glam, sexy and party | 953 | 29.0% |
| mode | Modern, minimal and avant-garde | 1,061 | 23.7% |

Best style 85.5%, worst 23.7%: a gap of 62 points.

## 2. By family of the poster's own tag (Street Fashion Style, held-out posters)

| family | photos | share of photos | family right |
|---|---|---|---|
| Casual and everyday | 8,651 | 46.5% | 27.4% |
| Vintage and retro | 2,457 | 13.2% | 27.3% |
| Classic, preppy and office | 2,299 | 12.4% | 37.8% |
| Bohemian, hippie and folk | 1,322 | 7.1% | 45.8% |
| Romantic, feminine and girly | 1,304 | 7.0% | 47.1% |
| Rock, punk, grunge and emo | 1,007 | 5.4% | 51.0% |
| Natural, country and outdoors | 503 | 2.7% | 28.9% |
| Glam, sexy and party | 404 | 2.2% | 50.8% |
| Modern, minimal and avant-garde | 239 | 1.3% | 61.5% |
| Street, hip-hop and sporty | 219 | 1.2% | 60.6% |
| Elegant and dressy | 133 | 0.7% | 28.6% |
| Goth and dark | 72 | 0.4% | 47.7% |

## 3. By poster (posters with 20 or more held-out photos)

- 196 posters. Family right per poster: 10th percentile 18.5%, median 33.8%, 90th percentile 46.8%.
- 5 posters are under 10% and 12 are over 50%.
- Spread between posters (standard deviation) 11.5 points. If only each poster's mix of tags and chance mattered it would be 7.9 points (1,000 simulations). After taking out what the tag mix predicts, 11.3 points remain.

## 4. By region of the location the poster typed

Raw, and with every region given the same mix of tag families as all held-out photos (families with under 20 photos in a region are left out of that region's adjusted figure, and the weights rescaled). Intervals are 95%, resampling posters.

| region | posters | photos | family right | adjusted for tag mix (95% interval) |
|---|---|---|---|---|
| United States and Canada | 185 | 8,241 | 36.6% | 36.2% (34.7% to 37.7%) |
| Western and Northern Europe | 130 | 5,741 | 34.3% | 34.0% (31.4% to 35.7%) |
| East and Southeast Asia | 57 | 1,699 | 29.2% | 28.5% (23.6% to 32.6%) |
| Eastern Europe and Central Asia | 41 | 1,435 | 27.9% | 28.1% (24.9% to 32.2%) |
| not matched | 72 | 864 | 30.8% | 29.6% (26.0% to 33.4%) |
| Latin America | 7 | 311 | 32.5% | 28.7% (25.4% to 32.9%) |
| Africa and Oceania | 20 | 262 | 28.8% | 27.1% (13.4% to 39.5%) |
| South Asia and Middle East | 7 | 57 | 32.2% | too few photos |

Difference from United States and Canada, adjusted for tag mix (95% interval, resampling posters):

- Western and Northern Europe: -2.2 points (-5.2 to +0.1)
- East and Southeast Asia: -7.8 points (-12.7 to -3.3)
- Eastern Europe and Central Asia: -8.2 points (-11.6 to -3.8)
- Latin America: -7.6 points (-10.7 to -2.8)

Where the street photos the network trains on come from (149,686 photos of the other nine poster groups): United States and Canada 47.8%; Western and Northern Europe 19.1%; not matched 12.2%; East and Southeast Asia 11.5%; Eastern Europe and Central Asia 5.5%; Latin America 1.8%; Africa and Oceania 1.7%; South Asia and Middle East 0.5%.

Locations not matched to a region (72 posters), most common: Riga, Riga (2); Lima, Lima (2); Brisbane, Queensland (2); Budapest, Budapest (2); CYPRUS (1); PA (1); Naga, Camarines Sur (1); Braè‘¯ov, Judeè‘²ul Braè‘¯ov (1); Frankfurt, Hessen (1); Giroc, TimiÃˆâ„¢ County (1); Sunny, PENANG ISLAND (1); Napoli (1).

Time elapsed: 51 s
