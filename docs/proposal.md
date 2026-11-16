# **VEGA: CS422 Project Topic Proposal**

## **1\. Real-World Problems**

An internet aesthetic is a recognizable visual style that bundles clothing, home decor, objects, music, and hobbies into one identity, such as cottagecore, dark academia, gorpcore, Y2K, or coquette. Oftentimes labelled with confusing or eclectic names, the internet aesthetic often serves as one of the main ways for a younger person to express their identity, and it will be a major factor in deciding what to wear, buy, and decorate with. Aesthetics, as an extension of the person’s identity, will often color items and things beyond clothes: a bedroom’s decoration, the way someone may communicate themselves online, the music they listen to, and the outfit they wear are all oftentimes downstream of the aesthetic a younger person cultivates for themselves.  Many aesthetics are named with the "-core" suffix, and the community-run Aesthetics Wiki has documented more than a thousand of them; the interest in aesthetics grew sharply during the COVID-19 pandemic as folks spent more time online looking for ways to express themselves. The labels for these aesthetics are fuzzy, overlapping, and are oftentimes ephemeral in nature, so it is hard for someone to own their own style or find related ones.

Commercial style recommenders mostly look at clothing alone and optimize for purchases, which pushes users toward more of what they already like. That narrows discovery and favors buying new over thrifting, DIY, or reusing what people own. We want to test whether machine learning can describe a person's aesthetic from their broader visual life, and point them to both nearby and unexpected aesthetics worth exploring.

There are many different problems that plague the ability to identify or utilize aesthetics well. With a number in the thousands of these overlapping, fast-changing labels, it will oftentimes be perilously hard for a person to find the terms that suit their own style and character \- and even harder for them to find the styles which they could also appreciate. For the younger folks, this exploration is done by scrolling through social media and taking guesses.

Secondly, algorithms which serve aesthetics packaged through social media posts, related shopping items, and other methods are tuned for total accuracy and repetition \- serving more of what a user would already like and appreciate. These recommendations serve to reinforce the same exact style to the user, differing only through content. This may be considered “boring” to any user \- and research shows that *serendipity* \- suggestions which may be both unexpected and relevant at the same time \- can be a better goal for helping people to discover new interests.

Thirdly, secondhand shopping \- “thrifting” is growing fast, and younger folks spearhead this change. ThredUp's 2025 Resale Report found that 48% of shoppers aged 18-44 look secondhand first, and that 59% of that group said better personalization, search, and discovery would make secondhand as easy as buying new. The issue is that thrift and resale items rarely have clean brand or product data, so describing and searching by aesthetic is a natural fit for this market.

By using machine learning, we may be able to help differentiate the different fuzzy, overlapping aesthetics which are hard to capture with rules that are handwritten and may not be universally agreed upon. Learned image representations can solve this issue. Because aesthetics blend into each other, it also makes sense to model them as regions in a vector embedding space rather than separate, clean boxes \- which allows for the nearest neighbor searching for aesthetics.

The research question, then, asks: *can a person's wider visual context (outfit plus room, car, and favorite objects) predict their aesthetic better than their outfit alone? And can an embedding-based map of aesthetics produce suggestions that are both relevant and genuinely surprising?*

## **2\. Intended Users**

We have multiple different kinds of users who would benefit from this platform.

1\. People exploring their style or identity, mainly teens and young adults, want words for their taste and ideas for where to go next without being boxed in. We plan to give them a ranked, aesthetic profile, nearby and serendipitous suggestions, and an explanation behind which photo drove each one.

2\. Thrift and resale shoppers need to search by vibe, because secondhand items rarely carry reliable brand or product data. Aesthetic labels can serve as search terms or filters.

3\. Resale platforms and independent sellers need better tagging and discovery for one-of-a-kind items. Our model can tag listing photos with aesthetics, and the aesthetic map can power "you might also like" suggestions.

4\. Stylists and small creators want to reach people in niche aesthetics. VEGA gives them a shared vocabulary and a map of how aesthetics relate to each other.

5\. Indirect stakeholders: the people pictured in public datasets, whose images are used to train the model, and the online communities that create and name these aesthetics. Our responsible-use plan (Section 6\) addresses both.

## **3\. Machine-learning objective and target outcome**

Given 3-5 images from one person (an outfit plus personal effects such as a room, car interior, or favorite objects), the system will:

1. **Classify** the image set with every aesthetic in our catalog (several hundred, see below) ranked by how well it fits the person, with scores and only the top 5 shown.  
2. **Recommend** the nearest related aesthetics, using distances between aesthetics in a shared embedding space.  
3. **Surprise** the user with "**serendipitous**" aesthetics: ones far from their top prediction but still strongly matched by at least one of their images.  
4. **Explain** the photo for each suggestion it contributed most to.

The value of this system is in the long tail of the data, where we have less represented and more eclectic, loose labels. The useful answer to a user and for our system is a niche aesthetic they have never heard of \- so we use every Aesthetics Wiki page that describes something visible, and drop only what a photo cannot show\!

If an aesthetic has a written description of visual traits such as clothing, decor, colors, and objects \- *and* a gallery of example images, we keep it\! If the aesthetic were to only be defined by music, a mood, a fandom of a video game, or a single meme blown way out of proportion, we drop it or merge it into another aesthetic.

After the pruning, we should still expect several hundred aesthetics to remain; the exact count will be known after scraping. Each aesthetic is represented by an aesthetic vector: the average CLIP embedding of its page description and its gallery images. Because no per-aesthetic hand labels are needed, the catalog's size is not limited by labeling effort, and a new aesthetic can be added just by adding its page.

We can also group the catalog into about 20-30 broad families (for example academia, nature-core, retro-digital, alternative) by clustering aesthetic vectors and naming the clusters by hand. Families are used to organize the map, break down results, and run the fairness check\!

1. Embed: each image goes through a pretrained CLIP image encoder (ViT-B/32), giving a 512-dimensional vector.  
2. User style profile: instead of judging each photo on its own, we build one profile vector per user. Each image's embedding is weighted by image type (outfit vs. context) and by how strongly it matches any aesthetic, so one outfit photo is not drowned out by four room photos. The weighted embeddings are pooled into a single profile, and per-image scores are kept for explanations.  
3. Zero-shot ranking: every aesthetic in the catalog is scored by cosine similarity between the user profile and its aesthetic vector. This ranking uses no training and is our main baseline.  
4. Learned re-ranking: each member's algorithm (Section 7\) takes the top 50 candidates from step 3 and re-scores them, using features of each (user, aesthetic) pair: similarity to the profile, the best similarity to any single photo, similarity by image type, the aesthetic's family, and the embeddings themselves. The re-rankers are trained on personas built from labeled wiki gallery images (Section 4).  
5. Aesthetic map: aesthetic vectors are clustered and laid out so that related aesthetics sit close together. Nearest and surprising suggestions are both computed from this map.  
6. Demo (stretch goal): a simple web app (Streamlit or Gradio) where a user uploads photos and sees their profile, suggestions, and explanations.

### **An example\!**

A user uploads four photos:

| Photo | Content | Strongest aesthetic match |
| ----- | ----- | ----- |
| 1 (outfit) | Flannel overshirt, cargo pants, hiking boots | Gorpcore 0.71 |
| 2 (room) | Plants, wooden desk, string lights, quilt | Cottagecore 0.64 |
| 3 (car) | Carabiner on the keys, national park sticker, trail map | Gorpcore 0.58 |
| 4 (object) | Shelf of small plush toys and figurines | Kawaii 0.66 |

* **Profile (top 5):** gorpcore, cottagecore, granola, goblincore, grunge. Goblincore (moss, mushrooms, found objects) is a niche aesthetic that a system limited to a short list of popular labels would never show.  
* **Nearest aesthetics:** naturecore, normcore, and other aesthetics near the gorpcore-cottagecore region of the map.  
* **Surprising aesthetic:** kawaii. It is far from gorpcore on the map, so it would never appear in a standard "more like this" list, but photo 4 matches it strongly. The app explains: *"suggested because of your plush shelf."*  
* **Outfit only:** from photo 1 alone, the system would return gorpcore and its close neighbors. The context photos add the cottagecore side and the kawaii surprise. This difference is what our main experiment measures.

### **Evaluation plan and metric definitions**

Why do we judge suggestions instead of assigning labels? Well, asking people to pick the correct aesthetics for an image out of several hundred is a patently unreliable method. Asking whether a given suggestion fits a set of photos is easy and consistent \- so human evaluation should be built around yes/no judgments of suggestions. We hold out part of each wiki gallery and build test personas from those held-out images, so we know which aesthetic(s) each persona came from.

* **Recall@5:** the share of personas whose source aesthetic appears in the top 5\.  
* **Mean reciprocal rank (MRR):** how high the source aesthetic is ranked, on average.

About 60 test personas are built from DeepFashion, Places365, and Open Images photos, which are realistic images not taken from the wiki. For each persona, human reviewers should pool the top 5 aesthetics from every system and setting, shuffle them, and hide which system produced each one. Each team member then rates every pooled aesthetic as **fits** or **doesn't fit**, using the photos and the aesthetic's wiki description.

* **Precision@5:** the share of a system's top 5 rated as fitting.  
* **nDCG@5:** like precision, but fitting aesthetics ranked higher count more.  
* **Rater agreement (Fleiss' kappa):** how often we agree with each other, which is a realistic ceiling for any system.

Every re-ranker should be compared against (a) a popularity ranking, which always suggests the aesthetics with the largest wiki galleries, and (b) the zero-shot CLIP ranking with no learning. Beating (b) shows that our trained models add something beyond the pretrained model.

Context-wise, each system is run three ways: outfit photo only, all photos averaged equally, and our weighted user profile. If personal context helps, the weighted profile should score higher than outfit-only\!

And for serendipity**,** following recommender-systems research that defines serendipity as relevance combined with unexpectedness, for a suggested aesthetic *a*:

* **Relevance:** rel(a) \= the highest similarity between *a* and any one of the user's photos (scaled to 0–1). A suggestion is relevant if at least one photo really matches it.  
* **Unexpectedness:** unexp(a) \= 1 − similarity between *a* and the user's top-ranked aesthetic. A suggestion is unexpected if it sits far from what the user already matches best.  
* **Serendipity score:** S(a) \= rel(a) × unexp(a). The surprising suggestions are those with the highest S(a) that pass minimum thresholds on both parts, tuned on validation personas.  
* **Human check:** surprising suggestions are added to the rating pool. We report the share rated as fitting, against a baseline of random aesthetics that are equally unexpected but not chosen for relevance.

### **Target outcomes**

* Learned re-rankers beat both baselines on precision@5 and MRR.  
* The weighted user profile beats outfit-only.  
* Suggestions cover a large share of the catalog, not just its most popular pages.  
* More than half of the surprising suggestions are rated as fitting, compared with a much lower rate for random ones.

## **4\. Proposed dataset and source**

| Source | Approx. size | What it provides | Role in project |
| ----- | ----- | ----- | ----- |
| Aesthetics Wiki | 1,000+ pages; several hundred visual aesthetics after filtering | Descriptions of each aesthetic and galleries of example images | The aesthetic catalog, aesthetic vectors, and labeled training images |
| DeepFashion (CUHK) | 800K+ images | Clothing and full-outfit photos | Outfit images for test personas |
| Fashionpedia | \~48K images | Full-body fashion photos in varied real-world settings | Outfit images for test personas |
| Places365 | \~1.8M images, 365 scene types | Bedrooms, living rooms, dorm rooms, car interiors | Room and car images for test personas |
| Open Images (subset) | Filtered to plush toys, books, figurines, plants, etc. | Everyday object photos | "Favorite object" images for test personas |

We will sample a manageable subset of the large datasets (a few thousand images each) rather than use them in full. Wiki pages and gallery images are used only for research and are not redistributed.

### **Labels without hand-labeling hundreds of classes**

* **Training labels come from the wiki itself.** Every gallery image on an aesthetic's page is a labeled example of that aesthetic, and images that appear on several pages get several labels. This gives labeled data for every aesthetic in the catalog with no hand labeling.  
* **Held-out wiki images** (about 20% of each gallery) are kept out of the aesthetic vectors and out of training, and are used for the automatic evaluation (recall@5, MRR).  
* **Human judgments** on realistic test personas are the main evaluation. The team rates pooled suggestions as fits or doesn't fit (Section 3), which stays manageable no matter how large the catalog is.

### **Building user profiles**

No public dataset contains sets of photos from one person, so we’ll have to build and construct synthetic personas. Each persona combines one outfit image with 2 \- 4 context images.

* Training and validation personas are built from wiki gallery images. Most of a persona's images come from one aesthetic, and some deliberately come from another, to reflect that real people mix styles. The persona's labels are the aesthetics its images came from. We plan about 2,000 training personas.  
* Test personas (about 60\) are built from DeepFashion, Fashionpedia, Places365, and Open Images. These photos come from outside the wiki, so the test checks whether the system works on realistic photos rather than curated examples. They have no labels and are evaluated only by human judgment.

### **Features and feasibility**

Every image and every aesthetic description is converted once into a 512-dimensional CLIP embedding, and all algorithms train on these same features. Embedding tens of thousands of images takes well under an hour on one free GPU (for example, Google Colab). Because the re-rankers only score the top 50 candidates per user, training and inference run in minutes on a laptop, even with several hundred aesthetics in the catalog.

## **5\. How this differs from existing work**

Well, here’s the existing work out there:

1. **Garment classification** is the standard tutorial task (Fashion-MNIST, DeepFashion categories): label one item as a shirt, dress, or sneaker. It says nothing about style or the person.  
2. **Clothing style classification** goes further. HipsterWars classifies outfit photos into a handful of broad styles, and later work learns style embeddings from fashion photos. Both still look at one outfit photo at a time, use older fixed style categories, and do not recommend anything.  
3. **Unsupervised style discovery** finds style clusters automatically from outfit photos, but those clusters do not match the named aesthetics people actually use, and personal context is not considered.  
4. **Outfit compatibility** models, often trained on Polyvore Outfits, predict whether clothing items go well together. They ignore who is wearing the items and any input beyond clothing.  
5. **Commercial recommenders** (Pinterest, shopping and resale apps) suggest similar items and are tuned for engagement and purchases. Their methods are not openly evaluated, they rarely explain suggestions, and they tend to over-specialize.

### What we’ve done differently is that we’re adding…

1. **Cross-domain, multi-image input.** We combine clothing with a person's room, car, and objects into a single style profile, rather than classifying one outfit photo at a time. Our main experiment directly tests whether this context improves predictions.  
2. **Current internet aesthetics as labels.** We rank across several hundred present-day aesthetics from a living community catalog, including niche long-tail ones, rather than classifying into a few broad, fixed fashion styles. New aesthetics can be added without new hand labels.  
3. **Serendipity as an evaluation criterion.** Serendipity has been studied mostly for movies and products. We apply it to style discovery, with a concrete score and a human check.  
4. **Explainable suggestions.** Every suggestion is traced back to the photo that caused it.  
5. **Fair algorithm comparison.** Three algorithm families are compared on identical features, splits, and metrics, including a CLIP zero-shot baseline.

## **6\. Anticipated challenges, limitations, and responsible use**

### **Main challenge: no clean ground truth**

Aesthetics are subjective and overlap, so there is no single correct answer for an image. Even our three labelers will sometimes disagree. **How we handle it:** instead of asking raters to pick the right aesthetics out of hundreds, we ask a simpler question: does this suggested aesthetic fit these photos? We take the majority judgment and report our agreement (Fleiss' kappa) as a realistic ceiling. A model that matches human agreement is as good as the task allows.

### **Other challenges and limitations**

* **Wiki quality varies.** The wiki is community-written, so some pages are thin and some gallery images are mislabeled or off-topic. We filter out pages without enough visual description or gallery images, merge duplicates, and spot-check galleries for the families we report on.  
* **Training images differ from real photos.** Wiki gallery images are curated and often styled, unlike casual phone photos. That is why the test personas come from separate real-world datasets, and why our main results are human-judged.  
* **Synthetic personas.** Our test users are built from separate dataset photos, not real people's photo sets. Results show whether the method works under controlled conditions; testing with real users is future work.  
* **Uneven gallery sizes.** Popular aesthetics have many more images than niche ones, so systems may drift toward popular answers. We include a popularity baseline, track long-tail coverage, and weight training examples by aesthetic.  
* **Aesthetics change fast.** Today's catalog may be outdated in a year. Because each aesthetic is defined by its page, the catalog can be refreshed by re-scraping, with no relabeling.

### **Responsible use**

* **Privacy and consent:** we use only public research datasets under their published licenses and collect no new photos of people. Images are used only for style classification: we do no face recognition or identity matching and store no personal metadata. Faces stay unaltered because hair, makeup, glasses, and accessories are unequivocally part of aesthetics and to remove them is to pollute the training data.  
* **Bias and fairness:** aesthetic labels can correlate with race, body type, gender presentation, or income, and CLIP is known to carry social biases from its training data. The model will not predict protected attributes. We will check whether similar outfits get different predictions depending on who is wearing them, and report any patterns we find.  
* **Stereotyping and identity:** results are framed as suggestions with confidence scores, not verdicts about who someone is. Users always see why a suggestion was made, so they can disagree with it.  
* **Cultural origins:** some aesthetics come from specific communities and subcultures. Our taxonomy uses the community's own names and descriptions rather than inventing labels.  
* **Harmful use:** the system describes style only. It is not designed to judge, rank, or profile people, and should not be used for targeting or surveillance.

## **7\. Preliminary plan for each member**

### **Shared experimental framework**

All three algorithms solve the same task: re-rank the top 50 zero-shot candidates for each user. They use the same CLIP embeddings, the same pair features, the same training and test personas, and the same metrics (recall@5, MRR, precision@5, nDCG@5, coverage, serendipity). Each member runs their model in all three input settings (outfit only, equal average, weighted profile), so all results land in one comparison table.

### **Individual algorithmsz**

**\[Member 1\]: k-nearest neighbors.** For a new user, find the k most similar training personas and score each candidate aesthetic by how often, and how strongly, it appears among their labels. This needs no training and shows how far similarity alone can go. They will tune k, the distance metric (cosine vs. Euclidean), and distance weighting. Shared responsibility: the data pipeline, including scraping and filtering the wiki, computing embeddings, and building personas.

**\[Member 2\]: logistic regression.** A pointwise re-ranker that predicts the probability that an aesthetic fits a user from the pair features. Its weights are easy to read, so it shows which signals matter most, for example outfit similarity versus room similarity. They will tune regularization strength, the feature set, and the ratio of negative to positive examples. Shared responsibility: evaluation, including judgment pooling, the rating sheet, rater agreement, and metric scripts.

**\[Member 3\]: small multilayer perceptron.** A nonlinear re-ranker that uses the same pair features plus the element-wise product of the user profile and aesthetic vector, which lets it learn interactions the linear model cannot. They will tune layer sizes, dropout, learning rate, and the loss (pointwise cross-entropy vs. a pairwise ranking loss). Shared responsibility: the aesthetic map, including family clustering, nearest and surprising suggestions, and serendipity scoring.

All members rate the pooled suggestions for the test personas, and co-write the final analysis. The stretch-goal demo app is shared work if time allows.

| Dates | Milestone |
| :---: | ----- |
| Oct 12 \- 25 | Scrape and filter Aesthetics Wiki; build catalog and families; compute CLIP embeddings |
| Oct 26 \- Nov 8 | Build training and test personas; run popularity and zero-shot baselines; automatic evaluation set up |
| Nov 9 \- 22 | Each member trains and tunes their algorithm; context experiment (three input settings) |
| Nov 23  \-  Dec 2 | Aesthetic map; pool and rate suggestions; rater agreement; serendipity scoring; fairness check |
| Dec 3 \- | Final comparison, report, presentation; demo app if time allows |

