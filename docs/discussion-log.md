# Discussion log

Required by the course: every substantive discussion during planning and after reviewing results.

Final version, copied from the report on 22 November 2026. The report is the document of record.

Each entry gives who took part, the question, the evidence, the decision and what we did. Three entries show a discussion changing an experiment or an interpretation: the clustering model's 0.3% and the fault behind it (20-21 November), how a person's photos are weighted (21 November), and the reading of the clustering model's test row (22 November). Two record proposals that were rejected, with the reason: labelling outside photos with our own model (19 November) and training on style families (20 November).

## Timeline

| Date | What happened |
|---|---|
| 28 Oct | Proposal's KNN and MLP replaced by logistic regression and a CNN. |
| By 16 Nov | Wiki scraped (1,161 pages, 23,056 images) and a first split made, then rebuilt so that near-duplicate images stay together (70 groups had straddled two parts). Clustering replaces logistic regression (K-Means on the 15th, hierarchical on the 16th). Team server and repository set up. |
| 16 to 17 Nov | The catalogue narrowed to 544 wearable aesthetics, 78 look-alikes merged, 503 labels. Shared loader, scorer and three baselines written. NN tuned (146 runs) and its settings fixed. The vector error found on the 17th and corrected; the team told on the 19th. |
| 19 Nov | A re-check on the corrected vectors finds one more near-duplicate pair across validation and test; split fixed and everything re-run. Labelling outside photos with our own model dropped (average confidence 9%); Street Fashion Style and FashionStyle14 fetched instead. Clustering model's first version pushed. The new encoder instruction was announced. Report skeleton started by Edmond. |
| 20 Nov | All vector-based results re-run with the new instruction. CNN tuned (four runs) and its five seeds trained; first look at its filters. Clustering model moved onto the shared scorer. Evening: the family level announced and argued through; real outfit photos added to the NN; agreement on training the final CNN without test pictures leaving the server and on who runs the tests; the freeze set for the next day at 6 pm. 11:32 pm: the clustering model's first shared score, 0.3%. |
| 20 Nov, night | The 20-family table made, checked against the wiki's own links and amended five times. Training on families rejected. Real outfit photos added to the NN at family level (+16 points on FashionStyle14). |
| 21 Nov | The fault in the clustering model's merge step found in a code review and fixed; cut re-tuned; 44.4%. CNN with street photos, then the rare-label recipe adopted. Encoder comparison with CLIP. Personas, surprise score and the local app built. 6 pm: freeze. NN scored on test. |
| 21 to 22 Nov, night | Baselines, CNN and clustering model scored on test, once each. The three input settings run for every model. Two fairness checks. Two rating pages. |
| 22 Nov | All three members rate the suggestions. Report written. |

## 28 October: Going over how our models will work

- Participants: Edmond, Andrew, Giancarlo
- Question or disagreement: How would all of our models work with a multiclass goal?
- Evidence considered: CNN works well with images and relates to the course more than a perceptron; KNN does not, while logistic regression can handle multiclass better
- Decision: We will utilize Logistic Regression and CNN
- Resulting action: Switched from KNN to Logistic Regression and from perceptron to CNN in our models

## 15 November: Better model discovered

- Participants: Edmond, Andrew, Giancarlo
- Question or disagreement: Since we learned about clustering and K-Means, should we switch to that?
- Evidence considered: We barely started on Logistic Regression, K-Means would identify the groups better as it was designed for that
- Decision: Utilize K-Means as a better model
- Resulting action: Switched from Logistic Regression to K-Means

## 15 November: How the shared repository is laid out

- Participants: Giancarlo, Edmond, Andrew (group chat, 10:55 to 11:25 pm)
- Question or disagreement: Giancarlo proposed a layout for the team repository and asked what the others thought. What goes into git, and which table library do we share?
- Evidence considered: The repository is public. Images, vectors and model weights are large; keys must never go in.
- Decision: Layout accepted by all three. Code, documents, configurations and small label files go in; images, vectors, weights and keys stay on the team server. Polars as the shared table library, since Edmond was already using it.
- Resulting action: Giancarlo created the repository that night. The same layout is the one submitted.

## 16 November: Switching out of K-Means

- Participants: Edmond, Andrew
- Question or disagreement: K-Means actually doesn't do well, can't identify a group. Hierarchical Clustering might be better.
- Evidence considered: K-Means records groups of similar images, not classes, so it is hard to tell whether every style is represented; hierarchical clustering can start from points of known classes and expand from there.
- Decision: Use Hierarchical Clustering over K-Means
- Resulting action: Switched from K-Means to Hierarchical Clustering as one of our models

## 16 November: What does the CNN read?

- Participants: Giancarlo, Edmond, Andrew
- Question or disagreement: Giancarlo asked whether the CNN should apply its filters to the image vectors, or needs the raw pictures.
- Evidence considered: A CNN's filters look for patterns across neighbouring pixels; a vector of 2,048 numbers has no such layout. If fed the vectors, a CNN would be just another NN.
- Decision: The CNN trains on the images themselves; the NN and the clustering model use the vectors.
- Resulting action: This is what makes the three models different in kind. It also meant the CNN needed its own image preparation and, later, a way to be tested without the test pictures leaving the server.

## 19 November: Corrected vectors, and the labelling plan is dropped

- Participants: Andrew, Edmond (group chat, 10:36 am and 5:12 pm)
- Question or disagreement: Two things reported the same day. First, keeping the end-of-text token when reading the vectors gave much better results. Second, the plan to grow the dataset by labelling outside fashion photos with a labeller trained on the wiki.
- Evidence considered: Corrected against old vectors on validation: nearest centroid 80.3% against 71.9% top-5, zero-shot text 42.8% against 14.6%. For the labelling plan: the labeller's average confidence on the outside photos was 9%.
- Decision: Every model uses the corrected vectors. The labelling plan is rejected: labels that weak cannot be trusted, and checking them by hand against 503 aesthetics was not something we could do.
- Resulting action: All vectors recomputed and the shared loader switched. Street Fashion Style and FashionStyle14, which come with labels made by people, were being downloaded the same afternoon and became the outside data instead. Fashionpedia and DeepFashion are used by no model.

## 19 November: Is the embedding preprocessing or a feature?

- Participants: Edmond, Giancarlo, Andrew (group chat, 6:01 to 6:45 pm)
- Question or disagreement: While drafting the data section, Edmond asked how to describe preprocessing when the images have already been turned into vectors by another model. Is that step our preprocessing?
- Evidence considered: Andrew: the images are the data and the 2048 dimensions allow for the encoded features to be expressed, made by a model we do not train. Giancarlo: preprocessing is what changes the inputs so they are uniform; for the CNN that means padding, colour scaling and handling odd file formats. Edmond: his model standardizes the vectors itself.
- Decision: The vectors are features from a frozen feature extractor. Each model documents its own preprocessing on top.
- Resulting action: The report's "Features" paragraph, the clustering model's standardization with training statistics only, and the CNN's own preparation step.

## 19 November: How large should the CNN's pictures be, and where does it train?

- Participants: Giancarlo, Andrew (group chat, 3:32 to 4:35 pm)
- Question or disagreement: On the team server's CPU, even 160 × 160 pictures would have taken until Sunday. How much should the pictures be shrunk?
- Evidence considered: Andrew: at 160 pixels the details of an outfit are squeezed away; heavy jobs run on Andrew's own PC, and only results are pushed to the server. Giancarlo wanted near full resolution with padding; 224 pixels is the usual size for this architecture.
- Decision: Train on Giancarlo's own GPU, with whole pictures padded to a square, and let a small tuning run choose between 128 and 224 pixels.
- Resulting action: 224 pixels won the tuning by 0.8 point (Table 3).

## 20 November: A second label level, and what would be unfair

- Participants: Andrew, Giancarlo, Edmond (group chat, 5:27 to 7:53 pm)
- Question or disagreement: Andrew announced 20 style families above the 503 aesthetics, so that outside photos with broad tags ("Rock", "Bohemian") could be used. Giancarlo: does the CNN have to be retrained, and what does its output layer become? Edmond: are families a second success criterion? Andrew then worried that the NN training on families and new photos would be unfair to the other two models.
- Evidence considered: A network trained to output the 20 families directly was 7.5 points worse than the 503-label network with its guesses grouped by family. Adding real outfit photos at weight 0.25 raised family accuracy on FashionStyle14 from 33.1% to 49.4% while wiki top-5 moved from 88.3% to 87.9%; at weight 1.0 wiki top-5 fell to 87.0%. The gain came from the photos, not from the families. "Casual and everyday" is heavily over-represented in those photos.
- Decision: Training on families is rejected: nobody retrains for them. A family is looked up from the aesthetic a model guesses (guess Goth for a Punk image: wrong aesthetic, right family). Wiki top-5 stays the primary metric, so the proposal's goal is not replaced. The new photos enter the NN at weight 0.25.
- Resulting action: The scorer gained two family columns for every model with no retraining. To keep the comparison fair, the NN without the new photos is reported as the like-for-like row, and Giancarlo trained a version of the CNN with the street photos.

## 20 to 21 November: The clustering model scores 0.3%, and why

- Participants: Edmond, Andrew
- Question or disagreement: Edmond's first row through the shared scorer was 0.3% top-5, below always guessing the most common aesthetics (5.8%). Edmond thought a model that cannot see the pictures will not win, and he would leave it there. Andrew disagreed, saying those are not the numbers of a weak model but of a bug.
- Evidence considered: The NN cannot see pictures either; it reads the same vectors and reaches 88%, and nearest centroid reaches 82%. Every setting tried up to then scored under 2%, with cuts between 0.65 and 0.85. The next day Andrew asked why the merging cut had to be so high and went through the merge step with Edmond. After a merge, the similarity to a third cluster was being averaged with the wrong pair of clusters, which pushed every similarity upward. Edmond traced it to a loop he had changed earlier to make the code faster.
- Decision: Treat 0.3% as a bug, not as the model's result. Fix the update, then re-tune the cut.
- Resulting action: Edmond wrote the fix himself. The first row after it scored 25.7% where the same setting had scored under 1%. He then re-tuned the cut, which on standardized vectors has to be far lower (0.59 gives 3.4%, 0.20 gives 44.4%), and settled on 0.16: validation top-5 44.4%. This is the entry where a discussion most clearly changed an experiment: without the shared table and the disagreement, the report would have carried 0.3% as a finding.

## 21 November: One clustering row in the results table

- Participants: Edmond, Giancarlo, with Andrew approving
- Question or disagreement: Edmond asked should the team's results table keep every clustering tuning attempt, or only the final model?
- Evidence considered: the table held 39 clustering tuning rows (cut values on standardized and unstandardized vectors), all on validation.
- Decision: keep the final model only. Everyone agreed.
- Resulting action: the 39 rows were moved unchanged to a separate file, and the table shows one clustering row (standardized, cut 0.16).

## 21 November: Oversampling helps the common labels and hurts the rare ones

- Participants: Giancarlo, Edmond (group chat, 1:56 to 2:08 pm); decision by Giancarlo about 3:20 pm
- Question or disagreement: Giancarlo asked the chat for input: oversampling rare labels raised the CNN's top-5 by about 2 points, but only through the common labels, and lowered the rare labels by about 2. Fold it into the other changes, or leave it out?
- Evidence considered: Giancarlo: the network had memorised the repeated pictures. Edmond: keep it and aim it at the rare labels, though it may help new pictures more than our validation set. Tested together with random crops, mixup or CutMix and label smoothing, top-5 fell about 1.3 points and rare-label top-5 rose about 5.5; the rule written beforehand (a two-seed mean of 32.5) failed.
- Decision: Adopt the combined recipe and report that the rule failed (Giancarlo: "We want a model that can identify things, not just the most common aesthetics.").
- Resulting action: All three CNN versions stayed in the table and were scored on test: level on top-5, with the adopted recipe 8 to 9 points ahead on rare labels.

## 21 November: Which of a person's photos should count?

- Participants: Andrew, Edmond, Giancarlo (group chat, 5:47 to 6:42 pm)
- Question or disagreement: Andrew: a person uploads three to five photos; which of them form their actual aesthetic, and should some weigh more? Giancarlo: that sounds like the user's choice of which pictures define them.
- Evidence considered: Weighting outfit photos above others, and weighting by the network's confidence, had both lost to a plain average (confidence by 3.5 to 12.7 points). Weighting each photo by how much its family reading overlaps with the other photos' gained 8.2 points on wiki personas and 2.8 on outfit-photo sets.
- Decision: A photo counts by its agreement with the others; a photo that stands apart counts less in the profile and is what the surprise suggestion draws on. This is the application layer, outside the three-model comparison, so the freeze did not cover it.
- Resulting action: A new experiment and a change in the app. When the same weighting was later run for every model (Table 9) it hurt where a person's photos share only a family, so the report's claim was narrowed to "not reliably better than the plain average".
20 November (announced), 21 November 6 pm (freeze): Freezing the models before the test

- Participants: Andrew, Giancarlo, Edmond (group chat)
- Question or disagreement: when do we stop changing the models, so that each one is scored on the test part only once?
- Evidence considered: the course rule, "Freeze all configurations before final test evaluation. Do not retune models after inspecting test results." No model had been scored on the test yet.
- Decision: one freeze for all models, Saturday 21 November at 6 pm Pacific. Everyone agreed.
- Resulting action: no setting changed after the freeze; each model was scored on the test once, the NN at the freeze and the others later that night.

## 22 November: Reading the test results

- Participants: Edmond, Giancarlo (through comments on the shared report draft)
- Question or disagreement: On validation the clustering model was ahead of the zero-shot text baseline (44.4% against 41.7%). On test it is behind (40.1% against 43.4%). What may the report claim? And what does "real outfit photos" in the recommendation mean?
- Evidence considered: The clustering model's drop of 4.3 points is the largest of any row; its scores depend on which pictures are scored together. The models were frozen, so nothing could be re-tuned.
- Decision: The report says the two are level and that neither lead is convincing, with Edmond's explanation. "Real outfit photos" is defined as photos from two published datasets, not photos from users.
- Resulting action: The sentence "the clustering model beats the text baseline" was removed from the results and the conclusion. No model was changed.

## 22 November: The three of us rate the same suggestions differently

- Participants: Andrew, Edmond, Giancarlo (group chat, 22 November, 7:48 to 8:52 am, where Edmond and Giancarlo posted their ratings; then the shared report draft)
- Question or disagreement: Edmond rated 70% of the profile's first aesthetics as fitting, Giancarlo 37% and Andrew 15%. On the second page the NN's precision@5 is 0.62 by one rater and 0.18 by the other. Giancarlo, on seeing Edmond's answers, remarked that he was a far harsher grader. Whose rating counts?
- Evidence considered: Fleiss' kappa 0.33 on the first page; Cohen's kappa 0.26 on the second. The one thing all three agree on is that the surprising suggestion fits more often than a random one (56% to 85% against 5% to 21%).
- Decision: Report every rater separately beside the pooled figure, give both the average and the majority, and claim only what holds for each rater.
- Resulting action: Tables 10 and 11 as they stand, and the limitation that the suggestions were rated only by the team, who agree "fairly, not well".
