# Contribution statement

Required by the course: each member's model, experiments and other work, plus the decisions made jointly.

Final version, copied from the report on 22 November 2026. The report is the document of record.

| Member | Model | Experiments | Other work |
|---|---|---|---|
| Andrew Nguyen | the neural network (MLP) on image vectors, with and without real outfit photos, including its tuning, error analysis and final test run. | the encoder comparison (Qwen against CLIP), the real-photo training experiments, the family-level scoring, and the application-layer experiments (photo pooling and the surprise suggestion), the three input settings for every model, and the two fairness checks. Ran the single test scoring of every model on the team server, the CNN and the clustering model with their owners' own code and agreement. | set up the shared team server; built the wiki dataset and added Street Fashion Style and FashionStyle14; computed the image vectors for about 316,000 images; wrote the shared loader, scorer and baselines that all models use; directed and reviewed the 20-family table; built the local demonstration app; built both rating pages and rated the first; wrote the repository's reproduction section, saved configurations and result tables. |
| Giancarlo Guevarra | the CNN, a ResNet-18 trained from raw pixels, with its own picture preparation (padding to a square, colour scaling and odd file formats). | its tuning on validation; three versions (plain, with street photos, and the rare-label recipe) with their final test runs; the CNN's error analysis; and the figure of what the CNN learned (its filters, and the dreams of six aesthetics next to their training pictures). | set up the team repo's folder layout and the course document templates; rated both rating pages and scored them; checked the report's numbers and citations; put the team's drafts together into the report; and built the first version of the slide deck. |
| Edmond Luong | modified hierarchical clustering with multiclass, his own code built from a ChatGPT starting framework. | 39 validation rows over 20 different settings, with and without standardization, ranging from 0.85 to 0.10, and ending at standardized vectors and cut 0.16. | wrote the discussion logs, worked on the skeleton of the report, rated the app's suggestions, and explanations related to hierarchical clustering, including the drop in test. |

## Joint decisions

- 28 October: logistic regression and a CNN as models.
- 15 November: K-Means in place of logistic regression; the repository layout, what stays out of git, and Polars.
- 16 November: hierarchical clustering in place of K-Means; the CNN trains on pixels, the other two models on vectors.
- 17 November: the catalogue narrowed and merged to 503 wearable labels.
- 19 November: corrected vectors for every model; the embedding is a feature and each model does its own preprocessing; the CNN trains on Giancarlo's GPU.
- 20 November: one shared scorer for all three models and no test scoring before the scoring is final; families are scored, not trained on, and wiki top-5 stays the primary metric; final CNN trained with the test pictures kept on the server.
- 20 to 21 November: the clustering model's 0.3% is treated as a fault and fixed.
- 21 November: the CNN keeps its rare-label recipe (Giancarlo, with Edmond's input); one clustering row in the table (Edmond, Giancarlo, Andrew); one freeze for all models at 6 pm (announced by Andrew on the 20th).
