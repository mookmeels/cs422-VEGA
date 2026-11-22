# Contribution statement

Required by the course: each member's model, experiments and other work, plus the decisions made jointly.

| Member | Model | Experiments | Other work |
|---|---|---|---|
| | NN | | |
| Giancarlo Guevarra | CNN: a ResNet-18 trained from raw pixels, no pretrained weights (`src/models/cnn/`) | Tuning on validation: half against full network width, and 128 against 224 px pictures. Five validation seeds and five final runs for three versions: the plain CNN; plus a second output trained on the Street Fashion Style photos' own tags; and the adopted recipe (random crop and stretch, mixup/CutMix, label smoothing 0.1, oversampling). Each recipe change was first tested alone (seed 422; label smoothing also on 423), then all four together on seeds 422 and 423. Pictures of what the filters and labels respond to. | Set up the team repo's folder layout and the course document templates |
| Edmond Luong | Hierarchical Clustering | | Wrote the discussion logs |


## Joint decisions

-
