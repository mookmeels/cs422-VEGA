The CNN: a ResNet trained from scratch on the wiki pictures. Owner: Giancarlo Guevarra
=====================================================================================

DRAFT, 20 Nov 2026. Everything marked [AFTER RUNS] is filled in once the runs on Gianni's PC are done.

What it is and why. Every other VEGA model starts from the Qwen image vectors, summaries made by a large model
trained on other data. This one learns only from the pixels of our ~9,400 training pictures: a convolutional
neural network (a ResNet-18 layout) whose weights all start random. It answers a question for the comparison:
how far does a network get that has seen nothing but this project's pictures? (A model that loses is still a
valid result if it is well supported, as the assignment says.)

How it was designed. Gianni decided every setting step by step with Claude (19 and 20 Nov 2026): for each step
the options were laid out with trade-offs and a recommendation, and Gianni chose. Nothing below was tuned on
test pictures; the two settings marked "tuned" were chosen on validation by a rule written before the runs.


FILES

  prepare_images.py   decodes every picture once and saves exact 8-bit pixels at 128 and 224 px (run first)
  train_cnn.py        the network, its training, and scoring through the team's vega_score.py
  predict_cnn.py      the final run's test predictions on ml-db, from weights trained on the PC (never scores them)
  prepare_street.py   the Street Fashion Style photos for the CNN's second output (their posters' own style tags)
  check_cnn.py        39 checks on made-up pictures (run after any change; all pass, about two minutes on a CPU)
  summarise_tuning.py the tuning table, the rule's verdict, seed means and the learning curves
  find_shared.py      finds the team's loader and scorer (src/common in the repo, code/shared on ml-db)
  show_filters.py     every first-layer filter drawn as a picture; for each of the 4 stages, a sample of 32 of its
                      filters (the most selective) with the training pictures that make each fire most
  dream_filters.py    for the same sample of filters, a dreamed-up picture of what each looks for: start from
                      faint noise and nudge the picture to make that filter fire as much as possible
                      (activation maximisation: Olah, Mordvintsev and Schubert 2017; Simonyan et al. 2013)
  dream_labels.py     the same for every one of the 503 labels: a dreamed-up picture that maximises the label's
                      score, 32 labels per page, each captioned with its number of training pictures

Outputs go to runs/cnn/ (prepared/, validation/, final/, filters/).


RUNNING IT

  Where it ran: Gianni's PC, AMD Radeon RX 7900 XTX (24 GB), Windows 11, Python 3.12, AMD's PyTorch 2.9.1 +
  ROCm 7.2.1 wheels, numpy 2.5.5, polars 2.1.0, pillow 13.0.0, with the team's train + validation pack
  (share/vega-trainval-pack.tar) unpacked and VEGA_ROOT pointing at it. Any CUDA or ROCm card, or a CPU
  (slowly), runs the same code; --device auto picks the graphics card with the most memory.
  Every PC run used --no-miopen: AMD's MIOpen library could not compile its batch-norm kernel on this card at
  run time (20 Nov 2026: "'type_traits' file not found", then miopenStatusUnknownError), so PyTorch's own GPU
  kernels do that work instead. Same arithmetic, different implementation; each run's log records it.

    python check_cnn.py
    python prepare_images.py                                        # 23 s on the PC (both sizes, 10,710 pictures)
    python train_cnn.py --width W --size S --seed 422 --curve --no-miopen       # one tuning run
    python train_cnn.py --width 64 --size 224 --seed 422 423 424 425 426 --curve --no-miopen --out <new folder>
                                                                    # the five seeds; a new folder, so the tuning
                                                                    # run's files stay (see REPEATS below)
    python summarise_tuning.py
    python show_filters.py --weights runs/cnn/validation/weights/<file>.pt --no-miopen   # about 1 minute
                                                                    # stage1_top.png to stage4_top.png: a sample
                                                                    # of 32 of the 64, 128, 256 and 512 filters
    python dream_filters.py --weights <the same .pt> --chosen <that run's chosen.json> --no-miopen
    python dream_labels.py --weights <the same .pt> --no-miopen                         # about 3.5 minutes

  The CNN + street photos (Gianni, 20 Nov 2026; see STREET PHOTOS below):
    python prepare_street.py --photos <folder with chictopia/> --manifest <that folder>/manifest.csv
    python train_cnn.py --width 64 --size 224 --seed 422 --curve --no-miopen --street runs/cnn/prepared/street
                                                                    # files named cnn-r18w64-224-street-...,
                                                                    # next to the plain CNN's in the same folders
    final: train_cnn.py --final --fit-only ... --street runs/cnn/prepared/street, as for the plain final run

  The prediction files (runs/cnn/validation/preds/*.npz) are copied to ml-db and scored there with
  vega_score.py, so the team table is made by the same code on the same images as every other model.

  The final run (Andrew's option A, 20 Nov 2026; the test pictures never leave ml-db):
    on ml-db:  python prepare_images.py --final --sizes 224         # fit = train + validation, eval = test
    to the PC: only fit-224.npy, fit-ids.txt and stats.json from runs/cnn/prepared/final (no test pixels,
               ids or picture outlines; boxes-224.npz holds test outlines too, and training does not need it)
    on the PC: python train_cnn.py --final --fit-only --width 64 --size 224 --seed 422 423 424 425 426
                   --no-miopen --prepared <that folder> --out <new folder>    # weights only, about 25 min a seed
    to ml-db:  the five weights files
    on ml-db:  python predict_cnn.py --weights <the five .pt files>          # test predictions, not scored
    then, once the team freezes settings: vega_score.py --final on each prediction file (scored once).
  [AFTER RUNS: timings of the final run.]


THE PICTURES (prepare_images.py)

  Every picture becomes a standard 3-channel sRGB picture:
    first frame of animated pictures (88 in train + validation); camera rotation tags applied; embedded colour
    profiles converted to sRGB; white behind see-through areas; CMYK converted; greyscale pictures stay grey
    (the same value in all three channels, nothing colourised).
  Shape: the whole picture is fitted inside a square (nothing is cut off) and the bars around it are filled
    with the average colour of the training pictures. Lanczos resampling.
  Colour scaling: each channel minus its mean, divided by its spread, both measured on training-picture pixels
    only (not the bars). After scaling the bars are zero.
  Sizes: 128 and 224 px, each prepared once and stored exactly (no re-compression).


THE NETWORK (train_cnn.py)

  ResNet-18 layout (He et al. 2015/2016), with W = 32 ("half width") or W = 64 (the standard ResNet-18):

    layer                                      filters   picture at 128 px   at 224 px
    start: 7 x 7, stride 2, BN, ReLU             W         64 x 64             112 x 112
           3 x 3 max-pool, stride 2              W         32 x 32             56 x 56
    stage 1: 2 residual blocks                   W         32 x 32             56 x 56
    stage 2: 2 blocks, first one stride 2        2W        16 x 16             28 x 28
    stage 3: 2 blocks, first one stride 2        4W        8 x 8               14 x 14
    stage 4: 2 blocks, first one stride 2        8W        4 x 4               7 x 7
    global average pooling, dropout 0.3, linear layer to 503 scores

  A residual block: 3 x 3 filters, batch norm, ReLU, 3 x 3 filters, batch norm, plus the block's input
  (a shortcut; a 1 x 1 stride-2 filter where the shape changes), then ReLU. Edges are zero-padded.
  Weights: 2,928,151 at W = 32; 11,434,551 at W = 64.


TRAINING, EVERY SETTING

    setting               value                               how it was set
    picture size          224 px                              tuned (decision 9; see TUNING)
    width                 64 (the standard ResNet-18,         tuned (decision 9; see TUNING)
                          11.4 M weights)
    starting weights      He normal; last BN of each block    Gianni, decision 3
                          starts at zero (pass-through)
    dropout               0.3 before the last layer           Gianni, decision 4
    loss                  the NN's: cross-entropy on the      Gianni, decision 5
                          loader's label rows, each picture
                          weighted by (pictures of its
                          label) ** -1 (balance 1); a
                          picture in two galleries counts
                          half to each label, as in the NN
    optimiser             AdamW, peak learning rate 1e-3,     Gianni, decision 6
                          weight decay 0.05 (not on BN
                          scales or biases)
    schedule              linear warm-up over the first 10%   Gianni, decision 6 (10% as the NN)
                          of steps, then cosine to zero
    batch, epochs         128 pictures, 60 epochs             Gianni, decision 7
    augmentation          random left-right flip, shift up    Gianni (training only; scoring sees the
                          to 10%, zoom in or out up to 10%,   plain picture)
                          gaps filled with the average colour (the plain and street CNNs; the adopted
                                                              recipe replaces it, see TRAINING OPTIONS)
    precision             fp32 (full precision)               Gianni, decision 8: bf16 only if at least
                                                              1.3x as fast and its epoch-2 loss within
                                                              0.05 of fp32's at both sizes. Measured (2
                                                              epochs, seed 422): bf16 was 1.19x as fast at
                                                              W 32 / 128 px (1,304 against 1,093 pictures/s)
                                                              and 1.88x at W 64 / 224 px (796 against 423);
                                                              losses within 0.020 and 0.010. Not 1.3x at
                                                              both sizes, so fp32.
    seeds                 422 to 426                          as the rest of the team


TUNING (validation only)

  Four runs, seed 422, 60 epochs: W = 32 and 64, each at 128 and 224 px. Rule, written before the runs: keep
  the best validation top-5; if another run is within 0.5 point, keep whichever trained faster.

    run (seed 422, fp32)     weights   top-5   top-1   MRR     per-label top-5   train minutes   peak GPU GB
    W 32, 128 px             2.9 M     24.7     9.5    0.182   22.1               8.2            1.2
    W 64, 128 px             11.4 M    29.8    10.2    0.203   24.6              13.3            1.8
    W 32, 224 px             2.9 M     29.1     9.5    0.198   27.3              14.0            3.3
    W 64, 224 px             11.4 M    30.6    11.8    0.217   26.6              22.1            4.8

  Verdict: W 64 at 224 px, best top-5 (30.6); the next best (29.8) is 0.8 point behind, outside the 0.5 margin.
  Larger pictures and more filters both helped, and most together. Gianni accepted the verdict (20 Nov 2026) and
  chose no further tuning ("We already got the results we wanted for which image size to use").
  Output of summarise_tuning.py: runs/cnn/from-pc-2026-11-20/validation/summary/.


RESULTS (validation, 1,351 pictures, 503 labels; team table runs/comparison/table.md, 20 Nov 2026)

    model                                   seeds   top-5        top-1        MRR             per-label top-5
    NN (MLP on the Qwen vectors)            5       88.3 ± 0.4   62.7 ± 0.5   0.741 ± 0.002   86.6 ± 0.5
    Baseline: nearest centroid (Qwen)       1       82.0         53.5         0.660           82.6
    Baseline: zero-shot text (Qwen)         1       41.7         21.0         0.315           41.1
    CNN (ResNet-18, 224 px, this folder)    5       30.9 ± 0.6   12.0 ± 0.3   0.220 ± 0.003   26.6 ± 0.3
    Baseline: popularity                    1        5.8          1.4         0.047            1.0

  Seeds 422 to 426, top-5: 30.1, 30.6, 31.2, 30.9, 31.8. Thin labels top-5 22.5 ± 0.8, other labels 31.5 ± 1.2;
  label coverage 99.8%: the CNN spreads its guesses over almost every label rather than a popular few.
  What it means: learning from our ~9,400 pictures alone, the network gets the right aesthetic in its top five
  about 3 times in 10, five times the popularity baseline, but far below every model built on the Qwen vectors,
  which were learned from vastly more pictures. That gap is the answer to the question the CNN was built for.
  Learning curve (curves.png in the summary folder): training loss keeps falling (to about 1.1 after 60
  epochs) while validation loss is lowest at epoch 22 (4.96) and then climbs (5.29 at the end); validation top-5 peaks
  around epoch 43 (31.1 in the seed-422 repeat) and ends at 30.0. The network memorises the training pictures
  late in training. The epoch count stayed at 60, as fixed before the runs; picking the best epoch by looking
  at validation would be tuning on it.
  [AFTER RUNS: where it goes wrong (error summary, waiting on Andrew's OK for nn/analyse_errors.py); the filter
  pictures (show_filters.py, being made on the PC).]


STREET PHOTOS (a second version of the CNN; Gianni, 20 Nov 2026)

  Why. The CNN learns only from ~9,400 wiki pictures. Street Fashion Style (Zenodo 833051, CC BY-NC 4.0) has about
  280,000 outfit photos, each with the poster's own style tag (117 tags, from Chic 20% and Casual 16% down to a
  handful of photos). Those tags are not the wiki's aesthetics, and Gianni ruled out grouping them into style
  families ("Scrap families. It ends up over representing certain families that are very popular").
  How. The network gets a second output: 39 scores, one per street tag with at least 500 training photos, after the
  same pooling and dropout as the 503 wiki scores. Each step trains on the usual 128 wiki pictures and adds 128
  street photos, drawn tag-evenly (a tag at random, then one of its photos) so the big tags cannot dominate; their
  tag cross-entropy counts 0.25 (Andrew's weight for his NN's real photos, copied, not tuned). The 503 outputs,
  their loss and every other setting are as above, with one difference found by the code scan (21 Nov 2026):
  one forward pass carries both the wiki pictures and the street photos, so every batch norm layer is fitted on that mix (about half street photos), and the saved running averages used at test time are the mix too. The street outputs are not scored in the team table; they let far more
  real outfit photos shape the network (multi-task learning). A street run's "train loss" includes 0.25 x the
  tag loss, so its curve is not comparable with the plain runs' training loss. Posters with sha1(name) % 10 == 0
  (as for Andrew's NN; about 30,000 photos) are never trained on and give the tag accuracy on unseen posters.
  Gate, written before any run (Gianni "2 a"): seed 422 first; continue to seeds 423 to 426 and the final run only
  if its wiki validation top-5 is at least 31.6 (the plain CNN's seed 422: 30.6, plus 1 point, twice the measured
  repeat gap). If adopted it is a new row, "CNN (ResNet-18, 224 px) + street photos"; the plain CNN stays as the
  row trained on the same pictures as the other models.
  Changed before the runs finished (Gianni, 21 Nov 2026, about 00:15 Pacific): "I think we fully train the model
  no matter what. Even if it's slower, we already got the weights from the last model." So all five seeds and the
  five final runs were trained whatever seed 422 scored; the 31.6 mark was reported, not used to stop.
  Results (21 Nov 2026, Gianni's PC, scored on ml-db from the copied predictions; each re-score identical):
    seed 422 wiki validation top-5 31.8: above the 31.6 mark (the plain CNN's seed 422: 30.6, its repeat 30.1).
    five seeds, wiki validation    top-5          top-1          MRR            family top-1
      CNN + street photos          31.8 +- 0.8    11.8 +- 0.4    0.221          26.6
      CNN (plain)                  30.9 +- 0.6    12.0 +- 0.3    0.220          26.8
    per seed, top-5 (street / plain): 422 31.8 / 30.1, 423 31.2 / 30.6, 424 31.1 / 31.2, 425 33.0 / 30.9,
      426 31.6 / 31.8. About one point higher on average, top-1 the same within noise; the spreads overlap.
  Tag accuracy on unseen posters (29,444 photos): top-1 2.8 to 3.1%, per-tag mean 6.0 to 6.7% (chance 2.6%).
    The tag output barely beats chance. Not a fault found in the code: on a 6,000-photo sample checked on ml-db,
    it scores 4.5% on photos it trained on and 2.8% on unseen ones, spread over 36 to 37 of the 39 tags (no
    collapse onto one tag, no memorising); the made-up tags in check_cnn.py reach above 60%. Reading (inferred):
    the posters' own tags are hard to tell apart from a photo, so the gain does not come from getting the tags
    right. Where it does come from is not shown: the street photos shape the shared filters, but they also share
    every batch norm layer with the wiki pictures (see How), and these runs cannot tell the two apart. Telling
    them apart would need a run that keeps the street photos out of the batch norm statistics.
  Preparing: 271,091 photos (241,647 training, 29,444 held out; 3 unreadable files skipped), 186 s on the PC,
    40.8 GB at 224 px.
  Times per run on the PC (fp32, --no-miopen): validation about 2,955 s (49 to 50 min; about 197 training
    pictures a second, each step 128 wiki pictures + 128 street photos read from disk); final about 3,265 s
    (54 min). Roughly twice the plain CNN's time.

TRAINING OPTIONS AND THE ADOPTED RECIPE (Gianni, 21 Nov 2026)

  Four options, all off by default (off = exactly the plain and street CNNs above, checked bit for bit):
    --crop-min 0.6 --stretch 0.15   a random crop keeping 60 to 100% of the picture, anywhere inside it, its
                                    width-to-height ratio between 1/1.15 and 1.15 (replaces shift and zoom; the
                                    PC's records of the adopted runs still say shift 0.1 and zoom 0.1, which were
                                    not applied); street photos get the same crop
    --mix mixup-cutmix              per wiki batch, mixup or CutMix at random (Zhang et al. 2018; Yun et al. 2019;
                                    strengths 0.8 and 1.0 as in DeiT, Touvron et al. 2021); street photos unmixed
    --smooth 0.1                    label smoothing (Szegedy et al. 2016): 10% of each label row spread evenly
    --sample oversample             rare labels' pictures drawn more often (each time freshly distorted) instead
                                    of weighted more in the loss; at balance 1 an epoch sees about half the
                                    pictures, the common labels' pictures about a quarter as often as before
  Tests (seed 422, plain CNN, top-5 / rare-label top-5; plain 30.6 / 23.0): crop and stretch 30.9 / 22.9; mixup
  30.8 / 23.7 (its loss weights were mixed in a way that weakened the balancing; fixed after the test, see the
  code scan); label smoothing 31.2 / 23.4 (seed 423: 31.2 / 24.6); oversampling 32.1 / 20.2. No single change
  reached the 1-point mark written before the tests.
  All four together on the street CNN, seeds 422 and 423: top-5 30.1 and 30.3 (street CNN 31.8 and 31.2),
  per-label top-5 29.0 and 29.0 (26.7), rare labels 28.3 and 28.0 (about 22.7). The rule written before this
  test (two-seed mean at least 32.5) failed; Gianni adopted the recipe anyway, 21 Nov about 15:20: "We want a
  model that can identify things, not just the most common aesthetics." So the team's CNN gives up about 1.3
  points of top-5 for about 5.5 points on the rare labels.
  The adopted recipe:
    python train_cnn.py --width 64 --size 224 --seed 422 423 424 425 426 --curve --no-miopen --precision fp32 \
        --street runs/cnn/prepared/street --crop-min 0.6 --stretch 0.15 --mix mixup-cutmix --smooth 0.1 \
        --sample oversample
    final: the same options with --final --fit-only --prepared <final fit-only folder>, then predict_cnn.py
  Results, five seeds (422 to 426), wiki validation, scored on ml-db from the copied predictions (each re-score
  identical to the PC's):
                                   top-5         top-1         per-label top-5   rare labels   other labels
    adopted recipe (street + 4)    30.1 +- 0.9   11.9 +- 0.1   29.0 +- 0.9       28.4 +- 0.9   29.8 +- 0.8
    street CNN                     31.8 +- 0.8   11.8 +- 0.4   26.7 +- 0.8       21.7 +- 1.3   32.6 +- 1.3
    plain CNN                      30.9 +- 0.6   12.0 +- 0.3   26.6 +- 0.3       22.5 +- 0.8   31.5 +- 1.2
  Per seed, top-5 / rare labels: 422 30.1 / 28.3, 423 30.3 / 28.0, 424 29.4 / 27.8, 425 29.1 / 27.7, 426 31.4 / 30.0.
  So against the street CNN: about 1.7 points less top-5, about 6.7 points more on the rare labels, top-1 level.
  Street tags on held-out posters stay near chance (top-1 2.9 to 3.4%), as for the street CNN.
  Times on the PC (fp32, --no-miopen, one run at a time): validation about 2,975 s (about 197 training pictures
  a second, as for the street CNN; the options cost no measurable time); final about 3,295 s. Two runs at once
  on the card stalled once (the -mix test, 21 Nov), so every run since was alone.
  Final runs: weights cnn-r18w64-224-street-crop60-st15-mix-ls10-os-test-seed422..426.pt; test predictions made
  on ml-db with predict_cnn.py, not scored.

REPEATS, TIME, MEMORY

  A graphics card adds numbers in varying order, so the same seed gives nearly, not exactly, the same scores.
  Measured gap between two runs of seed 422 at W 64, 224 px: top-5 30.6 then 30.1 (-0.5), top-1 11.8 then 12.1
  (+0.4). That is the same size as the tuning rule's 0.5 margin, so the margin sits at the level of this noise;
  the verdict did not depend on it (0.8 point gap), and the five-seed mean (30.9 ± 0.6) agrees.
  How the repeat happened: the five seeds were run in the tuning folder, and seed 422's repeat replaced the
  tuning run's prediction, weights and curve files (same names). The tuning run's scores survive in runs.jsonl
  and results.jsonl and its epochs in logs/tune-w64-224.log; the team table uses the repeat (latest run counts,
  as for every model). Since then train_cnn.py refuses to replace an earlier run's files, and
  summarise_tuning.py judges tuning on the first full-length fp32 run of each plain setting (street runs and
  other label sets never enter the verdict; code scan, 21 Nov 2026).
  Time per 60-epoch run, fp32 with a validation score after every epoch: W 32 / 128 px 8.7 min, W 64 / 128 px
  14.1, W 32 / 224 px 14.8, W 64 / 224 px 23.4 (22.1 of it training, 423 pictures/s). Peak GPU memory
  1.2 to 4.8 GB of 24. The four tuning runs and five seeds took about 3 hours of GPU time in all.


REFERENCES (each read on 20 Nov 2026; all primary sources)

  He, Zhang, Ren, Sun. "Deep Residual Learning for Image Recognition". arXiv 1512.03385, 10 Dec 2015.
    The layout (Table 1); deeper plain networks train worse, shortcuts fix it (sections 1, 4.1).
  He, Zhang, Ren, Sun. "Delving Deep into Rectifiers". arXiv 1502.01852, 6 Feb 2015.
    The starting weights for ReLU networks (section 2.2).
  Goyal, Dollár, Girshick, Noordhuis, Wesolowski, Kyrola, Tulloch, Jia, He. "Accurate, Large Minibatch SGD:
    Training ImageNet in 1 Hour". arXiv 1706.02677, 8 Jun 2017. Last BN of each block starts at zero
    (section 5.1); gradual warm-up (section 2.2).
  Krizhevsky, Sutskever, Hinton. "ImageNet Classification with Deep Convolutional Neural Networks". NIPS 2012.
    Flips and crops as augmentation; "Without this scheme, our network suffers from substantial overfitting"
    (section 4.1).
  Zhang, Bengio, Hardt, Recht, Vinyals. "Understanding deep learning requires rethinking generalization".
    ICLR 2017 (arXiv 1611.03530, 10 Nov 2016). Networks "easily fit random labels" (sections 1.1, 2.1): why
    a from-scratch network on few pictures needs augmentation and regularisation.
  Zagoruyko, Komodakis. "Wide Residual Networks". arXiv 1605.07146, 23 May 2016. Wider, shallower residual
    networks match much deeper thin ones and train faster (abstract, section 1): why width, not depth, is
    what tuning varies.
  Ioffe, Szegedy. "Batch Normalization". arXiv 1502.03167, 11 Feb 2015.
  Loshchilov, Hutter. "Decoupled Weight Decay Regularization" (AdamW). ICLR 2019 (arXiv 1711.05101, 14 Nov 2017).
  Lin, Chen, Yan. "Network In Network". arXiv 1312.4400, 16 Dec 2013. Global average pooling in place of
    fully connected layers, less prone to overfitting (section 3.2).
  Olah, Mordvintsev, Schubert. "Feature Visualization". Distill, 7 Nov 2017, doi 10.23915/distill.00007.
    "Starting from random noise, we optimize an image to activate a particular neuron"; unregularised results are
    "full of noise and nonsensical high-frequency patterns"; regularisers: jitter, rotation, scaling, blurring;
    a decorrelated Fourier parametrisation (sections "Feature Visualization by Optimization", "The Enemy of
    Feature Visualization", "Three Families of Regularization", "Preconditioning and Parameterization").
  Simonyan, Vedaldi, Zisserman. "Deep Inside Convolutional Networks: Visualising Image Classification Models and
    Saliency Maps". arXiv 1312.6034, 20 Dec 2013. Optimising the input image to maximise a class score, with L2
    regularisation, started from the zero image (section 2). The older Erhan, Bengio, Courville and Vincent 2009
    report (Université de Montréal, no. 1341) introduced the idea but could not be read here, so it is not cited.
