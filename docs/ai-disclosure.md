# Generative-AI disclosure

Required by the course: for each use, the tool, its purpose, the work it affected, and how the output was verified. Every member remains responsible for explaining their work.

Final version, copied from the report on 22 November 2026. The report is the document of record.

## Andrew Nguyen

| Tool | Purpose | Work affected | How the output was verified |
|---|---|---|---|
| Claude Code (Anthropic) | Wrote code at Andrew's direction; ran experiments and kept a dated log of each; proposed options with their trade-offs, with Andrew making each decision; drafted and rewrote report text from the logged results and the team's group chat; fact-checked report and slide passages. | The shared loader, scorer, table script and baselines; the split repair and duplicate checks; the NN and its tuning and error-analysis scripts; the encoder comparison; the application layer and local app; the fairness checks; both rating pages and their scoring; the every-model input-settings experiment; the scripts that ran each model's single test scoring; the repository's reproduction files; much of the report's problem, data, protocol, NN, comparison, application, limitations and conclusion text; discussion-log entries drafted from the group chat; Figures 1 to 5 and 8, and the example images on slides 4 and 18. | Self-tests on made-up data for the shared code and the NN; reruns that reproduced recorded prediction files byte for byte; comparisons written down before the runs that decided them; paired bootstrap intervals on every claimed difference; a work log of every run, including corrections of earlier mistakes. |
| Codex (OpenAI) | Early data preparation, 14 to 15 Nov. | The first version of the prepared dataset and its train, validation and test split. | The split was re-checked for near-duplicate images, 70 straddling groups were found, and it was repaired before any model was scored. |

## Giancarlo Guevarra

| Tool | Purpose | Work affected | How the output was verified |
|---|---|---|---|
| Claude Code (Anthropic), 16 Nov | Inventory of the files and environment on ml-db; explanation of the existing labeler; labeling plan; repo setup and the repo's folder layout | docs/data-layout.md, repo structure, labeling plan | Inventory commands were read-only and their output was reviewed; labeler behavior was re-checked by scoring 150 images |
| Claude Code (Anthropic), 19 to 22 Nov: Claude Opus as the main model; Claude Sonnet and Claude Fable for code scans | Laid out the design options for each step of the CNN, with sources (Giancarlo chose every one); wrote all the CNN code; ran and monitored the training runs on Giancarlo's PC; scanned the code for bugs; wrote the CNN README | Everything in src/models/cnn/ | 39 automatic self-checks (check_cnn.py); a check that each round of fixes left earlier results exactly unchanged; every result re-scored on ml-db with the shared scorer; two full code scans, each checked by a second model; Giancarlo approved every design decision |
| Claude Code (Anthropic), 22 Nov: Claude Opus | Drafted the CNN sections of the report from Giancarlo's results and decisions; ran the CNN error analysis with the team's script; built the report's CNN figure; checked the report's citations; scored the rating pages | The CNN parts of the report, its CNN figure, the rating results | Every number traced to a scored result file on ml-db; citations checked against the sources in a second session; Giancarlo reviewed and corrected the drafts |

## Edmond Luong

| Tool | Purpose | Work affected | How the output was verified |
|---|---|---|---|
| ChatGPT, Nov 16 | Provided a basic framework for hierarchical clustering and provide optimization suggestions | Affected `src/models/hclustering/hclustering.py` in the structure, though it was modified by Edmond to work with multiclass. | The structure of how it works matches with what we learned in class. When the code was written, it gave acceptable results. |
