# Generative-AI disclosure

Required by the course: for each use, the tool, its purpose, the work it affected, and how the output was verified. Every member remains responsible for explaining their work.

| Date | Member | Tool | Purpose | Work affected | How it was verified |
|---|---|---|---|---|---|
| 2026-11-16 | Giancarlo | Claude Code (Anthropic) | Inventory of the files and environment on ml-db; explanation of the existing labeler; labeling plan; repo setup and this folder layout | `docs/data-layout.md`, repo structure, labeling plan | Inventory commands were read-only and their output was reviewed; labeler behavior was re-checked by scoring 150 images |
| 2026-11-19 to 2026-11-22 | Giancarlo | Claude Code (Anthropic): Claude Opus as the main model; Claude Sonnet and Claude Fable for code scans | Laid out the design options for each step of the CNN, with sources (Giancarlo chose every one); wrote all the CNN code; ran and monitored the training runs on Giancarlo's PC; scanned the code for bugs; wrote the CNN README | Everything in `src/models/cnn/` | 39 automatic self-checks (`check_cnn.py`); a check that each round of fixes left earlier results exactly unchanged; every result re-scored on ml-db with the shared scorer; two full code scans, each checked by a second model; Giancarlo approved every design decision |

| 2026-11-16 | Edmond | ChatGPT | Provided a basic framework for hierarchical clustering and provide optimization suggestions | Affected `src/models/hclustering/hclustering.py` in the structure, but not reponsible for the specific code written | It matches what we have learned in class |