# Generative-AI disclosure

Required by the course: for each use, the tool, its purpose, the work it affected, and how the output was verified. Every member remains responsible for explaining their work.

| Date | Member | Tool | Purpose | Work affected | How it was verified |
|---|---|---|---|---|---|
| 2026-11-16 | Giancarlo | Claude Code (Anthropic) | Inventory of the files and environment on ml-db; explanation of the existing labeler; labeling plan; repo setup and this folder layout | `docs/data-layout.md`, repo structure, labeling plan | Inventory commands were read-only and their output was reviewed; labeler behavior was re-checked by scoring 150 images |

| 2026-11-16 | Edmond | ChatGPT | Provided a basic framework for hierarchical clustering and provide optimization suggestions | Affected `src/models/hclustering/hclustering.py` in the structure, but not reponsible for the specific code written | It matches what we have learned in class |