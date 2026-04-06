# AGENTS.md

## Repo purpose

Support workspace for a Kaggle competition. GPU training and submission happen on Kaggle; local work is editing, analysis, and docs.

## Primary artifact

[`kaggle-combined-lora-submission.ipynb`](kaggle-combined-lora-submission.ipynb) — the only notebook that runs on Kaggle. Everything else serves it.

## Workflow

1. Edit the combined notebook or supporting docs locally.
2. Sync to Kaggle, run on GPU.
3. Bring errors/logs/scores back; record in `reports/`.
4. Patch locally, rerun on Kaggle. Keep diffs small between runs.

## Rules

- **Nothing is validated until Kaggle confirms it.** Local completion is not a checkpoint.
- **One notebook for Kaggle.** Do not split into multiple notebooks or build a separate local pipeline.
- **Kaggle runs:** model loading, LoRA/RL training, submission packaging.
- **Local work:** notebook edits, `train.csv` analysis, [`docs/`](docs/README.md) updates, experiment notes.

## Context the user should share

Ask the user to attach or paste anything that improves debugging or decisions, when relevant:

- **Competition inputs:** `train.csv` / `test.csv` snippets, Kaggle dataset names, competition kernel links, official metric or submission demo notebooks.
- **Utilities:** NVIDIA utility script paths, CUTLASS / Triton setup notes, `kagglehub` model IDs, pinned package versions from a working run.
- **Custom data or artifacts:** filtered training JSONL, exported adapter configs, `submission.zip` structure, prior `reports/` entries.
- **Failure evidence:** full cell output, stderr, environment (GPU type, session time left).

Agents should use **Context7** (library/framework docs) and **web search** when fixing version-specific APIs, Kaggle environment quirks, or NeMo/TRL/vLLM behavior—not only training knowledge.

## When a Kaggle run fails

- Read the stack trace / log the user shares.
- Identify env mismatches, OOM, install issues, or formatting bugs.
- Patch the notebook here; user reruns on Kaggle.

## Docs index

[`docs/README.md`](docs/README.md) — competition rules, strategy/stack, training format, NeMo reference.

## Agent skill (project)

[`.agents/skills/nemotron-kaggle-engineer/SKILL.md`](.agents/skills/nemotron-kaggle-engineer/SKILL.md) — Nemotron/Kaggle workflow (YAML frontmatter + body per Cursor skill conventions). To author new skills, use your global **create-skill** / **skill-creator** guidance; keep this repo’s skills under `.agents/skills/`.

## Archive

Old multi-notebook pipeline, `src/`, configs, tests: [`docs/archive/`](docs/archive/README.md). Do not use unless explicitly reviving.
