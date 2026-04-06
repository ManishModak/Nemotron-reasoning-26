# Execution plan

Workspace date context: April 2026.

## Model

- **Kaggle is source of truth** for GPU memory, install quirks, timeouts, and submission behavior.
- The leaderboard runs **your LoRA** on the host’s **vLLM** stack; you do not ship inference code. Align SFT targets with `\boxed{}` and the metric’s user suffix ([`docs/training-data-format.md`](docs/training-data-format.md)). Doc index: [`docs/README.md`](docs/README.md).

## Primary deliverable

1. **[`kaggle-combined-lora-submission.ipynb`](kaggle-combined-lora-submission.ipynb)** — copy or upload to a Kaggle GPU notebook, run end-to-end, download **`submission.zip`**.
2. **First Kaggle checkpoint** — follow [`reports/kaggle-first-run-checklist.md`](reports/kaggle-first-run-checklist.md); record score, errors, and knob changes in `reports/`.

## Milestones

### M1 — First real submission (progress prize)

- Run the combined notebook on Kaggle (smoke with `MAX_TRAIN_SAMPLES` first if needed).
- Submit `submission.zip`; capture leaderboard feedback and runtime notes.

### M2 — Data and training quality

- Local analysis on `train.csv` (families, length, failure patterns).
- Improve SFT rows (optional `<redacted_thinking>`, filtering, synthetic data); keep [`docs/training-data-format.md`](docs/training-data-format.md) accurate.

### M3 — Training optimization

- Tune epochs, LR, `MAX_SEQ_LENGTH`, LoRA rank (≤ 32), gradient accumulation.
- Optional: lightweight RL / GRPO **only** after SFT is stable on Kaggle hardware.

### M4 — Final hardening

- Pick best adapter, re-verify zip contents, final dry run before deadline.

## Checkpoint loop

1. Edit the combined notebook (or docs) locally.
2. Sync to Kaggle and run.
3. Save logs, scores, and issues under `reports/`.
4. Repeat.

## Reference notebooks (do not duplicate logic in `src/` for submission)

- `nvidia-nemotron-submission-demo.ipynb` — LoRA attach + save pattern
- `nvidia-nemotron-metric.ipynb` — scoring, `\boxed{}` extraction, `verify()`
- `nvidia-utility-script.ipynb` — torch / mamba install pattern (cell 1 of combined notebook)

## Archived work

- Superseded markdown sources: [`docs/archive/streamlined-2026/`](docs/archive/streamlined-2026/README.md)
- Legacy pipeline code (if present in your tree): [`docs/archive/README.md`](docs/archive/README.md)
