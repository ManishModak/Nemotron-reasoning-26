# Execution plan

Workspace date context: April 2026. **~65 days to final deadline (June 15).**

## Model

- **Kaggle is source of truth** for GPU memory, install quirks, timeouts, and submission behavior.
- The leaderboard runs **your LoRA** on the host's **vLLM** stack; you do not ship inference code. Align SFT targets with `\boxed{}` and the metric's user suffix ([`docs/training-data-format.md`](docs/training-data-format.md)). Doc index: [`docs/README.md`](docs/README.md).

## Primary deliverable

1. **[`foundation-notebook.ipynb`](foundation-notebook.ipynb)** — copy or upload to a Kaggle GPU notebook, run end-to-end, download **`submission.zip`**.
2. **First scored leaderboard entry** — ✅ **Done** (0.58, rank 1333). See [`reports/2026-04-12-first-scored-submission.md`](reports/2026-04-12-first-scored-submission.md).

## Timeline

| Milestone | Date |
|-----------|------|
| Competition start | March 16, 2026 |
| ~~Open Progress Prize cutoff~~ | ~~April 9, 2026~~ (missed — removed from milestones) |
| Entry / team merger deadline | June 8, 2026 |
| **Final submission deadline** | **June 15, 2026** |

## Milestones

### ~~M1 — First scored submission~~ ✅ Complete (April 12, 2026)

- Full epoch trained (1188 steps, ~6 h on RTX Pro 6000).
- Loss curve: 10.03 → ~5.9 (plateau in 5.4–6.2 range).
- Leaderboard score: **0.58** (rank 1333, top 0.85).
- See [`reports/2026-04-12-first-scored-submission.md`](reports/2026-04-12-first-scored-submission.md).

### M2 — Data and training quality (current focus)

- Local analysis on `train.csv` (puzzle families, length distribution, failure patterns).
- Improve SFT rows (optional `<redacted_thinking>`, filtering, synthetic data); keep [`docs/training-data-format.md`](docs/training-data-format.md) accurate.
- Consider data augmentation or difficulty-based sampling.
- **Increase `MAX_SEQ_LENGTH`** beyond 1024 to capture full training signal.

### M3 — Training optimization

- Tune epochs, LR, `MAX_SEQ_LENGTH`, LoRA rank (≤ 32), gradient accumulation.
- Try multi-epoch training (2–3 epochs) once single-epoch baseline score is established.
- Optional: lightweight RL / GRPO **only** after SFT is stable on Kaggle hardware.

### M4 — Open Contribution angle

- Target **Best Data** or **Best Fine-Tuning** contribution awards (DGX Spark prize each).
- Document methodology, ablation results, and insights in a shareable format.

### M5 — Final hardening (before June 15)

- Pick best adapter from experiment history, re-verify zip contents, final dry run.
- Ensure 2 submissions are marked for final leaderboard consideration.

## Kaggle run workflow ("Save & Run All")

> **Always use "Save Version → Save & Run All (Commit)"** instead of manually running cells.
> This executes the notebook as a background job that survives browser disconnects.

1. Edit the foundation notebook (or docs) locally.
2. Sync to Kaggle (paste/upload).
3. **Save Version → Save & Run All (Commit)** — full background execution.
4. After completion (or timeout with checkpoints), download logs and `submission.zip`.
5. Record results under `reports/`.
6. If interrupted, reopen → re-commit. The notebook auto-resumes from the latest checkpoint.

## Reference notebooks (do not duplicate logic in `src/` for submission)

- `nvidia-nemotron-submission-demo.ipynb` — LoRA attach + save pattern
- `nvidia-nemotron-metric.ipynb` — scoring, `\boxed{}` extraction, `verify()`
- `nvidia-utility-script.ipynb` — torch / mamba install pattern (cell 1 of foundation notebook)

## Run history

### Run 1 — April 11, 2026 (foundation run, disconnected)

- Trained to **583/1188 steps (epoch 0.49)** before disconnect.
- Loss: 17.97 → ~5.0 (clearly learning).
- Lost all progress — `save_strategy="no"`, no checkpoints.
- See [`reports/2026-04-11-foundation-export-findings.md`](reports/2026-04-11-foundation-export-findings.md).

### Run 2 — April 11–12, 2026 (first scored submission) ✅

- Full epoch: 1188/1188 steps, ~6 h 12 min on RTX Pro 6000.
- Final training loss: 6.15 (average), loss plateau 5.4–6.2.
- Leaderboard score: **0.58**, rank 1333.
- See [`reports/2026-04-12-first-scored-submission.md`](reports/2026-04-12-first-scored-submission.md).

## Archived work

- Superseded `kaggle-combined-lora-submission.ipynb`: [`archive/`](archive/)
- Superseded markdown sources: [`docs/archive/streamlined-2026/`](docs/archive/streamlined-2026/README.md)
- Legacy pipeline code (if present in your tree): [`docs/archive/README.md`](docs/archive/README.md)
