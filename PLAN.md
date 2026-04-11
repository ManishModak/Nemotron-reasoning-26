# Execution plan

Workspace date context: April 2026. **~65 days to final deadline (June 15).**

## Model

- **Kaggle is source of truth** for GPU memory, install quirks, timeouts, and submission behavior.
- The leaderboard runs **your LoRA** on the host's **vLLM** stack; you do not ship inference code. Align SFT targets with `\boxed{}` and the metric's user suffix ([`docs/training-data-format.md`](docs/training-data-format.md)). Doc index: [`docs/README.md`](docs/README.md).

## Primary deliverable

1. **[`kaggle-combined-lora-submission.ipynb`](kaggle-combined-lora-submission.ipynb)** — copy or upload to a Kaggle GPU notebook, run end-to-end, download **`submission.zip`**.
2. **First scored leaderboard entry** — follow [`reports/kaggle-first-run-checklist.md`](reports/kaggle-first-run-checklist.md); record score, errors, and knob changes in `reports/`.

## Timeline

| Milestone | Date |
|-----------|------|
| Competition start | March 16, 2026 |
| ~~Open Progress Prize cutoff~~ | ~~April 9, 2026~~ (missed — removed from milestones) |
| Entry / team merger deadline | June 8, 2026 |
| **Final submission deadline** | **June 15, 2026** |

## Milestones

### M1 — First scored submission (immediate)

- Use **"Save Version → Save & Run All (Commit)"** to run the combined notebook end-to-end as a background job on Kaggle.
- Confirm checkpoints are saved every 100 steps under `/kaggle/working/sft_checkpoints`.
- If the run times out, reopen and re-commit — auto-resume kicks in.
- Submit `submission.zip`; capture leaderboard score and runtime notes in `reports/`.

### M2 — Data and training quality

- Local analysis on `train.csv` (puzzle families, length distribution, failure patterns).
- Improve SFT rows (optional `<redacted_thinking>`, filtering, synthetic data); keep [`docs/training-data-format.md`](docs/training-data-format.md) accurate.
- Consider data augmentation or difficulty-based sampling.

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

1. Edit the combined notebook (or docs) locally.
2. Sync to Kaggle (paste/upload).
3. **Save Version → Save & Run All (Commit)** — full background execution.
4. After completion (or timeout with checkpoints), download logs and `submission.zip`.
5. Record results under `reports/`.
6. If interrupted, reopen → re-commit. The notebook auto-resumes from the latest checkpoint.

## Reference notebooks (do not duplicate logic in `src/` for submission)

- `nvidia-nemotron-submission-demo.ipynb` — LoRA attach + save pattern
- `nvidia-nemotron-metric.ipynb` — scoring, `\boxed{}` extraction, `verify()`
- `nvidia-utility-script.ipynb` — torch / mamba install pattern (cell 1 of combined notebook)

## Foundation run record (April 11, 2026)

- Trained to **583/1188 steps (epoch 0.49)** before disconnect.
- Loss: 17.97 → ~5.0 (clearly learning).
- Lost all progress — `save_strategy="no"`, no checkpoints.
- **Fixed** in current combined notebook: checkpointing every 100 steps + auto-resume.
- See [`reports/2026-04-11-foundation-export-findings.md`](reports/2026-04-11-foundation-export-findings.md).

## Archived work

- Superseded markdown sources: [`docs/archive/streamlined-2026/`](docs/archive/streamlined-2026/README.md)
- Legacy pipeline code (if present in your tree): [`docs/archive/README.md`](docs/archive/README.md)
