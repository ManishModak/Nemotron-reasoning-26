# First Kaggle run checklist (foundation notebook)

Use [`foundation-notebook.ipynb`](../foundation-notebook.ipynb) as the only required runnable.

## Before you run

1. Create a Kaggle Notebook with these settings:
   - `Accelerator`: `GPU RTX Pro 6000`
   - `Persistence`: `Files only`
   - `Environment`: `Pin to original environment`
   - `Internet`: `On` (needed for first-time `kagglehub` if model input is not mounted)
2. **Add input data**
   - Competition: `nvidia-nemotron-3-reasoning-challenge` (or current official name) so `train.csv` is available.
   - Attach the Nemotron model input if available. The notebook prefers mounted Kaggle inputs and falls back to `kagglehub` only if no local model path is found.
   - Optional: attach **ryanholbrook/nvidia-utility-script** (or equivalent) so CUTLASS and Triton helper paths exist; the notebook skips those path injections if missing.
3. Upload the notebook file or paste cells from the repo copy into a new Kaggle notebook.

## How to run (Save & Run All)

> **Always use "Save Version -> Save & Run All (Commit)"** instead of manually running cells one-by-one. This runs the full notebook as a background job that survives browser disconnects and does not require your browser session to stay attached.

1. Verify all input data is attached.
2. For a smoke test, set `MAX_TRAIN_SAMPLES = 256` in the config cell, then commit with Save & Run All.
3. For a full training run, set `MAX_TRAIN_SAMPLES = None`, then use **Save Version -> Save & Run All (Commit)**.
4. Monitor progress from the "Version" tab and check logs for training loss and final adapter export.
5. Do not rely on mid-run resume. Full `Trainer` checkpoints are disabled because they exhausted Kaggle disk during committed runs.

## After the run completes

Confirm these files exist in the output:
- `/kaggle/working/final_adapter/adapter_config.json`
- `/kaggle/working/final_adapter/adapter_model.safetensors` or `.bin`
- `/kaggle/working/submission.zip`

Download `submission.zip` and submit to the competition.

## Capture back to this repo

- Runtime log, especially first-time installs, OOM, tokenizer errors, and any disk-space issues.
- Final `submission.zip` hash or score screenshot after leaderboard refresh.
- Any edits you made to knobs (`NUM_EPOCHS`, `MAX_SEQ_LENGTH`, LoRA rank).
- Whether the notebook found the mounted model input or fell back to `kagglehub`.
- Training loss at final step.

Update this file or add a dated report under `reports/` after the run.

## Progress prize

~~Open Progress Prize cutoff was April 9, 2026~~ - missed. Focus is now on final leaderboard placement and Open Contribution Awards.
