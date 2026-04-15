# Run A — Log Analysis

> **Date:** April 13, 2026  
> **Source:** Kaggle stdout excerpt (previously `download.txt`), notebook snapshot `[archive/run-a-notebook-kaggle-output.ipynb](archive/run-a-notebook-kaggle-output.ipynb)`, `[improved-RunA-Notebook.ipynb](improved-RunA-Notebook.ipynb)`, `[strategy_analysis.md](strategy_analysis.md)`

---

## 🔴 Critical Finding: Run A Timed Out — Training Did Not Complete

The notebook's papermill status for Cell 8 (training) is `**"status": "running"`** — it never reached `"completed"`. All subsequent cells (9–11: save adapter, zip, eval) have `"status": "pending"` with no outputs.

**This means:**

- **No adapter was saved** — no `submission.zip` was produced
- **No eval was run** — no `eval_results.json`
- **The Kaggle session expired** mid-training

---

## Training Progress Before Timeout

Training reached **step 1331 / 2250** (**59.2% complete**, epoch 1.18 / 2):


| Step | Loss          | Δ from prev      |
| ---- | ------------- | ---------------- |
| 50   | **19.94**     | — (warmup phase) |
| 100  | 4.91          | -15.03           |
| 150  | 4.32          | -0.58            |
| 200  | 3.73          | -0.59            |
| 250  | 3.87          | +0.15            |
| 300  | 3.91          | +0.04            |
| 350  | 3.59          | -0.33            |
| 400  | 3.57          | -0.02            |
| 450  | 3.40          | -0.17            |
| 500  | 3.38          | -0.02            |
| 550  | 3.37          | -0.01            |
| 600  | 3.25          | -0.11            |
| 650  | 3.15          | -0.11            |
| 700  | 3.38          | +0.24            |
| 750  | 2.85          | -0.54            |
| 800  | 3.12          | +0.27            |
| 850  | **2.69**      | -0.43            |
| 900  | 2.99          | +0.30            |
| 950  | 2.84          | -0.16            |
| 1000 | 2.86          | +0.02            |
| 1050 | 3.02          | +0.17            |
| 1100 | 2.63          | -0.40            |
| 1150 | 2.53          | -0.10            |
| 1200 | 2.55          | +0.02            |
| 1250 | **2.26**      | -0.29            |
| 1300 | **2.15**      | -0.11            |
| 1331 | *(timed out)* | —                |


### Loss Curve Observations

1. **Step 50 loss = 19.94** — extremely high initial loss. This is expected with `completion_only_loss=True` (response-only): the model is now only measured on the `\boxed{answer}` tokens (~5–10 tokens), which it has no prior knowledge of. The previous run (full-sequence loss) started at ~10 because it also counted prompt tokens the model was already decent at.
2. **Rapid convergence 50→150** — loss drops from 19.94 → 4.32 in 100 steps. The warmup (0.05 ratio ≈ 112 steps) did its job — no pathological spikes.
3. **Healthy continued decline** — loss trends from ~3.9 at step 200 down to ~2.15 at step 1300. The model is clearly still learning (not plateaued) when the session expired.
4. **Oscillation band ±0.3–0.5** — normal for batch_size=1 with grad_accum=8. The variance would shrink with larger effective batches.
5. **Comparison to Run 2:** Run 2 (full-sequence loss, 1 epoch) had loss 5.4–6.2 at plateau. The new loss ~2.15 is not directly comparable (different loss targets), but the overall trajectory is much healthier.

---

## Timing Breakdown


| Phase                  | Duration       | Notes                                      |
| ---------------------- | -------------- | ------------------------------------------ |
| TRL install            | 20s            | Offline wheel install ✅                    |
| Imports + config       | 57s            | CUTLASS path, torch, mamba_ssm             |
| Data loading           | 0.06s          | 9000 train + 500 eval from split dataset   |
| Tokenizer + formatting | 1s             | Prompt-completion format, sample printed ✅ |
| Model load + LoRA      | **7.6 min**    | 30B model, 880M trainable params (2.71%)   |
| Training (partial)     | **~11h 50min** | Reached 1331/2250 steps (59.2%)            |
| **Total elapsed**      | **~12h**       | Kaggle 12h GPU limit hit                   |


### ⏱ Speed Analysis

- **Steps completed:** 1331 in ~~11h 50min = **~~32 sec/step**
- **Total steps needed:** 2250 (9000 samples ÷ 8 grad_accum × 2 epochs)
- **Estimated full run time:** 2250 × 32s = **~20 hours**
- **Kaggle limit:** 12 hours
- **Overshoot:** ~8 hours — **the current config cannot finish in a single Kaggle session**

> [!CAUTION]
> **Run A as configured needs ~20h but Kaggle allows 12h.** This is the #1 issue to fix before rerunning.

---

## Config That Ran

```
Epochs:         2
LR:             0.0002
LR schedule:    cosine
Warmup ratio:   0.05
Optimizer:      adamw_torch  ⚠️ (strategy said adamw_8bit)
Max seq length: 512
Batch size:     1 × 8 grad accum
Response-only:  True ✅
Packing:        False ⚠️ (strategy said True, but correctly noted as incompatible)
Train samples:  9000
```

### Deviations from Strategy Plan


| Config    | Strategy Plan | Actual        | Impact                                                                  |
| --------- | ------------- | ------------- | ----------------------------------------------------------------------- |
| Optimizer | `adamw_8bit`  | `adamw_torch` | Slightly more VRAM usage, negligible quality difference                 |
| Packing   | `True`        | `False`       | **Major**: ~5-6x slower without packing → this is WHY the run timed out |


> [!IMPORTANT]
> **Root cause of timeout:** Packing was disabled because `completion_only_loss` (response-only) and `packing=True` are incompatible in TRL 1.1. The strategy expected packing would give 5-6x speedup, but without it, 2 epochs of 9000 samples at ~32s/step = 20h. The notebook comment acknowledges this: *"incompatible with DataCollatorForCompletionOnlyLM; seq len reduction compensates"* — but MAX_SEQ_LENGTH reduction from 1024 → 512 only gives 2x, not the needed 5-6x.

---

## What Went Right ✅

1. **TRL 1.1 installed successfully** offline
2. **Split dataset found** at `/kaggle/input/datasets/manishmodak/nemotron-split-data`
3. **Response-only loss is working** — the extremely high initial loss (19.94) confirms loss is only computed on completion tokens
4. **The loss is still declining** at step 1300 (2.15) — model has more to learn
5. **No crashes** — clean execution, only time limit was the issue
6. **Chat template with `enable_thinking=True`** — `<think>` tag present in sample prompt
7. **Tokenizer PAD/EOS alignment** happened automatically

## What Went Wrong ❌

1. **Session timed out at step 1331/2250** — no adapter, no eval, no submission
2. **Packing incompatibility** not addressed in advance — caused dramatic throughput loss
3. `**adamw_torch` instead of `adamw_8bit`** — minor but unnecessary VRAM cost

---

## `improved-RunA-Notebook.ipynb` vs archived Kaggle snapshot

Snapshot with committed Kaggle outputs: `[archive/run-a-notebook-kaggle-output.ipynb](archive/run-a-notebook-kaggle-output.ipynb)` (was root `run-a-notebook.ipynb`). Active notebook: `[improved-RunA-Notebook.ipynb](improved-RunA-Notebook.ipynb)`.


| Aspect              | Archived snapshot            | `improved-RunA-Notebook.ipynb` (current)                  |
| ------------------- | ---------------------------- | --------------------------------------------------------- |
| Outputs             | Has Kaggle execution outputs | Clean (no outputs)                                        |
| Duplicate eval cell | **Yes** — Cell 11 duplicated | Single eval cell                                          |
| `dataSources`       | TRL-offline + split + model  | Same pattern (attach TRL wheel + split + model on Kaggle) |
| `isInternetEnabled` | `false`                      | `false`                                                   |


---

## Actionable Next Steps

### Option A: Reduce to 1 epoch (fastest fix)

```python
NUM_EPOCHS = 1  # 1125 steps × 32s ≈ 10h → fits in 12h session
```

- **Pros:** Simplest change, guaranteed to finish
- **Cons:** Loss was clearly still declining — 1 epoch won't capture the full learning
- **Expected time:** ~10h (with margin)

### Option B: Reduce to 1 epoch + increase gradient accumulation

```python
NUM_EPOCHS = 1
GRADIENT_ACCUMULATION_STEPS = 16  # halves total steps
```

- **Steps:** 9000 ÷ 16 = 562 steps × 32s ≈ 5h
- **Pros:** Plenty of time for eval too; larger effective batch may smooth oscillation
- **Cons:** Larger effective batch with same LR may not converge as well

### Option C: Enable packing by dropping response-only loss

```python
packing = True
completion_only_loss = False  # revert to full-sequence loss
MAX_SEQ_LENGTH = 512
NUM_EPOCHS = 2
```

- **Estimated speedup:** ~3-4x with packing (fewer padded tokens)
- **Steps:** Similar count but each step processes more data
- **Cons:** Loses the response-only advantage (training on prompt tokens again)

### Option D: Checkpoint + resume across 2 sessions (robust)

```python
save_strategy = "steps"
save_steps = 500
save_total_limit = 2
```

- Run session 1: trains to step 1125 (end of epoch 1), saves checkpoint
- Download checkpoint, re-upload as Kaggle dataset
- Run session 2: `resume_from_checkpoint=True`, completes epoch 2 + eval
- **Pros:** Gets full 2-epoch training
- **Cons:** Manual checkpoint management

### 🏆 Recommended: Option A + eval time budget

```python
NUM_EPOCHS = 1                      # ~10h training
GRADIENT_ACCUMULATION_STEPS = 8     # keep current
# Budget: ~10h train + ~30min eval + ~30min save/zip = ~11h total (fits in 12h)
```

This gets a scored submission and eval baseline. Can then decide if epoch 2 is worth the checkpoint hassle based on the loss curve endpoint.