# Strategy Analysis: 0.58 → 0.85 (Revised v2)

> **Revision date:** April 13, 2026
> **Status:** Updated after Run A log analysis, bitsandbytes offline path, and throughput fixes.
> **Sources:** `reports/2026-04-12-first-scored-submission.md`, `[run_a_log_analysis.md](run_a_log_analysis.md)`, local `train.csv` analysis, `docs/studies/Qwopus3-5-27b-Colab_complete_guide_to_llm_finetuning.txt`, `docs/strategy-and-stack.md`, reviewer feedback.
> **Active Kaggle notebook:** `[improved-RunA-Notebook.ipynb](improved-RunA-Notebook.ipynb)` — `[foundation-notebook.ipynb](foundation-notebook.ipynb)` is kept unchanged as a safe backup.

---

## Current position


| Metric     | Value                                                                                            |
| ---------- | ------------------------------------------------------------------------------------------------ |
| Your score | **0.58** (rank 1333)                                                                             |
| Top score  | **0.85**                                                                                         |
| Gap        | 0.27                                                                                             |
| Training   | 1 epoch, plain `\boxed{answer}`, MAX_SEQ_LENGTH=1024, `transformers.Trainer`, full-sequence loss |


---

## Data-verified facts (from local train.csv analysis)

These findings correct assumptions made in the original strategy document.

### Sequence lengths are NOT a bottleneck


| Metric                                        | Value                   |
| --------------------------------------------- | ----------------------- |
| Max prompt length                             | 510 chars (~128 tokens) |
| Median prompt length                          | 281 chars (~70 tokens)  |
| Max total sequence (prompt + suffix + answer) | ~219 tokens             |
| Sequences exceeding 256 tokens                | **0**                   |
| Current MAX_SEQ_LENGTH                        | 1024                    |


> [!IMPORTANT]
> **Reasoning:** We verified this by running `df['prompt'].str.len().describe()` and estimating token counts at ~3.5–4 chars/token for Nemotron's tokenizer. Zero sequences exceed 256 tokens. The current `MAX_SEQ_LENGTH=1024` gives 4–5x headroom over every training example. Increasing it without adding longer training data (CoT traces) changes nothing — it only increases padding waste. This was incorrectly listed as a high-priority fix in the original strategy.

### Puzzle families are already perfectly balanced


| Family             | Count |
| ------------------ | ----- |
| Bit manipulation   | 1,602 |
| Gravity / physics  | 1,597 |
| Unit conversion    | 1,594 |
| Text encryption    | 1,576 |
| Numeral conversion | 1,576 |
| Equations          | 1,555 |


> [!IMPORTANT]
> **Reasoning:** Classified via keyword matching on prompt text (`"bit manipulation"`, `"gravitational constant"`, `"encryption"`, `"numeral system"`, `"unit conversion"/"converted"`, `"equation"/"operator"`). Ratio of biggest to smallest family: **1.03x**. Stratified sampling is unnecessary — the data is already near-perfectly balanced across all six families. This was incorrectly flagged as a priority in the original strategy.

---

## Critical training infrastructure problems discovered

After reviewing the Jackrong finetuning guide and comparing to the baseline training path, we identified **three major training efficiency issues** that the original strategy did not address.

### Problem 1: Full-sequence loss (training on prompt tokens)

**Current state:** The notebook uses `transformers.Trainer` with `DataCollatorForLanguageModeling(mlm=False)`. This computes loss on **every token** in the sequence — including the user prompt, chat template markers, and padding tokens. The model spends training capacity learning to reproduce puzzle text it will never generate at inference time.

**Why this matters:** At inference, the host provides the prompt via vLLM; the model only needs to generate the assistant response (`\boxed{answer}`). Training on prompt tokens is wasted gradient signal. For our data, the assistant response is only ~~10–30 tokens out of ~160 total — meaning **~~80% of the training signal is noise** from a task perspective.

**Fix:** Switch to TRL's `SFTTrainer` with `DataCollatorForCompletionOnlyLM` (response-only loss). This masks the prompt/template tokens so loss is computed only on the assistant response. The guide calls this "train on responses only" and recommends it as the default for all chat SFT.

### Problem 2: No packing (massive padding waste)

**Current state:** Each training example is padded to `MAX_SEQ_LENGTH=1024` tokens. Our data averages ~160 tokens per example.

**Why this matters:** Every batch slot wastes ~~860 tokens of padding. With `per_device_batch_size=1` and `grad_accum=8`, each gradient step processes 8 examples × 1024 slots = 8,192 token slots, but only ~1,280 tokens carry actual data. That's **~~84% wasted compute per step**.

**The efficiency gain from packing:** Packing concatenates multiple short examples into a single MAX_SEQ_LENGTH sequence with appropriate attention masking. At ~~160 tokens per example, we can fit **~~6 examples per 1024-token slot**. This effectively multiplies throughput by ~5–6x. Concretely: Run 2 took 6 hours for 1 epoch. With packing, we could potentially do **2–3 epochs in the same wall time**.

**Fix:** Enable `packing=True` in TRL's `SFTConfig`. TRL handles the attention masking correctly so packed examples don't attend to each other.

### Problem 3: Default optimizer and no warmup

**Current state:** Default AdamW optimizer (fp32 states), no warmup, linear LR schedule.

**Why this matters:** `adamw_8bit` halves optimizer memory usage with negligible quality difference for LoRA SFT. Cosine LR with warmup prevents loss spikes in early training (visible in our Run 2 loss curve: 10.03 at step 50 → 7.03 at step 100 — a very steep drop suggesting instability). Warmup ratio 0.03–0.05 stabilizes the first ~50 steps.

**Fix:** Set `optim="adamw_8bit"`, `lr_scheduler_type="cosine"`, `warmup_ratio=0.05`.

**Kaggle (no internet on accelerator):** ship a **Linux `manylinux` + Python 3.12** `bitsandbytes` wheel in the repo under `[bnb_offline/](bnb_offline/)` (same idea as `[trl_offline/](trl_offline/)`), upload that folder as a **Kaggle Dataset**, attach it to the notebook, and `pip install --no-index --find-links … bitsandbytes` in the first cell. If install fails, fall back to `optim="adamw_torch_fused"` (no extra packages; small speedup over `adamw_torch`).

---

## The 5 original SFT improvement items — re-prioritized with reasoning


| Item                            | Original priority | Revised priority                | Reasoning                                                                                                                                                                                                                                                                       |
| ------------------------------- | ----------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Response-only loss              | Not mentioned     | **#1 (critical)**               | ~80% of current training signal is prompt tokens the model never generates. This is the most direct fix for training efficiency — switches from "learn to reproduce everything" to "learn to answer correctly." Supported by Jackrong guide Section 6.2 and TRL best practices. |
| Packing                         | Not mentioned     | **#2 (critical)**               | 84% padding waste per step. 5–6x throughput gain means 2–3 epochs in the same GPU time budget. Supported by Jackrong guide and general SFT best practice for short sequences.                                                                                                   |
| Try 2–3 epochs                  | 1d (low)          | **#3 (high)**                   | Loss plateaued at 5.4–6.2 in epoch 1. With 9,500 short examples and LoRA, the model is clearly underfitting. Multi-epoch is standard for datasets this size. With packing, additional epochs are nearly free.                                                                   |
| Cosine LR + warmup + adamw_8bit | 1d (low)          | **#4 (high)**                   | Warmup prevents early instability (visible in Run 2 loss curve). Cosine schedule provides smoother decay than linear. adamw_8bit saves memory. All three are trivial config changes with no risk.                                                                               |
| Add reasoning traces (CoT)      | 1a (top)          | **#5 (after proving it helps)** | Directionally good but must be validated on a small pilot first. Generating traces for 9,500 rows costs significant API credits/time. Pilot 500 traces → compare eval accuracy → scale if positive ROI.                                                                         |
| Increase MAX_SEQ_LENGTH         | 1b (high)         | **#7 (only if needed)**         | Data fits in <256 tokens. Only matters after CoT traces are added, sized to actual trace length.                                                                                                                                                                                |
| Stratified sampling             | 1c (medium)       | **Dropped**                     | Families are balanced at 1.03x ratio. No action needed.                                                                                                                                                                                                                         |
| Data filtering                  | Listed            | **#6 (marginal)**               | With balanced families and short prompts, unclear what to filter. Revisit if per-family eval shows specific weaknesses.                                                                                                                                                         |


---

## Framework decision: TRL SFTTrainer, NOT Unsloth (for now)

### Why TRL SFTTrainer


| Factor                 | Reasoning                                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Response-only loss** | Built-in via `DataCollatorForCompletionOnlyLM`. No external dependency needed.                                                            |
| **Packing**            | Built-in via `packing=True` in `SFTConfig`. Handles attention masking correctly.                                                          |
| **Compatibility**      | TRL `SFTTrainer` extends `transformers.Trainer` — minimal migration from current notebook code. Same HF ecosystem, same PEFT integration. |
| **Kaggle environment** | TRL is pip-installable with no system dependencies. Current Kaggle images have it or can install it quickly.                              |


### Why NOT Unsloth (yet)


| Factor                  | Reasoning                                                                                                                                                                                                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model compatibility** | Nemotron-3-Nano-30B-A3B is a Mamba-based hybrid SSM architecture. Unsloth's `FastLanguageModel` targets standard transformer architectures (Llama, Mistral, Qwen, Gemma, Phi). We have **not verified** Unsloth works with Nemotron's `MambaForCausalLM` / hybrid layers. |
| **Risk**                | Rewriting the notebook around Unsloth and discovering it doesn't work on Kaggle = wasted GPU session.                                                                                                                                                                     |
| **Action plan**         | Test Unsloth compatibility LATER with `MAX_TRAIN_SAMPLES=256` smoke test. If it works, adopt for the speed gains (2x claimed). If not, TRL SFTTrainer is sufficient.                                                                                                      |


### What NOT to copy from the Jackrong guide


| Item                                         | Why skip                                                                                                                                        |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen `<think>...</think>` formatting         | Nemotron has its own chat template. Our format uses `\boxed{}` aligned with the competition metric. Qwen-specific thinking tags are irrelevant. |
| Generic mixed reasoning datasets (Opus/KIMI) | Competition puzzles are very specific; generic reasoning traces don't match the task distribution.                                              |
| Rank-64 LoRA                                 | Jackrong used rank-64 for Qwen-27B on A100/H100. Our LoRA rank 32 is already appropriate for Nemotron on Kaggle's RTX Pro 6000 VRAM budget.     |
| Large context settings (32768)               | Our data is <256 tokens. Even with CoT traces, 2048 is sufficient.                                                                              |
| WandB / Google Drive patterns                | Kaggle environment doesn't use Colab Drive mounts. Report to "none" for now.                                                                    |
| GGUF export                                  | We submit LoRA adapters, not merged models.                                                                                                     |


### What TO extract from the Jackrong guide


| Item                                                    | Application                                      | Guide reference                           |
| ------------------------------------------------------- | ------------------------------------------------ | ----------------------------------------- |
| TRL `SFTTrainer` + `SFTConfig`                          | Replace `transformers.Trainer`                   | Section 6.1                               |
| Response-only loss (`train_on_responses_only`)          | Mask prompt tokens                               | Section 6.2                               |
| Packing                                                 | Eliminate padding waste                          | Section 6.1 (`SFTConfig` packing support) |
| `adamw_8bit` optimizer                                  | Memory savings                                   | Section 6.1 training config               |
| `warmup_ratio=0.03–0.05`                                | Training stability                               | Section 6.1                               |
| Post-template format validation                         | Verify chat template correctness before training | Section 5.3                               |
| Label sanity check (decode labels, verify -100 masking) | Confirm response-only masking works              | Section 6.3                               |


---

## Revised execution plan (v3 — post Run A timeout)

### Principles

1. **Checkpointing on every run.** Never plan a run without a safety net. With `adamw_8bit`, each checkpoint is ~3–4 GB on disk (vs ~9 GB with full AdamW states); the cost of a timeout without any save is still a wasted 12h session.
2. **1 epoch default; 2 epochs optional in one session** when step count is halved (`per_device_train_batch_size=2`, `gradient_accumulation_steps=4`, `adamw_8bit`) — see Run A time budget. Otherwise use checkpointed multi-session.
3. **Measure every run.** Eval on 500 held-out rows after every training run.
4. **No fragile time estimates.** If a run *might* not fit in 12h, treat it as *won't fit* and plan for multi-session.

---

### Checkpointing — built into all runs

**Why:** Run A timed out at step 1331/2250 (59%) — see `[run_a_log_analysis.md](run_a_log_analysis.md)`. Without checkpointing, 12 hours of GPU time produced nothing — no adapter, no eval, no submission. This must never happen again.

**What gets saved** in each checkpoint folder (`checkpoint-XXX/`):


| File                        | Size (typical)                                 | Purpose                              |
| --------------------------- | ---------------------------------------------- | ------------------------------------ |
| `adapter_model.safetensors` | ~1.7 GB                                        | LoRA weights (only trainable params) |
| `optimizer.pt`              | ~1.8 GB (`adamw_8bit`) / ~7 GB (`adamw_torch`) | Optimizer state                      |
| `scheduler.pt`              | ~1 KB                                          | LR scheduler position                |
| `rng_state.pth`             | ~1 KB                                          | RNG state for reproducibility        |
| `trainer_state.json`        | ~10 KB                                         | Step counter, loss history           |
| **Total per checkpoint**    | **~3.5 GB** / **~8.7 GB**                      |                                      |


**Disk budget** (Kaggle allows 20 GB output; prefer `adamw_8bit` for smaller checkpoints):

```
1 checkpoint (save_total_limit=1, adamw_8bit):  ~3.5 GB
Final adapter:                                ~1.7 GB
submission.zip:                               ~1.6 GB
eval_results.json:                            ~0.01 GB
──────────────────────────────────────────────────────
Total:                                        ~6.8 GB  ✅ fits in 20 GB
```

**bitsandbytes offline (required for `adamw_8bit` without internet):**

1. Keep wheels in repo: `[bnb_offline/](bnb_offline/)` (see `[bnb_offline/README.md](bnb_offline/README.md)`).
2. Create a Kaggle **Dataset** from that folder and **Add Data** on the notebook.
3. Install in the notebook: `pip install --no-index --find-links <path-under-/kaggle/input/> bitsandbytes` (Linux + Python 3.12 wheels). Fallback optimizer: `adamw_torch_fused` (no extra wheel).

**Config (same for all runs):**

```python
sft_config = SFTConfig(
    # ... training params ...
    save_strategy="steps",
    save_steps=250,           # with ~562 steps/epoch, mid-epoch + late saves
    save_total_limit=1,
    optim="adamw_8bit",       # after offline bitsandbytes; else adamw_torch_fused
    ignore_data_skip=True,    # fast resume without dataloader fast-forward stall
)
```

**Multi-session resume workflow:**

```
Session 1 → trains, saves checkpoint-* → times out or completes
         → Kaggle auto-commits notebook output
         → You add committed output as input dataset to next session (Add Data → Your Notebooks / Datasets)
Session 2 → copies checkpoint from /kaggle/input/ to /kaggle/working/
         → trainer.train(resume_from_checkpoint=...) 
         → continues from exact step, correct LR position, full optimizer state
```

**Kaggle UI (session 2+):** Attach the artifact that contains `checkpoint-`*. Paths appear under `/kaggle/input/...`.

**Before `trainer.train`:** Confirm `trainer_state.json` exists and `global_step` in that file matches the checkpoint folder name if you care about sanity-checking.

**Resume code (auto-detects checkpoint, works for both session 1 and 2):**

```python
import shutil
from pathlib import Path

# Look for checkpoint in previous session's output (added as input dataset)
resume_path = None
prev_output = Path("/kaggle/input")
for p in sorted(prev_output.rglob("checkpoint-*"), reverse=True):
    if (p / "trainer_state.json").exists():
        # Copy to working dir (can't resume from read-only /kaggle/input)
        dest = Path("/kaggle/working") / p.name
        if not dest.exists():
            shutil.copytree(str(p), str(dest))
        resume_path = str(dest)
        print(f"Resuming from: {resume_path}")
        break

if not resume_path:
    print("No checkpoint found — training from scratch")

trainer.train(resume_from_checkpoint=resume_path)
```

> [!CAUTION]
> **Critical rules for multi-session resume:**
>
> - **Never change `num_train_epochs`, `learning_rate`, or `optim` between sessions.** The optimizer/scheduler state in the checkpoint assumes identical config.
> - **Always copy checkpoint to `/kaggle/working/`** — HF Trainer cannot resume from read-only `/kaggle/input/`.
> - **Delete the copied checkpoint after training completes** to free disk for the final adapter + zip.
> - Set `**ignore_data_skip=True`** in `SFTConfig` / `TrainingArguments` so resume does not spend a long time skipping dataloader batches.

---

### Optional: Liger Kernel (TRL)

If `pip install liger-kernel` is acceptable (or you vendor wheels like bitsandbytes), TRL supports `use_liger_kernel=True` in `SFTConfig` for fused kernels and possible extra headroom. Treat as an experiment after the baseline run is stable.

---

### Step 0: Eval split — pre-split locally, uploaded as Kaggle dataset

The split was performed **locally** using `sklearn.model_selection.train_test_split` with `random_state=42` and stratified by puzzle family. The resulting CSVs are committed to `datasets/` and uploaded to Kaggle as a custom dataset.

**Pre-split files (already generated):**


| File                                                 | Rows  | Purpose                                                                 |
| ---------------------------------------------------- | ----- | ----------------------------------------------------------------------- |
| `[datasets/train_9000.csv](datasets/train_9000.csv)` | 9,000 | Training data — CoT rows are selected from here                         |
| `[datasets/eval_500.csv](datasets/eval_500.csv)`     | 500   | Held-out eval — **never** trained on, **never** used for CoT generation |


Eval family distribution (balanced): bit 84, gravity 84, unit 84, numeral 83, encrypt 83, equation 82.

> [!WARNING]
> **The 500 eval rows are NEVER trained on and NEVER sent to teacher models for CoT.** They are your permanent ruler for comparing experiments.

---

### Run A — SFT baseline (1 epoch, plain data) 🏁

**Goal:** Get a scored submission + eval baseline with response-only loss. Checkpoint for safety.

**What changed vs the timed-out Run A:**


| Config        | Timed-out Run A          | This Run A                                                                              |
| ------------- | ------------------------ | --------------------------------------------------------------------------------------- |
| Epochs        | 2 (timed out at 59%)     | **1** default; **2** fits one session if steps halved (below)                           |
| Checkpointing | None                     | `**save_steps=250`, `save_total_limit=1`**                                              |
| Batch / accum | 1 × 8 (1125 steps/epoch) | **2 × 4** (562 steps/epoch, same effective batch = 8)                                   |
| Packing       | False                    | False (TRL 1.1 incompatible with response-only)                                         |
| Response-only | True ✅                   | True ✅                                                                                  |
| Optimizer     | adamw_torch              | `**adamw_8bit`** after `[bnb_offline/](bnb_offline/)` install; else `adamw_torch_fused` |


```python
args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,                          # or 2 if 562 steps/epoch (~5h/ep @ 32s/step)
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    optim="adamw_8bit",
    logging_steps=50,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    ignore_data_skip=True,
    bf16=True,
    report_to="none",
    completion_only_loss=True,
    max_length=512,
    packing=False,
)
```

**Time budget** (~32 s/step; measured in [run_a_log_analysis.md](run_a_log_analysis.md)):


| Phase                                   | Time (batch 2×4, 562 steps/epoch) | Time (batch 1×8, 1125 steps/epoch) |
| --------------------------------------- | --------------------------------- | ---------------------------------- |
| TRL + bnb wheels + imports + model load | ~10 min                           | ~10 min                            |
| Training 1 epoch                        | ~562×32s ≈ **5.0h**               | ~1125×32s ≈ **10.0h**              |
| Save adapter + zip                      | ~3 min                            | ~3 min                             |
| Eval (500 samples)                      | ~25–40 min                        | ~25–40 min                         |
| **Total (1 epoch)**                     | **~6–6.5h**                       | **~10.5–11h**                      |


**Outputs:** `submission.zip` + `eval_results.json` (score = X%)

> [!TIP]
> With checkpointing, a timeout mid-run still yields a `checkpoint-*` folder under `/kaggle/working/`. Commit, attach as input, resume — nothing is lost. Use `ignore_data_skip=True` on resume.

---

### Run B — CoT pilot (1 epoch, prove traces help before scaling)

**Goal:** Test if 500 reasoning traces improve accuracy over plain `\boxed{answer}` training.

**Prep (locally, before Kaggle run):**

1. Pick 500 rows from `datasets/train_9000.csv` (safe — no eval rows in this file)
2. Send each puzzle to a teacher model (GPT-4o, Claude, etc.) for step-by-step reasoning
3. **Filter:** Keep only traces where the teacher's final answer matches ground truth
4. Format as training data with reasoning before `\boxed{}`
5. Upload as Kaggle dataset

**Kaggle run config:**

```
Training data:    8,500 plain + 500 CoT-traced = 9,000 total
Epochs:           1
MAX_SEQ_LENGTH:   2048 (needed for CoT trace length)
Checkpointing:    save_steps=250, save_total_limit=1
Everything else:  same as Run A (batch 2×4, adamw_8bit when available)
```

**Time estimate:** With batch_size=1, only the 500 CoT samples (~~400–600 tokens) are slower than Run A's ~160-token samples. ~94% of steps run at the same speed. With Run A throughput settings, budget **~~6.5h** (short CoT mix) to **~11h** (if many long traces); checkpointing covers variance.

**Decision gate (compare to Run A):**


| Outcome                                     | Action                                 |
| ------------------------------------------- | -------------------------------------- |
| Y > X by ≥3% (≥15 more correct on 500 eval) | CoT helps → scale traces               |
| Y ≈ X (within 2%)                           | Skip CoT → go directly to RL           |
| Y < X                                       | CoT hurts → debug format/quality first |


---

### Run C — Scale CoT (only if Run B proves value) ⚠️ REQUIRES MULTI-SESSION

> [!IMPORTANT]
> **Full CoT (all 9000 traced samples, ~500 tokens avg) in 1 epoch takes ~25–31 hours.** This exceeds the 12h Kaggle limit regardless of other optimizations. Multi-session checkpointing is mandatory.

**Two approaches:**

#### Option C1: Curated subset (~3000–4000 best traces)

Instead of tracing all 9000 rows, select the highest-quality 3000–4000 traces (highest teacher confidence, cleanest reasoning). Mix with ~5000–6000 plain samples.

- **1 epoch time:** ~14–16h → needs 2 sessions with checkpointing
- **Pros:** Better quality per GPU-hour; less noise from bad traces
- **Cons:** Still requires multi-session

#### Option C2: Full 9000 traces

- **1 epoch time:** ~25–31h → needs 3 sessions with checkpointing
- **Pros:** Maximum coverage
- **Cons:** More sessions, more traces with wrong reasoning

**Multi-session workflow for Run C:**

```
Session C-1: Train steps 0–1000 → checkpoint saved → commit
Session C-2: Resume from checkpoint → train steps 1000–2000 → checkpoint → commit 
Session C-3 (if needed): Resume → finish epoch → eval → save adapter + zip
```

---

### Run D — RL / GRPO (separate session, best SFT adapter as starting point)

> [!IMPORTANT]
> **RL cannot share a session with SFT.** GRPO requires generating multiple completions per sample during training (~120–240s/step), making it 4–8x slower per step than SFT. It needs its own dedicated session(s).

**When to attempt:** After best SFT run, regardless of exact accuracy. Even at 0.58 we can test if RL adds value on a small subset.

**Config:**

```
Base model:       Nemotron-3-Nano-30B-A3B + best SFT LoRA adapter
RL method:        TRL GRPOTrainer
Training samples: 500–1000 (small — GRPO is expensive)
Generations:      4 per sample, temperature=0.7
Reward:           exact-match binary (correct=1, wrong=0)
Checkpointing:    save_steps=100 (GRPO steps are much slower)
```

**Time budget for GRPO:**


| Subset       | Steps      | Estimated time | Fits 12h?       |
| ------------ | ---------- | -------------- | --------------- |
| 500 samples  | 62 steps   | ~3–4h          | ✅               |
| 1000 samples | 125 steps  | ~6–8h          | ✅               |
| Full 9000    | 1125 steps | ~37–75h        | ❌ multi-session |


**Decision gate:** If RL on 500 samples improves eval by ≥2%, scale to 1000. If 1000 improves further, consider multi-session for larger subsets.

---

### Run E — Unsloth compatibility test (optional, parallel)

**Goal:** Determine if Unsloth works with Nemotron-3-Nano-30B-A3B on Kaggle.

1. Set `MAX_TRAIN_SAMPLES=256` for a quick test
2. Try loading model via `FastLanguageModel.from_pretrained()`
3. If it fails on model architecture → Unsloth is not compatible → continue with TRL
4. If it works → adopt for speed gains (2x claimed), rerun best config

**Reasoning:** Low-cost smoke test (<10 min). Mamba-hybrid may not be supported.

---

### Final submission run

1. Best config from experiments (SFT, or SFT+CoT, or SFT+CoT+RL)
2. `USE_EVAL = False` → train on all 9,500 rows
3. Checkpointing still enabled (never turn it off)
4. If multi-epoch is the best config → multi-session with checkpointing
5. Submit adapter

---

### Decision tree (visual)

```
Run A (1ep plain SFT) → score = X%
  │
  ├─→ Run B (1ep, 500 CoT pilot) → score = Y%
  │     │
  │     ├── Y > X+3%? → CoT helps → Run C (scale, multi-session)
  │     └── Y ≈ X?   → skip CoT
  │
  └─→ Run D (GRPO on best SFT adapter, separate session)
        │
        ├── RL improves? → scale RL (multi-session if needed)
        └── RL no help?  → focus on data quality / more SFT epochs

Final: best config, USE_EVAL=False, all 9500 rows
```

---

## The eval cell — what it does in the notebook

After training completes, add a cell that runs **only if `eval_df is not None`** (i.e., `USE_EVAL = True`; skipped entirely for final submission):

```python
if eval_df is not None:
    import json, re
    from pathlib import Path

    results = {"overall": {}, "per_family": {}, "failures": []}
    correct, total = 0, 0
    family_stats = {}  # family -> {correct: int, total: int}

    for _, row in eval_df.iterrows():
        # 1. Format prompt with boxed suffix (same as competition)
        # 2. Generate answer using the just-trained model (still in GPU memory)
        # 3. Extract \boxed{} content from generated text
        # 4. Compare to ground truth
        # 5. Track per-family accuracy
        # 6. Save first few failures for debugging
        pass  # (implementation in notebook)

    # Print structured results
    print("="*60)
    print(f"EVAL RESULTS ({len(eval_df)} held-out rows)")
    print("="*60)
    print(f"Overall accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print("\nPer-family breakdown:")
    for fam, stats in sorted(family_stats.items()):
        pct = 100*stats['correct']/stats['total']
        print(f"  {fam:20s}: {stats['correct']}/{stats['total']} ({pct:.1f}%)")
    print("="*60)

    # Save eval_results.json alongside submission.zip
    results["overall"] = {"correct": correct, "total": total, "accuracy": correct/total}
    results["per_family"] = family_stats
    eval_json_path = Path(OUTPUT_DIR) / "eval_results.json"
    with open(eval_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved eval results to {eval_json_path}")
else:
    print("Eval disabled (USE_EVAL=False). Skipping eval.")
```

### eval_results.json — safety measure for persistent results

**What it is:** A JSON file saved to `/kaggle/working/eval_results.json` containing accuracy numbers, per-family breakdown, and sample failures.

**Why it's needed:** Notebook cell outputs are visible in the committed version but get overwritten on the next commit run. The JSON file persists as a downloadable notebook artifact alongside `submission.zip`.

**No conflict with submission.zip:** The zip command explicitly zips only `adapter_config.json` + `adapter_model.safetensors` from `final_adapter/`. The `eval_results.json` sits at `/kaggle/working/eval_results.json` — completely separate.

**After each run, you download two things:**

1. `submission.zip` → submit to leaderboard
2. `eval_results.json` → copy numbers into `reports/` for permanent tracking

**Reasoning:** This adds ~25–40 minutes to the Kaggle run (500 inferences × ~3–5 sec each on a 30B model with `max_new_tokens=128`, greedy decoding). The diagnostic value is enormous: you get an accuracy number before submitting to the leaderboard, and per-family breakdown tells you where the model struggles.

---

## What NOT to do (with reasoning)


| Anti-pattern                                             | Why it wastes time                                                                                                                                                                           |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Plan a run without checkpointing**                     | Run A proved this: 12h timeout with no checkpoint = 12h wasted. Always `save_strategy="steps"`, `save_steps=250`, `save_total_limit=1`.                                                      |
| **Assume a run fits in 12h based on estimates**          | Step times vary. Any run estimated at >10h should be treated as multi-session. Checkpointing handles this automatically.                                                                     |
| **Use `adamw_8bit` without offline wheels**              | On no-internet accelerators, vendor `**bnb_offline/`** (or equivalent) and `pip install --no-index`. If that fails, use `**adamw_torch_fused`** (not plain `adamw_torch`) until wheels work. |
| Generate CoT for all 9,000 rows before proving they help | API cost with uncertain ROI. Pilot 500 traces first, compare eval accuracy, then scale.                                                                                                      |
| Combine SFT + RL in the same session                     | GRPO is 4–8x slower per step (generation at each step). SFT and RL need separate sessions.                                                                                                   |
| Increase MAX_SEQ_LENGTH without adding longer data       | Current data fits in <256 tokens. Only increase when adding CoT traces.                                                                                                                      |
| Stratify by puzzle family                                | Already balanced at 1.03x ratio.                                                                                                                                                             |
| Train on generic reasoning datasets as-is                | Wrong task format — puzzles require specific deduction.                                                                                                                                      |
| Jump to multi-epoch without checkpointing                | 2 epochs = ~20h (plain) to ~50h (CoT). Always multi-session.                                                                                                                                 |
| Copy Qwen-specific `<think>` tag formatting              | Nemotron has its own chat template. Use `\boxed{}` aligned with `training-data-format.md`.                                                                                                   |


---

## Concrete action checklist

```
Immediate (locally):
  ✅ Analyze train.csv — families balanced, seq length is fine
  ✅ Pre-split train.csv → datasets/train_9000.csv + datasets/eval_500.csv (stratified, seed=42)
  ✅ Upload split dataset to Kaggle
  ✅ Rewrite training cell: Trainer → SFTTrainer + response-only
  ✅ Add data loading cell: USE_EVAL flag
  ✅ Add eval cell (per-family accuracy + eval_results.json)
  [x] ADD CHECKPOINTING to improved notebook (save_steps=250, save_total_limit=1)
  [x] ADD RESUME LOGIC (auto-detect checkpoint from previous session)
  [x] OPTIMIZER: adamw_8bit + offline bitsandbytes in `bnb_offline/` (fallback: adamw_torch_fused)
  [x] THROUGHPUT: per_device_train_batch_size=2, gradient_accumulation_steps=4
  [ ] SET NUM_EPOCHS=1 or 2 (2 fits ~12h only with halved steps — see Run A time table)

Run A (1 epoch SFT baseline — next Kaggle run):
  [ ] Upload `bnb_offline/` as Kaggle dataset + attach to `[improved-RunA-Notebook.ipynb](improved-RunA-Notebook.ipynb)`
  [ ] Train 1 epoch (~6.5h with 2×4 batching, or ~10h with 1×8) with response-only loss + checkpoint safety
  [ ] Eval on 500 held-out rows (~25-40 min)
  [ ] Download eval_results.json + submission.zip
  [ ] Copy eval accuracy into reports/
  [ ] Submit submission.zip → record leaderboard score = X%

Run B (1 epoch CoT pilot — separate session):
  [ ] Pick 500 rows from train_9000.csv for CoT generation
  [ ] Generate CoT traces via teacher model (locally)
  [ ] Filter: keep only matching traces
  [ ] Upload traced data as Kaggle dataset
  [ ] Train 1 epoch: 8,500 plain + 500 traced, MAX_SEQ_LENGTH=2048
  [ ] Eval → score = Y%
  [ ] Decision: Y > X+3%? → scale CoT. Y ≈ X? → skip CoT.

Run C (CoT scaling — MULTI-SESSION with checkpointing):
  [ ] Generate traces for best 3000-4000 rows (curated, not all 9000)
  [ ] Session C-1: train steps 0 → ~1000, checkpoint, commit
  [ ] Session C-2: resume, finish epoch, eval, save adapter
  [ ] (Optional session C-3 if attempting 2 epochs)

Run D (RL / GRPO — dedicated session):
  [ ] Load best SFT adapter
  [ ] GRPO on 500-1000 samples (fits in ~4-8h)
  [ ] Eval → does RL improve over best SFT?
  [ ] If yes → scale RL (multi-session with checkpointing)

Run E (Unsloth smoke test — optional, parallel):
  [ ] MAX_TRAIN_SAMPLES=256, test FastLanguageModel compatibility
  [ ] If works: adopt for speed. If not: stay on TRL.

Final submission:
  [ ] Best config, USE_EVAL=False, all 9,500 rows
  [ ] Checkpointing still ON (multi-session if best config needs it)
  [ ] 2 submissions marked for final leaderboard
```

---

## Summary of key decisions and their reasoning


| Decision                                                   | Reasoning                                                                                                                                                                      |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Checkpointing on all runs**                              | Run A timeout proved: no checkpoint → 12h wasted. With `adamw_8bit`, checkpoints are ~3.5 GB each — still use `save_total_limit=1`.                                            |
| **1 epoch default; 2 epochs in-session when steps halved** | At ~32 s/step (see [run_a_log_analysis.md](run_a_log_analysis.md)), 562 steps/epoch ≈ 5 h/epoch with batch 2×4 + `adamw_8bit`; 1125 steps/epoch ≈ 10 h with batch 1×8. |
| **`adamw_8bit` + offline `bnb_offline/`** | Frees VRAM for **batch 2 × accum 4**, halving steps vs 1×8. Fallback: **`adamw_torch_fused`** if bitsandbytes install fails. |
| **Packing disabled (TRL 1.1 limitation)**                  | `packing=True` and `completion_only_loss=True` are incompatible in TRL 1.1. Response-only loss is more valuable than packing throughput. Future TRL versions may resolve this. |
| TRL SFTTrainer over Unsloth                                | Unsloth compatibility with Nemotron (Mamba SSM) is unverified. TRL provides response-only loss with no compatibility risk.                                                     |
| Response-only loss as #1 quality lever                     | ~80% of current training signal is prompt tokens the model never generates. Loss dropped from 10→2.15 with response-only.                                                      |
| 500-row eval split                                         | Small enough to not reduce training signal (5%), large enough for meaningful accuracy (~83 per family).                                                                        |
| CoT pilot before full generation                           | API cost control. 500 traces is enough to detect ≥3% accuracy improvement.                                                                                                     |
| RL in separate session                                     | GRPO is 4–8x slower per step (generation overhead). Cannot share a session with SFT training.                                                                                  |
| Exact-match reward for GRPO                                | Puzzles have single correct answers. Binary reward matches the competition metric exactly.                                                                                     |


---

## Resources

### Local study materials


| Resource                         | Path                                                                                                                              | Key sections                                                                                    | When to use              |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------ |
| Jackrong finetuning guide (text) | [Qwopus3-5-27b-Colab_complete_guide_to_llm_finetuning.txt](docs/studies/Qwopus3-5-27b-Colab_complete_guide_to_llm_finetuning.txt) | §6.1 Trainer config, §6.2 Response-only loss, §6.3 Label sanity check, §5.3 Template validation | Run A notebook rewrite   |
| Jackrong finetuning guide (PDF)  | [Qwopus3-5-27b-Colab_complete_guide_to_llm_finetuning.pdf](docs/studies/Qwopus3-5-27b-Colab_complete_guide_to_llm_finetuning.pdf) | Same as above, formatted version                                                                | Reference                |
| Jackrong notebook (code)         | [Qwopus3-5-27b-Colab.ipynb](docs/studies/Jackrong-llm-finetuning-guide/train_code/Qwopus3-5-27b-Colab.ipynb)                      | Working Unsloth + TRL SFTTrainer + response-only training implementation                        | Code reference for Run A |
| Jackrong repo (full)             | [Jackrong-llm-finetuning-guide/](docs/studies/Jackrong-llm-finetuning-guide/)                                                     | Data pipeline, dataset loading, normalization functions                                         | Data processing patterns |
| Composer 2 report                | [a-technical-report-on-composer2.txt](docs/studies/a-technical-report-on-composer2.txt)                                           | SFT → RL architecture pattern, realistic envs improve RL                                        | Background understanding |
| Real-time RL (Cursor)            | [real-time-rl-cursor.txt](docs/studies/real-time-rl-cursor.txt)                                                                   | On-policy RL, reward hacking pitfalls, fast iteration                                           | Run E (GRPO design)      |


### Repo docs


| Resource                       | Path                                                                                           | Purpose                                                          |
| ------------------------------ | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Training data format           | [training-data-format.md](docs/training-data-format.md)                                        | `\boxed{}` alignment with competition metric — must stay in sync |
| Strategy and stack             | [strategy-and-stack.md](docs/strategy-and-stack.md)                                            | Full tool stack reference, Unsloth/Axolotl/NeMo positioning      |
| Competition rules              | [competition-rules.md](docs/competition-rules.md)                                              | Submission format, scoring, timeline, prizes                     |
| NeMo tools reference           | [reference/nemo-tools.md](docs/reference/nemo-tools.md)                                        | NeMo Curator, DataDesigner, RL, Gym links                        |
| Foundation notebook (backup)   | [foundation-notebook.ipynb](foundation-notebook.ipynb)                                         | Frozen baseline; do not change for Kaggle runs                   |
| Run A notebook (active)        | [improved-RunA-Notebook.ipynb](improved-RunA-Notebook.ipynb)                                   | TRL SFTTrainer, eval split, checkpointing — sync to Kaggle       |
| Run A log analysis             | [run_a_log_analysis.md](run_a_log_analysis.md)                                                 | Timeout steps, loss curve, timing (~32 s/step)                   |
| bitsandbytes wheels            | [bnb_offline/](bnb_offline/)                                                                   | Offline install on no-internet Kaggle; upload as dataset         |
| First scored submission report | [reports/2026-04-12-first-scored-submission.md](reports/2026-04-12-first-scored-submission.md) | Run 2 results, loss curve, config details                        |


### External references


| Resource               | URL                                                                                                                                                   | When to use                                      |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| TRL SFTTrainer docs    | [huggingface.co/docs/trl/sft_trainer](https://huggingface.co/docs/trl/sft_trainer)                                                                    | Packing, completion-only loss, SFTConfig API     |
| TRL GRPO LoRA notebook | [github.com/huggingface/trl/.../grpo_trl_lora_qlora.ipynb](https://github.com/huggingface/trl/blob/main/examples/notebooks/grpo_trl_lora_qlora.ipynb) | GRPO implementation reference                    |
| Unsloth repo           | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)                                                                                  | Check model compatibility list for Nemotron      |
| Your `train.csv`       | Local (9,500 rows)                                                                                                                                    | Primary gold data — competition-specific puzzles |


