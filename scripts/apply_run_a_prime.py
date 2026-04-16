"""Apply Run A-prime edits to improved-RunA-Notebook.ipynb."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "improved-RunA-Notebook.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

by_id = {c.get("id"): i for i, c in enumerate(nb["cells"]) if c.get("id")}

def cell_src(cid: str) -> str:
    return "".join(nb["cells"][by_id[cid]]["source"])


def set_cell(cid: str, new_src: str) -> None:
    nb["cells"][by_id[cid]]["source"] = new_src.splitlines(keepends=True)


# --- config_imports ---
cfg = cell_src("config_imports")
cfg = cfg.replace(
    "# TRAINING CONFIG — Run A (infrastructure overhaul)\n",
    "# TRAINING CONFIG — Run A-prime (packing + multi-epoch + best checkpoint + longer eval gen)\n",
)
cfg = cfg.replace(
    "NUM_EPOCHS = 1                  # 2 fits ~12h with batch 2x4 (~562 steps/ep @ ~32s/step); old 2x1x8 timed out\n",
    "NUM_EPOCHS = 3                  # set to 2 if session time tight; packing lowers steps vs unpadded\n",
)
if "EVAL_STEPS = " not in cfg:
    cfg = cfg.replace(
        "SAVE_STEPS = 250\n",
        "SAVE_STEPS = 250                # used when USE_EVAL=False (step saves only)\n"
        "EVAL_STEPS = 250                # trainer eval + best-checkpoint cadence when USE_EVAL=True\n",
    )
cfg = cfg.replace(
    "MAX_SEQ_LENGTH = 512            # reduced from 1024 (data <256 tokens; cuts padding 2x)\n",
    "MAX_SEQ_LENGTH = 1024           # headroom for packed batches; plain rows still short\n"
    "EVAL_MAX_NEW_TOKENS = 2048      # post-train eval generate (host allows up to ~7680)\n",
)
set_cell("config_imports", cfg)

# --- tokenizer_fmt: add eval_ds ---
tok = cell_src("tokenizer_fmt")
marker = 'train_ds = Dataset.from_dict({"prompt": prompts, "completion": completions})\n\nprint(f"Training dataset:'
insert = '''train_ds = Dataset.from_dict({"prompt": prompts, "completion": completions})

eval_ds = None
if eval_df is not None:
    _ep = [format_prompt(r.prompt) for r in eval_df.itertuples(index=False)]
    _ec = [format_completion(r.answer) for r in eval_df.itertuples(index=False)]
    eval_ds = Dataset.from_dict({"prompt": _ep, "completion": _ec})
    print(f"Trainer eval dataset: {len(eval_ds)} examples (held-out; eval_loss + best checkpoint)")

print(f"Training dataset:'''
if "eval_ds = None" not in tok:
    if marker not in tok:
        raise SystemExit("tokenizer_fmt marker not found")
    tok = tok.replace(marker, insert, 1)
set_cell("tokenizer_fmt", tok)

# --- training cell ---
tr = cell_src("training")
old_sft = """# ─── SFTConfig (replaces TrainingArguments) ───
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,    # cosine (was linear)
    warmup_ratio=WARMUP_RATIO,              # 0.05 (was 0)
    optim=OPTIM,
    logging_steps=50,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=1,
    ignore_data_skip=True,
    bf16=True,
    report_to="none",
    dataset_text_field=None,             # Not used for prompt-completion format
    completion_only_loss=True,           # TRL 1.1 native response-only loss
    max_length=MAX_SEQ_LENGTH,
    packing=False,   # incompatible with DataCollatorForCompletionOnlyLM; seq len reduction compensates
)

# ─── Create trainer ───
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    processing_class=tokenizer,
)
"""

new_sft = """# ─── SFTConfig (Run A-prime: packing + optional best checkpoint on eval_loss) ───
_sft_common = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    optim=OPTIM,
    logging_steps=50,
    ignore_data_skip=True,
    bf16=True,
    report_to="none",
    dataset_text_field=None,
    completion_only_loss=True,
    max_length=MAX_SEQ_LENGTH,
    packing=True,
)

if eval_ds is not None:
    sft_config = SFTConfig(
        **_sft_common,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        per_device_eval_batch_size=1,
        save_strategy="best",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=1,
    )
else:
    sft_config = SFTConfig(
        **_sft_common,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=1,
    )

# ─── Create trainer ───
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)
"""

if "eval_ds is not None:" not in tr or "_sft_common" not in tr:
    if old_sft not in tr:
        raise SystemExit("training cell SFT block not found or already patched differently")
    tr = tr.replace(old_sft, new_sft, 1)

tr = tr.replace(
    'print("TRAINING CONFIG (Run A — Infrastructure Overhaul)")\n',
    'print("TRAINING CONFIG (Run A-prime)")\n',
)
tr = tr.replace(
    'print(f"  Packing:        False")\n',
    'print(f"  Packing:        True")\n',
)
tr = tr.replace(
    'print(f"  Save:           steps={SAVE_STEPS}, total_limit=1")\n',
    'print(f"  Save:           best-on-eval_loss (USE_EVAL=True) else steps={SAVE_STEPS}, total_limit=1")\n',
)

set_cell("training", tr)

# --- eval cell: max_new_tokens ---
ev = cell_src("eval_cell")
ev = ev.replace(
    "                max_new_tokens=128,\n",
    "                max_new_tokens=EVAL_MAX_NEW_TOKENS,\n",
)
set_cell("eval_cell", ev)

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("Wrote", NB)
