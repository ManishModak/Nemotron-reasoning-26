#!/usr/bin/env python3
"""
Generate the updated foundation-notebook.ipynb with Run A infrastructure changes.

Changes from Run 2:
  1. transformers.Trainer → TRL SFTTrainer + SFTConfig
  2. Response-only loss via DataCollatorForCompletionOnlyLM
  3. 2 epochs (was 1)
  4. Cosine LR + warmup (was linear, no warmup)
  5. adamw_8bit optimizer (was default adamw)
  6. MAX_SEQ_LENGTH 512 (was 1024 — data fits in <256 tokens)
  7. USE_EVAL flag + eval-aware data loading from pre-split CSVs
  8. Label sanity check cell (verify -100 masking)
  9. Per-family eval cell with eval_results.json output
"""

import json
import uuid
from pathlib import Path


def make_cell(source: str, cell_type: str = "code", cell_id: str | None = None) -> dict:
    """Create a notebook cell dict from a source code string."""
    # Split into lines and add \n to each line except the last
    lines = source.split("\n")
    formatted = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            formatted.append(line + "\n")
        else:
            formatted.append(line)
    # Remove trailing empty string if source ends with newline
    if formatted and formatted[-1] == "":
        formatted.pop()
        if formatted:
            # Remove the trailing \n from what is now the last line
            formatted[-1] = formatted[-1].rstrip("\n")

    return {
        "cell_type": cell_type,
        "execution_count": None,
        "id": cell_id or uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": formatted,
    }


# ════════════════════════════════════════════════════════════════
# CELL DEFINITIONS
# ════════════════════════════════════════════════════════════════

CELL_1_INSTALL = """\
# --- Cell 1: Install TRL (needed for SFTTrainer + response-only loss) ---
import subprocess, sys
try:
    import trl
    print(f"TRL already installed: {trl.__version__}")
except ImportError:
    print("Installing TRL...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "trl", "--quiet"],
        check=True,
    )
    import trl
    print(f"Installed TRL: {trl.__version__}")"""

CELL_2_CONFIG = """\
# --- Cell 2: Config + Imports ---
import site
import sys
import os
from pathlib import Path

if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

# CUTLASS path setup (from NVIDIA utility script)
candidate_cutlass_paths = [
    Path("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"),
    Path("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/nvidia_cutlass_dsl/python_packages/"),
]
for p in candidate_cutlass_paths:
    if p.exists():
        site.addsitedir(str(p))
        print(f"Added cutlass path: {p}")
        break

import pandas as pd
import torch
import kagglehub
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import mamba_ssm  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "mamba_ssm import failed. Run cell 1, restart kernel, then re-run from cell 2."
    ) from exc

# ═══════════════════════════════════════════════════════════
# TRAINING CONFIG — Run A (infrastructure overhaul)
# ═══════════════════════════════════════════════════════════
OUTPUT_DIR = "/kaggle/working"
FINAL_ADAPTER_DIR = f"{OUTPUT_DIR}/final_adapter"
MODEL_ID = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
LORA_RANK = 32

# Training knobs (set MAX_TRAIN_SAMPLES for smoke tests)
MAX_TRAIN_SAMPLES = None        # e.g. 256 for a quick run
NUM_EPOCHS = 2                  # was 1 → loss plateaued at 5.4-6.2, model underfitting
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512            # reduced from 1024 (data <256 tokens; cuts padding 2x)
LR_SCHEDULER_TYPE = "cosine"    # was "linear" → cosine gives smoother decay
WARMUP_RATIO = 0.05             # was 0 → prevents early loss spikes (10→7 in Run 2)
OPTIM = "adamw_8bit"            # was "adamw" → halves optimizer memory

# ═══════════════════════════════════════════════════════════
# EVAL CONFIG
# ═══════════════════════════════════════════════════════════
USE_EVAL = True   # False for final submission → train on all 9,500 rows
SPLIT_DATASET_SLUG = "your-username/nemotron-split-data"  # ← UPDATE after uploading

# Suffix — must match nvidia-nemotron-metric.ipynb exactly
BOXED_SUFFIX = (
    "\\nPlease put your final answer inside `\\\\boxed{}`. "
    "For example: `\\\\boxed{your answer}`"
)

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())"""

CELL_3_DATA_LOADING = """\
# --- Cell 3: Data loading (eval-split aware) ---
from pathlib import Path

def resolve_train_csv() -> Path:
    \"\"\"Find the competition train.csv on Kaggle or locally.\"\"\"
    candidates = [
        Path("/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv"),
        Path("/kaggle/input/competitions/nvidia-nemotron-3-reasoning-challenge/train.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    found = (
        list(Path("/kaggle/input").rglob("train.csv"))
        if Path("/kaggle/input").exists()
        else []
    )
    if found:
        return found[0]
    local = Path("train.csv")
    if local.exists():
        return local.resolve()
    raise FileNotFoundError(
        "train.csv not found. Add the competition dataset as a Kaggle input."
    )

if USE_EVAL:
    # Try Kaggle dataset first, then local datasets/ folder
    split_name = SPLIT_DATASET_SLUG.split("/")[-1]
    kaggle_split_root = Path(f"/kaggle/input/{split_name}")
    local_split_root = Path("datasets")

    if kaggle_split_root.exists():
        split_root = kaggle_split_root
    elif local_split_root.exists():
        split_root = local_split_root
    else:
        raise FileNotFoundError(
            f"Split data not found at {kaggle_split_root} or {local_split_root}. "
            "Upload datasets/ folder as a Kaggle dataset or ensure local files exist."
        )

    train_df = pd.read_csv(split_root / "train_9000.csv")
    eval_df = pd.read_csv(split_root / "eval_500.csv")
    print(f"\\u2705 Eval mode: Train={len(train_df)}, Eval={len(eval_df)} (from {split_root})")
else:
    # Final submission: use ALL competition data
    train_df = pd.read_csv(resolve_train_csv())
    eval_df = None
    print(f"\\U0001f3c1 Final submission mode: Training on ALL {len(train_df)} rows (no eval)")

# Apply sample limit if set
if MAX_TRAIN_SAMPLES is not None:
    train_df = train_df.head(int(MAX_TRAIN_SAMPLES)).copy()
    print(f"\\u26a0\\ufe0f Truncated to {len(train_df)} samples (MAX_TRAIN_SAMPLES={MAX_TRAIN_SAMPLES})")

print(f"Training rows: {len(train_df)}")"""

CELL_4_TOKENIZER = """\
# --- Cell 4: Tokenizer + data formatting ---
import os
from pathlib import Path

def resolve_model_path() -> str:
    \"\"\"Find the Nemotron model on Kaggle inputs or download via kagglehub.\"\"\"
    direct_candidates = [
        Path("/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default"),
        Path("/kaggle/input/nemotron-3-nano/transformers/default/1"),
    ]
    for path in direct_candidates:
        if path.exists():
            print(f"Found model in Kaggle inputs: {path}")
            return str(path)

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for config_path in input_root.rglob("config.json"):
            candidate = config_path.parent
            candidate_text = str(candidate).lower()
            if "nemotron" not in candidate_text:
                continue
            if not (candidate / "tokenizer_config.json").exists():
                continue
            print(f"Found model in Kaggle inputs: {candidate}")
            return str(candidate)

    print("Downloading model via kagglehub (requires internet)...")
    return kagglehub.model_download(MODEL_ID)

MODEL_PATH = resolve_model_path()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token


def example_to_text(prompt: str, answer: str) -> str:
    \"\"\"Format a single example as chat-template text (matching metric format).\"\"\"
    user_content = str(prompt).strip() + BOXED_SUFFIX
    assistant_content = f"\\\\boxed{{{str(answer).strip()}}}"
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=False)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=True)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


# Build training dataset
texts = [
    example_to_text(row.prompt, row.answer)
    for row in train_df.itertuples(index=False)
]
train_ds = Dataset.from_dict({"text": texts})

print(f"Training dataset: {len(train_ds)} examples")
print(f"Sample text length (chars): {len(texts[0]) if texts else 0}")
print(f"\\n--- Sample formatted text (first 500 chars) ---")
print(texts[0][:500] if texts else "(empty)")"""

CELL_5_RESPONSE_TEMPLATE = """\
# --- Cell 5: Detect response template + create response-only data collator ---
from trl import DataCollatorForCompletionOnlyLM

# Detect the assistant header by formatting a sample with a unique marker
_marker = "XASSISTANTRESPONSEX"
_test_msgs = [
    {"role": "user", "content": "test prompt here"},
    {"role": "assistant", "content": _marker},
]
_kwargs = dict(tokenize=False, add_generation_prompt=False)
try:
    _test_text = tokenizer.apply_chat_template(_test_msgs, **_kwargs, enable_thinking=True)
except TypeError:
    _test_text = tokenizer.apply_chat_template(_test_msgs, **_kwargs)

# Find the marker and extract what comes just before it (the assistant header)
_marker_idx = _test_text.index(_marker)
_prefix = _test_text[:_marker_idx]

print(f"Full template with marker:\\n{repr(_test_text)}")
print(f"\\nPrefix before assistant content:\\n{repr(_prefix)}")

# The response template: use the token IDs of the last structural element before
# the assistant content. This is typically the role marker (e.g. "Assistant\\n").
# Using token IDs is more robust than string matching.
_response_template_ids = tokenizer.encode(
    _prefix, add_special_tokens=False
)
# Take only the last few tokens as the response template (the assistant header)
# We want just the role marker, not the entire prefix
_full_prefix_tokens = tokenizer.encode(_prefix, add_special_tokens=False)

# Find a reasonable suffix: decode progressively fewer tokens from the end
# until we get just the assistant role marker
for _n_tokens in [3, 4, 5, 6, 8, 10]:
    _candidate_ids = _full_prefix_tokens[-_n_tokens:]
    _candidate_str = tokenizer.decode(_candidate_ids)
    if "assistant" in _candidate_str.lower() or "Assistant" in _candidate_str:
        _response_template_ids = _candidate_ids
        break
else:
    # Fallback: use last 4 tokens
    _response_template_ids = _full_prefix_tokens[-4:]

_response_template_str = tokenizer.decode(_response_template_ids)
print(f"\\nResponse template string: {repr(_response_template_str)}")
print(f"Response template token IDs: {_response_template_ids}")

# Create the data collator for response-only loss
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=_response_template_ids,
    tokenizer=tokenizer,
)
print("\\n\\u2705 DataCollatorForCompletionOnlyLM created (response-only loss enabled)")"""

CELL_6_SANITY_CHECK = """\
# --- Cell 6: Label sanity check (verify -100 masking on prompt tokens) ---
# Tokenize a sample and verify the data collator correctly masks prompt tokens

_sample = train_ds[0]
_tokenized = tokenizer(
    _sample["text"], truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt"
)

# Create a batch for the collator
_batch = [
    {
        "input_ids": _tokenized["input_ids"][0],
        "attention_mask": _tokenized["attention_mask"][0],
    }
]
_collated = data_collator(_batch)

_labels = _collated["labels"][0]
_input_ids = _collated["input_ids"][0]
_n_masked = (_labels == -100).sum().item()
_n_active = (_labels != -100).sum().item()
_n_total = len(_labels)

print(f"Total tokens: {_n_total}")
print(f"Masked (prompt, -100): {_n_masked} ({100*_n_masked/_n_total:.1f}%)")
print(f"Active (response):     {_n_active} ({100*_n_active/_n_total:.1f}%)")

if _n_active == 0:
    print("\\n\\u274c ERROR: All tokens are masked! Response template detection failed.")
    print("Falling back to full-sequence loss (no masking).")
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
elif _n_active > _n_total * 0.5:
    print("\\n\\u26a0\\ufe0f WARNING: >50% of tokens are active — masking may not be correct.")
    print("Check the response template detection in cell 5.")
else:
    print("\\n\\u2705 Masking looks correct — prompt tokens masked, only response tokens active.")

# Show what the model will actually learn to predict
print(f"\\n--- Active (trainable) tokens decoded ---")
_active_ids = _input_ids[_labels != -100]
print(tokenizer.decode(_active_ids))"""

CELL_7_MODEL = """\
# --- Cell 7: Load 30B model + attach LoRA (rank 32) ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=r".*\\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()"""

CELL_8_TRAINING = """\
# --- Cell 8: Training (SFTTrainer with response-only loss) ---
from trl import SFTTrainer, SFTConfig
import shutil
import subprocess

os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)

# ─── SFTConfig (replaces TrainingArguments) ───
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,    # cosine (was linear)
    warmup_ratio=WARMUP_RATIO,              # 0.05 (was 0)
    optim=OPTIM,                            # adamw_8bit (was adamw)
    logging_steps=50,
    save_strategy="no",
    bf16=True,
    report_to="none",
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,   # incompatible with DataCollatorForCompletionOnlyLM; seq len reduction compensates
)

# ─── Create trainer ───
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# ─── Fix Kaggle read-only Triton binary permissions ───
triton_bin_candidates = [
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/triton/backends/nvidia/bin",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin",
]
ro_bin_dir = next((path for path in triton_bin_candidates if os.path.exists(path)), None)
rw_bin_dir = "/kaggle/working/triton_bin"
if ro_bin_dir:
    os.makedirs(rw_bin_dir, exist_ok=True)
    for f in os.listdir(ro_bin_dir):
        src = os.path.join(ro_bin_dir, f)
        dst = os.path.join(rw_bin_dir, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        os.chmod(dst, 0o777)

    orig_popen = subprocess.Popen

    def patched_popen(*args, **kwargs):
        if args and isinstance(args[0], list) and isinstance(args[0][0], str):
            if args[0][0].startswith(ro_bin_dir):
                args[0][0] = args[0][0].replace(ro_bin_dir, rw_bin_dir)
        return orig_popen(*args, **kwargs)

    subprocess.Popen = patched_popen

# ─── Print config summary ───
print("=" * 60)
print("TRAINING CONFIG (Run A — Infrastructure Overhaul)")
print("=" * 60)
print(f"  Epochs:         {NUM_EPOCHS} (was 1)")
print(f"  LR:             {LEARNING_RATE}")
print(f"  LR schedule:    {LR_SCHEDULER_TYPE} (was linear)")
print(f"  Warmup ratio:   {WARMUP_RATIO} (was 0)")
print(f"  Optimizer:      {OPTIM} (was adamw)")
print(f"  Max seq length: {MAX_SEQ_LENGTH} (was 1024)")
print(f"  Batch size:     {PER_DEVICE_TRAIN_BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} grad accum")
print(f"  Response-only:  {not isinstance(data_collator, type(None))}")
print(f"  Packing:        False")
print(f"  Train samples:  {len(train_ds)}")
print("=" * 60)

trainer.train()"""

CELL_9_EXPORT = """\
# --- Cell 9: Save trained adapter weights ---
from pathlib import Path

final_adapter_dir = Path(FINAL_ADAPTER_DIR)
final_adapter_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(final_adapter_dir)
print("Saved adapter under", final_adapter_dir)"""

CELL_10_ZIP = """\
# --- Cell 10: Zip adapter files for Kaggle submission ---
import subprocess
from pathlib import Path

out = Path(OUTPUT_DIR)
adapter_dir = Path(FINAL_ADAPTER_DIR)
submission_zip = out / "submission.zip"
cfg = adapter_dir / "adapter_config.json"
weights = adapter_dir / "adapter_model.safetensors"
if not weights.exists():
    weights = adapter_dir / "adapter_model.bin"
if not cfg.exists() or not weights.exists():
    raise FileNotFoundError(f"Missing adapter files. Expected {cfg} and a weights file.")

if submission_zip.exists():
    submission_zip.unlink()

subprocess.run([
    "zip",
    "-j",
    str(submission_zip),
    str(cfg),
    str(weights),
], check=True)
print("Wrote", submission_zip)"""

CELL_11_EVAL = r"""
# --- Cell 11: Evaluation on 500 held-out rows ---
import json
import re
import math
from pathlib import Path

if eval_df is not None:
    print("=" * 60)
    print(f"RUNNING EVALUATION on {len(eval_df)} held-out rows")
    print("=" * 60)

    # ─── Family classification (keyword-based) ───
    def classify_family(prompt_text: str) -> str:
        p = str(prompt_text).lower()
        if "bit manipulation" in p:
            return "bit_manipulation"
        if "gravitational constant" in p:
            return "gravity"
        if "encryption rules" in p or "encryption" in p:
            return "encryption"
        if "numeral system" in p:
            return "numeral"
        if "unit conversion" in p or "converted" in p:
            return "unit_conversion"
        if "transformation rules" in p or "equation" in p or "operator" in p:
            return "equations"
        return "unknown"

    # ─── Answer extraction (matches nvidia-nemotron-metric.ipynb) ───
    def extract_final_answer(text):
        if text is None:
            return "NOT_FOUND"
        matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
        if matches:
            non_empty = [m.strip() for m in matches if m.strip()]
            if non_empty:
                return non_empty[-1]
            return matches[-1].strip()
        patterns = [
            r'The final answer is:\s*([^\n]+)',
            r'Final answer is:\s*([^\n]+)',
            r'Final answer\s*[:：]\s*([^\n]+)',
            r'final answer\s*[:：]\s*([^\n]+)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        matches = re.findall(r'-?\d+(?:\.\d+)?', text)
        if matches:
            return matches[-1]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else "NOT_FOUND"

    # ─── Verification (matches nvidia-nemotron-metric.ipynb) ───
    def verify(stored_answer, predicted):
        stored_answer = str(stored_answer).strip()
        predicted = str(predicted).strip()
        if re.fullmatch(r'[01]+', stored_answer):
            return predicted.lower() == stored_answer.lower()
        try:
            stored_num = float(stored_answer)
            predicted_num = float(predicted)
            return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
        except Exception:
            return predicted.lower() == stored_answer.lower()

    # ─── Run inference ───
    model.eval()
    correct = 0
    total = 0
    family_stats = {}
    failures = []

    for i, row in enumerate(eval_df.itertuples(index=False)):
        # Format prompt (same as metric: user message + generation prompt)
        user_content = str(row.prompt).strip() + BOXED_SUFFIX
        messages = [{"role": "user", "content": user_content}]
        _kw = dict(tokenize=False, add_generation_prompt=True)
        try:
            prompt_text = tokenizer.apply_chat_template(messages, **_kw, enable_thinking=True)
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(messages, **_kw)

        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=1.0,
                top_p=1.0,
                do_sample=False,
            )

        # Decode only the generated (new) tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted = extract_final_answer(generated_text)
        ground_truth = str(row.answer).strip()
        is_correct = verify(ground_truth, predicted)

        # Track per-family stats
        family = classify_family(row.prompt)
        if family not in family_stats:
            family_stats[family] = {"correct": 0, "total": 0}
        family_stats[family]["total"] += 1
        if is_correct:
            correct += 1
            family_stats[family]["correct"] += 1
        elif len(failures) < 10:
            failures.append({
                "id": row.id if hasattr(row, "id") else i,
                "family": family,
                "expected": ground_truth,
                "predicted": predicted,
                "generated": generated_text[:200],
            })

        total += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(eval_df)}] accuracy so far: {correct}/{total} ({100*correct/total:.1f}%)")

    # ─── Print results ───
    print("\n" + "=" * 60)
    print(f"EVAL RESULTS ({total} held-out rows)")
    print("=" * 60)
    print(f"Overall accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print("\nPer-family breakdown:")
    for fam, stats in sorted(family_stats.items()):
        pct = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {fam:20s}: {stats['correct']}/{stats['total']} ({pct:.1f}%)")
    if failures:
        print(f"\nSample failures (first {min(len(failures), 5)}):")
        for f in failures[:5]:
            print(f"  [{f['family']}] expected={f['expected']!r} got={f['predicted']!r}")
    print("=" * 60)

    # ─── Save eval_results.json ───
    results = {
        "overall": {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0,
        },
        "per_family": family_stats,
        "failures": failures,
    }
    eval_json_path = Path(OUTPUT_DIR) / "eval_results.json"
    with open(eval_json_path, "w") as f_out:
        json.dump(results, f_out, indent=2)
    print(f"\nSaved eval results to {eval_json_path}")

else:
    print("Eval disabled (USE_EVAL=False). Skipping eval.")
"""

# ════════════════════════════════════════════════════════════════
# ASSEMBLE NOTEBOOK
# ════════════════════════════════════════════════════════════════

cells = [
    make_cell(CELL_1_INSTALL, cell_id="install_trl"),
    make_cell(CELL_2_CONFIG, cell_id="config_imports"),
    make_cell(CELL_3_DATA_LOADING, cell_id="data_loading"),
    make_cell(CELL_4_TOKENIZER, cell_id="tokenizer_fmt"),
    make_cell(CELL_5_RESPONSE_TEMPLATE, cell_id="response_tmpl"),
    make_cell(CELL_6_SANITY_CHECK, cell_id="sanity_check"),
    make_cell(CELL_7_MODEL, cell_id="model_lora"),
    make_cell(CELL_8_TRAINING, cell_id="training"),
    make_cell(CELL_9_EXPORT, cell_id="export_adapter"),
    make_cell(CELL_10_ZIP, cell_id="zip_submit"),
    make_cell(CELL_11_EVAL, cell_id="eval_cell"),
]

# Kaggle metadata (preserved from the original notebook)
kaggle_metadata = {
    "kaggle": {
        "accelerator": "nvidiaRtxPro6000",
        "dataSources": [
            {
                "databundleVersionId": 16082784,
                "sourceId": 129716,
                "sourceType": "competition",
            },
            {
                "databundleVersionId": 16059913,
                "modelId": 611168,
                "modelInstanceId": 598905,
                "sourceId": 784907,
                "sourceType": "modelInstanceVersion",
            },
            {
                "sourceId": 306236690,
                "sourceType": "kernelVersion",
            },
        ],
        "dockerImageVersionId": 31329,
        "isGpuEnabled": True,
        "isInternetEnabled": True,
        "language": "python",
        "sourceType": "notebook",
    },
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12.12",
    },
}

notebook = {
    "cells": cells,
    "metadata": kaggle_metadata,
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Write the notebook (repo root — this file lives in scripts/)
output_path = Path(__file__).resolve().parent.parent / "foundation-notebook.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"[OK] Wrote {output_path}")
print(f"   Cells: {len(cells)}")
print(f"   Cell IDs: {[c['id'] for c in cells]}")
