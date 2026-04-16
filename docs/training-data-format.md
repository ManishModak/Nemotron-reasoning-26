# SFT training data format (Kaggle submission alignment)

Official scoring loads **Nemotron-3-Nano-30B-A3B** with your LoRA in vLLM, appends a fixed instruction to each **test** `prompt`, then extracts the answer from `\boxed{...}` (see `nvidia-nemotron-metric.ipynb` and [competition-rules.md](competition-rules.md)).

## User message (must match the metric)

After the raw puzzle text from `train.csv`, append **exactly** the same suffix the metric uses:

```text
\nPlease put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
```

In Python (as in `foundation-notebook.ipynb`):

```python
BOXED_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
user_content = prompt.strip() + BOXED_SUFFIX
```

## Assistant message (supervision target)

Train the model to emit the ground-truth answer in boxed form:

```text
\boxed{<answer column from train.csv>}
```

Use the literal `answer` string from the dataset (no extra normalization unless you intentionally change training and accept metric risk).

## Chat template

Build a two-turn chat and render with the model tokenizer:

```python
messages = [
    {"role": "user", "content": user_content},
    {"role": "assistant", "content": f"\\boxed{{{answer.strip()}}}"},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
```

Pass `enable_thinking=True` when the tokenizer supports it (Nemotron demos); otherwise omit the argument.

## Optional extensions (later milestones)

- Wrap reasoning in `<redacted_thinking>...</redacted_thinking>` **before** `\boxed{...}` if you want process-style supervision and the template supports it.
- Filter or augment rows locally; keep this document in sync with whatever the combined notebook actually trains on.

## Why this shape

- At submission time you **do not** control the prompt beyond what the host appends; aligning SFT user text with that suffix reduces train/serve skew.
- The metric grades extracted `\boxed{}` content; teaching the adapter to answer in that format directly targets the leaderboard objective.
