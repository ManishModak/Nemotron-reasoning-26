---
name: nemotron-kaggle-engineer
description: Use for this repo’s NVIDIA Nemotron Kaggle challenge—single combined LoRA notebook, TRL/PEFT/HF training, metric-aligned SFT, Kaggle env debugging, and submission.zip packaging. Not for generic production MLOps.
---

# Nemotron Kaggle engineer (project skill)

Derived from a general AI-engineer profile, **narrowed to this workspace only**. Read [`AGENTS.md`](../../../AGENTS.md) first.

## Scope

- **Primary artifact:** [`foundation-notebook.ipynb`](../../../foundation-notebook.ipynb) — env install → LoRA SFT → save adapter → zip.
- **Submission:** `submission.zip` with `adapter_config.json` + weights; host evaluates with vLLM + LoRA (see [`docs/competition-rules.md`](../../../docs/competition-rules.md)).
- **Training text:** User suffix + `\boxed{answer}` alignment — [`docs/training-data-format.md`](../../../docs/training-data-format.md).
- **Reference notebooks in repo:** `nvidia-utility-script.ipynb`, `nvidia-nemotron-submission-demo.ipynb`, `nvidia-nemotron-metric.ipynb`.

## What to do

1. **Kaggle-first:** Assume GPU training and scoring run on Kaggle; local edits are not validation until a Kaggle run confirms.
2. **One notebook:** Do not introduce a second parallel pipeline unless the user explicitly asks.
3. **Constraints:** LoRA rank ≤ 32; match metric prompt suffix when formatting SFT data; prefer `bfloat16` / patterns from the demo notebooks.
4. **Documentation:** Use **Context7** for TRL, Transformers, PEFT, vLLM API drift; use **web search** for Kaggle session limits, CUDA/torch wheels, competition rule updates.
5. **Ask the user** for competition datasets, utility kernels, full tracebacks, GPU type, and pinned versions when debugging OOM, import, or tokenizer errors.

## What to skip (unless the user asks)

- Generic production checklists (SLAs, A/B infra, multi-tenant serving).
- Mandatory “bias testing on all demographic groups” on every change.
- Repo layouts like `ai/memory-bank/` that do not exist here.

## Stack touchpoints

- **Training:** HF `AutoModelForCausalLM`, `AutoTokenizer`, PEFT LoRA (`in_proj|out_proj|up_proj|down_proj`), TRL `SFTTrainer` / `SFTConfig`.
- **Env:** `uv pip`, `/kaggle/working` torch target, `mamba_ssm`, optional CUTLASS path from NVIDIA utility script.
- **Data:** `train.csv` (`id`, `prompt`, `answer`); resolve paths for `/kaggle/input/...` and local fallback.

## On failure

Parse stderr → env vs OOM vs API mismatch → smallest patch to the combined notebook → suggest `MAX_TRAIN_SAMPLES` smoke test → record outcomes under [`reports/`](../../../reports/).
