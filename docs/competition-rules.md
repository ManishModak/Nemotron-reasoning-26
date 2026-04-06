# Competition rules and scoring

Single reference for the NVIDIA Nemotron Model Reasoning Challenge on Kaggle. Confirm dates and rules on the [live competition page](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview).

## What this is

**Tagline:** Advance reasoning techniques using NVIDIA Nemotron open models on a novel benchmark.  
**Host:** NVIDIA (Google Cloud partner).

**Goal:** Improve reasoning accuracy on a logical-reasoning benchmark from NVIDIA Research. The baseline is **Nemotron-3-Nano-30B-A3B** (~30B total / ~3B active MoE, hybrid Mamba–Transformer).

**Task:** Puzzles ask the model to infer hidden rules from examples (bit ops, strings, equations, units, physics-style rules, etc.) and answer test cases. Leaderboard uses a **private** test set.

**Typical levers:** prompting / CoT, data curation, synthetic data, RL, **LoRA fine-tuning (rank ≤ 32)**.

## Timeline (verify on Kaggle)

| Milestone | Date (workspace notes) |
|-----------|-------------------------|
| Competition start | March 16, 2026 |
| Open Progress Prize cutoff | April 9, 2026 |
| Entry / team merger deadline | June 8, 2026 |
| Final submission deadline | June 15, 2026 |

## Prizes (approximate)

Total pool ~**$106,388** plus hardware (DGX Spark ~$4,699; max one hardware prize per team):

- 1st: $25,000 + 5× DGX Spark  
- 2nd: $15,000 + 2× DGX Spark  
- 3rd: $5,000 + 1× DGX Spark  
- Open Progress Prize: $5,000 + 1× DGX Spark  
- Open Contribution Awards (Best Data / RL / Fine-Tuning): 1× DGX Spark each  

## What you submit

- Package a **LoRA adapter** (rank **≤ 32**) for **Nemotron-3-Nano-30B** into **`submission.zip`**.
- The archive must include **`adapter_config.json`** (and weight files as produced by PEFT / your trainer).
- You can start from the official **NVIDIA Nemotron Submission Demo** pattern (see repo `nvidia-nemotron-submission-demo.ipynb`).

**Submission limits:** You may mark up to **2** submissions for the final leaderboard; if fewer are selected, Kaggle can auto-pick from your best runs.

## How scoring works (host-side)

- The host loads **Nemotron-3-Nano-30B** with **your LoRA** in **vLLM**.
- Each row is prompted; the model is told to put the final answer in **`\boxed{...}`**.
- The metric **extracts** the answer (prefers `\boxed{}`, else heuristics / last numeric token).
- A prediction is **correct** if it matches the label **exactly as a string**, or for numeric answers if values match within **relative tolerance 1e-2** (and a small absolute tolerance near zero; see official metric notebook for the exact `verify` logic).
- **Final score** = fraction of test items correct.

### Reported inference settings (from metric notebook)

| Parameter | Value |
|-----------|--------|
| max_lora_rank | 32 |
| max_tokens | 7680 |
| top_p | 1.0 |
| temperature | 0.0 |
| max_num_seqs | 64 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 8192 |

You do **not** control these at submit time; they define how your adapter is evaluated.

## Compute

Runs on **Kaggle** (including **Blackwell-class** VMs when allocated). Plan training and installs for session time and GPU memory limits (see [strategy-and-stack.md](strategy-and-stack.md)).

## Repo pointers

- Training text alignment with the metric: [training-data-format.md](training-data-format.md)  
- Runnable Kaggle path: root `kaggle-combined-lora-submission.ipynb`  
- Metric reference: `nvidia-nemotron-metric.ipynb`  
