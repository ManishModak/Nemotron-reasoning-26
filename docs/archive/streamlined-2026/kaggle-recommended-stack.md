# Kaggle Workflow & Recommended Stack

A comprehensive 5-notebook pipeline optimized for Kaggle, leveraging both open-source tools and official NVIDIA frameworks.

## Notebook 0: Baseline + EDA + Local Eval Harness
- **Goal**: Start with official submission demo, add strong validation split, prompt optimization, and error analysis. Use `<think>...</think>` tags.
- **Tools**: Official submission demo, NeMo Evaluator, **DSPy** (for automated CoT/prompt optimization).

## Notebook 1: Data Preparation & Pipelines
- **Goal**: Collect, clean, filter, and structure logic puzzle data.
- **Tools**: 
  - **NeMo Curator** (Primary): GPU-accelerated dedup, quality filtering, fastText classifiers. (16x faster than pandas).
  - pandas / polars (for speed) / numpy.
  - re (Regular Expressions)
  - Hugging Face datasets

## Notebook 2: Synthetic Data Generation & Lightweight Fine-Tuning
- **Goal**: Generate high-quality reasoning examples via a teacher model, then fine-tune base model efficiently.
- **Tools**:
  - **NeMo DataDesigner**: Structured schemas, validators, LLM-as-judge scoring.
  - **Distilabel**: Lightweight synthetic-data pipeline (faster on T4).
  - **Axolotl**: YAML-config driven SFT/LoRA. Highly reproducible.
  - unsloth: Massive speedups for LoRA/QLoRA.
  - HF transformers, peft, trl (SFTTrainer), pytorch.
  - API clients (Nemotron-3-Ultra / larger frontier models).

## Notebook 3: Reinforcement Learning (RL)
- **Goal**: Train fine-tuned checkpoint through trial and error (especially GRPO) to maximize puzzle accuracy.
- **Tools**:
  - **NeMo RL**: First-class support for GRPO/DAPO on Nemotron-3-Nano.
  - **NeMo Gym (Reasoning Gym)**: 100+ logic puzzle environments with verifiable rewards.
  - HF trl (GRPO), Axolotl configs.

## Notebook 4: Inference, Prompting & Testing
- **Goal**: Final inference and structured puzzle outputs under Kaggle limits.
- **Tools**:
  - **SGLang / TensorRT-LLM**: Best speed/throughput on G4 Blackwell. SGLang is great for long reasoning traces.
  - **vLLM**: Fallback on T4.
  - outlines / Instructor: Pydantic-based structured outputs.
  - torch.compile + FlashAttention-2/3.
  - DSPy: Automatic prompt optimization.

## Key Principles
1. Prioritize **verifiable rewards** and **process supervision** (`<think>` tags) in all stages.
2. Use **Weights & Biases (wandb)** across all notebooks for contribution prize tracking.
3. Keep **LoRA rank ≤32**.
4. Use a **hybrid symbolic + LLM approach** (decode puzzle types to deterministic solvers as fallback).
