# Nemotron Model Reasoning Challenge

Working repo scaffold for the NVIDIA Nemotron Model Reasoning Challenge on Kaggle.

## Goal

Build a reproducible pipeline for:

- Kaggle-first evaluation and inference
- local support for error analysis and shared code iteration
- data cleaning and curation
- synthetic data generation and lightweight fine-tuning
- reinforcement learning experiments
- final inference and submission packaging

## Structure

- `docs/`: research notes and summaries already collected
- `notebooks/`: staged Kaggle workflow from baseline to submission
- `src/`: shared Python modules for evaluation, data processing, prompts, and solvers
- `configs/`: training and experiment configs
- `artifacts/`: saved adapters, intermediate outputs, and packaged submissions
- `reports/`: validation summaries, ablations, and milestone notes

## Pipeline

1. `00_baseline_eval.ipynb`
2. `01_data_prep.ipynb`
3. `02_synthetic_sft.ipynb`
4. `03_rl_grpo.ipynb`
5. `04_inference_submission.ipynb`

## Execution Rule

- build reusable logic in `src/` and keep notebooks thin
- design the runnable path for Kaggle from the start
- use local runs for CPU-safe iteration and debugging
- use Kaggle for inference, training, runtime validation, and submission-critical checks

## Immediate Priorities

1. Build a strong held-out evaluation harness.
2. Add a hybrid symbolic plus LLM inference path.
3. Produce a reliable first submission before the progress-prize cutoff.
