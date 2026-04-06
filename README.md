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

1. `00_eval_and_submission.ipynb`
2. `00_baseline_eval.ipynb`
3. `01_data_prep.ipynb`
4. `02_synthetic_sft.ipynb`
5. `03_rl_grpo.ipynb`
6. `04_inference_submission.ipynb`

## Default Notebook

Use `notebooks/00_eval_and_submission.ipynb` as the default Kaggle handoff notebook for Milestone 1.

It is the simplest path to:

- run held-out validation on `train.csv`
- write eval artifacts and the handoff bundle
- generate `submission.csv` from `test.csv`

The split notebooks remain available when you want to isolate evaluation from submission generation.

## Execution Rule

- build reusable logic in `src/` and keep notebooks thin
- design the runnable path for Kaggle from the start
- use local runs for CPU-safe iteration and debugging
- use Kaggle for inference, training, runtime validation, and submission-critical checks

## Immediate Priorities

1. Build a strong held-out evaluation harness.
2. Add a hybrid symbolic plus LLM inference path.
3. Produce a reliable first submission before the progress-prize cutoff.
4. Keep the combined Kaggle notebook path runnable at all times.
