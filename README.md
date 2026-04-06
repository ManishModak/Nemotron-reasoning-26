# Nemotron Model Reasoning Challenge (Kaggle)

Repo for the NVIDIA Nemotron reasoning competition. **Competition runs happen on Kaggle**; this workspace holds the runnable notebook, data copies, and notes.

## Default runnable (submit this path)

| Artifact | Purpose |
|----------|---------|
| [`kaggle-combined-lora-submission.ipynb`](kaggle-combined-lora-submission.ipynb) | One notebook: PyTorch env setup → LoRA SFT on `train.csv` → `submission.zip` (adapter only) |

Supporting docs:

- [`docs/README.md`](docs/README.md) — index of all docs
- [`docs/competition-rules.md`](docs/competition-rules.md) — submission, scoring, timeline
- [`docs/training-data-format.md`](docs/training-data-format.md) — SFT text aligned with the metric
- [`reports/kaggle-first-run-checklist.md`](reports/kaggle-first-run-checklist.md) — Kaggle run checklist

## What you submit to Kaggle

A **`submission.zip`** containing your LoRA adapter (`adapter_config.json` + weights). The host runs **vLLM + Nemotron-3-Nano-30B-A3B + your adapter** and scores `\boxed{}` extraction vs ground truth (see [`docs/competition-rules.md`](docs/competition-rules.md) and `nvidia-nemotron-metric.ipynb`).

## Repo layout

- `docs/` — [index](docs/README.md): rules, strategy, training format; **`docs/archive/`** — legacy docs + old `src/` / notebooks
- `artifacts/`, `reports/` — outputs and run notes
- `train.csv`, `test.csv` — local copies for analysis (Kaggle uses competition inputs)
- `nvidia-*.ipynb` — upstream NVIDIA / metric / utility references

## Local vs Kaggle

- **Kaggle:** full GPU train, `kagglehub` model download, `submission.zip`.
- **Local:** read `train.csv`, experiment design, update the combined notebook, refresh docs/reports after each Kaggle attempt.

## Legacy code

The previous `notebooks/`, `src/`, `configs/`, and `tests/` tree lives under [`docs/archive/`](docs/archive/README.md). Use it only if you revive the old eval/solver harness.
