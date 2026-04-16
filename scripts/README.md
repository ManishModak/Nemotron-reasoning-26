# `scripts/` index

Run from repo root unless noted.

## CoT / benchmark APIs

| File | Purpose |
|------|---------|
| `nvidia_taskcot.py` | Main NVIDIA NIM CoT pipeline (streaming OpenAI SDK by default, tier1/tier2, CSV + run log). |
| `openrouter_taskcot.py` | Multi-phase / OpenRouter-oriented CoT benchmark (see docstring). |
| `requirements-nvidia-taskcot.txt` | `pip install -r …` — `openai` for default streaming in `nvidia_taskcot.py`. |

```powershell
pip install -r scripts/requirements-nvidia-taskcot.txt
python scripts/nvidia_taskcot.py --sample datasets/benchmark_sample_500.csv
```

## Data + notebooks

| File | Purpose |
|------|---------|
| `make_benchmark_sample.py` | Build `datasets/benchmark_sample_500.csv` from `train_9000.csv`. |
| `generate_run_a_notebook.py` | Regenerate / patch `foundation-notebook.ipynb` (Run A TRL/SFT layout). |
| `apply_run_a_prime.py` | Patch `improved-RunA-Notebook.ipynb` (Run A-prime tweaks). |

## Environment

`nvidia_taskcot.py` / `openrouter_taskcot.py` load keys from `.env` in repo root or cwd (see each script’s docstring).
