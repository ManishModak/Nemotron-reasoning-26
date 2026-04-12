# Archived pipeline (pre–single-notebook plan)

These paths were the previous multi-notebook + `src/` layout. They are **not** used for the current Kaggle submission path.

| Path | Note |
|------|------|
| `notebooks/` | Staged notebooks (`00_*` … `04_*`) |
| `src/` | Shared eval, prompts, solvers, predictors |
| `configs/` | Experiment JSON/YAML (e.g. baseline eval) |
| `tests/` | Pytest suite for `src/` |
| `00-eval-and-submission.ipynb` | Prompt-only eval + CSV submission (not LoRA `submission.zip`) |

**Active path:** root [`foundation-notebook.ipynb`](../../foundation-notebook.ipynb), plus [`docs/training-data-format.md`](../training-data-format.md) and [`reports/kaggle-first-run-checklist.md`](../../reports/kaggle-first-run-checklist.md).

Restore pieces from here only if you intentionally revive the old harness.
