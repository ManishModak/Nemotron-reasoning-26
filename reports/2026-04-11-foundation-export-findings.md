# Kaggle foundation export findings - 2026-04-11

- Source: [`foundation-notebook.ipynb`](../foundation-notebook.ipynb) exported from a real Kaggle session.
- The competition dataset was mounted and resolved successfully: `/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv`.
- Training progressed to `583/1188` steps at about `Epoch 0.49/1`, so the core model load, tokenizer path, data formatting, and trainer loop were functional on Kaggle.
- Kaggle notebook metadata showed a Nemotron model input was attached, but the notebook still printed `Downloading model via kagglehub (requires internet)...`, which indicates the current model-path detection missed the mounted input.
- The export cell later failed with `NameError: name 'model' is not defined`, which means the save/export step was run after notebook state had been lost and no recoverable trainer/model state remained in memory.
- Prior notebook behavior used `save_strategy="no"`, so no intermediate trainer checkpoints were available to resume from after the disconnect.

## Implication

Checkpointing plus automatic resume are mandatory before the next serious Kaggle training run. Mounted Kaggle model inputs should also be preferred over `kagglehub` downloads to reduce startup cost and network dependence.
