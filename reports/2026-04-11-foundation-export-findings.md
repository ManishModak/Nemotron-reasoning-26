# Kaggle foundation export findings - 2026-04-11

- Source: [`foundation-notebook.ipynb`](../foundation-notebook.ipynb) exported from a real Kaggle session.
- The competition dataset was mounted and resolved successfully: `/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv`.
- Training progressed to `583/1188` steps at about `Epoch 0.49/1`, so the core model load, tokenizer path, data formatting, and trainer loop were functional on Kaggle.
- A later committed run log showed the improved model-path detection working correctly: `Found model in Kaggle inputs: /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1`.
- The export cell later failed with `NameError: name 'model' is not defined`, which means the save/export step was run after notebook state had been lost and no recoverable trainer/model state remained in memory.
- A later committed run with full `Trainer` checkpoints failed with `OSError: [Errno 28] No space left on device`, so full checkpointing is not viable on Kaggle for this notebook as currently configured.

## Implication

Committed background runs via `Save Version -> Save & Run All` are the safer default than interactive sessions. Mounted Kaggle model inputs should be preferred over `kagglehub` downloads, and full `Trainer` checkpoints should stay disabled unless replaced with a lighter-weight save strategy.
