# bitsandbytes offline wheels (Kaggle, no internet)

Kaggle **Python 3.12**, **Linux x86_64**, CUDA image (see notebook `language_info` / Kaggle docker).

## What to put here (current pin)

Current wheel (Linux **manylinux_2_24_x86_64**, includes CUDA 12.8 binaries for Blackwell):

- `bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl`

No dependency wheels needed — `torch`, `numpy`, `scipy` are already on Kaggle.

Re-download with:

```bash
pip download "bitsandbytes>=0.45.3" --no-deps -d . --python-version 312 --platform manylinux_2_24_x86_64 --implementation cp --abi cp312 --only-binary=:all:
```

> **v0.42.0 does NOT work** — missing `libbitsandbytes_cuda128.so` for CUDA 12.8. Need **v0.45.3+**.

## How to refresh wheels (developer machine)

Use a **Linux** env that matches Kaggle (WSL2 Ubuntu, Linux VM, or a one-off Kaggle notebook with **Internet ON**):

```bash
mkdir -p bnb_offline && cd bnb_offline
pip download 'bitsandbytes>=0.43.0' -d . --python-version 312 --platform manylinux_2_28_x86_64 --implementation cp --abi cp312 --only-binary=:all:
```

If `--platform manylinux_2_28_x86_64` fails, try `manylinux2014_x86_64`.

**From Windows without Linux:** run the same `pip download` in **WSL Ubuntu** against Kaggle’s Python 3.12, or use a short **Internet ON** Kaggle notebook:

```python
!pip download 'bitsandbytes>=0.43.0' -d /kaggle/working/bnb_offline
```

Then download `/kaggle/working/bnb_offline` from the run output and copy files into this repo folder.

## Kaggle usage

1. Zip this folder or upload as a **Dataset**.
2. **Add Data** on the training notebook.
3. Notebook installs with:

```text
pip install --no-index --find-links /kaggle/input/<your-dataset-slug>/bnb_offline bitsandbytes
```

(Adjust path — auto-discovery is implemented in `improved-RunA-Notebook.ipynb`.)

## Fallback

If install fails, set optimizer to `**adamw_torch_fused**` (PyTorch-only, no bitsandbytes).