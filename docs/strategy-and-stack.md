# Strategy, hardware, and tool stack

Consolidated reference for how we approach the Nemotron reasoning challenge on Kaggle. For submission mechanics and scoring, see [competition-rules.md](competition-rules.md). For SFT text alignment with the metric, see [training-data-format.md](training-data-format.md).

## Hardware reality

| Environment | Notes |
|-------------|--------|
| **Standard (T4)** | ~16 GB VRAM, ~5–9 h sessions, tight weekly quota. |
| **G4 (when allocated)** | RTX PRO 6000 Blackwell-class; more VRAM and speed; still bounded by time and disk. |

**Principles**

- Maximize **signal per GPU-hour**: LoRA rank ≤ 32, consider 4-bit/8-bit where it helps, use Unsloth/Axolotl-style memory patterns when training.
- **Hybrid stack**: full NeMo (especially Docker-heavy RL) is powerful on G4 but risky on T4; mix NVIDIA data tools with lightweight training paths.

## Current “locked-in” stack (recommended)

What we actually rely on when Kaggle stability matters:

| Stage | Tools | Why |
|-------|--------|-----|
| **Data prep** | **NeMo Curator** + pandas/polars | GPU-accelerated dedup/filtering; high signal for messy puzzle text. |
| **Synthetic data** | **Distilabel** (+ teacher API, e.g. larger Nemotron) | Lighter on T4 than full DataDesigner; prefer **process supervision** (`<redacted_thinking>`), not answers-only. |
| **SFT / LoRA** | **Unsloth** + **HF TRL** (`SFTTrainer`, etc.) + optional **Axolotl** YAML | Robust LoRA ≤ 32 on tight VRAM; matches HF/vLLM ecosystems. |
| **RL (optional)** | **HF TRL GRPO** first; **NeMo RL** + **NeMo Gym** if G4 + time | Prefer **verifiable** rewards (exact output / env checks). |
| **Prompt tuning (cheap lift)** | **DSPy** | Little extra GPU cost; improves CoT/prompts before or without weight changes. |
| **Local eval mirror** | **vLLM** | Close to Kaggle’s hosted evaluator (still vLLM + LoRA). |

### Deprioritized or “use with care”

| Item | Note |
|------|------|
| NeMo-Skills, heavy Ray-only stacks | Often too heavy for a single Kaggle session. |
| Full DeepSpeed | Unsloth/TRL usually enough for LoRA. |
| NeMo DataDesigner / NeMo RL | Strong for contribution prizes; can be fragile on Kaggle—budget debug time, prefer G4. |
| **SGLang / TensorRT-LLM** | Great for **local** throughput on Blackwell; **hosted scoring is still vLLM**—validate adapters on vLLM/HF. |
| **Instructor / heavy JSON schema decoding** | Risky vs. simple `\boxed{}` extraction the metric uses; if you experiment, keep it **local only** and do not assume the competition server uses it. |

**Self-consistency:** majority vote over a few samples at inference can help accuracy; it costs more tokens/time (fine for local eval; not something you ship in `submission.zip`).

## Multi-stage workflow (conceptual)

This is the logical pipeline the old multi-notebook scaffold mapped to. The **default repo submission path** is now one notebook (`foundation-notebook.ipynb`); you can still split work across sessions mentally using these stages:

1. **Baseline + EDA** — validation split, error buckets, optional DSPy on prompts.  
2. **Data** — Curator / polars cleaning, HF `datasets` export.  
3. **Synthetic + SFT** — Distilabel (or DataDesigner) → TRL SFT / Unsloth LoRA.  
4. **RL (optional)** — GRPO with verifiable rewards; NeMo Gym ideas for reward design.  
5. **Inference checks** — vLLM locally; optional SGLang on G4 for speed experiments only.

Cross-cutting: **wandb** (or similar) helps if you target Open Contribution prizes and need ablation trails.

## High-impact tactics

1. **Held-out eval** before every submit; track accuracy by puzzle family.  
2. **Hybrid symbolic + LLM** — deterministic solvers for repeated pattern families where you can reverse-engineer rules.  
3. **Process supervision** — train with reasoning traces (`<redacted_thinking>`) when the tokenizer/template supports it.  
4. **Verifiable RL rewards** — match final answer and/or structured checks; NeMo Gym–style envs as inspiration.  
5. **Structured decoding** — prefer prompts + `\boxed{}` alignment with the metric; see [training-data-format.md](training-data-format.md).  
6. **DSPy** for prompt/CoT iteration without always retraining.

## How we got here (short history)

1. **Phase 1 — Generic Kaggle LLM stack:** pandas/datasets → Unsloth/TRL SFT → TRL RL → vLLM + outlines-class tools. Solid, not Nemotron-specific.  
2. **Phase 2 — NeMo pivot:** Curator, DataDesigner, NeMo RL, NeMo Gym for official alignment and contribution angles.  
3. **Phase 3 — Kaggle reality:** T4 limits → **hybrid** — keep Curator (and Gym *ideas* for rewards), use Distilabel + Unsloth/TRL for execution, DSPy + vLLM for validation; defer heavy NeMo RL to G4/time-boxed attempts.

## NVIDIA tool blurbs (detail)

For links and extra bullets per product, see [reference/nemo-tools.md](reference/nemo-tools.md).

- **NeMo Curator** — GPU curation, dedup, quality filters.  
- **NeMo DataDesigner** — schema-first synthetic data, validators, LLM-as-judge.  
- **NeMo RL** — GRPO/DAPO-scale RL aligned with Nemotron post-training.  
- **NeMo Gym** — many verifiable reasoning-style environments for RL reward design.
