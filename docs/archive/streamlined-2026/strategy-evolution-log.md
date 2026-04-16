# Strategy Evolution Log

This document traces the evolution of our technical strategy for the NVIDIA Nemotron Model Reasoning Challenge, from initial brainstorming to the final Kaggle-optimized architecture.

## Phase 1: The Initial Standard LLM Stack
We began by outlining a standard, reliable 4-notebook pipeline commonly used in Kaggle LLM competitions:
- **Data Prep:** `pandas`, `numpy`, `re`, `datasets`.
- **Synthetic Data & SFT:** Custom API scripts for distillation, followed by `unsloth` and HF `TRL` (SFTTrainer) for fine-tuning.
- **Reinforcement Learning:** HF `TRL` utilizing PPO/DPO/GRPO.
- **Inference:** `vLLM` with structured output libraries like `outlines`.

*Verdict:* A solid baseline, but lacked competition-specific optimizations and didn't leverage the official ecosystem NVIDIA provided for this specific model (Nemotron-3-Nano).

## Phase 2: The "Official NVIDIA Boost" Pivot
After reviewing the competition host's (Jamil C Semaan) recommendations, we discovered that NVIDIA provided the exact tools used to train the baseline model. We pivoted heavily towards the **NeMo Ecosystem** to maximize chances of winning Open Contribution Prizes:
- **Data Prep:** Swapped to **NeMo Curator** (GPU-accelerated, massive speedup, built-in semantic dedup).
- **Synthetic Data:** Swapped to **NeMo DataDesigner** (structured schemas, dependency graphs, LLM-as-judge).
- **RL:** Swapped to **NeMo RL** and **NeMo Gym (Reasoning Gym)** to guarantee verifiable rewards (crucial for logical puzzles).

*Verdict:* Extremely powerful and aligned with the judges' expectations, but potentially too heavy and complex (Docker dependencies) for standard Kaggle environments.

## Phase 3: The Kaggle Hardware Reality Check (T4 vs. G4)
We re-evaluated the NeMo-heavy stack against the actual constraints of the competition:
- **T4 VMs:** 16GB VRAM, strict 9-hour timeouts, limited weekly quota. Too weak for full NeMo RL Docker stacks.
- **G4 VMs:** Special allocation with RTX PRO 6000 Blackwell GPUs. Powerful, but still subject to session timeouts.

To maximize signal-per-GPU-hour without blowing the quota, we adopted a **Hybrid Approach**:
1. **Retain High-Value NeMo Tools:** Keep `NeMo Curator` for lightning-fast data prep. Keep `NeMo Gym` conceptual verifiable rewards.
2. **Introduce Lightweight Alternatives:** Brought in **Distilabel** as a faster, cheaper alternative to NeMo DataDesigner for synthetic data generation on T4s.
3. **Revert to Reliable Training Frameworks:** Shifted back to **Unsloth + HF TRL** (with **Axolotl** for config management) because it is the most robust way to guarantee LoRA rank ≤32 training within Kaggle's memory limits.
4. **Optimize Inference (Locally):** Added **DSPy** for zero-cost prompt optimization and identified **SGLang / TensorRT-LLM** for maximum throughput on Blackwell G4s (though acknowledging Kaggle's backend evaluator ultimately uses vLLM).

## Conclusion
The final hybrid stack is a direct result of balancing NVIDIA's high-quality NeMo ecosystem with the harsh, resource-constrained reality of Kaggle's execution environment. We use NVIDIA's tools where they provide the biggest data quality boost (Curator), and Kaggle-proven tools where stability and memory efficiency are paramount (Unsloth/TRL).
