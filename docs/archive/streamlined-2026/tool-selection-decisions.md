# Tool Selection & Framework Decisions

Based on Kaggle's resource constraints (T4 vs. G4 Blackwells) and the competition's unique submission format (LoRA adapters instead of inference scripts), we evaluated multiple frameworks. Here is the final decision matrix on what we kept, what we discarded, and why.

## ❌ Discarded / Deprioritized Tools

| Tool / Framework | Reason for Discarding / Deprioritizing |
| :--- | :--- |
| **NeMo-Skills** | Too heavy and complex to set up within standard Kaggle session limits. |
| **Full DeepSpeed** | Unnecessary overhead. Unsloth already handles the required memory and speed optimizations much better for LoRA training. |
| **Ray / Stable-Baselines3** | Overkill for this competition. HF TRL (or NeMo RL) natively supports PPO/GRPO for language models and covers 99% of our needs. |
| **Llama-Factory, Datatrove, Argilla, verl** | Redundant. They offer similar capabilities to our chosen stack but are either too heavy or duplicate functionality we already have. |
| **Complex Structured Output Libraries (e.g., Instructor)** | Kaggle's evaluator extracts answers from a simple LaTeX `\boxed{}` tag. Forcing strict JSON formats with heavy libraries risks breaking the final submission evaluation. Simple prompt engineering (via DSPy) is safer. |

## ⚖️ The "It Depends" Category

| Tool / Framework | When to Use |
| :--- | :--- |
| **NeMo DataDesigner & NeMo RL** | Highly capable and officially recommended by NVIDIA (great for Open Contribution Prizes). **However**, they can be very heavy and prone to dependency issues on Kaggle VMs. Use only if you have G4 access and plenty of time to debug environments. |
| **SGLang / TensorRT-LLM** | Incredible for speed on Blackwell GPUs, but **only for local/offline validation**. The actual Kaggle submission uses their backend (typically vLLM), so you must ensure your LoRA adapter works natively with standard HF/vLLM. |

## ✅ The Final "Locked-In" Stack (Highly Recommended)

This is the most reliable, fast, and Kaggle-optimized stack we decided to keep:

### 1. Data Preparation
- **Kept:** **NeMo Curator** + **Pandas/Polars**
- **Why:** NeMo Curator is genuinely best-in-class for GPU-accelerated deduplication and quality filtering. It's fast and turns noisy puzzles into high-signal training examples.

### 2. Synthetic Data Generation
- **Kept:** **Distilabel** (using a frontier model like Nemotron-3-Ultra via API)
- **Why:** It's a much lighter, faster pipeline than full NeMo DataDesigner on Kaggle VMs. **Crucial strategy:** Use it to generate *process supervision* (`<think>` traces), not just final answers.

### 3. Fine-Tuning & Reinforcement Learning (SFT + GRPO)
- **Kept:** **Unsloth** + **Hugging Face TRL** (+ **Axolotl** for configs)
- **Why:** This is the safest, fastest, and most robust way to train a 30B model on Kaggle. Unsloth provides massive speed/memory wins. It easily guarantees we meet the competition's "LoRA rank ≤ 32" rule. 

### 4. Validation & Prompting
- **Kept:** **DSPy** + **vLLM** (for local offline eval)
- **Why:** DSPy provides automated prompt/CoT optimization (which is practically free accuracy). vLLM closely mirrors the Kaggle backend evaluation environment, ensuring our adapter will work when submitted.
