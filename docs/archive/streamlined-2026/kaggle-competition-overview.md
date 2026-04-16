# NVIDIA Nemotron Model Reasoning Challenge

**Tagline**: Advance reasoning techniques using NVIDIA Nemotron open models on a novel benchmark.
**Host**: NVIDIA (with Google Cloud as a partner).

## Competition Goal
Develop and experiment with techniques that improve reasoning accuracy on a novel logical reasoning benchmark created by NVIDIA Research. 
The baseline model is **Nemotron-3-Nano-30B-A3B** (a ~30B total / ~3B active parameter Mixture-of-Experts / hybrid Mamba-Transformer model).

**Key Focus Areas:**
- Advanced prompting and chain-of-thought (CoT)
- Data pipelines and filtering
- Synthetic data generation
- Reinforcement learning (RL)
- Lightweight fine-tuning (e.g., LoRA adapters with rank ≤32)

## Task Details
- Dataset consists of logical reasoning puzzles requiring identifying hidden transformation rules.
- Puzzles include symbolic/string/equation changes, bit manipulation, encryption, unit conversions, physics rules, etc.
- Models must infer rules from examples and apply them to test cases, outputting specific formats (structured/LaTeX).
- Evaluation is based on accuracy against a private test set.

## Timeline
- **Start**: March 16, 2026
- **Midpoint Cutoff** (Open Progress Prize): April 9, 2026
- **Final Submission Deadline**: June 15, 2026

## Prizes (Total Pool ≈ $106,388 + hardware)
- **1st Place**: $25,000 + 5× DGX Spark
- **2nd Place**: $15,000 + 2× DGX Spark
- **3rd Place**: $5,000 + 1× DGX Spark
- **Open Progress Prize**: $5,000 + 1× DGX Spark
- **3× Open Contribution Awards** (Best Data, Best RL, Best Fine-Tuning): 1× DGX Spark each
*(DGX Spark value ≈ $4,699. Max 1 hardware prize per team).*

## Participation & Rules
- **Submission**: Kaggle notebooks predicting on hidden test set (LoRA adapters rank ≤32).
- **Compute**: Kaggle environments, including Google Cloud resources (Blackwell GPUs).
- **Format**: Final submission is a LoRA adapter (rank ≤32) + any required files in a `.zip`.
