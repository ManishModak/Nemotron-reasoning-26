# Nemotron 3 Nano Post-Training Guide

**Title:** Post-Training Nemotron 3 Nano with NeMo RL

This technical guide provides a step-by-step walkthrough for post-training the Nemotron 3 Nano model using the NeMo RL framework.

*   **Key Steps:**
    *   **Data Preparation:** Instructions for downloading the `Nemotron-3-Nano-RL-Training-Blend` dataset and preparing `.jsonl` files for training and validation.
    *   **Code Setup:** Requires cloning the NeMo RL repository and initializing submodules (Megatron and NeMo).
    *   **Training Configuration:** Uses a Slurm-based launch script (`launch.sh`) to run GRPO training across 32 nodes (256 GPUs).
    *   **Technical Note:** Highlights a specific workaround for vLLM logprob divergence (prior to version 0.17.0) to ensure training stability.

**Source:** [Nemotron 3 Nano Guide](https://docs.nvidia.com/nemo/rl/nightly/guides/nemotron-3-nano.html)