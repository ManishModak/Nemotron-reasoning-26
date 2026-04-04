# NVIDIA NeMo RL

**Title:** NeMo RL: A Scalable and Efficient Post-Training Library

NeMo RL is an open-source library designed to scale reinforcement learning (RL) methods for large language models (LLMs) and multimodal models. It is built for flexibility, reproducibility, and massive multi-GPU/multi-node deployments.

*   **Key Features:**
    *   **Scalability:** Uses Ray for resource management and Megatron Core for high-performance training with 6D parallelism.
    *   **Algorithms Supported:** GRPO (Group Relative Policy Optimization), DAPO, SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), and Reward Model (RM) training.
    *   **Backends:** Supports both PyTorch-native DTensor and NVIDIA’s Megatron Core.
    *   **Inference Integration:** Seamlessly integrates with vLLM for high-throughput generation during RL rollouts.
    *   **Advanced Capabilities:** Supports multi-turn RL, FP8 low-precision training, and Vision Language Models (VLM).

**Source:** [NVIDIA-NeMo/RL GitHub](https://github.com/NVIDIA-NeMo/RL)