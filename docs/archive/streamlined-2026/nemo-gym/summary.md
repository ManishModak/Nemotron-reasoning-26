# NVIDIA NeMo Gym

**Title:** NeMo Gym: Build RL Environments for LLM Training

NeMo Gym provides the infrastructure to build, test, and scale reinforcement learning environments specifically tailored for training LLMs. It simplifies the process of creating complex, multi-turn, and tool-using scenarios.

*   **Key Information:**
    *   **Purpose:** Accelerates environment development for "Reinforcement Learning from Verifiable Reward" (RLVR).
    *   **Ecosystem:** Integrates with training frameworks like NeMo RL, OpenRLHF, and Unsloth.
    *   **Available Environments:** Includes a wide variety of pre-built environments for math (GSM8k, Lean4), coding (SWE-bench, Spider2), reasoning (Reasoning Gym), and agentic tasks (Calendar scheduling, Web search).
    *   **Quick Start:** Provides tools (`ng_run`, `ng_collect_rollouts`) to start servers and collect verified training data with minimal setup.
    *   **Requirements:** Runs on Linux, macOS, and Windows (WSL2), requiring Python 3.12+.

**Source:** [NVIDIA-NeMo/Gym GitHub](https://github.com/NVIDIA-NeMo/gym)