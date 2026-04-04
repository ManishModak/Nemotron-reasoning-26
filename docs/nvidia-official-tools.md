# NVIDIA NeMo Ecosystem Tools for the Challenge

NVIDIA highly recommends utilizing their open-source NeMo tools. They were used to train the Nemotron-3-Nano baseline, making them purpose-built for this challenge.

## NeMo Curator
- **Use Case**: Data Filtering and Curation.
- **Features**: GPU-accelerated (RAPIDS), 16x faster deduplication, semantic dedup, 30+ built-in quality/safety/domain filters, fastText classifiers.
- **Advantage**: Massively outperforms simple `pandas` or `re` for cleaning messy puzzle data into high-signal examples.

## NeMo DataDesigner
- **Use Case**: Synthetic Data Generation.
- **Features**: Structured schemas, dependency graphs, built-in Python/SQL validators, LLM-as-judge scoring, preview mode, statistical samplers.
- **Advantage**: Produces reproducible, higher-quality reasoning examples compared to raw API prompt scripts.
- **Installation**: `pip install data-designer`

## NeMo RL
- **Use Case**: Reinforcement Learning (at scale).
- **Features**: Official GRPO/DAPO configs specifically designed for Nemotron-3-Nano. Used directly for post-training the baseline model.
- **Advantage**: Superior to raw `TRL` implementations as it natively integrates verifiable rewards and official NVIDIA optimizations.

## NeMo Gym
- **Use Case**: RL Environments.
- **Features**: The **Reasoning Gym** offers 100+ logic, math, graph, and puzzle environments with ground-truth verification.
- **Advantage**: Crucial for defining verifiable rewards in RL workflows to prevent reward hacking.
