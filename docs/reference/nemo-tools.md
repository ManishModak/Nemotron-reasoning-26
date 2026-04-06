# NeMo and Nemotron reference (consolidated)

Quick reference for NVIDIA tools relevant to the reasoning challenge. For when to use what on Kaggle, see [../strategy-and-stack.md](../strategy-and-stack.md).

---

## NeMo Curator

GPU-accelerated curation for LLM / VLM / multimodal datasets.

- **Text:** exact/fuzzy/semantic dedup, 30+ quality filters, language ID.  
- **Image / video / audio:** aesthetic/NSFW filters, clips, ASR, WER, etc.  
- **Performance:** RAPIDS + Ray; often much faster than CPU-only pandas workflows for dedup.  
- **Source:** [NVIDIA-NeMo/Curator](https://github.com/NVIDIA-NeMo/Curator)

---

## NeMo DataDesigner

Synthetic data with schemas and validation (beyond one-off prompting).

- Statistical samplers, LLMs, or seed data; correlated fields.  
- Validators: Python, SQL, custom.  
- **LLM-as-judge** scoring; preview / iteration workflows.  
- **Install:** `pip install data-designer` (providers: NVIDIA Build API, OpenAI, OpenRouter, etc.).  
- **Source:** [NVIDIA-NeMo/DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner)

---

## NeMo RL

Large-scale post-training RL for LLMs and VLMs.

- **Scale:** Ray + Megatron Core, multi-GPU/node.  
- **Algorithms:** GRPO, DAPO, SFT, DPO, reward modeling.  
- **Backends:** PyTorch DTensor, Megatron Core.  
- **Rollouts:** vLLM integration for generation during RL.  
- **Extras:** multi-turn RL, FP8, VLM.  
- **Source:** [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)

---

## NeMo Gym

RL environments aimed at **verifiable rewards (RLVR)** for LLM training.

- Integrations: NeMo RL, OpenRLHF, Unsloth, etc.  
- **Pre-built env families:** math (e.g. GSM8K, Lean4), coding (e.g. SWE-bench, Spider2), reasoning (**Reasoning Gym**), agents (calendar, web search).  
- **CLI:** `ng_run`, `ng_collect_rollouts` for servers and rollout collection.  
- **Requirements:** Python 3.12+; Linux / macOS / Windows (WSL2).  
- **Source:** [NVIDIA-NeMo/gym](https://github.com/NVIDIA-NeMo/gym)

---

## NeMo Nemotron (hub)

Asset hub for the Nemotron model family: recipes, cookbooks, datasets.

- **Tiers:** Nano (edge/PC), Super (single-GPU throughput), Ultra (datacenter multi-GPU).  
- **Repo layout:** `src/nemotron/recipes/` (pretrain → SFT → RL), `usage-cookbook/`, `use-case-examples/` (RAG, agents).  
- **Datasets:** broad open catalogue (code, math, science, safety, multimodal).  
- **Source:** [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron)

---

## Nemotron 3 Nano post-training (NeMo RL guide)

Official walkthrough for **Nemotron 3 Nano** + **NeMo RL** (not Kaggle-sized by default).

- **Data:** `Nemotron-3-Nano-RL-Training-Blend` → `.jsonl` train/val.  
- **Setup:** clone NeMo RL + submodules (Megatron, NeMo).  
- **Training:** example Slurm `launch.sh` at **32 nodes / 256 GPUs** for GRPO.  
- **Note:** historical vLLM logprob caveats for versions before **0.17.0** (check current NVIDIA docs).  
- **Source:** [Nemotron 3 Nano RL guide](https://docs.nvidia.com/nemo/rl/nightly/guides/nemotron-3-nano.html)
