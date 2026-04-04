# Execution Plan

Current date in workspace context: April 5, 2026

## Milestones

### Milestone 1: Progress-prize path

Target date: April 9, 2026

Deliverables:

- baseline evaluation notebook running end to end
- held-out validation split and error report
- prompt-only improved submission
- hybrid symbolic plus LLM submission
- short report comparing the best prompt and routing variants

Acceptance criteria:

- one reproducible local evaluation path exists
- answer extraction and normalization are standardized
- at least one Kaggle submission variant is ready from repo outputs

### Milestone 2: Lightweight fine-tuning path

Target date: April 10 to May 10, 2026

Deliverables:

- cleaned and versioned training dataset
- synthetic data generation workflow
- first LoRA rank <= 32 training run
- ablation report for baseline vs cleaned vs synthetic-enhanced training data

Acceptance criteria:

- training config is reproducible
- adapter artifact is saved under `artifacts/`
- validation lift over prompt-only baseline is measured

### Milestone 3: RL path

Target date: May 11 to June 1, 2026

Deliverables:

- narrow GRPO experiment with verifiable rewards
- reward design notes
- comparison against best SFT-only adapter

Acceptance criteria:

- RL run is stable on at least one Kaggle hardware path
- reward function is grounded in exact output correctness or deterministic checks

### Milestone 4: Final submission hardening

Target date: June 2 to June 15, 2026

Deliverables:

- final adapter package
- inference throughput comparison on available hardware
- final validation summary and submission checklist

Acceptance criteria:

- final artifact packaging is verified
- final inference path stays within Kaggle runtime constraints

## Workstreams

### 1. Evaluation

Owner: `notebooks/00_baseline_eval.ipynb`, `src/eval/`

Tasks:

- define validation split
- log per-puzzle-family errors
- standardize prompt and output parsing
- track best prompt variants

### 2. Data

Owner: `notebooks/01_data_prep.ipynb`, `src/data/`

Tasks:

- ingest competition data
- clean and normalize examples
- add optional puzzle-type labels
- generate high-confidence subsets for training

### 3. Prompting and Solvers

Owner: `src/prompts/`, `src/solvers/`, `notebooks/04_inference_submission.ipynb`

Tasks:

- define core reasoning prompt
- add answer-format enforcement
- implement deterministic handlers for easy puzzle families
- add routing logic for symbolic fallback

### 4. Training

Owner: `notebooks/02_synthetic_sft.ipynb`, `configs/axolotl/`

Tasks:

- create synthetic-data workflow
- run small smoke-test SFT
- scale to LoRA training with tracked ablations

### 5. RL

Owner: `notebooks/03_rl_grpo.ipynb`

Tasks:

- test lightweight GRPO path first
- add verifiable reward checks
- compare RL against best non-RL checkpoint

## Recommended Stack

Default path:

- eval and inference: Hugging Face or vLLM first
- data prep: pandas or polars first, optional NeMo Curator later
- synthetic data: Distilabel first, optional NeMo DataDesigner later
- SFT: Unsloth + TRL, optionally managed with Axolotl configs
- RL: TRL GRPO first, NeMo RL only if environment setup is stable

## Checkpoint Cadence

- daily: update validation score, runtime notes, blockers
- every training run: record config, dataset version, adapter path, and result
- before each Kaggle submission: confirm prompt version, model artifact, and answer formatting

## Immediate Next Actions

1. Implement the evaluation harness in `00_baseline_eval.ipynb`.
2. Build shared parsing utilities under `src/eval/`.
3. Start a simple symbolic routing prototype under `src/solvers/`.
4. Keep training work blocked until evaluation is trustworthy.

