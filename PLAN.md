# Execution Plan

Current date in workspace context: April 5, 2026

## Execution Model

This repo exists to support Kaggle execution, not to replace it.

Default execution model:

- develop shared code, prompts, parsing, configs, and notebook structure here
- keep the code Kaggle-compatible from the start
- run compute-bound validation and submission-critical paths on Kaggle
- bring failures, logs, runtime notes, and outputs back here for targeted fixes

Practical rule:

- local work is for code quality, iteration speed, and reproducibility
- Kaggle is the source of truth for inference, training, runtime limits, and submission behavior
- local completion is not enough
- each major phase needs a Kaggle checkpoint before it is treated as validated

## Development Rule

Use a Kaggle-first workflow with local support:

- if the task needs GPU, model loading, adapter training, throughput checks, or submission packaging, develop the runnable path for Kaggle directly
- if the task is parser logic, evaluation logic, prompt construction, data normalization, solver code, or report generation, build and test it locally first, then hand it off to Kaggle
- avoid local-only assumptions in paths, dependencies, or notebook structure
- prefer thin notebooks and reusable `src/` modules so Kaggle handoff stays small and easy to debug

## Milestones

### Milestone 1: Progress-prize path

Target date: April 9, 2026

Deliverables:

- combined Kaggle notebook running held-out eval and submission generation end to end
- baseline evaluation notebook running end to end
- held-out validation split and error report
- prompt-only improved submission
- hybrid symbolic plus LLM submission
- short report comparing the best prompt and routing variants
- at least one Kaggle notebook test pass for the baseline or best available variant

Acceptance criteria:

- one reproducible local evaluation path exists
- answer extraction and normalization are standardized
- at least one Kaggle submission variant is ready from repo outputs
- the prepared notebook has been tested on Kaggle and its result is recorded

### Milestone 2: Lightweight fine-tuning path

Target date: April 10 to May 10, 2026

Deliverables:

- cleaned and versioned training dataset
- synthetic data generation workflow
- first LoRA rank <= 32 training run
- ablation report for baseline vs cleaned vs synthetic-enhanced training data
- Kaggle test notes for installation, runtime, and memory behavior

Acceptance criteria:

- training config is reproducible
- adapter artifact is saved under `artifacts/`
- validation lift over prompt-only baseline is measured
- at least one SFT path has been attempted on Kaggle and the outcome is documented

### Milestone 3: RL path

Target date: May 11 to June 1, 2026

Deliverables:

- narrow GRPO experiment with verifiable rewards
- reward design notes
- comparison against best SFT-only adapter
- Kaggle RL run notes covering setup issues, runtime, and reward behavior

Acceptance criteria:

- RL run is stable on at least one Kaggle hardware path
- reward function is grounded in exact output correctness or deterministic checks
- failures are documented if RL is not viable within Kaggle constraints

### Milestone 4: Final submission hardening

Target date: June 2 to June 15, 2026

Deliverables:

- final adapter package
- inference throughput comparison on available hardware
- final validation summary and submission checklist
- final Kaggle notebook dry run before submission

Acceptance criteria:

- final artifact packaging is verified
- final inference path stays within Kaggle runtime constraints
- submission steps have been validated in Kaggle notebook context

## Kaggle Checkpoints

Each major workstream follows this loop:

1. prepare the smallest runnable Kaggle unit here
2. transfer the notebook or module snapshot to Kaggle
3. run it on Kaggle as early as possible
4. capture logs, errors, metrics, and runtime constraints
5. patch the underlying code here
6. rerun on Kaggle

The assistant should support each checkpoint by helping interpret failures and updating the local codebase accordingly.

## Workstreams

### 1. Evaluation

Owner: `notebooks/00_eval_and_submission.ipynb`, `notebooks/00_baseline_eval.ipynb`, `src/eval/`

Tasks:

- define validation split
- log per-puzzle-family errors
- standardize prompt and output parsing
- track best prompt variants
- keep the baseline flow notebook-safe and Kaggle-runnable at every checkpoint
- keep the combined eval-plus-submission notebook as the default Kaggle entrypoint

### 2. Data

Owner: `notebooks/01_data_prep.ipynb`, `src/data/`

Tasks:

- ingest competition data
- clean and normalize examples
- add optional puzzle-type labels
- generate high-confidence subsets for training
- verify dataset loading and export format on Kaggle

### 3. Prompting and Solvers

Owner: `src/prompts/`, `src/solvers/`, `notebooks/00_eval_and_submission.ipynb`, `notebooks/04_inference_submission.ipynb`

Tasks:

- define core reasoning prompt
- add answer-format enforcement
- implement deterministic handlers for easy puzzle families
- add routing logic for symbolic fallback
- validate notebook-safe packaging of the prompt and solver path on Kaggle

### 4. Training

Owner: `notebooks/02_synthetic_sft.ipynb`, `configs/axolotl/`

Tasks:

- create synthetic-data workflow locally
- prepare the runnable SFT notebook for Kaggle
- execute the actual smoke test on Kaggle
- scale to LoRA training with tracked ablations

### 5. RL

Owner: `notebooks/03_rl_grpo.ipynb`

Tasks:

- test lightweight GRPO path first
- add verifiable reward checks
- compare RL against best non-RL checkpoint
- keep RL blocked until Kaggle environment viability is confirmed

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
- after each Kaggle test: record the exact error, runtime limit, install issue, or quality regression before patching

## Immediate Next Actions

1. Make `notebooks/00_eval_and_submission.ipynb` the default Kaggle notebook and keep the split notebooks thin.
2. Point the evaluation path at the real competition dataset instead of smoke-only defaults.
3. Build shared parsing utilities under `src/` for the known puzzle families.
4. Start a simple symbolic routing prototype under `src/solvers/`.
5. Transfer the first baseline checkpoint to Kaggle early and record the result before expanding scope.
6. Keep training work blocked until evaluation is trustworthy on Kaggle as well as locally.
