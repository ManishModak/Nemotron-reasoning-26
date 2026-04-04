# AGENTS.md

## Operating Model

This repo is the local planning and coding workspace.

The default workflow is:

1. implement and organize code here
2. stop at a defined checkpoint
3. move the relevant notebook or files to Kaggle
4. run training, inference, or submission checks on Kaggle
5. bring back errors, logs, screenshots, or outputs into this repo context
6. debug and revise here
7. repeat until the checkpoint passes

## Execution Boundary

Competition-critical runs are expected to happen on Kaggle notebooks, not in this local workspace.

That includes:

- baseline competition runs
- GPU-backed inference validation
- LoRA fine-tuning
- GRPO or other RL runs
- final submission packaging checks that depend on Kaggle environment behavior

Local work in this repo should focus on:

- notebook structure
- prompt design
- parsing and evaluation utilities
- data transformation code
- config authoring
- symbolic solver logic
- experiment planning and result tracking

## Checkpoint Rule

Do not treat code as complete just because it is written locally.

A task reaches a real checkpoint only when:

- the relevant notebook or module is prepared here
- it has been transferred to Kaggle
- the Kaggle run has been attempted
- the result is recorded in `reports/` or the active working notes

## Kaggle Handoff Pattern

For each milestone:

1. prepare the minimal runnable notebook here
2. identify the exact Kaggle dependencies and runtime assumptions
3. run it on Kaggle
4. collect failures, runtime limits, and output quality issues
5. patch locally
6. rerun on Kaggle

Keep changes small between Kaggle tests so regressions are easy to isolate.

## Support Expectation

When a Kaggle run fails or behaves unexpectedly, the assistant should help with:

- reading stack traces and notebook errors
- identifying environment mismatches
- adapting code for Kaggle storage, package, and GPU limits
- reducing runtime or memory pressure
- adjusting prompts, parsing, batching, or training configs
- converting working local logic into Kaggle-safe notebook cells

The expected debugging loop is collaborative:

- user runs or tests on Kaggle
- user shares the error, log, output, or observed behavior
- assistant analyzes it and proposes the next patch
- assistant updates repo files here
- user reruns on Kaggle

## Planning Constraint

All plans in this repo should assume:

- coding and organization happen locally in this repository
- environment validation happens on Kaggle at defined checkpoints
- no major modeling decision is considered validated until Kaggle confirms it

