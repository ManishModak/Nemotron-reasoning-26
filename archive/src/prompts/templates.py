"""Prompt template definitions for Sprint 1."""

from __future__ import annotations

from src.eval.schemas import PromptVariant


def baseline_direct_prompt() -> PromptVariant:
    return PromptVariant(
        name="baseline_direct",
        instruction=(
            "Infer the hidden rule from the demonstrations and apply it once to the task input. "
            "Do not add extra formatting beyond the final boxed answer."
        ),
    )


def reasoned_boxed_prompt() -> PromptVariant:
    return PromptVariant(
        name="reasoned_boxed",
        instruction=(
            "Reason briefly about the rule using the demonstrations, then provide the final "
            "answer in boxed form."
        ),
    )


def self_check_boxed_prompt() -> PromptVariant:
    return PromptVariant(
        name="self_check_boxed",
        instruction=(
            "Infer the rule, perform one short self-check against the demonstrations, and then "
            "return the final answer in boxed form."
        ),
    )

