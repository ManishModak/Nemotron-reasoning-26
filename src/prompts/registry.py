"""Registry for the fixed Sprint 1 prompt variants."""

from __future__ import annotations

from src.eval.schemas import PromptVariant
from src.prompts.templates import (
    baseline_direct_prompt,
    reasoned_boxed_prompt,
    self_check_boxed_prompt,
)


def get_prompt_variants() -> list[PromptVariant]:
    """Return the three fixed prompt variants for this sprint."""

    return [
        baseline_direct_prompt(),
        reasoned_boxed_prompt(),
        self_check_boxed_prompt(),
    ]


def get_prompt_variant(name: str) -> PromptVariant:
    """Return a prompt variant by name."""

    for variant in get_prompt_variants():
        if variant.name == name:
            return variant
    raise KeyError(f"Unknown prompt variant: {name}")
