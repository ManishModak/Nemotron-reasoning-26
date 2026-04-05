"""Shared schemas for evaluation, prompting, and routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class FewShotExample:
    """One demonstration pair used to infer the hidden transformation."""

    input_text: str
    output_text: str


@dataclass(frozen=True)
class EvalExample:
    """A normalized evaluation item independent of the raw competition format."""

    example_id: str
    task_text: str
    few_shot_examples: tuple[FewShotExample, ...]
    gold_answer: str | None = None
    family_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptVariant:
    """Prompt variant metadata plus its renderer."""

    name: str
    instruction: str
    expects_boxed_answer: bool = True
    stop_tokens: tuple[str, ...] = ()

    def render_prompt(self, example: EvalExample) -> str:
        lines = [
            f"Variant: {self.name}",
            "You are solving a transformation puzzle.",
            self.instruction.strip(),
            "",
            "Demonstrations:",
        ]
        for index, shot in enumerate(example.few_shot_examples, start=1):
            lines.extend(
                [
                    f"Example {index} Input: {shot.input_text}",
                    f"Example {index} Output: {shot.output_text}",
                ]
            )
        lines.extend(
            [
                "",
                f"Task Input: {example.task_text}",
                "Return one final answer.",
                "Put the final answer in \\boxed{...}.",
            ]
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class PredictionRecord:
    """One evaluation record for a single example and prompt variant."""

    example_id: str
    variant_name: str
    solver_name: str
    raw_output: str
    parsed_answer: str | None
    normalized_answer: str | None
    gold_answer: str | None
    normalized_gold_answer: str | None
    is_correct: bool
    latency_s: float
    family_hint: str | None = None


@dataclass
class EvalRunResult:
    """Aggregate output of an evaluation run."""

    run_name: str
    predictions: list[PredictionRecord]
    metrics: dict[str, Any]
    error_buckets: dict[str, list[str]]
    artifact_paths: dict[str, Path] = field(default_factory=dict)


class PredictorProtocol(Protocol):
    """Protocol for prompt-only or adapter-backed text generation backends."""

    def __call__(
        self,
        prompts: list[str],
        *,
        stop: tuple[str, ...] = (),
    ) -> list[str]:
        """Generate one raw output per prompt."""

