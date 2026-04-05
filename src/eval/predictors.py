"""Local predictor implementations used for smoke tests and notebook scaffolding."""

from __future__ import annotations

import re

from src.eval.schemas import EvalExample, FewShotExample
from src.solvers.arithmetic import AffineArithmeticSolver
from src.solvers.base_conversion import BaseConversionSolver
from src.solvers.formatting import UnitConversionSolver
from src.solvers.string_shift import CaesarShiftSolver

TASK_PATTERN = re.compile(r"^Task Input:\s*(.*)$", re.MULTILINE)
VARIANT_PATTERN = re.compile(r"^Variant:\s*(.*)$", re.MULTILINE)
EXAMPLE_INPUT_PATTERN = re.compile(r"^Example (\d+) Input:\s*(.*)$", re.MULTILINE)
EXAMPLE_OUTPUT_PATTERN = re.compile(r"^Example (\d+) Output:\s*(.*)$", re.MULTILINE)


class HeuristicPredictor:
    """A lightweight predictor that keeps the evaluation path executable locally."""

    def __call__(self, prompts: list[str], *, stop: tuple[str, ...] = ()) -> list[str]:
        return [self._predict(prompt) for prompt in prompts]

    def _predict(self, prompt: str) -> str:
        variant = _extract_pattern(VARIANT_PATTERN, prompt) or "baseline_direct"
        example = _example_from_prompt(prompt)
        answer = self._solve(example)
        return _format_output(variant, answer)

    def _solve(self, example: EvalExample) -> str:
        for solver in (
            AffineArithmeticSolver(),
            BaseConversionSolver(),
            CaesarShiftSolver(),
            UnitConversionSolver(),
        ):
            result = solver.solve(example)
            if result.handled and result.answer is not None:
                return result.answer

        if _supports_reverse(example):
            return example.task_text[::-1]
        return "unknown"


def _example_from_prompt(prompt: str) -> EvalExample:
    task_text = _extract_pattern(TASK_PATTERN, prompt)
    if task_text is None:
        raise ValueError("Prompt missing task input.")

    inputs = {int(index): value for index, value in EXAMPLE_INPUT_PATTERN.findall(prompt)}
    outputs = {int(index): value for index, value in EXAMPLE_OUTPUT_PATTERN.findall(prompt)}

    shots = [
        FewShotExample(input_text=inputs[index], output_text=outputs[index])
        for index in sorted(set(inputs) & set(outputs))
    ]
    return EvalExample(
        example_id="prompt-derived",
        task_text=task_text,
        few_shot_examples=tuple(shots),
    )


def _extract_pattern(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def _supports_reverse(example: EvalExample) -> bool:
    if not example.few_shot_examples:
        return False
    return all(
        shot.output_text == shot.input_text[::-1]
        for shot in example.few_shot_examples
    )


def _format_output(variant: str, answer: str) -> str:
    if variant == "baseline_direct":
        return f"\\boxed{{{answer}}}"
    if variant == "reasoned_boxed":
        return f"The pattern is consistent across the demonstrations.\nFinal answer: \\boxed{{{answer}}}"
    if variant == "self_check_boxed":
        return (
            "I verified the inferred rule against the demonstrations once.\n"
            f"Final answer: \\boxed{{{answer}}}"
        )
    return f"\\boxed{{{answer}}}"
