"""Deterministic numeric base conversion solver."""

from __future__ import annotations

from src.eval.schemas import EvalExample
from src.solvers.base import SolverResult

BASE_FORMATTERS = {
    2: lambda value: format(value, "b"),
    8: lambda value: format(value, "o"),
    16: lambda value: format(value, "x"),
}


class BaseConversionSolver:
    """Handle decimal-to-base transformations with exact examples."""

    name = "base_conversion"

    def solve(self, example: EvalExample) -> SolverResult:
        integers: list[tuple[int, str]] = []
        for shot in example.few_shot_examples:
            try:
                integers.append((int(shot.input_text.strip()), shot.output_text.strip().lower()))
            except ValueError:
                return SolverResult(handled=False, trace="non-decimal demonstration input")

        if len(integers) < 2:
            return SolverResult(handled=False, trace="need at least two demonstrations")

        try:
            query_value = int(example.task_text.strip())
        except ValueError:
            return SolverResult(handled=False, trace="non-decimal task input")

        for base, formatter in BASE_FORMATTERS.items():
            if all(formatter(value) == output for value, output in integers):
                return SolverResult(
                    handled=True,
                    answer=formatter(query_value),
                    confidence=0.985,
                    trace=f"decimal to base-{base}",
                )

        return SolverResult(handled=False, trace="no supported base conversion matched")

