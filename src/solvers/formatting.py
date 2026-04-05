"""Deterministic formatting and unit conversion solvers."""

from __future__ import annotations

import re

from src.eval.schemas import EvalExample
from src.solvers.base import SolverResult

UNIT_FACTORS = {
    ("km", "m"): 1000,
    ("m", "cm"): 100,
    ("kg", "g"): 1000,
    ("hr", "min"): 60,
}
UNIT_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$")


class UnitConversionSolver:
    """Handle simple multiplicative unit conversions with exact demonstrations."""

    name = "unit_conversion"

    def solve(self, example: EvalExample) -> SolverResult:
        inferred_rule: tuple[str, str, float] | None = None
        for shot in example.few_shot_examples:
            current = _parse_unit_pair(shot.input_text, shot.output_text)
            if current is None:
                return SolverResult(handled=False, trace="unsupported unit example")
            if inferred_rule is None:
                inferred_rule = current
            elif inferred_rule != current:
                return SolverResult(handled=False, trace="inconsistent unit conversion")

        if inferred_rule is None:
            return SolverResult(handled=False, trace="no unit rule inferred")

        source_unit, target_unit, factor = inferred_rule
        parsed_task = UNIT_PATTERN.match(example.task_text)
        if parsed_task is None or parsed_task.group(2).lower() != source_unit:
            return SolverResult(handled=False, trace="task input does not match inferred source unit")

        value = float(parsed_task.group(1))
        converted = value * factor
        converted_text = str(int(converted)) if converted.is_integer() else str(converted)
        return SolverResult(
            handled=True,
            answer=f"{converted_text} {target_unit}",
            confidence=0.96,
            trace=f"{source_unit} -> {target_unit} x {factor}",
        )


def _parse_unit_pair(source_text: str, target_text: str) -> tuple[str, str, float] | None:
    source = UNIT_PATTERN.match(source_text)
    target = UNIT_PATTERN.match(target_text)
    if source is None or target is None:
        return None

    source_value = float(source.group(1))
    target_value = float(target.group(1))
    source_unit = source.group(2).lower()
    target_unit = target.group(2).lower()
    expected_factor = UNIT_FACTORS.get((source_unit, target_unit))
    if expected_factor is None:
        return None
    if target_value != source_value * expected_factor:
        return None
    return source_unit, target_unit, float(expected_factor)

