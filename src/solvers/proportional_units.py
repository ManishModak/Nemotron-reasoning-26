"""Deterministic proportional unit-conversion solver."""

from __future__ import annotations

import re
from statistics import fmean

from src.eval.schemas import EvalExample
from src.solvers.base import SolverResult

INPUT_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$")
OUTPUT_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*$")


class ProportionalUnitSolver:
    """Infer a per-row multiplicative conversion factor from demonstrations."""

    name = "proportional_unit_conversion"

    def solve(self, example: EvalExample) -> SolverResult:
        if example.family_hint not in (None, "unit_conversion"):
            return SolverResult(handled=False, trace="family not supported")

        factors: list[float] = []
        source_unit: str | None = None
        for shot in example.few_shot_examples:
            parsed_input = INPUT_PATTERN.match(shot.input_text)
            parsed_output = OUTPUT_PATTERN.match(shot.output_text)
            if parsed_input is None or parsed_output is None:
                return SolverResult(handled=False, trace="unsupported unit demonstration")
            value = float(parsed_input.group(1))
            unit = parsed_input.group(2).lower()
            converted = float(parsed_output.group(1))
            if value == 0:
                return SolverResult(handled=False, trace="zero-valued conversion not supported")
            if source_unit is None:
                source_unit = unit
            elif source_unit != unit:
                return SolverResult(handled=False, trace="mixed units in demonstrations")
            factors.append(converted / value)

        if not factors or source_unit is None:
            return SolverResult(handled=False, trace="no unit demonstrations")
        factor = fmean(factors)

        for shot in example.few_shot_examples:
            parsed_input = INPUT_PATTERN.match(shot.input_text)
            parsed_output = OUTPUT_PATTERN.match(shot.output_text)
            assert parsed_input is not None
            assert parsed_output is not None
            value = float(parsed_input.group(1))
            observed = round(float(parsed_output.group(1)), 2)
            expected = round(value * factor, 2)
            if abs(expected - observed) > 0.02:
                return SolverResult(handled=False, trace="unit demonstrations are inconsistent")

        query_match = INPUT_PATTERN.match(example.task_text)
        if query_match is None:
            return SolverResult(handled=False, trace="unsupported unit task")
        if query_match.group(2).lower() != source_unit:
            return SolverResult(handled=False, trace="task unit does not match demonstrations")
        query_value = float(query_match.group(1))
        answer = query_value * factor
        return SolverResult(
            handled=True,
            answer=f"{answer:.2f}",
            confidence=0.98,
            trace=f"inferred factor {factor:.6f}",
        )
