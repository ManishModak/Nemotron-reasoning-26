"""Deterministic gravity-distance solver."""

from __future__ import annotations

import re
from statistics import fmean

from src.eval.schemas import EvalExample
from src.solvers.base import SolverResult

TIME_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)s\s*$", re.IGNORECASE)
DISTANCE_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*m\s*$", re.IGNORECASE)


class GravityDistanceSolver:
    """Infer the per-row gravity constant from demonstrations."""

    name = "gravity_distance"

    def solve(self, example: EvalExample) -> SolverResult:
        if example.family_hint not in (None, "gravity_distance"):
            return SolverResult(handled=False, trace="family not supported")

        g_values: list[float] = []
        for shot in example.few_shot_examples:
            parsed_time = TIME_PATTERN.match(shot.input_text)
            parsed_distance = DISTANCE_PATTERN.match(shot.output_text)
            if parsed_time is None or parsed_distance is None:
                return SolverResult(handled=False, trace="unsupported gravity demonstration")
            time_value = float(parsed_time.group(1))
            distance_value = float(parsed_distance.group(1))
            if time_value == 0:
                return SolverResult(handled=False, trace="zero time not supported")
            g_values.append((2.0 * distance_value) / (time_value**2))

        if not g_values:
            return SolverResult(handled=False, trace="no gravity demonstrations")
        gravity = fmean(g_values)

        for shot in example.few_shot_examples:
            parsed_time = TIME_PATTERN.match(shot.input_text)
            parsed_distance = DISTANCE_PATTERN.match(shot.output_text)
            assert parsed_time is not None
            assert parsed_distance is not None
            time_value = float(parsed_time.group(1))
            observed = round(float(parsed_distance.group(1)), 2)
            expected = round(0.5 * gravity * (time_value**2), 2)
            if abs(expected - observed) > 0.02:
                return SolverResult(handled=False, trace="gravity demonstrations are inconsistent")

        query_match = TIME_PATTERN.match(example.task_text)
        if query_match is None:
            return SolverResult(handled=False, trace="unsupported gravity task")
        query_time = float(query_match.group(1))
        answer = 0.5 * gravity * (query_time**2)
        return SolverResult(
            handled=True,
            answer=f"{answer:.2f}",
            confidence=0.985,
            trace=f"inferred gravity {gravity:.4f}",
        )
