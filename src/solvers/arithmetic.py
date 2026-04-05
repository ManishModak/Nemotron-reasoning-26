"""Deterministic arithmetic transform solver."""

from __future__ import annotations

from fractions import Fraction

from src.eval.schemas import EvalExample
from src.solvers.base import SolverResult


class AffineArithmeticSolver:
    """Solve integer affine rules of the form y = a * x + b."""

    name = "affine_arithmetic"

    def solve(self, example: EvalExample) -> SolverResult:
        pairs: list[tuple[int, int]] = []
        for shot in example.few_shot_examples:
            try:
                pairs.append((int(shot.input_text.strip()), int(shot.output_text.strip())))
            except ValueError:
                return SolverResult(handled=False, trace="non-integer demonstration")

        if len(pairs) < 2:
            return SolverResult(handled=False, trace="need at least two demonstrations")

        try:
            query_value = int(example.task_text.strip())
        except ValueError:
            return SolverResult(handled=False, trace="non-integer task input")

        coefficient, intercept = _fit_affine_rule(pairs)
        if coefficient is None or intercept is None:
            return SolverResult(handled=False, trace="no affine rule matched all pairs")

        answer = coefficient * query_value + intercept
        if answer.denominator != 1:
            return SolverResult(handled=False, trace="non-integer affine result")

        return SolverResult(
            handled=True,
            answer=str(answer.numerator),
            confidence=0.99,
            trace=f"y = {coefficient} * x + {intercept}",
        )


def _fit_affine_rule(pairs: list[tuple[int, int]]) -> tuple[Fraction | None, Fraction | None]:
    first_x, first_y = pairs[0]
    for other_x, other_y in pairs[1:]:
        if other_x == first_x:
            continue
        coefficient = Fraction(other_y - first_y, other_x - first_x)
        intercept = Fraction(first_y) - coefficient * first_x
        if all(coefficient * x + intercept == y for x, y in pairs):
            return coefficient, intercept
    return None, None

