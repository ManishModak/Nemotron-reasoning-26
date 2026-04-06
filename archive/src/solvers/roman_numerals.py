"""Deterministic Roman numeral conversion solver."""

from __future__ import annotations

from src.eval.schemas import EvalExample
from src.solvers.base import SolverResult

ROMAN_TABLE = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)


class RomanNumeralSolver:
    """Convert integers into standard Roman numerals."""

    name = "roman_numeral"

    def solve(self, example: EvalExample) -> SolverResult:
        if example.family_hint not in (None, "roman_numeral"):
            return SolverResult(handled=False, trace="family not supported")

        pairs: list[tuple[int, str]] = []
        for shot in example.few_shot_examples:
            try:
                value = int(shot.input_text.strip())
            except ValueError:
                return SolverResult(handled=False, trace="non-integer Roman example")
            pairs.append((value, shot.output_text.strip().upper()))

        if not pairs:
            return SolverResult(handled=False, trace="no Roman demonstrations")
        if not all(_to_roman(value) == numeral for value, numeral in pairs):
            return SolverResult(handled=False, trace="Roman demonstrations are inconsistent")

        try:
            query_value = int(example.task_text.strip())
        except ValueError:
            return SolverResult(handled=False, trace="non-integer Roman task")
        if query_value <= 0:
            return SolverResult(handled=False, trace="Roman numerals require positive integers")

        return SolverResult(
            handled=True,
            answer=_to_roman(query_value),
            confidence=0.995,
            trace="standard Roman numeral conversion",
        )


def _to_roman(value: int) -> str:
    if value <= 0:
        raise ValueError("Roman numerals require a positive integer")
    remainder = value
    pieces: list[str] = []
    for numeral_value, symbol in ROMAN_TABLE:
        while remainder >= numeral_value:
            remainder -= numeral_value
            pieces.append(symbol)
    return "".join(pieces)
