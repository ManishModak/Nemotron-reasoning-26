"""Deterministic Caesar-style string shift solver."""

from __future__ import annotations

from src.eval.schemas import EvalExample
from src.solvers.base import SolverResult


class CaesarShiftSolver:
    """Detect and apply a uniform alphabetical shift."""

    name = "caesar_shift"

    def solve(self, example: EvalExample) -> SolverResult:
        shift: int | None = None
        for shot in example.few_shot_examples:
            inferred = _infer_shift(shot.input_text.strip(), shot.output_text.strip())
            if inferred is None:
                return SolverResult(handled=False, trace="examples do not share one alpha shift")
            if shift is None:
                shift = inferred
            elif shift != inferred:
                return SolverResult(handled=False, trace="inconsistent shifts across demos")

        if shift is None:
            return SolverResult(handled=False, trace="no shift inferred")

        answer = _apply_shift(example.task_text.strip(), shift)
        if answer is None:
            return SolverResult(handled=False, trace="task input contains unsupported chars")

        return SolverResult(
            handled=True,
            answer=answer,
            confidence=0.97,
            trace=f"uniform alpha shift {shift}",
        )


def _infer_shift(source: str, target: str) -> int | None:
    if len(source) != len(target) or not source:
        return None

    inferred_shift: int | None = None
    for left, right in zip(source, target, strict=True):
        if not left.isalpha() or not right.isalpha():
            return None
        if left.islower() != right.islower():
            return None
        base = ord("a") if left.islower() else ord("A")
        current_shift = (ord(right) - base) - (ord(left) - base)
        current_shift %= 26
        if inferred_shift is None:
            inferred_shift = current_shift
        elif inferred_shift != current_shift:
            return None
    return inferred_shift


def _apply_shift(text: str, shift: int) -> str | None:
    shifted_chars: list[str] = []
    for char in text:
        if not char.isalpha():
            return None
        base = ord("a") if char.islower() else ord("A")
        shifted_chars.append(chr(base + ((ord(char) - base + shift) % 26)))
    return "".join(shifted_chars)

