"""Base solver types and interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.eval.schemas import EvalExample


@dataclass(frozen=True)
class SolverResult:
    """Result of a deterministic solver attempt."""

    handled: bool
    answer: str | None = None
    confidence: float = 0.0
    trace: str = ""


class Solver(Protocol):
    """Protocol implemented by deterministic rule solvers."""

    name: str

    def solve(self, example: EvalExample) -> SolverResult:
        """Try to solve an example deterministically."""

