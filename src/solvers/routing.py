"""Symbolic routing for easy puzzle families."""

from __future__ import annotations

from dataclasses import dataclass

from src.eval.schemas import EvalExample
from src.solvers.arithmetic import AffineArithmeticSolver
from src.solvers.base import Solver, SolverResult
from src.solvers.base_conversion import BaseConversionSolver
from src.solvers.formatting import UnitConversionSolver
from src.solvers.gravity import GravityDistanceSolver
from src.solvers.proportional_units import ProportionalUnitSolver
from src.solvers.roman_numerals import RomanNumeralSolver
from src.solvers.string_shift import CaesarShiftSolver


@dataclass
class ConservativeRouter:
    """Try a small deterministic solver set before falling back to the predictor."""

    confidence_threshold: float = 0.95
    enabled_families: tuple[str, ...] | None = None
    solvers: tuple[Solver, ...] = (
        RomanNumeralSolver(),
        GravityDistanceSolver(),
        ProportionalUnitSolver(),
        AffineArithmeticSolver(),
        BaseConversionSolver(),
        CaesarShiftSolver(),
        UnitConversionSolver(),
    )

    def route(self, example: EvalExample) -> tuple[str, SolverResult]:
        if (
            self.enabled_families is not None
            and example.family_hint is not None
            and example.family_hint not in self.enabled_families
        ):
            return "llm_fallback", SolverResult(
                handled=False,
                trace=f"family {example.family_hint} disabled for routing",
            )
        best_name = "llm_fallback"
        best_result = SolverResult(handled=False, trace="no solver attempted")
        for solver in self.solvers:
            result = solver.solve(example)
            if not result.handled:
                continue
            if result.confidence > best_result.confidence:
                best_name = solver.name
                best_result = result

        if best_result.handled and best_result.confidence >= self.confidence_threshold:
            return best_name, best_result

        return "llm_fallback", SolverResult(
            handled=False,
            trace=f"best confidence {best_result.confidence:.3f} below threshold",
        )
