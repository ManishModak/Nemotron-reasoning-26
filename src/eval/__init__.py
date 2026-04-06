"""Evaluation helpers and score reporting."""

from src.eval.predictors import HeuristicPredictor, TransformersKagglePredictor, build_predictor
from src.eval.runner import evaluate_examples
from src.eval.schemas import EvalExample, EvalRunResult, FewShotExample, PredictionRecord

__all__ = [
    "EvalExample",
    "EvalRunResult",
    "FewShotExample",
    "HeuristicPredictor",
    "PredictionRecord",
    "TransformersKagglePredictor",
    "build_predictor",
    "evaluate_examples",
]
