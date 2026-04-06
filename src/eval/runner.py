"""Evaluation runner for prompt and symbolic baselines."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from src.eval.parsing import answers_match, extract_final_answer, normalize_answer
from src.eval.reporting import build_error_buckets, write_run_artifacts
from src.eval.schemas import (
    EvalExample,
    EvalRunResult,
    PredictionRecord,
    PredictorProtocol,
    PromptVariant,
)
from src.solvers.routing import ConservativeRouter


def evaluate_examples(
    examples: list[EvalExample],
    *,
    prompt_variants: list[PromptVariant],
    predictor: PredictorProtocol,
    router: ConservativeRouter | None = None,
    run_name: str = "baseline_eval",
    output_dir: str | Path | None = None,
    failure_sample_size: int = 5,
) -> EvalRunResult:
    """Run evaluation across all examples and prompt variants."""

    predictions: list[PredictionRecord] = []
    for variant in prompt_variants:
        for example in examples:
            predictions.append(
                _evaluate_one(
                    example,
                    variant=variant,
                    predictor=predictor,
                    router=router,
                )
            )

    metrics = _build_metrics(predictions)
    result = EvalRunResult(
        run_name=run_name,
        predictions=predictions,
        metrics=metrics,
        error_buckets=build_error_buckets(predictions),
    )
    if output_dir is not None:
        write_run_artifacts(result, output_dir, failure_sample_size=failure_sample_size)
    return result


def _evaluate_one(
    example: EvalExample,
    *,
    variant: PromptVariant,
    predictor: PredictorProtocol,
    router: ConservativeRouter | None,
) -> PredictionRecord:
    start_time = time.perf_counter()
    solver_name = "llm_fallback"
    raw_output = ""
    parsed_answer: str | None = None

    if router is not None:
        solver_name, solver_result = router.route(example)
        if solver_result.handled and solver_result.answer is not None:
            raw_output = f"\\boxed{{{solver_result.answer}}}"
            parsed_answer = solver_result.answer

    if not raw_output:
        prompt = variant.render_prompt(example)
        raw_output = predictor([prompt], stop=variant.stop_tokens)[0]
        parsed_answer = extract_final_answer(raw_output)

    latency_s = time.perf_counter() - start_time
    normalized_answer = normalize_answer(parsed_answer)
    normalized_gold = normalize_answer(example.gold_answer)
    is_correct = answers_match(parsed_answer, example.gold_answer)

    return PredictionRecord(
        example_id=example.example_id,
        variant_name=variant.name,
        solver_name=solver_name,
        raw_output=raw_output,
        parsed_answer=parsed_answer,
        normalized_answer=normalized_answer,
        gold_answer=example.gold_answer,
        normalized_gold_answer=normalized_gold,
        is_correct=is_correct,
        latency_s=latency_s,
        family_hint=example.family_hint,
    )


def _build_metrics(predictions: list[PredictionRecord]) -> dict[str, object]:
    total_predictions = len(predictions)
    correct_predictions = sum(prediction.is_correct for prediction in predictions)
    accuracy_by_variant: dict[str, float] = {}
    route_breakdown: dict[str, int] = {}
    errors_by_family: dict[str, int] = {}
    routed_predictions = sum(prediction.solver_name != "llm_fallback" for prediction in predictions)
    latency_values = [prediction.latency_s for prediction in predictions]

    for variant_name in sorted({prediction.variant_name for prediction in predictions}):
        variant_predictions = [
            prediction for prediction in predictions if prediction.variant_name == variant_name
        ]
        variant_total = len(variant_predictions)
        variant_correct = sum(prediction.is_correct for prediction in variant_predictions)
        accuracy_by_variant[variant_name] = (
            variant_correct / variant_total if variant_total else 0.0
        )

    for prediction in predictions:
        route_breakdown[prediction.solver_name] = route_breakdown.get(prediction.solver_name, 0) + 1
        if not prediction.is_correct:
            family_key = prediction.family_hint or "unknown_family"
            errors_by_family[family_key] = errors_by_family.get(family_key, 0) + 1

    return {
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "overall_accuracy": (
            correct_predictions / total_predictions if total_predictions else 0.0
        ),
        "accuracy_by_variant": accuracy_by_variant,
        "routed_predictions": routed_predictions,
        "fallback_predictions": total_predictions - routed_predictions,
        "route_breakdown": route_breakdown,
        "errors_by_family": errors_by_family,
        "average_latency_s": statistics.fmean(latency_values) if latency_values else 0.0,
    }
