"""Artifact and report generation for evaluation runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.eval.schemas import EvalRunResult, PredictionRecord


def build_error_buckets(predictions: list[PredictionRecord]) -> dict[str, list[str]]:
    """Bucket incorrect predictions into a few actionable categories."""

    buckets = {
        "parse_failure": [],
        "symbolic_miss": [],
        "llm_miss": [],
    }
    for prediction in predictions:
        if prediction.is_correct:
            continue
        if prediction.parsed_answer is None:
            buckets["parse_failure"].append(prediction.example_id)
        elif prediction.solver_name == "llm_fallback":
            buckets["llm_miss"].append(prediction.example_id)
        else:
            buckets["symbolic_miss"].append(prediction.example_id)
    return buckets


def write_run_artifacts(
    result: EvalRunResult,
    output_dir: str | Path,
    *,
    failure_sample_size: int = 5,
) -> dict[str, Path]:
    """Write markdown, CSV, and JSONL artifacts for one evaluation run."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions_csv = output_path / "predictions.csv"
    predictions_jsonl = output_path / "predictions.jsonl"
    summary_md = output_path / "summary.md"
    metrics_json = output_path / "metrics.json"

    _write_predictions_csv(result.predictions, predictions_csv)
    _write_predictions_jsonl(result.predictions, predictions_jsonl)
    summary_md.write_text(
        render_markdown_summary(result, failure_sample_size=failure_sample_size),
        encoding="utf-8",
    )
    metrics_json.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    result.artifact_paths.update(
        {
            "predictions_csv": predictions_csv,
            "predictions_jsonl": predictions_jsonl,
            "summary_md": summary_md,
            "metrics_json": metrics_json,
        }
    )
    return result.artifact_paths


def render_markdown_summary(
    result: EvalRunResult,
    *,
    failure_sample_size: int = 5,
) -> str:
    """Render a short markdown summary for reports and notebook output."""

    failure_sample = [
        prediction.example_id
        for prediction in result.predictions
        if not prediction.is_correct
    ][:failure_sample_size]
    failures_by_family = build_failure_samples(result.predictions, sample_size=failure_sample_size)
    lines = [
        f"# Evaluation Summary: {result.run_name}",
        "",
        f"- Total predictions: {result.metrics['total_predictions']}",
        f"- Overall accuracy: {result.metrics['overall_accuracy']:.3f}",
        f"- Routed predictions: {result.metrics['routed_predictions']}",
        f"- Fallback predictions: {result.metrics['fallback_predictions']}",
        "",
        "## Accuracy By Variant",
    ]
    for variant_name, accuracy in result.metrics["accuracy_by_variant"].items():
        lines.append(f"- {variant_name}: {accuracy:.3f}")
    lines.extend(["", "## Routed-vs-Fallback Breakdown"])
    for route_name, count in result.metrics["route_breakdown"].items():
        lines.append(f"- {route_name}: {count}")
    lines.extend(["", "## Errors By Family"])
    for family_name, count in result.metrics["errors_by_family"].items():
        lines.append(f"- {family_name}: {count}")
    lines.extend(["", "## Error Buckets"])
    for bucket_name, example_ids in result.error_buckets.items():
        lines.append(f"- {bucket_name}: {len(example_ids)}")
    lines.extend(["", "## Failure Sample"])
    if failure_sample:
        for example_id in failure_sample:
            lines.append(f"- {example_id}")
    else:
        lines.append("- none")
    lines.extend(["", "## Failure Samples By Family"])
    if failures_by_family:
        for family_name, example_ids in failures_by_family.items():
            lines.append(f"- {family_name}: {', '.join(example_ids)}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def export_handoff_bundle(
    result: EvalRunResult,
    output_dir: str | Path,
    *,
    selected_variant: str,
) -> Path:
    """Write a compact JSON bundle for the Kaggle handoff notebook."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bundle = {
        "run_name": result.run_name,
        "selected_variant": selected_variant,
        "metrics": result.metrics,
        "artifact_paths": {key: str(path) for key, path in result.artifact_paths.items()},
    }
    bundle_path = output_path / "handoff_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    result.artifact_paths["handoff_bundle"] = bundle_path
    return bundle_path


def build_failure_samples(
    predictions: list[PredictionRecord],
    *,
    sample_size: int = 5,
) -> dict[str, list[str]]:
    """Return a small sample of failing ids for each family."""

    failures: dict[str, list[str]] = {}
    for prediction in predictions:
        if prediction.is_correct:
            continue
        family_name = prediction.family_hint or "unknown_family"
        bucket = failures.setdefault(family_name, [])
        if len(bucket) < sample_size and prediction.example_id not in bucket:
            bucket.append(prediction.example_id)
    return failures


def _write_predictions_csv(predictions: list[PredictionRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "example_id",
                "variant_name",
                "solver_name",
                "raw_output",
                "parsed_answer",
                "normalized_answer",
                "gold_answer",
                "normalized_gold_answer",
                "is_correct",
                "latency_s",
                "family_hint",
            ]
        )
        for prediction in predictions:
            writer.writerow(
                [
                    prediction.example_id,
                    prediction.variant_name,
                    prediction.solver_name,
                    prediction.raw_output,
                    prediction.parsed_answer,
                    prediction.normalized_answer,
                    prediction.gold_answer,
                    prediction.normalized_gold_answer,
                    prediction.is_correct,
                    f"{prediction.latency_s:.6f}",
                    prediction.family_hint,
                ]
            )


def _write_predictions_jsonl(predictions: list[PredictionRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            payload = {
                "example_id": prediction.example_id,
                "variant_name": prediction.variant_name,
                "solver_name": prediction.solver_name,
                "raw_output": prediction.raw_output,
                "parsed_answer": prediction.parsed_answer,
                "normalized_answer": prediction.normalized_answer,
                "gold_answer": prediction.gold_answer,
                "normalized_gold_answer": prediction.normalized_gold_answer,
                "is_correct": prediction.is_correct,
                "latency_s": prediction.latency_s,
                "family_hint": prediction.family_hint,
            }
            handle.write(json.dumps(payload) + "\n")
