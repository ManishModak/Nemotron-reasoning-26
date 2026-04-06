from pathlib import Path

from src.data.competition_io import load_eval_examples, resolve_path
from src.eval.predictors import HeuristicPredictor
from src.eval.runner import evaluate_examples
from src.prompts.registry import get_prompt_variants
from src.solvers.routing import ConservativeRouter


def test_runner_writes_artifacts(tmp_path: Path) -> None:
    examples = load_eval_examples(resolve_path("artifacts/samples/smoke_eval_examples.jsonl"))
    validation_examples = examples[:4]

    result = evaluate_examples(
        validation_examples,
        prompt_variants=get_prompt_variants(),
        predictor=HeuristicPredictor(),
        router=ConservativeRouter(confidence_threshold=0.95),
        output_dir=tmp_path,
        failure_sample_size=2,
    )

    assert result.metrics["total_predictions"] == 12
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert result.metrics["routed_predictions"] > 0
