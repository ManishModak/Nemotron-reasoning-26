from src.data.competition_io import load_eval_examples, resolve_path
from src.solvers.routing import ConservativeRouter


def test_router_uses_symbolic_solver_for_easy_example() -> None:
    examples = load_eval_examples(resolve_path("artifacts/samples/smoke_eval_examples.jsonl"))
    arithmetic = next(example for example in examples if example.example_id == "arith_add_1")

    router = ConservativeRouter(confidence_threshold=0.95)
    solver_name, result = router.route(arithmetic)

    assert solver_name == "affine_arithmetic"
    assert result.answer == "12"


def test_router_falls_back_for_unknown_rule() -> None:
    examples = load_eval_examples(resolve_path("artifacts/samples/smoke_eval_examples.jsonl"))
    nonlinear = next(example for example in examples if example.example_id == "unknown_1")

    router = ConservativeRouter(confidence_threshold=0.95)
    solver_name, result = router.route(nonlinear)

    assert solver_name == "llm_fallback"
    assert not result.handled

