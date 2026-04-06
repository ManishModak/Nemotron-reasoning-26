from src.data.competition_io import load_eval_examples, resolve_path
from src.solvers.routing import ConservativeRouter
from src.eval.schemas import EvalExample, FewShotExample


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


def test_router_uses_new_family_specific_solver() -> None:
    example = EvalExample(
        example_id="roman-1",
        task_text="38",
        few_shot_examples=(
            FewShotExample(input_text="11", output_text="XI"),
            FewShotExample(input_text="15", output_text="XV"),
        ),
        family_hint="roman_numeral",
    )

    router = ConservativeRouter(
        confidence_threshold=0.95,
        enabled_families=("roman_numeral",),
    )
    solver_name, result = router.route(example)

    assert solver_name == "roman_numeral"
    assert result.answer == "XXXVIII"


def test_router_respects_enabled_families_gate() -> None:
    example = EvalExample(
        example_id="gravity-1",
        task_text="2.00s",
        few_shot_examples=(
            FewShotExample(input_text="1.00s", output_text="4.90 m"),
            FewShotExample(input_text="2.00s", output_text="19.60 m"),
        ),
        family_hint="gravity_distance",
    )

    router = ConservativeRouter(
        confidence_threshold=0.95,
        enabled_families=("roman_numeral",),
    )
    solver_name, result = router.route(example)

    assert solver_name == "llm_fallback"
    assert not result.handled
