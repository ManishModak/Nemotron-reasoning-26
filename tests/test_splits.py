from src.eval.schemas import EvalExample
from src.eval.splits import split_examples, stable_fraction


def make_example(example_id: str) -> EvalExample:
    return EvalExample(example_id=example_id, task_text="x", few_shot_examples=tuple())


def test_stable_fraction_is_repeatable() -> None:
    assert stable_fraction("sample-1", seed=13) == stable_fraction("sample-1", seed=13)


def test_split_examples_is_seed_stable() -> None:
    examples = [make_example(f"example-{index}") for index in range(20)]
    split_a = split_examples(examples, validation_ratio=0.3, seed=7)
    split_b = split_examples(examples, validation_ratio=0.3, seed=7)
    assert [item.example_id for item in split_a[1]] == [item.example_id for item in split_b[1]]

