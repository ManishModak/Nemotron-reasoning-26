from src.eval.schemas import EvalExample, FewShotExample
from src.solvers.gravity import GravityDistanceSolver
from src.solvers.proportional_units import ProportionalUnitSolver
from src.solvers.roman_numerals import RomanNumeralSolver


def test_roman_numeral_solver_handles_standard_conversion() -> None:
    example = EvalExample(
        example_id="roman-1",
        task_text="38",
        few_shot_examples=(
            FewShotExample(input_text="11", output_text="XI"),
            FewShotExample(input_text="15", output_text="XV"),
            FewShotExample(input_text="19", output_text="XIX"),
        ),
        family_hint="roman_numeral",
    )

    result = RomanNumeralSolver().solve(example)

    assert result.handled
    assert result.answer == "XXXVIII"


def test_gravity_distance_solver_infers_gravity() -> None:
    example = EvalExample(
        example_id="gravity-1",
        task_text="4.41s",
        few_shot_examples=(
            FewShotExample(input_text="1.37s", output_text="14.92 m"),
            FewShotExample(input_text="4.27s", output_text="144.96 m"),
            FewShotExample(input_text="3.28s", output_text="85.54 m"),
        ),
        family_hint="gravity_distance",
    )

    result = GravityDistanceSolver().solve(example)

    assert result.handled
    assert result.answer == "154.62"


def test_proportional_unit_solver_infers_factor() -> None:
    example = EvalExample(
        example_id="unit-1",
        task_text="25.09 m",
        few_shot_examples=(
            FewShotExample(input_text="10.08 m", output_text="6.69"),
            FewShotExample(input_text="17.83 m", output_text="11.83"),
            FewShotExample(input_text="35.85 m", output_text="23.79"),
        ),
        family_hint="unit_conversion",
    )

    result = ProportionalUnitSolver().solve(example)

    assert result.handled
    assert result.answer == "16.65"
