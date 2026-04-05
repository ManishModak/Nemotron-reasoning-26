from src.eval.parsing import answers_match, extract_final_answer, normalize_answer


def test_extracts_last_boxed_answer() -> None:
    text = "Draft \\boxed{3}\nFinal answer: \\boxed{12}"
    assert extract_final_answer(text) == "12"


def test_extracts_plain_final_line() -> None:
    text = "Reasoning...\nFinal answer: 42"
    assert extract_final_answer(text) == "42"


def test_normalize_answer_handles_whitespace_and_case() -> None:
    assert normalize_answer("  Hello   World. ") == "hello world"


def test_answers_match_handles_numeric_formatting() -> None:
    assert answers_match("1,000", "1000")

