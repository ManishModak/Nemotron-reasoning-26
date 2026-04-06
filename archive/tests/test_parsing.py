from src.eval.parsing import answers_match, extract_final_answer, normalize_answer


def test_extracts_last_boxed_answer() -> None:
    text = "Draft \\boxed{3}\nFinal answer: \\boxed{12}"
    assert extract_final_answer(text) == "12"


def test_extracts_metric_style_final_answer_prefix() -> None:
    text = "Reasoning...\nThe final answer is: 42"
    assert extract_final_answer(text) == "42"


def test_extracts_last_number_when_no_structured_format_exists() -> None:
    text = "Reasoning step 1 gives 5\nReasoning step 2 gives 17"
    assert extract_final_answer(text) == "17"


def test_extracts_last_nonempty_line_when_no_number_exists() -> None:
    text = "Reasoning...\nanswer candidate"
    assert extract_final_answer(text) == "answer candidate"


def test_normalize_answer_preserves_binary_strings() -> None:
    assert normalize_answer("00011011") == "00011011"


def test_normalize_answer_handles_whitespace_and_case() -> None:
    assert normalize_answer("  Hello   World. ") == "hello world"


def test_answers_match_handles_numeric_tolerance() -> None:
    assert answers_match("24.6401", "24.64")


def test_answers_match_treats_binary_as_exact_string() -> None:
    assert not answers_match("00011011", "11011")
    assert answers_match("10011000", "10011000")


def test_answers_match_falls_back_to_case_insensitive_strings() -> None:
    assert answers_match("xlvii", "XLVII")
