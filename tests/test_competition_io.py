from src.data.competition_io import normalize_row


def test_normalize_row_parses_competition_prompt() -> None:
    row = {
        "id": "00066667",
        "prompt": (
            "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.\n\n"
            "Here are some examples of input -> output:\n"
            "01010001 -> 11011101\n"
            "00001001 -> 01101101\n"
            "00010101 -> 01010101\n"
            "\nNow, determine the output for: 00110100"
        ),
        "answer": "10010111",
    }

    example = normalize_row(row)

    assert example.task_text == "00110100"
    assert len(example.few_shot_examples) == 3
    assert example.few_shot_examples[0].input_text == "01010001"
    assert example.few_shot_examples[0].output_text == "11011101"
    assert example.gold_answer == "10010111"
    assert example.family_hint == "bit_manipulation"
    assert "raw_prompt" in example.metadata
