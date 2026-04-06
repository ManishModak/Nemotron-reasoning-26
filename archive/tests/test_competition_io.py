from src.data.competition_io import normalize_row


def test_normalize_row_parses_bit_manipulation_prompt() -> None:
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
    assert example.family_hint == "bit_manipulation"
    assert "raw_prompt" in example.metadata


def test_normalize_row_parses_gravity_prompt() -> None:
    row = {
        "id": "gravity-1",
        "prompt": (
            "In Alice's Wonderland, the gravitational constant has been secretly changed. "
            "Here are some example observations:\n"
            "For t = 1.37s, distance = 14.92 m\n"
            "For t = 4.27s, distance = 144.96 m\n"
            "For t = 3.28s, distance = 85.54 m\n"
            "Now, determine the falling distance for t = 4.41s given d = 0.5*g*t^2."
        ),
        "answer": "154.62",
    }

    example = normalize_row(row)

    assert example.task_text == "4.41s"
    assert len(example.few_shot_examples) == 3
    assert example.few_shot_examples[0].input_text == "1.37s"
    assert example.few_shot_examples[0].output_text == "14.92 m"
    assert example.family_hint == "gravity_distance"


def test_normalize_row_parses_unit_conversion_prompt() -> None:
    row = {
        "id": "unit-1",
        "prompt": (
            "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:\n"
            "10.08 m becomes 6.69\n"
            "17.83 m becomes 11.83\n"
            "35.85 m becomes 23.79\n"
            "Now, convert the following measurement: 25.09 m"
        ),
        "answer": "16.65",
    }

    example = normalize_row(row)

    assert example.task_text == "25.09 m"
    assert len(example.few_shot_examples) == 3
    assert example.few_shot_examples[0].input_text == "10.08 m"
    assert example.few_shot_examples[0].output_text == "6.69"
    assert example.family_hint == "unit_conversion"


def test_normalize_row_parses_roman_prompt() -> None:
    row = {
        "id": "roman-1",
        "prompt": (
            "In Alice's Wonderland, numbers are secretly converted into a different numeral system. "
            "Some examples are given below:\n"
            "11 -> XI\n"
            "15 -> XV\n"
            "94 -> XCIV\n"
            "19 -> XIX\n"
            "Now, write the number 38 in the Wonderland numeral system."
        ),
        "answer": "XXXVIII",
    }

    example = normalize_row(row)

    assert example.task_text == "38"
    assert len(example.few_shot_examples) == 4
    assert example.few_shot_examples[0].input_text == "11"
    assert example.few_shot_examples[0].output_text == "XI"
    assert example.family_hint == "roman_numeral"


def test_normalize_row_parses_text_decryption_prompt() -> None:
    row = {
        "id": "text-1",
        "prompt": (
            "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:\n"
            "ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley\n"
            "pqrsfv pqorzg wvgwpo trgbjo -> dragon dreams inside castle\n"
            "bxo sfjpov pqrsfv dfjjfig -> the golden dragon follows\n"
            "Now, decrypt the following text: trb wzrswvog hffk"
        ),
        "answer": "cat imagines book",
    }

    example = normalize_row(row)

    assert example.task_text == "trb wzrswvog hffk"
    assert len(example.few_shot_examples) == 3
    assert example.family_hint == "text_decryption"


def test_normalize_row_parses_equation_transform_prompt() -> None:
    row = {
        "id": "eq-1",
        "prompt": (
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
            "Below are a few examples:\n"
            "`!*[{ = '\"[`\n"
            "\\'*'> = ![@\n"
            "'-!` = \\\n"
            "Now, determine the result for: [[-!'"
        ),
        "answer": "@&",
    }

    example = normalize_row(row)

    assert example.task_text == "[[-!'"
    assert len(example.few_shot_examples) == 3
    assert example.few_shot_examples[0].input_text == "`!*[{"
    assert example.few_shot_examples[0].output_text == "'\"[`"
    assert example.family_hint == "equation_transform"
