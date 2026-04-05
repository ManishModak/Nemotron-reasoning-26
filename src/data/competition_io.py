"""Dataset loading and normalization for local and Kaggle evaluation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from src.eval.schemas import EvalExample, FewShotExample

ID_KEYS = ("example_id", "id", "uid")
TASK_KEYS = ("task_text", "task", "prompt", "question", "query", "input")
ANSWER_KEYS = ("gold_answer", "answer", "target", "output")
FAMILY_KEYS = ("family_hint", "family", "category", "type")
SHOT_KEYS = ("few_shot_examples", "examples", "demos", "shots")
PAIR_PATTERN = re.compile(r"(.+?)\s*->\s*(.+)")
BIT_PAIR_PATTERN = re.compile(r"([01]{8})\s*->\s*([01]{8})")
BIT_QUERY_PATTERN = re.compile(r"Now,\s*determine the output for:\s*([01]{8})", re.IGNORECASE)


def repo_root() -> Path:
    """Return the repository root based on this module location."""

    return Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path) -> Path:
    """Resolve repo-relative or absolute paths."""

    path = Path(path_like)
    if path.is_absolute():
        return path
    return repo_root() / path


def discover_dataset_paths(base_dir: str | Path | None = None) -> dict[str, Path]:
    """Discover likely competition dataset files."""

    root = resolve_path(base_dir or ".")
    candidates = {
        "train": ["data/train.jsonl", "data/train.json", "data/train.csv"],
        "validation": [
            "data/validation.jsonl",
            "data/validation.json",
            "data/validation.csv",
        ],
        "test": ["data/test.jsonl", "data/test.json", "data/test.csv"],
        "smoke": ["artifacts/samples/smoke_eval_examples.jsonl"],
    }
    found: dict[str, Path] = {}
    for name, paths in candidates.items():
        for relative in paths:
            candidate = root / relative
            if candidate.exists():
                found[name] = candidate
                break
    return found


def load_eval_examples(path_like: str | Path) -> list[EvalExample]:
    """Load evaluation examples from CSV, JSON, or JSONL."""

    path = resolve_path(path_like)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("examples", [])
        else:
            rows = payload
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported dataset format: {path}")

    return [normalize_row(row, default_index=index) for index, row in enumerate(rows)]


def normalize_row(row: dict[str, Any], *, default_index: int = 0) -> EvalExample:
    """Normalize a raw dataset row into the internal evaluation schema."""

    row_copy = dict(row)
    example_id = _first_present(row_copy, ID_KEYS) or f"example-{default_index:04d}"
    raw_prompt = _first_present(row_copy, ("prompt",))
    task_text = _first_present(row_copy, TASK_KEYS)
    if task_text is None:
        raise ValueError(f"Missing task text in row {example_id}.")

    gold_answer = _first_present(row_copy, ANSWER_KEYS)
    family_hint = _first_present(row_copy, FAMILY_KEYS)
    raw_shots = _first_present(row_copy, SHOT_KEYS)
    if raw_shots is not None:
        few_shot_examples = _parse_shots(raw_shots)
    else:
        task_text, few_shot_examples, inferred_family = _parse_prompt_text(str(task_text))
        if family_hint is None:
            family_hint = inferred_family

    metadata = {
        key: value
        for key, value in row_copy.items()
        if key
        not in {
            *ID_KEYS,
            *TASK_KEYS,
            *ANSWER_KEYS,
            *FAMILY_KEYS,
            *SHOT_KEYS,
        }
    }
    if raw_prompt is not None:
        metadata["raw_prompt"] = str(raw_prompt)

    return EvalExample(
        example_id=str(example_id),
        task_text=str(task_text),
        few_shot_examples=tuple(few_shot_examples),
        gold_answer=None if gold_answer is None else str(gold_answer),
        family_hint=None if family_hint is None else str(family_hint),
        metadata=metadata,
    )


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_shots(raw_shots: Any) -> list[FewShotExample]:
    if isinstance(raw_shots, str):
        raw_shots = raw_shots.strip()
        if not raw_shots:
            return []
        raw_shots = json.loads(raw_shots)

    parsed: list[FewShotExample] = []
    for item in raw_shots:
        if isinstance(item, dict):
            input_text = item.get("input_text") or item.get("input") or item.get("x")
            output_text = item.get("output_text") or item.get("output") or item.get("y")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            input_text, output_text = item
        else:
            raise ValueError(f"Unsupported few-shot example format: {item!r}")
        parsed.append(FewShotExample(input_text=str(input_text), output_text=str(output_text)))
    return parsed


def _parse_prompt_text(prompt_text: str) -> tuple[str, list[FewShotExample], str | None]:
    """Extract task query and demonstrations from free-form competition prompts."""

    bit_query = BIT_QUERY_PATTERN.search(prompt_text)
    bit_pairs = BIT_PAIR_PATTERN.findall(prompt_text)
    if bit_query is not None and bit_pairs:
        return (
            bit_query.group(1),
            [
                FewShotExample(input_text=input_text, output_text=output_text)
                for input_text, output_text in bit_pairs
            ],
            "bit_manipulation",
        )

    lines = [line.strip() for line in prompt_text.splitlines() if line.strip()]
    pairs: list[FewShotExample] = []
    query = prompt_text
    for line in lines:
        pair_match = PAIR_PATTERN.fullmatch(line)
        if pair_match is not None:
            pairs.append(
                FewShotExample(
                    input_text=pair_match.group(1).strip(),
                    output_text=pair_match.group(2).strip(),
                )
            )
            continue
        lowered = line.casefold()
        if "determine the output for:" in lowered:
            query = line.split(":", 1)[-1].strip()

    if pairs and query != prompt_text:
        return query, pairs, "parsed_from_prompt"

    return prompt_text, [], None
