"""Dataset loading, discovery, and normalization for local and Kaggle evaluation."""

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
GRAVITY_PAIR_PATTERN = re.compile(
    r"For t =\s*(-?\d+(?:\.\d+)?)s,\s*distance =\s*(-?\d+(?:\.\d+)?)\s*m",
    re.IGNORECASE,
)
GRAVITY_QUERY_PATTERN = re.compile(
    r"Now,\s*determine the falling distance for t =\s*(-?\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)
UNIT_PAIR_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([A-Za-z]+)\s+becomes\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
UNIT_QUERY_PATTERN = re.compile(
    r"Now,\s*convert the following measurement:\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z]+)",
    re.IGNORECASE,
)
ROMAN_QUERY_PATTERN = re.compile(
    r"Now,\s*write the number\s+(-?\d+)\s+in the Wonderland numeral system\.?",
    re.IGNORECASE,
)
TEXT_QUERY_PATTERN = re.compile(
    r"Now,\s*decrypt the following text:\s*(.+)",
    re.IGNORECASE,
)
EQUATION_PAIR_PATTERN = re.compile(r"(.+?)\s*=\s*(.+)")
EQUATION_QUERY_PATTERN = re.compile(
    r"Now,\s*determine the result for:\s*(.+)",
    re.IGNORECASE,
)
COMPETITION_HINTS = ("nvidia", "nemotron", "reasoning", "challenge", "competition")


def repo_root() -> Path:
    """Return the repository root based on this module location."""

    return Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path) -> Path:
    """Resolve repo-relative or absolute paths."""

    path = Path(path_like)
    if path.is_absolute():
        return path
    return repo_root() / path


def discover_dataset_file(
    filename: str,
    *,
    base_dir: str | Path | None = None,
    search_roots: tuple[str | Path, ...] = (),
) -> Path | None:
    """Discover one dataset file by name, preferring Kaggle competition inputs."""

    roots: list[Path] = []
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        roots.append(kaggle_root)
    for item in search_roots:
        candidate = resolve_path(item)
        if candidate.exists() and candidate not in roots:
            roots.append(candidate)
    base_root = resolve_path(base_dir or ".")
    if base_root.exists() and base_root not in roots:
        roots.append(base_root)
    repo = repo_root()
    if repo not in roots:
        roots.append(repo)

    candidates: list[Path] = []
    for search_root in roots:
        direct = search_root / filename
        if direct.is_file():
            candidates.append(direct)
        candidates.extend(path for path in search_root.rglob(filename) if path.is_file())

    if not candidates:
        return None
    return max(candidates, key=_score_dataset_candidate)


def discover_dataset_paths(base_dir: str | Path | None = None) -> dict[str, Path]:
    """Discover likely competition dataset files."""

    found: dict[str, Path] = {}
    for name, filename in (
        ("train", "train.csv"),
        ("validation", "validation.csv"),
        ("test", "test.csv"),
    ):
        candidate = discover_dataset_file(filename, base_dir=base_dir, search_roots=(Path.cwd(),))
        if candidate is not None:
            found[name] = candidate

    smoke = resolve_path("artifacts/samples/smoke_eval_examples.jsonl")
    if smoke.exists():
        found["smoke"] = smoke
    return found


def resolve_dataset_path(
    path_like: str | Path | None,
    *,
    fallback_filename: str | None = None,
    auto_discover: bool = False,
    base_dir: str | Path | None = None,
) -> Path:
    """Resolve an explicit or discovered dataset path."""

    if path_like not in (None, ""):
        explicit = resolve_path(path_like)
        if explicit.exists():
            return explicit
    if auto_discover and fallback_filename is not None:
        discovered = discover_dataset_file(
            fallback_filename,
            base_dir=base_dir,
            search_roots=(Path.cwd(),),
        )
        if discovered is not None:
            return discovered
    target = path_like or fallback_filename or "<unknown>"
    raise FileNotFoundError(f"Could not resolve dataset path for: {target}")


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
    """Extract task query and demonstrations from the competition prompt templates."""

    lowered = prompt_text.casefold()
    for parser in (
        _parse_bit_prompt,
        _parse_gravity_prompt,
        _parse_unit_prompt,
        _parse_roman_prompt,
        _parse_text_prompt,
        _parse_equation_prompt,
        _parse_generic_prompt,
    ):
        parsed = parser(prompt_text, lowered)
        if parsed is not None:
            return parsed
    return prompt_text, [], None


def _parse_bit_prompt(
    prompt_text: str,
    lowered: str,
) -> tuple[str, list[FewShotExample], str] | None:
    if "bit manipulation rule transforms 8-bit binary numbers" not in lowered:
        return None
    query_match = BIT_QUERY_PATTERN.search(prompt_text)
    pairs = BIT_PAIR_PATTERN.findall(prompt_text)
    if query_match is None or not pairs:
        return None
    return (
        query_match.group(1),
        [
            FewShotExample(input_text=input_text, output_text=output_text)
            for input_text, output_text in pairs
        ],
        "bit_manipulation",
    )


def _parse_gravity_prompt(
    prompt_text: str,
    lowered: str,
) -> tuple[str, list[FewShotExample], str] | None:
    if "gravitational constant has been secretly changed" not in lowered:
        return None
    query_match = GRAVITY_QUERY_PATTERN.search(prompt_text)
    pairs = GRAVITY_PAIR_PATTERN.findall(prompt_text)
    if query_match is None or not pairs:
        return None
    shots = [
        FewShotExample(input_text=f"{time_value}s", output_text=f"{distance_value} m")
        for time_value, distance_value in pairs
    ]
    return f"{query_match.group(1)}s", shots, "gravity_distance"


def _parse_unit_prompt(
    prompt_text: str,
    lowered: str,
) -> tuple[str, list[FewShotExample], str] | None:
    if "secret unit conversion is applied to measurements" not in lowered:
        return None
    query_match = UNIT_QUERY_PATTERN.search(prompt_text)
    pairs = UNIT_PAIR_PATTERN.findall(prompt_text)
    if query_match is None or not pairs:
        return None
    shots = [
        FewShotExample(input_text=f"{value} {unit}", output_text=converted)
        for value, unit, converted in pairs
    ]
    return f"{query_match.group(1)} {query_match.group(2)}", shots, "unit_conversion"


def _parse_roman_prompt(
    prompt_text: str,
    lowered: str,
) -> tuple[str, list[FewShotExample], str] | None:
    if "different numeral system" not in lowered:
        return None
    query_match = ROMAN_QUERY_PATTERN.search(prompt_text)
    if query_match is None:
        return None
    lines = [line.strip() for line in prompt_text.splitlines() if line.strip()]
    pairs = _collect_arrow_pairs(lines)
    if not pairs:
        return None
    return query_match.group(1), pairs, "roman_numeral"


def _parse_text_prompt(
    prompt_text: str,
    lowered: str,
) -> tuple[str, list[FewShotExample], str] | None:
    if "secret encryption rules are used on text" not in lowered:
        return None
    query_match = TEXT_QUERY_PATTERN.search(prompt_text)
    if query_match is None:
        return None
    lines = [line.strip() for line in prompt_text.splitlines() if line.strip()]
    pairs = _collect_arrow_pairs(lines)
    if not pairs:
        return None
    return query_match.group(1).strip(), pairs, "text_decryption"


def _parse_equation_prompt(
    prompt_text: str,
    lowered: str,
) -> tuple[str, list[FewShotExample], str] | None:
    if "secret set of transformation rules is applied to equations" not in lowered:
        return None
    query_match = EQUATION_QUERY_PATTERN.search(prompt_text)
    if query_match is None:
        return None
    lines = [line.strip() for line in prompt_text.splitlines() if line.strip()]
    pairs: list[FewShotExample] = []
    for line in lines:
        pair_match = EQUATION_PAIR_PATTERN.fullmatch(line)
        if pair_match is None:
            continue
        pairs.append(
            FewShotExample(
                input_text=pair_match.group(1).strip(),
                output_text=pair_match.group(2).strip(),
            )
        )
    if not pairs:
        return None
    return query_match.group(1).strip(), pairs, "equation_transform"


def _parse_generic_prompt(
    prompt_text: str,
    _: str,
) -> tuple[str, list[FewShotExample], str] | None:
    lines = [line.strip() for line in prompt_text.splitlines() if line.strip()]
    pairs = _collect_arrow_pairs(lines)
    query: str | None = None
    for line in lines:
        lowered_line = line.casefold()
        if "determine the output for:" in lowered_line or "decrypt the following text:" in lowered_line:
            query = line.split(":", 1)[-1].strip()
    if pairs and query is not None:
        return query, pairs, "parsed_from_prompt"
    return None


def _collect_arrow_pairs(lines: list[str]) -> list[FewShotExample]:
    pairs: list[FewShotExample] = []
    for line in lines:
        pair_match = PAIR_PATTERN.fullmatch(line)
        if pair_match is None:
            continue
        pairs.append(
            FewShotExample(
                input_text=pair_match.group(1).strip(),
                output_text=pair_match.group(2).strip(),
            )
        )
    return pairs


def _score_dataset_candidate(path: Path) -> tuple[int, int, str]:
    lowered = path.as_posix().casefold()
    score = 0
    if lowered.startswith("/kaggle/input/"):
        score += 100
    if any(hint in lowered for hint in COMPETITION_HINTS):
        score += 20
    if any(noisy in lowered for noisy in ("sample", "smoke", "artifact", "artifacts", "docs")):
        score -= 30
    return score, -len(lowered), lowered
