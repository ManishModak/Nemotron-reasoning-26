"""Answer extraction and metric-aligned matching helpers."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)(?:\}|$)")
FINAL_ANSWER_PATTERNS = (
    re.compile(r"The final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
)
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
BINARY_PATTERN = re.compile(r"[01]+")


def extract_boxed_answer(text: str | None) -> str | None:
    """Return the last boxed answer if one is present."""

    if text is None:
        return None
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    non_empty = [match.strip() for match in matches if match.strip()]
    if non_empty:
        return non_empty[-1]
    stripped = matches[-1].strip()
    return stripped or None


def extract_final_answer(text: str | None) -> str | None:
    """Extract the final answer using the same priority as the official metric."""

    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed
    if text is None:
        return None

    for pattern in FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            candidate = matches[-1].strip()
            if candidate:
                return candidate

    matches = NUMBER_PATTERN.findall(text)
    if matches:
        return matches[-1]

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return non_empty_lines[-1] if non_empty_lines else None


def _normalize_numeric(candidate: str) -> str | None:
    stripped = candidate.replace(",", "")
    try:
        value = Decimal(stripped)
    except InvalidOperation:
        return None
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def normalize_answer(answer: str | None) -> str | None:
    """Normalize answers for reporting while preserving binary-string semantics."""

    if answer is None:
        return None

    candidate = answer.strip()
    candidate = candidate.strip("$")
    candidate = candidate.strip("`")
    candidate = candidate.strip('"')
    candidate = candidate.strip("'")
    candidate = candidate.strip()
    candidate = re.sub(r"\s+", " ", candidate)
    candidate = candidate.rstrip(".;,")

    if BINARY_PATTERN.fullmatch(candidate):
        return candidate

    numeric = _normalize_numeric(candidate)
    if numeric is not None:
        return numeric

    return candidate.casefold()


def answers_match(predicted: str | None, gold: str | None) -> bool:
    """Return True when the prediction matches using the official metric semantics."""

    if predicted is None or gold is None:
        return False

    gold_clean = gold.strip()
    predicted_clean = predicted.strip()

    if BINARY_PATTERN.fullmatch(gold_clean):
        return predicted_clean.lower() == gold_clean.lower()

    try:
        gold_num = float(gold_clean)
        predicted_num = float(predicted_clean)
        return math.isclose(gold_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted_clean.lower() == gold_clean.lower()
