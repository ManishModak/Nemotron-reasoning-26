"""Answer extraction and normalization helpers."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

BOXED_PATTERN = re.compile(r"\\boxed\s*{([^}]*)}")
FINAL_PREFIX_PATTERN = re.compile(
    r"^(final answer|answer|output|prediction)\s*[:=-]\s*",
    re.IGNORECASE,
)


def extract_boxed_answer(text: str) -> str | None:
    """Return the last boxed answer if one is present."""

    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip() or None


def extract_final_answer(text: str) -> str | None:
    """Extract the best final answer candidate from model output."""

    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        return None

    candidate = FINAL_PREFIX_PATTERN.sub("", non_empty_lines[-1]).strip()
    return candidate or None


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
    """Normalize model and gold answers for exact-match scoring."""

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

    numeric = _normalize_numeric(candidate)
    if numeric is not None:
        return numeric

    return candidate.casefold()


def answers_match(predicted: str | None, gold: str | None) -> bool:
    """Return True when the normalized answers match exactly."""

    normalized_predicted = normalize_answer(predicted)
    normalized_gold = normalize_answer(gold)
    if normalized_predicted is None or normalized_gold is None:
        return False
    return normalized_predicted == normalized_gold

