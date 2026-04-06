"""Deterministic evaluation splitting helpers."""

from __future__ import annotations

import hashlib

from src.eval.schemas import EvalExample


def stable_fraction(example_id: str, seed: int = 0) -> float:
    """Map an example id and seed to a stable float in [0, 1)."""

    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64)


def split_examples(
    examples: list[EvalExample],
    *,
    validation_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[EvalExample], list[EvalExample]]:
    """Split examples into train and validation partitions."""

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    train: list[EvalExample] = []
    validation: list[EvalExample] = []
    for example in examples:
        bucket = stable_fraction(example.example_id, seed=seed)
        target = validation if bucket < validation_ratio else train
        target.append(example)
    return train, validation

