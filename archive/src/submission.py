"""Submission helpers for Kaggle inference outputs."""

from __future__ import annotations

import csv
from pathlib import Path

from src.eval.schemas import PredictionRecord


def write_submission_csv(
    predictions: list[PredictionRecord],
    output_path: str | Path,
) -> Path:
    """Write a submission CSV from one-prediction-per-example records."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "answer"])
        for prediction in predictions:
            writer.writerow([prediction.example_id, prediction.parsed_answer or ""])
    return path
