from __future__ import annotations

import json
from pathlib import Path


def _execute_notebook(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__notebook__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, str(path), "exec"), namespace)
    return namespace


def test_baseline_notebook_smoke_executes() -> None:
    namespace = _execute_notebook(Path("notebooks/00_baseline_eval.ipynb"))
    assert "result" in namespace
    assert namespace["result"].metrics["total_predictions"] > 0


def test_inference_notebook_smoke_executes() -> None:
    namespace = _execute_notebook(Path("notebooks/04_inference_submission.ipynb"))
    assert "sample_result" in namespace
    assert namespace["sample_result"].metrics["total_predictions"] > 0
