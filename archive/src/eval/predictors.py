"""Predictor backends for smoke tests and Kaggle execution."""

from __future__ import annotations

import os
import re
import site
import shutil
from pathlib import Path
from typing import Any

from src.eval.schemas import EvalExample, FewShotExample
from src.solvers.arithmetic import AffineArithmeticSolver
from src.solvers.base_conversion import BaseConversionSolver
from src.solvers.formatting import UnitConversionSolver
from src.solvers.string_shift import CaesarShiftSolver

TASK_PATTERN = re.compile(r"^Task Input:\s*(.*)$", re.MULTILINE)
VARIANT_PATTERN = re.compile(r"^Variant:\s*(.*)$", re.MULTILINE)
EXAMPLE_INPUT_PATTERN = re.compile(r"^Example (\d+) Input:\s*(.*)$", re.MULTILINE)
EXAMPLE_OUTPUT_PATTERN = re.compile(r"^Example (\d+) Output:\s*(.*)$", re.MULTILINE)


class HeuristicPredictor:
    """A lightweight predictor that keeps the evaluation path executable locally."""

    def __call__(self, prompts: list[str], *, stop: tuple[str, ...] = ()) -> list[str]:
        return [self._predict(prompt) for prompt in prompts]

    def _predict(self, prompt: str) -> str:
        variant = _extract_pattern(VARIANT_PATTERN, prompt) or "baseline_direct"
        example = _example_from_prompt(prompt)
        answer = self._solve(example)
        return _format_output(variant, answer)

    def _solve(self, example: EvalExample) -> str:
        for solver in (
            AffineArithmeticSolver(),
            BaseConversionSolver(),
            CaesarShiftSolver(),
            UnitConversionSolver(),
        ):
            result = solver.solve(example)
            if result.handled and result.answer is not None:
                return result.answer

        if _supports_reverse(example):
            return example.task_text[::-1]
        return "unknown"


class TransformersKagglePredictor:
    """Kaggle-backed Hugging Face predictor for the real baseline checkpoint."""

    def __init__(
        self,
        *,
        model_handle: str,
        max_new_tokens: int = 128,
        batch_size: int = 1,
        temperature: float = 0.0,
        do_sample: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _prepare_kaggle_runtime()
        model_path = _resolve_model_path(model_handle)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token or self._tokenizer.unk_token
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            dtype=_resolve_torch_dtype(torch, torch_dtype),
            device_map=device_map,
        )
        self._batch_size = max(1, int(batch_size))
        self._max_new_tokens = max(1, int(max_new_tokens))
        self._temperature = float(temperature)
        self._do_sample = bool(do_sample)

    def __call__(self, prompts: list[str], *, stop: tuple[str, ...] = ()) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(prompts), self._batch_size):
            batch = prompts[start : start + self._batch_size]
            tokenized = self._tokenizer(batch, return_tensors="pt", padding=True)
            prompt_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
            tokenized = {
                key: value.to(self._model.device)
                for key, value in tokenized.items()
            }
            generate_kwargs: dict[str, Any] = {
                **tokenized,
                "max_new_tokens": self._max_new_tokens,
                "do_sample": self._do_sample,
                "pad_token_id": self._tokenizer.pad_token_id,
            }
            if self._do_sample:
                generate_kwargs["temperature"] = self._temperature
            with self._torch.inference_mode():
                generated = self._model.generate(**generate_kwargs)
            decoded: list[str] = []
            for row_index, prompt_length in enumerate(prompt_lengths):
                completion_tokens = generated[row_index, int(prompt_length) :]
                decoded.append(
                    self._tokenizer.decode(completion_tokens, skip_special_tokens=True)
                )
            outputs.extend(_apply_stop_tokens(text, stop) for text in decoded)
        return outputs


def build_predictor(config: dict[str, Any]) -> HeuristicPredictor | TransformersKagglePredictor:
    """Build a predictor instance from experiment config."""

    predictor_config = config.get("predictor", config)
    predictor_type = predictor_config.get("type", "heuristic")
    if predictor_type == "heuristic":
        return HeuristicPredictor()
    if predictor_type == "transformers_kaggle":
        return TransformersKagglePredictor(
            model_handle=predictor_config["model_handle"],
            max_new_tokens=predictor_config.get("max_new_tokens", 128),
            batch_size=predictor_config.get("batch_size", 1),
            temperature=predictor_config.get("temperature", 0.0),
            do_sample=predictor_config.get("do_sample", False),
            trust_remote_code=predictor_config.get("trust_remote_code", True),
            torch_dtype=predictor_config.get("torch_dtype", "bfloat16"),
            device_map=predictor_config.get("device_map", "auto"),
        )
    raise ValueError(f"Unsupported predictor type: {predictor_type}")


def _example_from_prompt(prompt: str) -> EvalExample:
    task_text = _extract_pattern(TASK_PATTERN, prompt)
    if task_text is None:
        raise ValueError("Prompt missing task input.")

    inputs = {int(index): value for index, value in EXAMPLE_INPUT_PATTERN.findall(prompt)}
    outputs = {int(index): value for index, value in EXAMPLE_OUTPUT_PATTERN.findall(prompt)}

    shots = [
        FewShotExample(input_text=inputs[index], output_text=outputs[index])
        for index in sorted(set(inputs) & set(outputs))
    ]
    return EvalExample(
        example_id="prompt-derived",
        task_text=task_text,
        few_shot_examples=tuple(shots),
    )


def _extract_pattern(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def _supports_reverse(example: EvalExample) -> bool:
    if not example.few_shot_examples:
        return False
    return all(
        shot.output_text == shot.input_text[::-1]
        for shot in example.few_shot_examples
    )


def _format_output(variant: str, answer: str) -> str:
    if variant == "baseline_direct":
        return f"\\boxed{{{answer}}}"
    if variant == "reasoned_boxed":
        return f"The pattern is consistent across the demonstrations.\nFinal answer: \\boxed{{{answer}}}"
    if variant == "self_check_boxed":
        return (
            "I verified the inferred rule against the demonstrations once.\n"
            f"Final answer: \\boxed{{{answer}}}"
        )
    return f"\\boxed{{{answer}}}"


def _resolve_model_path(model_handle: str) -> str:
    handle_path = Path(model_handle)
    if handle_path.exists():
        return str(handle_path)
    import kagglehub

    return kagglehub.model_download(model_handle)


def _prepare_kaggle_runtime() -> None:
    """Mirror the organizer demo setup for Nemotron dependencies on Kaggle."""

    kaggle_root = Path("/kaggle")
    if not kaggle_root.exists():
        return

    candidate_paths = (
        Path(
            "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/"
            "nvidia_cutlass_dsl/python_packages/"
        ),
        Path(
            "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/"
            "nvidia_cutlass_dsl/python_packages/"
        ),
    )
    for candidate in candidate_paths:
        if candidate.exists():
            site.addsitedir(str(candidate))

    ptxas_candidates = (
        Path("/usr/local/cuda/bin/ptxas"),
        Path("/usr/bin/ptxas"),
        Path(
            "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/"
            "triton/backends/nvidia/bin/ptxas"
        ),
        Path(
            "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/"
            "triton/backends/nvidia/bin/ptxas"
        ),
    )
    executable_ptxas = next(
        (
            str(candidate)
            for candidate in ptxas_candidates
            if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK)
        ),
        shutil.which("ptxas"),
    )
    if executable_ptxas:
        os.environ.setdefault("TRITON_PTXAS_PATH", executable_ptxas)
        os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", executable_ptxas)


def _resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    normalized = dtype_name.casefold()
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[normalized]


def _apply_stop_tokens(text: str, stop_tokens: tuple[str, ...]) -> str:
    truncated = text
    for token in stop_tokens:
        index = truncated.find(token)
        if index != -1:
            truncated = truncated[:index]
    return truncated
