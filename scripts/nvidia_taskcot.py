#!/usr/bin/env python3
"""
Single-pass CoT generation pipeline via NVIDIA build API (or any OpenAI-compatible endpoint).

Flow per row (1-2 API calls, not 3):
  1. Ask model for step-by-step CoT + \\boxed{answer}.
  2. Verify: extract \\boxed{}, compare to gold.
     - Correct  -> tier1_genuine (1 call).
     - Wrong    -> tier2: default ``session`` = one chat with [user, assistant, user] so the
       model sees its wrong CoT; ``standalone`` = fresh user-only message (old behavior).

CSV checkpoint after every row — re-run to resume. Press 'q' + Enter to gracefully stop.

Setup:
  NVIDIA_API_KEY=nvapi-...   (or NVAPI_KEY)  in .env or environment

Usage:
  python scripts/nvidia_taskcot.py                                    # all 500 rows, default model
  python scripts/nvidia_taskcot.py --sample datasets/benchmark_sample_500.csv
  python scripts/nvidia_taskcot.py --model minimaxai/minimax-m2.7 --max-tokens 16384
  python scripts/nvidia_taskcot.py --log-file nvidia_benchmark_outputs/my_run.log
  python scripts/nvidia_taskcot.py --no-run-log

Outputs: nvidia_benchmark_outputs/ (override with --out-dir)
  cot_all_<model>.csv           — every row (tier1 + tier2, verified or not)
  cot_traces_verified.csv       — only verified rows (for training)
  summary.txt                   — per-family stats
  run_<model>_<UTC>.log         — real-time detailed log (HTTP, retries, full assistant text)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import select
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

RATE_LIMIT_BACKOFF_MAX_S = 300.0

DEFAULT_MODEL = "minimaxai/minimax-m2.7"
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TIMEOUT_S = 360

# ─── Prompts ───

COT_PROMPT = """{question}

Think through this step by step. Show your complete reasoning process,
then put your final answer inside \\boxed{{}}.
For example: \\boxed{{your answer}}"""

CORRECTION_PROMPT = """{question}

Your previous answer was incorrect. The objectively correct answer is: {gold_answer}

Recalibrate your thinking. Generate a rigorous, step-by-step Chain of Thought
that logically and flawlessly arrives at this correct answer.
Explicitly point out the hidden trap or complex step where a system might typically fail.
Put your final answer inside \\boxed{{}}. For example: \\boxed{{your answer}}"""

# Second user turn when using multi-turn correction (question already in first user message).
CORRECTION_FOLLOWUP = """The objectively correct answer for this puzzle is: {gold_answer}

Your previous response did not match this answer. Recalibrate your thinking. Generate a rigorous,
step-by-step Chain of Thought that logically arrives at this correct answer.
Explicitly point out where prior reasoning went wrong if applicable.
Put your final answer inside \\boxed{{}}. For example: \\boxed{{your answer}}"""

# ─── Terminal colors (ANSI) ───

_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def _dim(t: str) -> str:   return _c("2", t)
def _bold(t: str) -> str:  return _c("1", t)
def _green(t: str) -> str: return _c("32", t)
def _red(t: str) -> str:   return _c("31", t)
def _yellow(t: str) -> str:return _c("33", t)
def _cyan(t: str) -> str:  return _c("36", t)

# ─── Env loading ───

def _parse_env_line(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.startswith("export "):
        s = s[7:].lstrip()
    if "=" not in s:
        return None
    key, _, val = s.partition("=")
    key = key.strip()
    if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
        return None
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        val = val[1:-1]
    return key, val


def load_env_file(path: Path, *, override: bool = False) -> int:
    if not path.is_file():
        return 0
    n = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw)
        if not parsed:
            continue
        k, v = parsed
        if not override and k in os.environ and os.environ[k] != "":
            continue
        os.environ[k] = v
        n += 1
    return n


def load_env() -> None:
    for p in [Path.cwd() / ".env", REPO_ROOT / ".env"]:
        if load_env_file(p):
            return


# ─── Helpers ───

def normalize_text(x: str) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    text = unicodedata.normalize("NFKC", str(x)).strip()
    return re.sub(r"\s+", " ", text).casefold()


def normalized_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def classify_family(prompt_text: str) -> str:
    p = str(prompt_text).lower()
    if "bit manipulation" in p:        return "bit_manipulation"
    if "gravitational constant" in p:   return "gravity"
    if "encryption rules" in p or "encryption" in p: return "encryption"
    if "numeral system" in p:           return "numeral"
    if "unit conversion" in p or "converted" in p:   return "unit_conversion"
    if "transformation rules" in p or "equation" in p or "operator" in p: return "equations"
    return "unknown"


def extract_boxed(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not matches:
        return ""
    non_empty = [m.strip() for m in matches if m.strip()]
    return non_empty[-1] if non_empty else matches[-1].strip()


def extract_final_answer(raw: str, gold: str) -> str:
    if not raw:
        return ""
    tail = raw
    if "</think>" in raw:
        tail = raw.rsplit("</think>", 1)[-1].strip()

    boxed = extract_boxed(tail) or extract_boxed(raw)
    if boxed:
        return boxed.strip()

    gold_s = str(gold).strip()
    n_bits = len(gold_s) if re.fullmatch(r"[01]+", gold_s) else 0
    if n_bits:
        found = re.findall(rf"\b[01]{{{n_bits}}}\b", tail or raw)
        if found:
            return found[-1]

    lines = [ln.strip() for ln in (tail or raw).splitlines() if ln.strip()]
    return lines[-1] if lines else str(raw).strip()


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def _short(s: str, n: int = 80) -> str:
    s = str(s).replace("\n", " ")
    return (s[: n - 3] + "...") if len(s) > n else s


def _thinking_preview(raw: str, max_chars: int = 120) -> str:
    """Extract a short preview of the <think> block content."""
    if "<think>" not in raw:
        return ""
    m = re.search(r"<think>(.*?)(?:</think>|$)", raw, re.DOTALL)
    if not m:
        return ""
    inner = m.group(1).strip().replace("\n", " ")
    inner = re.sub(r"\s+", " ", inner)
    if len(inner) > max_chars:
        inner = inner[:max_chars - 3] + "..."
    return inner


# ─── Run file log (plain text, flush every line) ───


class RunFileLog:
    """Append-style detailed log: timestamps, HTTP metadata, full assistant generations."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = path.open("w", encoding="utf-8", newline="\n")
        self._closed = False

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def line(self, msg: str) -> None:
        if self._closed:
            return
        for part in msg.splitlines() or [""]:
            self._f.write(f"{self._ts()} {part}\n")
        self._f.flush()

    def block(self, header: str, body: str) -> None:
        if self._closed:
            return
        self._f.write(f"{self._ts()} === {header} ===\n")
        self._f.write(body)
        if body and not body.endswith("\n"):
            self._f.write("\n")
        self._f.write(f"{self._ts()} === end {header} ===\n")
        self._f.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._f.close()


def _messages_char_count(messages: list[dict[str, str]]) -> tuple[int, int]:
    n = len(messages)
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    return n, chars


# ─── Graceful quit via 'q' key ───

_quit_flag = threading.Event()

def _quit_listener() -> None:
    """Background thread: read stdin, set flag on 'q'."""
    try:
        while not _quit_flag.is_set():
            if sys.platform == "win32":
                import msvcrt
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch.lower() == "q":
                        _quit_flag.set()
                        return
                time.sleep(0.2)
            else:
                r, _, _ = select.select([sys.stdin], [], [], 0.3)
                if r:
                    line = sys.stdin.readline().strip().lower()
                    if line == "q":
                        _quit_flag.set()
                        return
    except Exception:
        pass


def should_quit() -> bool:
    return _quit_flag.is_set()


# ─── API call ───

def _parse_retry_after(header_val: str | None) -> float | None:
    if not header_val:
        return None
    s = header_val.strip()
    if not s.isdigit():
        return None
    return float(min(int(s), int(RATE_LIMIT_BACKOFF_MAX_S)))


def _backoff_wait(*, attempt: int, sleep_s: float, err: str, meta: dict) -> float:
    jitter = random.uniform(0, 0.75)
    base = sleep_s * (2 ** attempt) + jitter
    ra = meta.get("retry_after_s")
    if isinstance(ra, (int, float)) and ra > 0:
        base = max(base, float(ra))
    code = meta.get("http_status")
    if code in (408, 429, 503, 504):
        base = max(base, sleep_s * (2 ** (attempt + 1)))
    el = err.lower()
    if "429" in err or "rate" in el or "too many" in el or "throttl" in el:
        base = max(base, sleep_s * (2 ** (attempt + 2)))
    return min(float(base), RATE_LIMIT_BACKOFF_MAX_S)


def chat(
    api_url: str,
    api_key: str,
    model: str,
    user_content: str,
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
    log: RunFileLog | None = None,
    log_tag: str = "",
) -> tuple[str, str, dict[str, Any]]:
    meta: dict[str, Any] = {}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    if log:
        log.line(
            f"HTTP_REQUEST kind=single_user tag={log_tag} url={api_url} model={model} "
            f"user_chars={len(user_content)} json_bytes={len(data)} "
            f"max_tokens={max_tokens} temperature={temperature} timeout_s={timeout}"
        )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            dt = time.monotonic() - t0
            meta["http_dt_s"] = dt
            payload = json.loads(raw)
    except urllib.error.HTTPError as e:
        dt = time.monotonic() - t0
        meta["http_dt_s"] = dt
        meta["http_status"] = int(e.code)
        hdrs = getattr(e, "headers", None)
        ra = _parse_retry_after(hdrs.get("Retry-After") if hdrs else None)
        if ra is not None:
            meta["retry_after_s"] = ra
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = str(e)
        err_one = f"HTTP {e.code}: {err_body[:2000]}"
        if log:
            eb = err_body[:12000].replace("\n", " | ").replace("\r", "")
            log.line(
                f"HTTP_FAIL tag={log_tag} status={e.code} dt_s={dt:.3f} "
                f"retry_after={meta.get('retry_after_s')} body={eb}"
            )
        return "", err_one, meta
    except Exception as e:
        dt = time.monotonic() - t0
        meta["http_dt_s"] = dt
        if log:
            log.line(f"HTTP_EXCEPTION tag={log_tag} dt_s={dt:.3f} err={repr(e)}")
        return "", repr(e), meta

    try:
        ch0 = payload["choices"][0]
        content = str(ch0.get("message", {}).get("content", "")).strip()
        meta["finish_reason"] = str(ch0.get("finish_reason") or "")
        meta["response_id"] = str(payload.get("id") or "")
        if "usage" in payload:
            meta["usage"] = payload["usage"]
        if log:
            usage_s = json.dumps(payload.get("usage") or {}, default=str)
            log.line(
                f"HTTP_OK tag={log_tag} dt_s={dt:.3f} id={meta['response_id']} "
                f"finish_reason={meta['finish_reason']} usage={usage_s} "
                f"assistant_chars={len(content)}"
            )
            log.block(f"ASSISTANT {log_tag}", content)
        return content, "", meta
    except (KeyError, IndexError, TypeError):
        bad = f"Bad response: {json.dumps(payload)[:1500]}"
        if log:
            log.line(f"PARSE_FAIL tag={log_tag} {bad}")
        return "", bad, meta


def chat_messages(
    api_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
    log: RunFileLog | None = None,
    log_tag: str = "",
) -> tuple[str, str, dict[str, Any]]:
    meta: dict[str, Any] = {}
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    n_msg, msg_chars = _messages_char_count(messages)
    roles = ",".join(str(m.get("role", "?")) for m in messages)
    if log:
        log.line(
            f"HTTP_REQUEST kind=multi_turn tag={log_tag} url={api_url} model={model} "
            f"messages={n_msg} roles=[{roles}] total_content_chars={msg_chars} "
            f"json_bytes={len(data)} max_tokens={max_tokens} temperature={temperature} "
            f"timeout_s={timeout}"
        )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            dt = time.monotonic() - t0
            meta["http_dt_s"] = dt
            payload = json.loads(raw)
    except urllib.error.HTTPError as e:
        dt = time.monotonic() - t0
        meta["http_dt_s"] = dt
        meta["http_status"] = int(e.code)
        hdrs = getattr(e, "headers", None)
        ra = _parse_retry_after(hdrs.get("Retry-After") if hdrs else None)
        if ra is not None:
            meta["retry_after_s"] = ra
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = str(e)
        err_one = f"HTTP {e.code}: {err_body[:2000]}"
        if log:
            eb = err_body[:12000].replace("\n", " | ").replace("\r", "")
            log.line(
                f"HTTP_FAIL tag={log_tag} status={e.code} dt_s={dt:.3f} "
                f"retry_after={meta.get('retry_after_s')} body={eb}"
            )
        return "", err_one, meta
    except Exception as e:
        dt = time.monotonic() - t0
        meta["http_dt_s"] = dt
        if log:
            log.line(f"HTTP_EXCEPTION tag={log_tag} dt_s={dt:.3f} err={repr(e)}")
        return "", repr(e), meta

    try:
        ch0 = payload["choices"][0]
        content = str(ch0.get("message", {}).get("content", "")).strip()
        meta["finish_reason"] = str(ch0.get("finish_reason") or "")
        meta["response_id"] = str(payload.get("id") or "")
        if "usage" in payload:
            meta["usage"] = payload["usage"]
        if log:
            usage_s = json.dumps(payload.get("usage") or {}, default=str)
            log.line(
                f"HTTP_OK tag={log_tag} dt_s={dt:.3f} id={meta['response_id']} "
                f"finish_reason={meta['finish_reason']} usage={usage_s} "
                f"assistant_chars={len(content)}"
            )
            log.block(f"ASSISTANT {log_tag}", content)
        return content, "", meta
    except (KeyError, IndexError, TypeError):
        bad = f"Bad response: {json.dumps(payload)[:1500]}"
        if log:
            log.line(f"PARSE_FAIL tag={log_tag} {bad}")
        return "", bad, meta


def chat_with_retries(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
    sleep_s: float,
    max_retries: int,
    label: str = "",
    log: RunFileLog | None = None,
    log_tag: str = "",
) -> tuple[str, str]:
    last_err = ""
    for attempt in range(max_retries):
        tag = f"{log_tag} attempt={attempt + 1}/{max_retries}" if log_tag else f"attempt={attempt + 1}/{max_retries}"
        text, err, meta = chat(
            api_url, api_key, model, prompt,
            max_tokens=max_tokens, temperature=temperature, timeout=timeout,
            log=log, log_tag=tag,
        )
        if not err:
            return text, ""
        last_err = err
        wait_s = _backoff_wait(attempt=attempt, sleep_s=sleep_s, err=err, meta=meta)
        if log:
            log.line(
                f"RETRY_SCHEDULED tag={log_tag} attempt={attempt + 1}/{max_retries} "
                f"wait_s={wait_s:.3f} http_status={meta.get('http_status')} "
                f"err={last_err[:8000]}"
            )
        if label:
            print(
                f"    {_yellow('retry')} {attempt+1}/{max_retries} in {wait_s:.1f}s: {_short(err, 90)}",
                flush=True,
            )
        time.sleep(wait_s)
    if log:
        log.line(f"RETRY_EXHAUSTED tag={log_tag} last_err={last_err[:8000]}")
    return "", last_err


def chat_messages_with_retries(
    api_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
    sleep_s: float,
    max_retries: int,
    label: str = "",
    log: RunFileLog | None = None,
    log_tag: str = "",
) -> tuple[str, str]:
    last_err = ""
    for attempt in range(max_retries):
        tag = f"{log_tag} attempt={attempt + 1}/{max_retries}" if log_tag else f"attempt={attempt + 1}/{max_retries}"
        text, err, meta = chat_messages(
            api_url, api_key, model, messages,
            max_tokens=max_tokens, temperature=temperature, timeout=timeout,
            log=log, log_tag=tag,
        )
        if not err:
            return text, ""
        last_err = err
        wait_s = _backoff_wait(attempt=attempt, sleep_s=sleep_s, err=err, meta=meta)
        if log:
            log.line(
                f"RETRY_SCHEDULED tag={log_tag} attempt={attempt + 1}/{max_retries} "
                f"wait_s={wait_s:.3f} http_status={meta.get('http_status')} "
                f"err={last_err[:8000]}"
            )
        if label:
            print(
                f"    {_yellow('retry')} {attempt+1}/{max_retries} in {wait_s:.1f}s: {_short(err, 90)}",
                flush=True,
            )
        time.sleep(wait_s)
    if log:
        log.line(f"RETRY_EXHAUSTED tag={log_tag} last_err={last_err[:8000]}")
    return "", last_err


# ─── CSV helpers ───

FIELDS = [
    "row_id", "question", "gold_answer", "family",
    "cot_response", "extracted_answer", "verified", "tier",
    "cot_response_t2", "extracted_answer_t2", "verified_t2",
    "correction_mode", "model", "task_error",
]


def load_done_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    if "row_id" not in df.columns:
        return set()
    return set(df["row_id"].astype(str))


def append_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


# ─── Main loop ───

def run(
    df: pd.DataFrame,
    model: str,
    out_dir: Path,
    api_key: str,
    api_url: str,
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
    sleep_s: float,
    max_retries: int,
    correction_mode: str,
    enable_run_log: bool = True,
    run_log_path: Path | None = None,
) -> Path:
    slug = model_slug(model)
    all_csv = out_dir / f"cot_all_{slug}.csv"
    done = load_done_ids(all_csv)

    if all_csv.exists():
        try:
            peek = pd.read_csv(all_csv, nrows=0)
            if "correction_mode" not in peek.columns:
                full_prev = pd.read_csv(all_csv)
                for c in FIELDS:
                    if c not in full_prev.columns:
                        full_prev[c] = ""
                full_prev[FIELDS].to_csv(all_csv, index=False)
                print(_yellow(f"Migrated {all_csv.name}: added correction_mode column"), flush=True)
                done = load_done_ids(all_csv)
        except Exception:
            pass

    rows = [r for _, r in df.iterrows() if str(r["row_id"]) not in done]
    n_total = len(rows)
    n_done_prev = len(done)

    print(flush=True)
    print(_bold(f"  Model:    {model}"), flush=True)
    print(_bold(f"  Rows:     {n_total} to run, {n_done_prev} cached"), flush=True)
    print(_bold(f"  Tokens:   max_tokens={max_tokens}, timeout={timeout}s"), flush=True)
    print(_bold(f"  Output:   {all_csv}"), flush=True)
    print(_bold(f"  Tier2:    correction_mode={correction_mode}"), flush=True)

    run_log: RunFileLog | None = None
    run_log_file: Path | None = None
    if enable_run_log:
        run_log_file = run_log_path or (
            out_dir / f"run_{slug}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        )
        run_log = RunFileLog(run_log_file)
        run_log.line(
            f"PIPELINE_START model={model} api_url={api_url} sample_rows={n_total} "
            f"cached={n_done_prev} max_tokens={max_tokens} temperature={temperature} "
            f"timeout_s={timeout} sleep_s={sleep_s} max_retries={max_retries} "
            f"correction_mode={correction_mode}"
        )
        print(_bold(f"  Run log:  {run_log_file}"), flush=True)
    print(f"  Press {_cyan('q + Enter')} to stop gracefully.", flush=True)
    print(flush=True)

    stats = {"t1_ok": 0, "t2_ok": 0, "t2_fail": 0, "errors": 0, "total": 0}
    fam_stats: dict[str, dict[str, int]] = {}

    try:
        for i, row in enumerate(rows, start=1):
            if should_quit():
                print(f"\n{_yellow('Stopping')} (q pressed). {stats['total']} rows done this run.", flush=True)
                if run_log:
                    run_log.line(f"QUIT_REQUESTED after_row={stats['total']} iter={i}/{n_total}")
                break

            full_rid = str(row["row_id"])
            rid = full_rid[:12]
            q = str(row["question"]).strip()
            gold = str(row["answer"]).strip()
            fam = classify_family(q)
            base_tag = f"row_id={full_rid} iter={i}/{n_total} family={fam}"

            if fam not in fam_stats:
                fam_stats[fam] = {"total": 0, "verified": 0}
            fam_stats[fam]["total"] += 1

            if run_log:
                run_log.line(
                    f"ROW_START {base_tag} gold={gold!r} "
                    f"question_chars={len(q)} preview={_short(q, 240)!r}"
                )

            # ── Header ──
            print(
                f"{_bold(f'[{i}/{n_total}]')} {_cyan(fam):20s} id={_dim(rid)}",
                end="  ",
                flush=True,
            )

            # ── Pass 1: CoT ──
            t0 = time.monotonic()
            prompt1 = COT_PROMPT.format(question=q)
            cot_raw, err1 = chat_with_retries(
                api_url, api_key, model, prompt1,
                max_tokens=max_tokens, temperature=temperature, timeout=timeout,
                sleep_s=sleep_s, max_retries=max_retries, label=f"[{i}]",
                log=run_log, log_tag=f"{base_tag} phase=T1",
            )
            dt1 = time.monotonic() - t0

            if err1:
                print(_red(f"ERROR {_short(err1, 60)}  ({dt1:.0f}s)"), flush=True)
                if run_log:
                    run_log.line(f"ROW_END {base_tag} tier=error total_dt_s={dt1:.3f} err={err1[:8000]}")
                rec = {
                    "row_id": row["row_id"], "question": q, "gold_answer": gold, "family": fam,
                    "cot_response": "", "extracted_answer": "", "verified": False, "tier": "error",
                    "cot_response_t2": "", "extracted_answer_t2": "", "verified_t2": False,
                    "correction_mode": "", "model": model, "task_error": err1,
                }
                append_row(all_csv, rec)
                stats["errors"] += 1
                stats["total"] += 1
                if sleep_s:
                    time.sleep(sleep_s)
                continue

            ext1 = extract_final_answer(cot_raw, gold)
            ok1 = normalized_match(ext1, gold)

            # ── Thinking preview ──
            think_prev = _thinking_preview(cot_raw, 100)
            has_think = "<think>" in cot_raw
            think_len = len(cot_raw.split("</think>")[0]) if has_think else 0

            if run_log:
                run_log.line(
                    f"ROW_VERIFY_T1 {base_tag} extracted={ext1!r} gold={gold!r} match={ok1} "
                    f"total_dt_s={dt1:.3f} response_chars={len(cot_raw)} think_block={has_think}"
                )

            if ok1:
                # Tier 1 success
                fam_stats[fam]["verified"] += 1
                stats["t1_ok"] += 1
                stats["total"] += 1
                tag = _green("T1 OK")
                print(
                    f"{tag}  ans={_short(ext1, 40)!r}  {dt1:.0f}s"
                    + (f"  {_dim(f'think={think_len}ch')}" if has_think else ""),
                    flush=True,
                )
                if think_prev:
                    print(f"    {_dim(think_prev)}", flush=True)

                if run_log:
                    run_log.line(f"ROW_END {base_tag} tier=tier1_genuine verified=True")

                rec = {
                    "row_id": row["row_id"], "question": q, "gold_answer": gold, "family": fam,
                    "cot_response": cot_raw, "extracted_answer": ext1, "verified": True,
                    "tier": "tier1_genuine",
                    "cot_response_t2": "", "extracted_answer_t2": "", "verified_t2": False,
                    "correction_mode": "", "model": model, "task_error": "",
                }
                append_row(all_csv, rec)
                if sleep_s:
                    time.sleep(sleep_s)
                continue

            # ── Pass 2: Correction ──
            print(
                f"{_yellow('T1 MISS')} got={_short(ext1, 30)!r} want={_short(gold, 30)!r}  {dt1:.0f}s"
                + (f"  {_dim(f'think={think_len}ch')}" if has_think else ""),
                flush=True,
            )
            if think_prev:
                print(f"    {_dim(think_prev)}", flush=True)

            if should_quit():
                rec = {
                    "row_id": row["row_id"], "question": q, "gold_answer": gold, "family": fam,
                    "cot_response": cot_raw, "extracted_answer": ext1, "verified": False,
                    "tier": "tier1_wrong",
                    "cot_response_t2": "", "extracted_answer_t2": "", "verified_t2": False,
                    "correction_mode": "", "model": model, "task_error": "",
                }
                append_row(all_csv, rec)
                stats["t2_fail"] += 1
                stats["total"] += 1
                if run_log:
                    run_log.line(f"ROW_END {base_tag} tier=tier1_wrong quit_before_T2")
                print(f"\n{_yellow('Stopping')} (q pressed).", flush=True)
                break

            t0b = time.monotonic()
            cm = correction_mode.strip().lower()
            if cm == "session":
                msgs = [
                    {"role": "user", "content": prompt1},
                    {"role": "assistant", "content": cot_raw},
                    {"role": "user", "content": CORRECTION_FOLLOWUP.format(gold_answer=gold)},
                ]
                cot2_raw, err2 = chat_messages_with_retries(
                    api_url, api_key, model, msgs,
                    max_tokens=max_tokens, temperature=temperature, timeout=timeout,
                    sleep_s=sleep_s, max_retries=max_retries, label=f"[{i} T2]",
                    log=run_log, log_tag=f"{base_tag} phase=T2",
                )
                cm_out = "session"
            else:
                prompt2 = CORRECTION_PROMPT.format(question=q, gold_answer=gold)
                cot2_raw, err2 = chat_with_retries(
                    api_url, api_key, model, prompt2,
                    max_tokens=max_tokens, temperature=temperature, timeout=timeout,
                    sleep_s=sleep_s, max_retries=max_retries, label=f"[{i} T2]",
                    log=run_log, log_tag=f"{base_tag} phase=T2",
                )
                cm_out = "standalone"
            dt2 = time.monotonic() - t0b

            ext2 = extract_final_answer(cot2_raw, gold) if not err2 else ""
            ok2 = normalized_match(ext2, gold) if not err2 else False

            tier = "tier2_verified" if ok2 else "tier2_unverified"
            if ok2:
                fam_stats[fam]["verified"] += 1
                stats["t2_ok"] += 1
                tag2 = _green("T2 OK")
            else:
                stats["t2_fail"] += 1
                tag2 = _red("T2 FAIL")

            stats["total"] += 1
            think2_prev = _thinking_preview(cot2_raw, 100)

            if run_log:
                run_log.line(
                    f"ROW_VERIFY_T2 {base_tag} extracted={ext2!r} gold={gold!r} match={ok2} "
                    f"tier={tier} t2_dt_s={dt2:.3f} err={err2[:4000] if err2 else ''}"
                )

            print(
                f"    {tag2}  ans={_short(ext2, 30)!r}  {dt2:.0f}s"
                + (f"  err={_short(err2, 40)}" if err2 else ""),
                flush=True,
            )
            if think2_prev:
                print(f"    {_dim(think2_prev)}", flush=True)

            best_cot = cot2_raw if ok2 else cot_raw
            best_ext = ext2 if ok2 else ext1
            rec = {
                "row_id": row["row_id"], "question": q, "gold_answer": gold, "family": fam,
                "cot_response": best_cot, "extracted_answer": best_ext, "verified": ok2,
                "tier": tier,
                "cot_response_t2": cot2_raw, "extracted_answer_t2": ext2, "verified_t2": ok2,
                "correction_mode": cm_out, "model": model, "task_error": err2,
            }
            append_row(all_csv, rec)
            if run_log:
                run_log.line(f"ROW_END {base_tag} tier={tier} verified={ok2}")
            if sleep_s:
                time.sleep(sleep_s)

        # ── Summary ──
        print(flush=True)
        print(_bold("=" * 60), flush=True)
        print(_bold("SUMMARY"), flush=True)
        print(_bold("=" * 60), flush=True)
        total = stats["total"]
        v = stats["t1_ok"] + stats["t2_ok"]
        attempted_cot = stats["t1_ok"] + stats["t2_ok"] + stats["t2_fail"]
        t1_rate = 100 * stats["t1_ok"] / max(1, attempted_cot)
        t2_pool = stats["t2_ok"] + stats["t2_fail"]
        t2_recover = 100 * stats["t2_ok"] / max(1, t2_pool) if t2_pool else 0.0
        print(f"  Processed:   {total} rows (+ {n_done_prev} cached from prior runs)", flush=True)
        print(f"  Tier1 OK:    {_green(str(stats['t1_ok']))}", flush=True)
        print(f"  Tier2 OK:    {_green(str(stats['t2_ok']))}", flush=True)
        print(f"  Tier2 fail:  {_red(str(stats['t2_fail']))}", flush=True)
        print(f"  Errors:      {stats['errors']}", flush=True)
        print(f"  Final accuracy (verified/this run): {v}/{total} ({100*v/max(1,total):.1f}%)", flush=True)
        print(
            f"  Tier1-only accuracy (excl. errors):   {stats['t1_ok']}/{attempted_cot} ({t1_rate:.1f}%)",
            flush=True,
        )
        if t2_pool:
            print(
                f"  Tier2 recovery (wrong T1 -> T2 OK):  {stats['t2_ok']}/{t2_pool} ({t2_recover:.1f}%)",
                flush=True,
            )
        print(flush=True)
        print("  Per-family:", flush=True)
        for fam in sorted(fam_stats):
            fs = fam_stats[fam]
            pct = 100 * fs["verified"] / max(1, fs["total"])
            bar = _green(f"{fs['verified']}/{fs['total']}") if pct > 50 else _yellow(f"{fs['verified']}/{fs['total']}")
            print(f"    {fam:20s} {bar} ({pct:.0f}%)", flush=True)
        print(_bold("=" * 60), flush=True)

        if run_log:
            run_log.line(
                f"PIPELINE_SUMMARY processed={total} t1_ok={stats['t1_ok']} t2_ok={stats['t2_ok']} "
                f"t2_fail={stats['t2_fail']} errors={stats['errors']} verified={v} "
                f"final_acc_pct={100*v/max(1,total):.2f} tier1_only_pct={t1_rate:.2f}"
            )
            if t2_pool:
                run_log.line(f"PIPELINE_SUMMARY tier2_recovery_pct={t2_recover:.2f}")

        # ── Export verified ──
        if all_csv.exists():
            full = pd.read_csv(all_csv)
            ok_mask = full["verified"].astype(str).str.lower().isin(("true", "1", "yes"))
            verified_df = full.loc[ok_mask, [
                "row_id", "question", "gold_answer", "cot_response", "tier", "family", "model",
            ]].copy()
            export_path = out_dir / "cot_traces_verified.csv"
            verified_df.to_csv(export_path, index=False)
            print(f"\n  Verified export: {len(verified_df)} rows -> {export_path}", flush=True)

            summary_path = out_dir / "summary.txt"
            with open(summary_path, "w") as sf:
                sf.write(f"model: {model}\n")
                sf.write(f"correction_mode: {correction_mode}\n")
                sf.write(f"total_rows: {len(full)}\n")
                sf.write(f"verified: {len(verified_df)}\n")
                sf.write(
                    f"this_run_final_accuracy: {v}/{total} ({100*v/max(1,total):.1f}%)\n"
                )
                sf.write(
                    f"this_run_tier1_only: {stats['t1_ok']}/{attempted_cot} ({t1_rate:.1f}%)\n"
                )
                if t2_pool:
                    sf.write(
                        f"this_run_tier2_recovery: {stats['t2_ok']}/{t2_pool} ({t2_recover:.1f}%)\n"
                    )
                for fam in sorted(fam_stats):
                    fs = fam_stats[fam]
                    sf.write(f"{fam}: {fs['verified']}/{fs['total']}\n")
            print(f"  Summary: {summary_path}", flush=True)

        if run_log:
            run_log.line("PIPELINE_END ok=1")

        return all_csv
    finally:
        if run_log is not None:
            run_log.close()


def main() -> int:
    load_env()

    ap = argparse.ArgumentParser(
        description="Single-pass CoT generation (NVIDIA build API default)"
    )
    ap.add_argument("--sample", default="datasets/benchmark_sample_500.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-url", default=None)
    ap.add_argument("--out-dir", default=None, type=Path)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--sleep", type=float, default=0.5, help="Pause between requests (rate limit)")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument(
        "--correction-mode",
        choices=("session", "standalone"),
        default="session",
        help="Tier2: session = same chat [user, assistant, user]; standalone = fresh one-shot prompt",
    )
    ap.add_argument(
        "--no-run-log",
        action="store_true",
        help="Disable detailed run_<model>_<UTC>.log (HTTP, retries, full assistant text)",
    )
    ap.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write detailed log to this path instead of a timestamped file under --out-dir",
    )
    args = ap.parse_args()

    api_key = ""
    for name in ("NVIDIA_API_KEY", "NVAPI_KEY"):
        api_key = os.environ.get(name, "").strip()
        if api_key:
            break
    if not api_key:
        print("Set NVIDIA_API_KEY (or NVAPI_KEY) in .env or environment.", file=sys.stderr)
        return 1

    api_url = (args.api_url or "").strip() or NVIDIA_CHAT_URL

    sample_path = Path(args.sample)
    if not sample_path.exists():
        print(f"Sample not found: {sample_path}", file=sys.stderr)
        return 1

    raw = pd.read_csv(sample_path)
    id_col = "id" if "id" in raw.columns else None
    if id_col is None:
        raw["id"] = range(len(raw))
        id_col = "id"
    df = raw[[id_col, "prompt", "answer"]].rename(
        columns={id_col: "row_id", "prompt": "question", "answer": "answer"}
    )
    for c in ["row_id", "question", "answer"]:
        df[c] = df[c].astype(str)

    out_dir = args.out_dir or Path("nvidia_benchmark_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start quit listener
    qt = threading.Thread(target=_quit_listener, daemon=True)
    qt.start()

    print(_bold("=" * 60), flush=True)
    print(_bold("  NVIDIA CoT Generation Pipeline"), flush=True)
    print(_bold("=" * 60), flush=True)

    try:
        run(
            df, args.model, out_dir, api_key, api_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            sleep_s=args.sleep,
            max_retries=args.max_retries,
            correction_mode=args.correction_mode,
            enable_run_log=not args.no_run_log,
            run_log_path=args.log_file,
        )
    except KeyboardInterrupt:
        print(f"\n{_yellow('Interrupted')} (Ctrl+C). Progress saved to {out_dir}.", flush=True)
        return 130

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        raise SystemExit(130)
