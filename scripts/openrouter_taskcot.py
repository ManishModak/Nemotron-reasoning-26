#!/usr/bin/env python3
"""
Benchmark + tiered CoT pipeline over an OpenAI-compatible chat/completions API.

Providers:
  --provider nvidia (default)
    URL: https://integrate.api.nvidia.com/v1/chat/completions (override with --api-url)
    Key: NVIDIA_API_KEY or NVAPI_KEY (Bearer nvapi-...)
    Default model: minimaxai/minimax-m2.7
    Default token budgets: phase1=8192 (thinking + answer on NVIDIA), CoT=16384

  --provider openrouter
    URL: https://openrouter.ai/api/v1/chat/completions
    Key: OPENROUTER_API_KEY
    Default model: minimax/minimax-m2.5:free
    Docs: https://openrouter.ai/minimax/minimax-m2.5:free/api

  Or: python scripts/nvidia_taskcot.py  (forces --provider nvidia if omitted)

Setup (.env or environment; real env wins over .env):
  NVIDIA_API_KEY=              # default provider (or NVAPI_KEY)

  OPENROUTER_API_KEY=          # --provider openrouter
  OPENROUTER_HTTP_REFERER=
  OPENROUTER_APP_TITLE=

Usage:
  python scripts/openrouter_taskcot.py --phase 1 --sample datasets/benchmark_sample_500.csv
  python scripts/openrouter_taskcot.py --provider openrouter --phase both
  python scripts/nvidia_taskcot.py --phase 2 --phase1-csv nvidia_benchmark_outputs/phase1_....csv

Outputs default to nvidia_benchmark_outputs/ or openrouter_benchmark_outputs/ (override --out-dir).
Each phase appends to per-model CSVs for resume after Ctrl+C or rate limits.

Rate limits: default --workers 1, --sleep 0.35; retries on 429/503 with backoff (RATE_LIMIT_BACKOFF_MAX_S).

Ctrl+C: exit 130; completed rows remain on disk — re-run to resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Max seconds to sleep on one rate-limit backoff (avoid multi-hour stalls if header is huge)
RATE_LIMIT_BACKOFF_MAX_S = 300.0


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
    """Load KEY=VALUE pairs into os.environ. Returns count of variables set."""
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


def preload_env_from_argv() -> Path | None:
    """Parse --env-file / --no-env from argv before argparse so defaults see os.environ."""
    if "--no-env" in sys.argv:
        return None
    env_path: Path | None = None
    for i, a in enumerate(sys.argv):
        if a.startswith("--env-file="):
            env_path = Path(a.split("=", 1)[1]).strip().expanduser()
            break
        if a == "--env-file" and i + 1 < len(sys.argv):
            env_path = Path(sys.argv[i + 1]).expanduser()
            break
    candidates: list[Path] = []
    if env_path is not None:
        candidates.append(env_path)
    else:
        candidates.extend([Path.cwd() / ".env", REPO_ROOT / ".env"])
    loaded_from: Path | None = None
    for p in candidates:
        if load_env_file(p):
            loaded_from = p
            break
    if loaded_from:
        print(f"Loaded environment from {loaded_from}")
    return loaded_from

# Default models per provider (slugs differ between hosts)
DEFAULT_MODEL_OPENROUTER = "minimax/minimax-m2.5:free"
DEFAULT_MODEL_NVIDIA = "minimaxai/minimax-m2.7"

# Token budgets: NVIDIA MiniMax often emits <think>…</think> before the answer;
# phase1 needs room to finish thinking + final token. OpenRouter free tiers stay conservative.
DEFAULT_MAX_TOKENS_PHASE1_NVIDIA = 8192
DEFAULT_MAX_TOKENS_PHASE1_OPENROUTER = 256
DEFAULT_MAX_TOKENS_COT_NVIDIA = 16384
DEFAULT_MAX_TOKENS_COT_OPENROUTER = 4096

# Long CoT generations on NVIDIA can exceed a short socket timeout.
DEFAULT_TIMEOUT_S_NVIDIA = 360
DEFAULT_TIMEOUT_S_OPENROUTER = 120

ANSWER_ONLY_SUFFIX_OPENROUTER = (
    "\n\nReturn only the final answer. No explanation, no steps, no extra words."
)

# MiniMax on NVIDIA often uses Nemotron-style thinking tags; allow them, then parse the tail for scoring.
ANSWER_ONLY_SUFFIX_NVIDIA = (
    "\n\nYou may reason privately inside <think>...</think>.\n"
    "When you are done, close the thinking block, then output only the final answer on the last line "
    "(exact puzzle format: no labels, no quotes, no extra words). "
    "Optionally wrap the final answer in \\boxed{}."
)

COT_PROMPT_TIER1 = """{question}

Think through this step by step. Show your complete reasoning process,
then put your final answer inside \\boxed{{}}.
For example: \\boxed{{your answer}}"""

COT_PROMPT_TIER2 = """{question}

Your previous answer was incorrect. The objectively correct answer is: {gold_answer}

Recalibrate your thinking. Generate a rigorous, step-by-step Chain of Thought
that logically and flawlessly arrives at this correct answer.
Explicitly point out the hidden trap or complex step where a system might typically fail.
Put your final answer inside \\boxed{{}}. For example: \\boxed{{your answer}}"""


def normalize_text(x: str) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    text = unicodedata.normalize("NFKC", str(x)).strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def normalized_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def classify_family(prompt_text: str) -> str:
    p = str(prompt_text).lower()
    if "bit manipulation" in p:
        return "bit_manipulation"
    if "gravitational constant" in p:
        return "gravity"
    if "encryption rules" in p or "encryption" in p:
        return "encryption"
    if "numeral system" in p:
        return "numeral"
    if "unit conversion" in p or "converted" in p:
        return "unit_conversion"
    if "transformation rules" in p or "equation" in p or "operator" in p:
        return "equations"
    return "unknown"


def extract_boxed(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not matches:
        return ""
    non_empty = [m.strip() for m in matches if m.strip()]
    return non_empty[-1] if non_empty else matches[-1].strip()


def _strip_after_redacted_thinking(text: str) -> str:
    """Keep only content after the last closing thinking tag (answer tail)."""
    s = str(text).strip()
    if "</think>" in s:
        return s.rsplit("</think>", 1)[-1].strip()
    return s


def extract_phase1_final_answer(raw: str, gold: str) -> str:
    """Turn model output into a single string comparable to gold (thinking stripped, boxed / tail heuristics)."""
    if not raw:
        return ""
    tail = _strip_after_redacted_thinking(raw)
    boxed = extract_boxed(tail) or extract_boxed(raw)
    if boxed:
        return boxed.strip()

    gold_s = str(gold).strip()
    n_bits = len(gold_s) if re.fullmatch(r"[01]+", gold_s) else 0
    if n_bits:
        pool = tail if tail else raw
        found = re.findall(rf"\b[01]{{{n_bits}}}\b", pool)
        if found:
            return found[-1]

    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()] if tail else []
    if lines:
        return lines[-1]
    lines2 = [ln.strip() for ln in str(raw).splitlines() if ln.strip()]
    return lines2[-1] if lines2 else str(raw).strip()


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def _parse_retry_after_seconds(header_val: str | None) -> float | None:
    """Parse Retry-After as integer seconds; ignore HTTP-date form."""
    if not header_val:
        return None
    s = header_val.strip()
    if not s.isdigit():
        return None
    try:
        return float(min(int(s), int(RATE_LIMIT_BACKOFF_MAX_S)))
    except ValueError:
        return None


def _rate_limit_backoff_wait(
    *,
    attempt: int,
    sleep_s: float,
    err: str,
    meta: dict[str, Any],
) -> float:
    """Seconds to wait before retrying after a failed request."""
    jitter = random.uniform(0, 0.75)
    base = sleep_s * (2**attempt) + jitter
    ra = meta.get("retry_after_s")
    if isinstance(ra, (int, float)) and ra > 0:
        base = max(base, float(ra))
    code = meta.get("http_status")
    if code in (408, 429, 503):
        base = max(base, sleep_s * (2 ** (attempt + 1)))
    el = err.lower()
    if "429" in err or "rate" in el or "too many requests" in el or "throttl" in el or "quota" in el:
        base = max(base, sleep_s * (2 ** (attempt + 2)))
    return min(float(base), RATE_LIMIT_BACKOFF_MAX_S)


def openai_chat_completions(
    api_url: str,
    api_key: str,
    model: str,
    user_content: str,
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
    referer: str | None = None,
    app_title: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """POST chat/completions. Returns (assistant_text, error_message, meta).

    meta may include: http_status (int), retry_after_s (float) from Retry-After header.
    OpenRouter-only headers (HTTP-Referer, X-Title) are sent only when both referer/app_title are set.
    """
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
    if referer:
        headers["HTTP-Referer"] = referer
    if app_title:
        headers["X-Title"] = app_title

    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        meta["http_status"] = int(e.code)
        _hdrs = getattr(e, "headers", None)
        _ra_hdr = _hdrs.get("Retry-After") if _hdrs else None
        ra = _parse_retry_after_seconds(_ra_hdr)
        if ra is not None:
            meta["retry_after_s"] = ra
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = str(e)
        return "", f"HTTP {e.code}: {err_body[:2000]}", meta
    except Exception as e:
        return "", repr(e), meta

    try:
        choice = payload["choices"][0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        return str(content).strip(), "", meta
    except (KeyError, IndexError, TypeError):
        return "", f"Unexpected response shape: {json.dumps(payload)[:1500]}", meta


def load_done_ids(csv_path: Path, id_col: str = "row_id") -> set[str]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        return set()
    return set(df[id_col].astype(str))


def append_row(csv_path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def _short(s: str, n: int = 72) -> str:
    s = str(s).replace("\n", " ")
    return (s[: n - 3] + "...") if len(s) > n else s


def _print_interrupt_help(out_dir: Path, phase: str) -> None:
    print("\nInterrupted (Ctrl+C).", flush=True)
    print(
        "Partial progress is already on disk (each completed row is appended to CSV).",
        flush=True,
    )
    print(f"  Output directory: {out_dir.resolve()}", flush=True)
    print("Re-run the same command to resume; completed row_ids are skipped.", flush=True)
    print(f"(phase was: {phase})", flush=True)


def run_phase1(
    df: pd.DataFrame,
    model: str,
    out_dir: Path,
    api_key: str,
    *,
    api_url: str,
    phase1_suffix: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    workers: int,
    sleep_s: float,
    referer: str | None,
    app_title: str | None,
    max_retries: int,
    verbose: bool,
) -> Path:
    out_path = out_dir / f"phase1_{model_slug(model)}.csv"
    done = load_done_ids(out_path)
    fields = [
        "row_id",
        "question",
        "gold_answer",
        "prediction",
        "prediction_raw",
        "normalized_match",
        "task_error",
        "model",
        "family",
    ]

    rows_todo = [r for _, r in df.iterrows() if str(r["row_id"]) not in done]
    n_total = len(rows_todo)
    print(f"Phase 1 [{model}]: {len(done)} cached, {n_total} to run -> {out_path}", flush=True)
    if verbose and n_total:
        est = n_total * (timeout + sleep_s + 2)
        print(
            f"  (verbose) Up to ~{n_total} HTTP calls; worst-case ~{est / 60:.0f} min if each hits timeout. "
            f"Lines appear after each response.",
            flush=True,
        )

    def one(row: pd.Series, *, idx: int, log: bool) -> dict[str, Any]:
        q = str(row["question"]).strip()
        gold = str(row["answer"]).strip()
        rid = str(row["row_id"])
        fam = classify_family(q)
        prompt = q + phase1_suffix
        if log:
            print(
                f"  [{idx}/{n_total}] POST row_id={rid[:16]}... family={fam} "
                f"(max_tokens={max_tokens}, timeout={timeout}s)",
                flush=True,
            )
        t0 = time.monotonic()
        err = ""
        pred = ""
        for attempt in range(max_retries):
            pred, err, meta = openai_chat_completions(
                api_url,
                api_key,
                model,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                referer=referer,
                app_title=app_title,
            )
            if not err:
                break
            wait_s = _rate_limit_backoff_wait(attempt=attempt, sleep_s=sleep_s, err=err, meta=meta)
            if log:
                print(
                    f"  [{idx}/{n_total}]   retry {attempt + 1}/{max_retries} in {wait_s:.1f}s: "
                    f"{_short(err, 100)}",
                    flush=True,
                )
            time.sleep(wait_s)
        dt = time.monotonic() - t0
        raw_pred = pred
        parsed = extract_phase1_final_answer(raw_pred, gold) if not err else ""
        nm = normalized_match(parsed, gold) if not err else False
        if log:
            pred_disp = _short(parsed, 60) if parsed else "(empty)"
            err_disp = _short(err, 80) if err else ""
            print(
                f"  [{idx}/{n_total}] <- {dt:.1f}s  match={nm}  pred={pred_disp!r}  err={err_disp!r}",
                flush=True,
            )
        return {
            "row_id": rid,
            "question": q,
            "gold_answer": gold,
            "prediction": parsed,
            "prediction_raw": raw_pred if not err else "",
            "normalized_match": nm,
            "task_error": err,
            "model": model,
            "family": fam,
        }

    write_lock = threading.Lock()
    done_parallel = [0]

    def one_and_save(row: pd.Series, *, idx: int, log: bool) -> None:
        rec = one(row, idx=idx, log=log)
        with write_lock:
            append_row(out_path, fields, rec)

    def one_and_save_parallel(row: pd.Series) -> None:
        rec = one(row, idx=0, log=False)
        with write_lock:
            append_row(out_path, fields, rec)
            done_parallel[0] += 1
            c = done_parallel[0]
        if verbose:
            nm = rec["normalized_match"]
            print(
                f"  [{c}/{n_total}] saved row_id={str(rec['row_id'])[:16]}... "
                f"match={nm} in parallel batch",
                flush=True,
            )

    if workers <= 1:
        for i, row in enumerate(rows_todo, start=1):
            one_and_save(row, idx=i, log=verbose)
            if sleep_s:
                time.sleep(sleep_s)
    else:
        if verbose:
            print(f"  Starting {workers} workers (progress on completion only)...", flush=True)
        ex = ThreadPoolExecutor(max_workers=workers)
        _pool_state = "ok"
        try:
            futs = [ex.submit(one_and_save_parallel, row) for row in rows_todo]
            for fut in as_completed(futs):
                fut.result()
        except KeyboardInterrupt:
            _pool_state = "interrupt"
            if verbose:
                print("\n  (Ctrl+C) Cancelling pending parallel tasks...", flush=True)
            raise
        except BaseException:
            _pool_state = "error"
            raise
        finally:
            if _pool_state == "ok":
                ex.shutdown(wait=True)
            else:
                # Abrupt exit: do not wait on hung HTTP; cancel not-yet-started futures (Py 3.9+)
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    ex.shutdown(wait=False)

    if n_total:
        print(f"Phase 1 [{model}] finished this run: {n_total} new rows -> {out_path}", flush=True)
    return out_path


def run_phase2(
    sample_df: pd.DataFrame,
    phase1_csv: Path,
    model: str,
    out_dir: Path,
    api_key: str,
    *,
    api_url: str,
    export_csv_name: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    workers: int,
    sleep_s: float,
    referer: str | None,
    app_title: str | None,
    max_retries: int,
    verbose: bool,
) -> None:
    p1 = pd.read_csv(phase1_csv)
    p1 = p1[p1["model"] == model] if "model" in p1.columns else p1
    _nm = p1["normalized_match"].astype(str).str.lower().isin(("true", "1", "yes"))
    correct_ids = set(p1.loc[_nm, "row_id"].astype(str))

    tier1_path = out_dir / f"cot_tier1_{model_slug(model)}.csv"
    tier2_path = out_dir / f"cot_tier2_{model_slug(model)}.csv"
    done1 = load_done_ids(tier1_path)
    done2 = load_done_ids(tier2_path)

    fields = [
        "row_id",
        "question",
        "gold_answer",
        "cot_response",
        "extracted_answer",
        "verified",
        "tier",
        "model",
        "family",
        "task_error",
    ]

    tier1_rows: list[pd.Series] = []
    tier2_rows: list[pd.Series] = []
    for _, row in sample_df.iterrows():
        rid = str(row["row_id"])
        if rid in correct_ids:
            tier1_rows.append(row)
        else:
            tier2_rows.append(row)

    t1_todo = [r for r in tier1_rows if str(r["row_id"]) not in done1]
    t2_todo = [r for r in tier2_rows if str(r["row_id"]) not in done2]
    print(
        f"Phase 2 [{model}]: tier1={len(tier1_rows)} tier2={len(tier2_rows)} "
        f"(cached t1={len(done1)} t2={len(done2)}; to_run t1={len(t1_todo)} t2={len(t2_todo)})",
        flush=True,
    )
    if verbose and (t1_todo or t2_todo):
        n_calls = len(t1_todo) + len(t2_todo)
        print(
            f"  (verbose) CoT uses max_tokens={max_tokens} (long generations). "
            f"~{n_calls} API calls; lines print after each completes.",
            flush=True,
        )

    def call_cot(user_prompt: str, *, log_prefix: str) -> tuple[str, str]:
        last_err = ""
        for attempt in range(max_retries):
            text, err, meta = openai_chat_completions(
                api_url,
                api_key,
                model,
                user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                referer=referer,
                app_title=app_title,
            )
            if not err:
                return text, ""
            last_err = err
            wait_s = _rate_limit_backoff_wait(attempt=attempt, sleep_s=sleep_s, err=err, meta=meta)
            if verbose:
                print(
                    f"  {log_prefix}  retry {attempt + 1}/{max_retries} in {wait_s:.1f}s: "
                    f"{_short(err, 100)}",
                    flush=True,
                )
            time.sleep(wait_s)
        return "", last_err

    def process_tier(row: pd.Series, tier: str, *, idx: int, n_seg: int, log_prefix: str) -> dict[str, Any]:
        q = str(row["question"]).strip()
        gold = str(row["answer"]).strip()
        rid = str(row["row_id"])
        fam = classify_family(q)
        if tier == "tier1_genuine":
            user = COT_PROMPT_TIER1.format(question=q)
        else:
            user = COT_PROMPT_TIER2.format(question=q, gold_answer=gold)
        if verbose:
            print(
                f"  {log_prefix} [{idx}/{n_seg}] POST row_id={rid[:16]}... family={fam} "
                f"(timeout={timeout}s, max_tokens={max_tokens})",
                flush=True,
            )
        t0 = time.monotonic()
        cot, err = call_cot(user, log_prefix=log_prefix)
        dt = time.monotonic() - t0
        ext = extract_boxed(cot)
        verified = normalized_match(ext, gold) if not err else False
        if verbose:
            print(
                f"  {log_prefix} [{idx}/{n_seg}] <- {dt:.1f}s  verified={verified}  "
                f"boxed={_short(ext, 40)!r}  err={_short(err, 60)!r}",
                flush=True,
            )
        return {
            "row_id": rid,
            "question": q,
            "gold_answer": gold,
            "cot_response": cot,
            "extracted_answer": ext,
            "verified": verified,
            "tier": tier,
            "model": model,
            "family": fam,
            "task_error": err,
        }

    n1 = len(t1_todo)
    for i, row in enumerate(t1_todo, start=1):
        rec = process_tier(row, "tier1_genuine", idx=i, n_seg=n1, log_prefix="Tier1")
        append_row(tier1_path, fields, rec)
        if sleep_s:
            time.sleep(sleep_s)
    if n1 and verbose:
        print(f"  Tier1: wrote {n1} rows this run -> {tier1_path}", flush=True)

    n2 = len(t2_todo)
    for i, row in enumerate(t2_todo, start=1):
        rec = process_tier(row, "tier2_correction", idx=i, n_seg=n2, log_prefix="Tier2")
        append_row(tier2_path, fields, rec)
        if sleep_s:
            time.sleep(sleep_s)
    if n2 and verbose:
        print(f"  Tier2: wrote {n2} rows this run -> {tier2_path}", flush=True)

    # Merge verified export
    parts = []
    if tier1_path.exists():
        parts.append(pd.read_csv(tier1_path))
    if tier2_path.exists():
        parts.append(pd.read_csv(tier2_path))
    if not parts:
        print("Phase 2: no outputs to merge.", flush=True)
        return
    all_cot = pd.concat(parts, ignore_index=True)
    _ok = all_cot["verified"].astype(str).str.lower().isin(("true", "1", "yes"))
    verified = all_cot.loc[_ok].copy()
    export = verified[
        ["row_id", "question", "gold_answer", "cot_response", "tier", "family", "model"]
    ]
    export_path = out_dir / export_csv_name
    export.to_csv(export_path, index=False)
    print(f"Phase 2 [{model}] export: {len(export)} verified CoT rows -> {export_path}", flush=True)


def build_leaderboard(
    out_dir: Path, models: list[str], *, leaderboard_csv: str, verbose: bool
) -> None:
    frames = []
    for m in models:
        p = out_dir / f"phase1_{model_slug(m)}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        print("No phase1 files for leaderboard.", flush=True)
        return
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out_dir / "phase1_all_models.csv", index=False)
    lb = (
        all_df.groupby("model", as_index=False)
        .agg(
            rows=("row_id", "count"),
            matches=("normalized_match", "sum"),
            errors=("task_error", lambda s: int((s.astype(str) != "").sum())),
        )
    )
    lb["match_pct"] = (100.0 * lb["matches"] / lb["rows"]).round(2)
    lb = lb.sort_values("match_pct", ascending=False)
    lb_path = out_dir / leaderboard_csv
    lb.to_csv(lb_path, index=False)
    print("Leaderboard (Phase 1):", flush=True)
    print(lb.to_string(index=False), flush=True)
    if not verbose:
        print(f"(wrote {lb_path})", flush=True)


def _resolve_api_key(provider: str) -> tuple[str, str]:
    """Returns (api_key, env_name_used_for_message)."""
    if provider == "nvidia":
        for name in ("NVIDIA_API_KEY", "NVAPI_KEY"):
            v = os.environ.get(name, "").strip()
            if v:
                return v, name
        return "", "NVIDIA_API_KEY or NVAPI_KEY"
    v = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return v, "OPENROUTER_API_KEY"


def main() -> int:
    preload_env_from_argv()

    ap = argparse.ArgumentParser(
        description="Benchmark + tiered CoT (OpenRouter or NVIDIA build API, OpenAI-compatible)"
    )
    ap.add_argument(
        "--provider",
        choices=("openrouter", "nvidia"),
        default="nvidia",
        help="API host: nvidia (default, integrate.api.nvidia.com) or openrouter",
    )
    ap.add_argument(
        "--api-url",
        default=None,
        metavar="URL",
        help="Override chat/completions URL (default depends on --provider)",
    )
    ap.add_argument(
        "--env-file",
        default=None,
        metavar="PATH",
        help="Load env vars from this file (default: .env in cwd, then repo root). Skipped if --no-env.",
    )
    ap.add_argument(
        "--no-env",
        action="store_true",
        help="Do not load any .env file (only real process environment).",
    )
    ap.add_argument("--sample", default="datasets/benchmark_sample_500.csv", help="CSV with id,prompt,answer")
    ap.add_argument("--phase", choices=["1", "2", "both"], default="both")
    ap.add_argument(
        "--models",
        default=None,
        help="Comma-separated model ids (defaults: openrouter minimax free; nvidia minimaxai/minimax-m2.7)",
    )
    ap.add_argument("--phase1-csv", default=None, help="Phase1 results for phase2 (single model file)")
    ap.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Output directory (default: openrouter_benchmark_outputs or nvidia_benchmark_outputs)",
    )
    ap.add_argument(
        "--max-tokens-phase1",
        type=int,
        default=None,
        help="Max completion tokens for phase 1 (defaults: 8192 nvidia, 256 openrouter)",
    )
    ap.add_argument(
        "--max-tokens-cot",
        type=int,
        default=None,
        help="Max completion tokens for CoT tiers (defaults: 16384 nvidia, 4096 openrouter)",
    )
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-request HTTP timeout in seconds (defaults: 360 nvidia, 120 openrouter)",
    )
    ap.add_argument("--workers", type=int, default=1, help="Parallel requests (use 1 to respect free-tier limits)")
    ap.add_argument("--sleep", type=float, default=0.35, help="Seconds between sequential requests")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Less console output (no per-row progress; retries/errors may still print in Phase 2).",
    )
    ap.add_argument("--referer", default=os.environ.get("OPENROUTER_HTTP_REFERER"))
    ap.add_argument("--app-title", default=os.environ.get("OPENROUTER_APP_TITLE", "nemotron-openrouter-taskcot"))
    args = ap.parse_args()
    verbose = not args.quiet

    provider = args.provider
    api_url = (args.api_url or "").strip() or (
        NVIDIA_CHAT_URL if provider == "nvidia" else OPENROUTER_CHAT_URL
    )

    api_key, key_hint = _resolve_api_key(provider)
    if not api_key:
        print(f"Set {key_hint} in the environment (provider={provider}).", file=sys.stderr)
        return 1

    referer = args.referer if provider == "openrouter" else None
    app_title = args.app_title if provider == "openrouter" else None

    leaderboard_csv = (
        "leaderboard_nvidia.csv" if provider == "nvidia" else "leaderboard_openrouter.csv"
    )
    cot_export_name = (
        "cot_traces_verified_nvidia.csv" if provider == "nvidia" else "cot_traces_verified_openrouter.csv"
    )

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

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (
            Path("nvidia_benchmark_outputs")
            if provider == "nvidia"
            else Path("openrouter_benchmark_outputs")
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    models_raw = args.models
    if not models_raw:
        models_raw = (
            DEFAULT_MODEL_NVIDIA if provider == "nvidia" else DEFAULT_MODEL_OPENROUTER
        )
    models = [m.strip() for m in models_raw.split(",") if m.strip()]
    if not models:
        models = [
            DEFAULT_MODEL_NVIDIA if provider == "nvidia" else DEFAULT_MODEL_OPENROUTER
        ]

    max_tok_p1 = (
        args.max_tokens_phase1
        if args.max_tokens_phase1 is not None
        else (
            DEFAULT_MAX_TOKENS_PHASE1_NVIDIA
            if provider == "nvidia"
            else DEFAULT_MAX_TOKENS_PHASE1_OPENROUTER
        )
    )
    max_tok_cot = (
        args.max_tokens_cot
        if args.max_tokens_cot is not None
        else (
            DEFAULT_MAX_TOKENS_COT_NVIDIA
            if provider == "nvidia"
            else DEFAULT_MAX_TOKENS_COT_OPENROUTER
        )
    )
    timeout_s = (
        args.timeout
        if args.timeout is not None
        else (
            DEFAULT_TIMEOUT_S_NVIDIA
            if provider == "nvidia"
            else DEFAULT_TIMEOUT_S_OPENROUTER
        )
    )

    if verbose:
        print(f"Provider={provider} api_url={api_url}", flush=True)
        print(
            f"Token budgets: phase1 max_tokens={max_tok_p1}, CoT max_tokens={max_tok_cot}; "
            f"HTTP timeout={timeout_s}s",
            flush=True,
        )

    try:
        if args.phase in ("1", "both"):
            for model in models:
                run_phase1(
                    df,
                    model,
                    out_dir,
                    api_key,
                    api_url=api_url,
                    phase1_suffix=(
                        ANSWER_ONLY_SUFFIX_NVIDIA
                        if provider == "nvidia"
                        else ANSWER_ONLY_SUFFIX_OPENROUTER
                    ),
                    max_tokens=max_tok_p1,
                    temperature=args.temperature,
                    timeout=timeout_s,
                    workers=args.workers,
                    sleep_s=args.sleep,
                    referer=referer,
                    app_title=app_title,
                    max_retries=args.max_retries,
                    verbose=verbose,
                )
            build_leaderboard(out_dir, models, leaderboard_csv=leaderboard_csv, verbose=verbose)

        if args.phase in ("2", "both"):
            for model in models:
                p1 = (
                    Path(args.phase1_csv)
                    if args.phase1_csv
                    else out_dir / f"phase1_{model_slug(model)}.csv"
                )
                if not p1.exists():
                    print(f"Missing phase1 file for {model}: {p1}", file=sys.stderr)
                    return 1
                run_phase2(
                    df,
                    p1,
                    model,
                    out_dir,
                    api_key,
                    api_url=api_url,
                    export_csv_name=cot_export_name,
                    max_tokens=max_tok_cot,
                    temperature=args.temperature,
                    timeout=timeout_s,
                    workers=max(1, args.workers),
                    sleep_s=args.sleep,
                    referer=referer,
                    app_title=app_title,
                    max_retries=args.max_retries,
                    verbose=verbose,
                )

        if verbose:
            print("Done.", flush=True)
        return 0
    except KeyboardInterrupt:
        _print_interrupt_help(out_dir, args.phase)
        return 130


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # If interrupt happens before/during argparse or outside main()'s handler
        print("\nInterrupted.", flush=True)
        raise SystemExit(130)
