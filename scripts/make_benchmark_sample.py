"""Build stratified benchmark_sample_500.csv from datasets/train_9000.csv."""

from __future__ import annotations

import pandas as pd

N_TOTAL = 500
OUTPUT = "datasets/benchmark_sample_500.csv"


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


def main() -> None:
    root = pd.read_csv("datasets/train_9000.csv")
    root["family"] = root["prompt"].map(classify_family)
    unknown = int((root["family"] == "unknown").sum())
    if unknown:
        raise SystemExit(f"Unexpected unknown families: {unknown}")

    families = sorted(root["family"].unique())
    n_total = N_TOTAL
    base = n_total // len(families)
    rem = n_total % len(families)
    counts = {f: base + (1 if i < rem else 0) for i, f in enumerate(families)}

    parts: list[pd.DataFrame] = []
    for fam in families:
        n = counts[fam]
        sub = root[root["family"] == fam]
        if len(sub) < n:
            raise SystemExit(f"Family {fam!r} has only {len(sub)} rows, need {n}")
        parts.append(sub.sample(n=n, random_state=42))

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out = out.drop(columns=["family"])
    out.to_csv(OUTPUT, index=False)
    print("Wrote", OUTPUT, "rows:", len(out))
    print("Per-family counts:", out.assign(_f=out["prompt"].map(classify_family))["_f"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
