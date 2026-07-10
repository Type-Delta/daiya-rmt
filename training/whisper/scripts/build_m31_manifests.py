#!/usr/bin/env python
"""Build deterministic, conversation-disjoint M3.1 split/gate/benchmark manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


TRAIN_SOURCES = {"01", "02", "05", "06", "07", "08", "09", "10"}
VALIDATION_SOURCES = {"03", "04"}
BENCHMARK_SOURCES = {"11"}
SOURCE_RE = re.compile(r"Th-En_sample_(\d{2})", re.IGNORECASE)
LATIN_RE = re.compile(r"[A-Za-z]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def source_number(row: dict[str, Any]) -> str:
    value = str(row.get("source_file") or row.get("file_name") or "")
    match = SOURCE_RE.search(value)
    if not match:
        raise ValueError(f"Cannot derive source number from {value!r}")
    return match.group(1)


def source_basename(row: dict[str, Any]) -> str:
    return str(row["source_file"]).replace("\\", "/").rsplit("/", 1)[-1]


def stratum(row: dict[str, Any]) -> str:
    short = float(row.get("speech_duration") or 0.0) <= 3.0
    latin = bool(LATIN_RE.search(str(row.get("text") or "")))
    return f"{'short' if short else 'long'}_{'latin' if latin else 'no_latin'}"


def stable_rank(seed: str, row: dict[str, Any]) -> str:
    sample_id = str(row["file_name"])
    return hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).hexdigest()


def choose_stratified(
    rows: list[dict[str, Any]],
    *,
    sources: set[str],
    per_source_per_stratum: int,
    seed: str,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for source in sorted(sources):
        for bucket in ("short_latin", "short_no_latin", "long_latin", "long_no_latin"):
            candidates = [row for row in rows if source_number(row) == source and stratum(row) == bucket]
            candidates.sort(key=lambda row: stable_rank(seed, row))
            if len(candidates) < per_source_per_stratum:
                raise ValueError(f"Source {source} stratum {bucket} has only {len(candidates)} rows")
            chosen.extend(candidates[:per_source_per_stratum])
    return sorted(chosen, key=lambda row: (source_number(row), float(row.get("source_start") or 0.0)))


def manifest_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "sample_id": row["file_name"],
        "source_file": source_basename(row),
        "source_start": row.get("source_start"),
        "source_end": row.get("source_end"),
        "speech_duration": row.get("speech_duration"),
        "stratum": stratum(row),
        "split": split,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen_sources = {source_number(row) for row in rows}
    expected_sources = TRAIN_SOURCES | VALIDATION_SOURCES | BENCHMARK_SOURCES
    if seen_sources != expected_sources:
        raise ValueError(f"Unexpected sources: got {sorted(seen_sources)}, expected {sorted(expected_sources)}")

    split_rows = []
    for source in sorted(expected_sources):
        sample = next(row for row in rows if source_number(row) == source)
        split = "train" if source in TRAIN_SOURCES else "validation" if source in VALIDATION_SOURCES else "benchmark"
        split_rows.append({"source_file": source_basename(sample), "split": split})

    selector = choose_stratified(
        rows,
        sources=VALIDATION_SOURCES,
        per_source_per_stratum=16,
        seed="m31-gate-v1",
    )
    benchmark = choose_stratified(
        rows,
        sources=BENCHMARK_SOURCES,
        per_source_per_stratum=32,
        seed="m31-benchmark-v1",
    )

    write_jsonl(args.output_dir / "m31-split-v1.jsonl", split_rows)
    write_jsonl(args.output_dir / "m31-generation-gate-v1.jsonl", [manifest_row(row, "validation") for row in selector])
    write_jsonl(args.output_dir / "m31-benchmark-v1.jsonl", [manifest_row(row, "benchmark") for row in benchmark])

    metadata_hash = hashlib.sha256(args.metadata.read_bytes()).hexdigest()
    print(json.dumps({
        "metadata_sha256": metadata_hash,
        "split_count": len(split_rows),
        "selector_count": len(selector),
        "benchmark_count": len(benchmark),
    }, indent=2))


if __name__ == "__main__":
    main()
