#!/usr/bin/env python
"""Lab-only faster-whisper ASR probe for the Daiya clean labeled dataset.

Example:
    python lab/asr_eval.py --model large-v3 --limit 20 --device cuda --compute-type float16
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "training" / "dataset" / "hf_datasets" / "whisper"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "lab" / "artifacts" / "asr_eval"
STRATEGIES = {
    "isolated",
    "rolling_initial_prompt",
    "left_audio_context",
    "merged_deferred_short",
}
TECHNICAL_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]*")
THAI_GAP_RE = re.compile(r"(?<=[\u0e00-\u0e7f]) +(?=[\u0e00-\u0e7f])")
COMMON_ENGLISH_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "before",
    "be",
    "but",
    "by",
    "context",
    "current",
    "do",
    "does",
    "first",
    "for",
    "from",
    "hint",
    "in",
    "into",
    "is",
    "it",
    "label",
    "labels",
    "latest",
    "memory",
    "metadata",
    "note",
    "notes",
    "of",
    "only",
    "on",
    "or",
    "previous",
    "prompt",
    "recent",
    "repeat",
    "said",
    "spoken",
    "static",
    "successful",
    "term",
    "terms",
    "text",
    "that",
    "the",
    "this",
    "to",
    "topic",
    "transcript",
    "unless",
    "use",
    "with",
    "word",
    "words",
    "you",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate faster-whisper models against Daiya's clean labeled ASR chunks."
    )
    parser.add_argument(
        "--model", default=None, help="faster-whisper model name or local model path."
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated faster-whisper model names/paths to benchmark with identical config.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum metadata rows to evaluate."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Held-out manifest/sample allowlist. Supports JSON, JSONL, CSV/TSV, or one sample ID per line. "
            "Rows are evaluated exactly once in manifest order."
        ),
    )
    parser.add_argument(
        "--sample-ids",
        default=None,
        help="Comma-separated stable sample IDs to evaluate exactly once. Cannot be combined with --manifest.",
    )
    parser.add_argument(
        "--sample-id-field",
        default="sample_id",
        help="Preferred metadata/manifest field used as the stable sample ID.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Training split manifest used to prove benchmark rows are held out by source/conversation.",
    )
    parser.add_argument(
        "--required-split",
        default="benchmark",
        choices=("test", "benchmark"),
        help="Every selected benchmark row must belong to this split in --split-manifest.",
    )
    parser.add_argument(
        "--compare-raw",
        nargs="+",
        type=Path,
        default=None,
        help="Load one or more existing details JSONL files and summarize/compare compatible raw outputs.",
    )
    parser.add_argument(
        "--device", default="auto", help="faster-whisper device, e.g. auto, cpu, cuda."
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="faster-whisper compute type, e.g. default, int8, float16, int8_float16.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language hint passed to transcribe().",
    )
    parser.add_argument(
        "--initial-prompt", default=None, help="Optional prompt passed to transcribe()."
    )
    parser.add_argument(
        "--include-context-technical-terms",
        action="store_true",
        help="Add technical terms derived only from each row's context_before to its prompt.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset root containing metadata.jsonl and WAV chunks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for details JSONL and summary JSON.",
    )
    parser.add_argument(
        "--beam-size", type=int, default=5, help="Beam size passed to transcribe()."
    )
    parser.add_argument(
        "--no-condition-on-previous-text",
        action="store_true",
        help="Disable conditioning on previous text for independent chunk scoring.",
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES),
        default="isolated",
        help="Chunk decode strategy to evaluate. Default preserves isolated chunk scoring.",
    )
    parser.add_argument(
        "--benchmark-strategies",
        default=None,
        help=(
            "Comma-separated strategies to compare in one run, e.g. "
            "isolated,rolling_initial_prompt,left_audio_context,merged_deferred_short."
        ),
    )
    parser.add_argument(
        "--short-utterance-seconds",
        type=float,
        default=3.0,
        help="Speech-duration threshold for short-utterance subset summaries and merge decisions.",
    )
    parser.add_argument(
        "--rolling-prompt-turns",
        type=int,
        default=3,
        help="Number of previous predictions to include for rolling_initial_prompt.",
    )
    parser.add_argument(
        "--rolling-prompt-chars",
        type=int,
        default=600,
        help="Maximum rolling transcript characters appended to initial_prompt.",
    )
    parser.add_argument(
        "--left-audio-context-seconds",
        type=float,
        default=4.0,
        help="Maximum previous chunk audio seconds prepended for left_audio_context.",
    )
    parser.add_argument(
        "--context-max-gap-seconds",
        type=float,
        default=1.5,
        help="Maximum source timestamp gap when borrowing previous/next chunks for context strategies.",
    )
    parser.add_argument(
        "--merge-max-seconds",
        type=float,
        default=12.0,
        help="Maximum concatenated audio duration for merged_deferred_short groups.",
    )
    parser.add_argument(
        "--merge-max-chunks",
        type=int,
        default=3,
        help="Maximum number of adjacent chunks in a merged_deferred_short group.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Deterministic paired bootstrap samples used for micro CER/WER-like confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=1337,
        help="Seed for deterministic bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-block-size",
        type=int,
        default=8,
        help="Contiguous moving-block size used for deterministic bootstrap resampling.",
    )
    return parser.parse_args()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "model"


def model_set_suffix(models: list[str]) -> str:
    """Return a short deterministic filename component; full model identities stay in JSON."""
    if len(models) == 1:
        leaf = Path(models[0]).name or models[0]
        safe_leaf = safe_name(leaf)
        if len(safe_leaf) <= 64:
            return safe_leaf
    return f"models-{len(models)}-{sha256_json(models)[:12]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_models(args: argparse.Namespace) -> list[str]:
    models: list[str] = []
    raw_values = []
    if args.model:
        raw_values.append(args.model)
    if args.models:
        raw_values.extend(args.models.split(","))
    for raw_model in raw_values:
        model = raw_model.strip()
        if model and model not in models:
            models.append(model)
    if args.compare_raw:
        return models
    if not models:
        raise ValueError("Provide --model or --models unless using --compare-raw.")
    return models


def read_metadata(dataset_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    metadata_path = dataset_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found at {metadata_path}")

    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on {metadata_path}:{line_number}: {exc}"
                ) from exc
            record["_line_number"] = line_number
            rows.append(record)
    return rows


def record_audio_name(record: dict[str, Any]) -> str | None:
    file_name = record.get("file_name") or record.get("path")
    audio = record.get("audio")
    if not file_name and isinstance(audio, dict):
        file_name = audio.get("path")
    return str(file_name) if file_name else None


def sample_id_for_record(
    record: dict[str, Any], preferred_field: str = "sample_id"
) -> str:
    for key in (preferred_field, "sample_id", "id", "uid", "uuid"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    audio_name = record_audio_name(record)
    if audio_name:
        return audio_name
    source_file = record.get("source_file")
    source_start = record.get("source_start")
    source_end = record.get("source_end")
    if source_file is not None and source_start is not None and source_end is not None:
        return f"{source_file}:{source_start}-{source_end}:{audio_name or ''}"
    line_number = record.get("_line_number")
    if line_number is not None:
        return f"metadata-line:{line_number}"
    raise ValueError(f"Cannot derive stable sample ID for record: {record}")


def manifest_id_from_row(row: dict[str, Any], preferred_field: str) -> str:
    for key in (preferred_field, "sample_id", "id", "uid", "uuid", "file_name", "path"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"Manifest row has no sample ID field: {row}")


def read_manifest_ids(manifest_path: Path, preferred_field: str) -> list[str]:
    suffix = manifest_path.suffix.lower()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if suffix == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get(
                "samples", payload.get("items", payload.get("ids", payload))
            )
        if isinstance(payload, list):
            ids = []
            for item in payload:
                ids.append(
                    manifest_id_from_row(item, preferred_field)
                    if isinstance(item, dict)
                    else str(item).strip()
                )
            return [sample_id for sample_id in ids if sample_id]
        if isinstance(payload, dict):
            return [manifest_id_from_row(payload, preferred_field)]
        raise ValueError(f"Unsupported JSON manifest shape in {manifest_path}")

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames:
                return [manifest_id_from_row(row, preferred_field) for row in reader]

    ids: list[str] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                try:
                    ids.append(manifest_id_from_row(json.loads(line), preferred_field))
                    continue
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL manifest row {manifest_path}:{line_number}: {exc}"
                    ) from exc
            ids.append(line.split(",", 1)[0].split("\t", 1)[0].strip())
    return ids


def parse_requested_sample_ids(
    args: argparse.Namespace,
) -> tuple[list[str] | None, str, str | None]:
    if args.manifest and args.sample_ids:
        raise ValueError("--manifest and --sample-ids are mutually exclusive.")
    if (args.manifest or args.sample_ids) and args.limit is not None:
        raise ValueError(
            "--limit cannot be combined with a fixed manifest/sample allowlist."
        )
    if args.manifest:
        return (
            read_manifest_ids(args.manifest, args.sample_id_field),
            "manifest",
            str(args.manifest.resolve()),
        )
    if args.sample_ids:
        sample_ids = [
            item.strip() for item in args.sample_ids.split(",") if item.strip()
        ]
        return sample_ids, "sample_ids", None
    return None, "limit" if args.limit is not None else "metadata", None


def validate_unique_sample_ids(sample_ids: list[str], *, source: str) -> None:
    counts = Counter(sample_ids)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{source} contains duplicate sample IDs: {duplicates[:10]}")


def select_metadata(
    metadata: list[dict[str, Any]],
    requested_sample_ids: list[str] | None,
    sample_id_field: str,
) -> list[dict[str, Any]]:
    for record in metadata:
        record["_sample_id"] = sample_id_for_record(record, sample_id_field)
    if requested_sample_ids is None:
        return metadata

    validate_unique_sample_ids(
        requested_sample_ids, source="Requested manifest/sample allowlist"
    )
    by_id: dict[str, list[dict[str, Any]]] = {}
    for record in metadata:
        by_id.setdefault(str(record["_sample_id"]), []).append(record)

    missing = [
        sample_id for sample_id in requested_sample_ids if sample_id not in by_id
    ]
    duplicate_records = sorted(
        sample_id
        for sample_id in requested_sample_ids
        if len(by_id.get(sample_id, [])) > 1
    )
    if missing or duplicate_records:
        parts = []
        if missing:
            parts.append(f"missing IDs: {missing[:10]}")
        if duplicate_records:
            parts.append(f"metadata duplicates: {duplicate_records[:10]}")
        raise ValueError(
            "Manifest/sample allowlist validation failed; " + "; ".join(parts)
        )
    return [by_id[sample_id][0] for sample_id in requested_sample_ids]


def portable_source_id(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def read_split_manifest_assignments(
    path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    sample_splits: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid split manifest JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Split manifest row must be an object at {path}:{line_number}"
                )
            split = str(row.get("split") or row.get("partition") or "").strip().lower()
            split = {
                "val": "validation",
                "eval": "validation",
                "dev": "validation",
            }.get(split, split)
            if split not in {"train", "validation", "test", "benchmark"}:
                raise ValueError(f"Invalid split {split!r} at {path}:{line_number}")
            sample_id = row.get("sample_id") or row.get("id") or row.get("uid")
            explicit_group = (
                row.get("conversation")
                or row.get("conversation_id")
                or row.get("group_id")
                or row.get("session_id")
            )
            source_file = row.get("source_file")
            group_id = (
                str(explicit_group)
                if explicit_group
                else portable_source_id(str(source_file))
                if source_file
                else None
            )
            if not sample_id and not group_id:
                raise ValueError(
                    f"Split manifest row has no sample/group identity at {path}:{line_number}"
                )
            for mapping, key, label in (
                (sample_splits, sample_id, "sample"),
                (group_splits, group_id, "group"),
            ):
                if not key:
                    continue
                key = str(key)
                previous = mapping.setdefault(key, split)
                if previous != split:
                    raise ValueError(
                        f"Split manifest {label} {key!r} maps to both {previous!r} and {split!r}"
                    )
    return sample_splits, group_splits


def validate_selected_split(
    metadata: list[dict[str, Any]],
    split_manifest: Path,
    required_split: str,
) -> dict[str, Any]:
    sample_splits, group_splits = read_split_manifest_assignments(split_manifest)
    assignments: list[str] = []
    for record in metadata:
        sample_id = str(record.get("_sample_id") or sample_id_for_record(record))
        group_id = portable_source_id(
            str(record.get("source_file") or source_group_for_record(record))
        )
        split = sample_splits.get(sample_id) or group_splits.get(group_id)
        if split != required_split:
            raise ValueError(
                f"Benchmark sample {sample_id!r} belongs to split {split!r}, expected {required_split!r}."
            )
        assignments.append(f"{sample_id}\0{group_id}\0{split}")
    return {
        "path": str(split_manifest.resolve()),
        "sha256": sha256_file(split_manifest),
        "required_split": required_split,
        "validated_count": len(metadata),
        "assignment_sha256": sha256_json(assignments),
    }


def validate_rolling_order(metadata: list[dict[str, Any]]) -> None:
    previous_group: str | None = None
    previous_start: float | None = None
    closed_groups: set[str] = set()
    for record in metadata:
        group = portable_source_id(
            str(record.get("source_file") or source_group_for_record(record))
        )
        start = record.get("source_start")
        if group != previous_group:
            if group in closed_groups:
                raise ValueError(
                    f"Rolling manifest re-enters closed source group {group!r}."
                )
            if previous_group is not None:
                closed_groups.add(previous_group)
            previous_group = group
            previous_start = None
        if (
            isinstance(start, int | float)
            and previous_start is not None
            and float(start) < previous_start
        ):
            raise ValueError(
                f"Rolling manifest is out of source-time order for {group!r}."
            )
        if isinstance(start, int | float):
            previous_start = float(start)


def dataset_hash(dataset_dir: Path) -> str | None:
    metadata_path = dataset_dir / "metadata.jsonl"
    return sha256_file(metadata_path) if metadata_path.exists() else None


def sample_set_hash(sample_ids: list[str]) -> str:
    return sha256_json(sample_ids)


def decode_config_payload(
    args: argparse.Namespace, strategies: list[str]
) -> dict[str, Any]:
    return {
        "language": args.language,
        "initial_prompt": args.initial_prompt,
        "include_context_technical_terms": args.include_context_technical_terms,
        "beam_size": args.beam_size,
        "condition_on_previous_text": not args.no_condition_on_previous_text,
        "strategies": strategies,
        "short_utterance_seconds": args.short_utterance_seconds,
        "rolling_prompt_turns": args.rolling_prompt_turns,
        "rolling_prompt_chars": args.rolling_prompt_chars,
        "left_audio_context_seconds": args.left_audio_context_seconds,
        "context_max_gap_seconds": args.context_max_gap_seconds,
        "merge_max_seconds": args.merge_max_seconds,
        "merge_max_chunks": args.merge_max_chunks,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_block_size": args.bootstrap_block_size,
    }


def model_identity(model_arg: str) -> dict[str, Any]:
    path = Path(model_arg)
    exists = path.exists()
    resolved = str(path.resolve()) if exists else None
    fingerprint_payload: dict[str, Any] = {"model": model_arg, "path": resolved}
    content_hash = None
    if exists and path.is_file():
        content_hash = sha256_file(path)
    elif exists and path.is_dir():
        entries = []
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            try:
                stat = child.stat()
            except OSError:
                continue
            rel = str(child.relative_to(path)).replace("\\", "/")
            entries.append(
                {"path": rel, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            )
        fingerprint_payload["files"] = entries
    return {
        "name": model_arg,
        "path": resolved,
        "exists": exists,
        "fingerprint": content_hash or sha256_json(fingerprint_payload),
    }


def resolve_audio_path(dataset_dir: Path, record: dict[str, Any]) -> Path | None:
    file_name = record.get("file_name") or record.get("path")
    audio = record.get("audio")
    if not file_name and isinstance(audio, dict):
        file_name = audio.get("path")
    if not file_name:
        return None

    audio_path = Path(str(file_name))
    if not audio_path.is_absolute():
        audio_path = dataset_dir / audio_path
    return audio_path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_thai_spacing(text: str) -> str:
    thai_chars = sum("\u0e00" <= char <= "\u0e7f" for char in text)
    if thai_chars < 10:
        return text
    gaps = len(THAI_GAP_RE.findall(text))
    if gaps / thai_chars <= 0.12:
        return text
    return THAI_GAP_RE.sub("", text)


def normalize_no_space(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def is_combining_or_tone(char: str) -> bool:
    return unicodedata.category(char).startswith("M")


def token_units(text: str) -> list[str]:
    """Small language-agnostic tokenization for WER-like scoring.

    Latin words stay as words. Thai base characters carry following marks as
    one unit because Thai whitespace is phrase-like rather than word-like.
    """
    normalized = normalize_text(text)
    tokens: list[str] = []
    index = 0
    while index < len(normalized):
        char = normalized[index]
        if char.isspace() or is_punctuation_or_symbol(char):
            index += 1
            continue

        if is_latin_or_digit(char):
            start = index
            while index < len(normalized) and is_latin_or_digit(normalized[index]):
                index += 1
            tokens.append(normalized[start:index])
            continue

        if is_thai(char):
            unit = [char]
            index += 1
            while (
                index < len(normalized)
                and is_thai(normalized[index])
                and is_combining_or_tone(normalized[index])
            ):
                unit.append(normalized[index])
                index += 1
            tokens.append("".join(unit))
            continue

        if is_combining_or_tone(char) and tokens:
            tokens[-1] += char
        else:
            tokens.append(char)
        index += 1
    return tokens


def levenshtein(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_item in enumerate(reference, start=1):
        current = [row_index]
        for col_index, hyp_item in enumerate(hypothesis, start=1):
            insertion = current[col_index - 1] + 1
            deletion = previous[col_index] + 1
            substitution = previous[col_index - 1] + (ref_item != hyp_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def error_rate(distance: int, reference_length: int) -> float | None:
    if reference_length == 0:
        return None
    return distance / reference_length


def is_thai(char: str) -> bool:
    return "\u0e00" <= char <= "\u0e7f"


def is_basic_latin(char: str) -> bool:
    return "A" <= char <= "Z" or "a" <= char <= "z"


def is_latin_or_digit(char: str) -> bool:
    return is_basic_latin(char) or char.isdigit()


def is_punctuation_or_symbol(char: str) -> bool:
    return unicodedata.category(char)[0] in {"P", "S"}


def script_name(char: str) -> str | None:
    if is_thai(char):
        return "Thai"
    if is_basic_latin(char):
        return "Latin"
    codepoint = ord(char)
    if 0x3040 <= codepoint <= 0x309F:
        return "Hiragana"
    if 0x30A0 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
        return "Katakana"
    if 0x4E00 <= codepoint <= 0x9FFF or 0x3400 <= codepoint <= 0x4DBF:
        return "CJK"
    if 0xAC00 <= codepoint <= 0xD7AF:
        return "Hangul"
    if 0x0400 <= codepoint <= 0x04FF:
        return "Cyrillic"
    if 0x0370 <= codepoint <= 0x03FF:
        return "Greek"
    if 0x0600 <= codepoint <= 0x06FF:
        return "Arabic"
    if 0x0900 <= codepoint <= 0x097F:
        return "Devanagari"
    if char.isdigit():
        return "Digit"
    return None


def non_thai_english_scripts(text: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for char in unicodedata.normalize("NFKC", text):
        if (
            char.isspace()
            or is_punctuation_or_symbol(char)
            or is_combining_or_tone(char)
        ):
            continue
        script = script_name(char)
        if script in {None, "Thai", "Latin", "Digit"}:
            continue
        counts[script] += 1
    return dict(sorted(counts.items()))


def language_and_mixed_bucket(
    record: dict[str, Any], reference: str
) -> tuple[str | None, str]:
    raw_language = record.get("language") or record.get("language_label")
    language = str(raw_language).strip() if raw_language not in (None, "") else None
    label = (language or "").lower().replace("-", "_")
    if "thai" in label and ("english" in label or "_en" in label or " en" in label):
        return language, "thai_english"
    if ("japanese" in label or label.startswith("ja")) and (
        "english" in label or "_en" in label or " en" in label
    ):
        return language, "japanese_english"
    if "thai" in label or label.startswith("th"):
        return language, "thai"
    if "japanese" in label or label.startswith("ja"):
        return language, "japanese"
    if "english" in label or label.startswith("en"):
        return language, "english"

    scripts = Counter(
        script
        for char in unicodedata.normalize("NFKC", reference)
        if not char.isspace()
        and not is_punctuation_or_symbol(char)
        and not is_combining_or_tone(char)
        for script in [script_name(char)]
        if script is not None
    )
    has_latin = scripts.get("Latin", 0) > 0
    has_thai = scripts.get("Thai", 0) > 0
    has_japanese = any(
        scripts.get(script, 0) > 0 for script in ("Hiragana", "Katakana", "CJK")
    )
    if has_thai and has_latin:
        return language, "thai_english"
    if has_japanese and has_latin:
        return language, "japanese_english"
    if has_thai:
        return language, "thai"
    if has_japanese:
        return language, "japanese"
    if has_latin:
        return language, "english"
    return language, "other"


def source_group_for_record(record: dict[str, Any]) -> str:
    for key in (
        "group",
        "source_group",
        "source_file",
        "session_id",
        "conversation_id",
    ):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    audio_name = record_audio_name(record)
    if audio_name:
        return str(
            Path(audio_name).parent
            if Path(audio_name).parent != Path(".")
            else Path(audio_name).stem
        )
    return "unknown"


def english_terms_from_text(*texts: str) -> list[str]:
    terms: dict[str, str] = {}
    for text in texts:
        for match in TECHNICAL_TERM_RE.finditer(unicodedata.normalize("NFKC", text)):
            term = match.group(0).strip("._-")
            if not is_useful_english_term(term):
                continue
            terms.setdefault(term.lower(), term)
    return sorted(terms.values(), key=lambda value: value.lower())


def is_useful_english_term(term: str) -> bool:
    normalized = term.lower()
    if len(term) < 2 or normalized in COMMON_ENGLISH_WORDS:
        return False
    if normalized.isdigit():
        return False
    if len(term) == 2 and not term.isupper():
        return False
    if "." in term or "_" in term or "+" in term or "#" in term:
        return True
    if term.isupper() and len(term) >= 2:
        return True
    return len(term) >= 4


def text_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    ref_no_space = normalize_no_space(reference)
    hyp_no_space = normalize_no_space(hypothesis)
    ref_tokens = token_units(reference)
    hyp_tokens = token_units(hypothesis)

    cer_distance = levenshtein(ref_norm, hyp_norm)
    cer_no_space_distance = levenshtein(ref_no_space, hyp_no_space)
    wer_like_distance = levenshtein(ref_tokens, hyp_tokens)

    return {
        "cer_distance": cer_distance,
        "cer_reference_length": len(ref_norm),
        "cer": error_rate(cer_distance, len(ref_norm)),
        "cer_no_space_distance": cer_no_space_distance,
        "cer_no_space_reference_length": len(ref_no_space),
        "cer_no_space": error_rate(cer_no_space_distance, len(ref_no_space)),
        "wer_like_distance": wer_like_distance,
        "wer_like_reference_length": len(ref_tokens),
        "wer_like": error_rate(wer_like_distance, len(ref_tokens)),
    }


def load_faster_whisper() -> Any | None:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    return WhisperModel


def memory_snapshot() -> dict[str, Any]:
    ram_bytes = None
    try:
        import psutil  # type: ignore[import-not-found]

        ram_bytes = psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        ram_bytes = None

    gpu_bytes = process_gpu_memory_bytes()
    return {
        "ram_rss_bytes": ram_bytes,
        "gpu_peak_bytes": gpu_bytes,
        "gpu_measurement": "nvidia-smi_process_snapshot"
        if gpu_bytes is not None
        else "unavailable",
    }


def process_gpu_memory_bytes() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    current_pid = os.getpid()
    values: list[int] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            mebibytes = int(parts[1])
        except ValueError:
            continue
        if pid == current_pid:
            values.append(mebibytes * 1024 * 1024)
    return sum(values) if values else None


def memory_peak(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "ram_rss_bytes": max(
            [
                value
                for value in (before.get("ram_rss_bytes"), after.get("ram_rss_bytes"))
                if isinstance(value, int)
            ],
            default=None,
        ),
        "gpu_peak_bytes": max(
            [
                value
                for value in (before.get("gpu_peak_bytes"), after.get("gpu_peak_bytes"))
                if isinstance(value, int)
            ],
            default=None,
        ),
        "gpu_measurement": (
            "nvidia-smi_process_snapshot"
            if before.get("gpu_peak_bytes") is not None
            or after.get("gpu_peak_bytes") is not None
            else "unavailable"
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, default=json_default
                )
            )
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL row {path}:{line_number}: {exc}"
                ) from exc
    return rows


def json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()  # type: ignore[no-any-return]
    return str(value)


def nan_safe_mean(values: list[float | None]) -> float | None:
    real_values = [
        value for value in values if value is not None and not math.isnan(value)
    ]
    if not real_values:
        return None
    return sum(real_values) / len(real_values)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate_latency_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [
        row.get("latency_seconds")
        for row in rows
        if isinstance(row.get("latency_seconds"), int | float)
    ]
    rtfs = [row.get("rtf") for row in rows if isinstance(row.get("rtf"), int | float)]
    processed_rtfs = [
        row.get("processed_audio_rtf")
        for row in rows
        if isinstance(row.get("processed_audio_rtf"), int | float)
    ]
    durations = [
        row.get("duration_seconds")
        for row in rows
        if isinstance(row.get("duration_seconds"), int | float)
    ]
    processed_durations = [
        row.get("processed_audio_duration_seconds")
        for row in rows
        if isinstance(row.get("processed_audio_duration_seconds"), int | float)
    ]
    return {
        "latency_mean_seconds": nan_safe_mean(latencies),
        "latency_p50_seconds": percentile(latencies, 50),
        "latency_p95_seconds": percentile(latencies, 95),
        "rtf_mean": nan_safe_mean(rtfs),
        "rtf_p50": percentile(rtfs, 50),
        "rtf_p95": percentile(rtfs, 95),
        "processed_audio_rtf_mean": nan_safe_mean(processed_rtfs),
        "processed_audio_rtf_p50": percentile(processed_rtfs, 50),
        "processed_audio_rtf_p95": percentile(processed_rtfs, 95),
        "duration_seconds": sum(durations) if durations else None,
        "processed_audio_duration_seconds": sum(processed_durations)
        if processed_durations
        else None,
    }


def aggregate_memory_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ram_values = []
    gpu_values = []
    for row in rows:
        peak = row.get("peak_memory") or {}
        if isinstance(peak.get("ram_rss_bytes"), int):
            ram_values.append(peak["ram_rss_bytes"])
        if isinstance(peak.get("gpu_peak_bytes"), int):
            gpu_values.append(peak["gpu_peak_bytes"])
    max_ram = max(ram_values) if ram_values else None
    max_gpu = max(gpu_values) if gpu_values else None
    return {
        "memory_aggregation_method": "max_endpoint_snapshots",
        "max_endpoint_ram_rss_bytes": max_ram,
        "max_endpoint_gpu_bytes": max_gpu,
        # Compatibility aliases; these are endpoint snapshots, not sampled process peaks.
        "peak_ram_rss_bytes": max_ram,
        "peak_gpu_bytes": max_gpu,
    }


def contiguous_block_indices(
    length: int, block_size: int, rng: random.Random
) -> list[int]:
    if block_size <= 0:
        raise ValueError("bootstrap block size must be positive")
    if length <= 0:
        return []
    width = min(block_size, length)
    indices: list[int] = []
    while len(indices) < length:
        start = rng.randrange(length - width + 1)
        indices.extend(range(start, start + width))
    return indices[:length]


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_distance = sum(row["metrics"]["cer_distance"] for row in rows)
    aggregate_length = sum(row["metrics"]["cer_reference_length"] for row in rows)
    aggregate_no_space_distance = sum(
        row["metrics"]["cer_no_space_distance"] for row in rows
    )
    aggregate_no_space_length = sum(
        row["metrics"]["cer_no_space_reference_length"] for row in rows
    )
    aggregate_word_distance = sum(row["metrics"]["wer_like_distance"] for row in rows)
    aggregate_word_length = sum(
        row["metrics"]["wer_like_reference_length"] for row in rows
    )
    return {
        "count": len(rows),
        "cer_edit_count": aggregate_distance,
        "cer_reference_count": aggregate_length,
        "mean_cer": nan_safe_mean([row["metrics"]["cer"] for row in rows]),
        "micro_cer": error_rate(aggregate_distance, aggregate_length),
        "cer_no_space_edit_count": aggregate_no_space_distance,
        "cer_no_space_reference_count": aggregate_no_space_length,
        "mean_cer_no_space": nan_safe_mean(
            [row["metrics"]["cer_no_space"] for row in rows]
        ),
        "micro_cer_no_space": error_rate(
            aggregate_no_space_distance, aggregate_no_space_length
        ),
        "wer_like_edit_count": aggregate_word_distance,
        "wer_like_reference_count": aggregate_word_length,
        "mean_wer_like": nan_safe_mean([row["metrics"]["wer_like"] for row in rows]),
        "micro_wer_like": error_rate(aggregate_word_distance, aggregate_word_length),
        **aggregate_latency_rows(rows),
        **aggregate_memory_rows(rows),
    }


def bootstrap_micro_ci(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    samples: int,
    seed: int,
    block_size: int = 8,
) -> dict[str, Any] | None:
    if not rows or samples <= 0:
        return None
    distance_key = f"{metric}_distance"
    length_key = f"{metric}_reference_length"
    pairs = [
        (row["metrics"][distance_key], row["metrics"][length_key])
        for row in rows
        if row.get("metrics") and row["metrics"].get(length_key, 0) > 0
    ]
    if not pairs:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        distance = 0
        length = 0
        for index in contiguous_block_indices(len(pairs), block_size, rng):
            sample_distance, sample_length = pairs[index]
            distance += sample_distance
            length += sample_length
        rate = error_rate(distance, length)
        if rate is not None:
            values.append(rate)
    return {
        "seed": seed,
        "samples": samples,
        "block_size": block_size,
        "method": "contiguous_moving_block_bootstrap",
        "p2_5": percentile(values, 2.5),
        "p50": percentile(values, 50),
        "p97_5": percentile(values, 97.5),
    }


def add_uncertainty(
    aggregate: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
    block_size: int,
) -> dict[str, Any]:
    aggregate = dict(aggregate)
    aggregate["uncertainty"] = {
        "method": "contiguous_moving_block_bootstrap",
        "block_size": block_size,
        "micro_cer": bootstrap_micro_ci(
            rows, metric="cer", samples=samples, seed=seed, block_size=block_size
        ),
        "micro_wer_like": bootstrap_micro_ci(
            rows,
            metric="wer_like",
            samples=samples,
            seed=seed + 1,
            block_size=block_size,
        ),
    }
    return aggregate


def paired_bootstrap_delta(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    metric: str,
    samples: int,
    seed: int,
    block_size: int = 8,
) -> dict[str, Any]:
    left_by_id = {str(row.get("sample_id")): row for row in left_rows}
    right_by_id = {str(row.get("sample_id")): row for row in right_rows}
    if len(left_by_id) != len(left_rows) or len(right_by_id) != len(right_rows):
        raise ValueError(
            "Paired comparison contains duplicate sample IDs within a model/strategy."
        )
    if set(left_by_id) != set(right_by_id):
        missing_left = sorted(set(right_by_id) - set(left_by_id))[:5]
        missing_right = sorted(set(left_by_id) - set(right_by_id))[:5]
        raise ValueError(
            f"Paired comparison sample sets differ; missing left={missing_left}, missing right={missing_right}."
        )
    sample_ids = [str(row.get("sample_id")) for row in left_rows]
    distance_key = f"{metric}_distance"
    length_key = f"{metric}_reference_length"

    def delta(indices: list[int]) -> float | None:
        left_distance = left_length = right_distance = right_length = 0
        for index in indices:
            sample_id = sample_ids[index]
            left_metric = left_by_id[sample_id]["metrics"]
            right_metric = right_by_id[sample_id]["metrics"]
            left_distance += left_metric[distance_key]
            left_length += left_metric[length_key]
            right_distance += right_metric[distance_key]
            right_length += right_metric[length_key]
        left_rate = error_rate(left_distance, left_length)
        right_rate = error_rate(right_distance, right_length)
        return (
            None if left_rate is None or right_rate is None else left_rate - right_rate
        )

    point = delta(list(range(len(sample_ids))))
    rng = random.Random(seed)
    values = []
    for _ in range(max(0, samples)):
        value = delta(contiguous_block_indices(len(sample_ids), block_size, rng))
        if value is not None:
            values.append(value)
    return {
        "left_minus_right": point,
        "sample_count": len(sample_ids),
        "seed": seed,
        "samples": samples,
        "block_size": block_size,
        "method": "paired_contiguous_moving_block_bootstrap",
        "p2_5": percentile(values, 2.5),
        "p50": percentile(values, 50),
        "p97_5": percentile(values, 97.5),
    }


def paired_model_delta_summaries(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
    block_size: int,
) -> dict[str, Any]:
    by_strategy_model: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        strategy = str(row.get("strategy") or "unknown")
        model = str(row.get("model_name") or "unknown")
        by_strategy_model.setdefault(strategy, {}).setdefault(model, []).append(row)
    result: dict[str, Any] = {}
    offset = 0
    for strategy, by_model in sorted(by_strategy_model.items()):
        models = sorted(by_model)
        if len(models) < 2:
            continue
        strategy_result: dict[str, Any] = {}
        for left_index, left in enumerate(models):
            for right in models[left_index + 1 :]:
                key = f"{left}__minus__{right}"
                strategy_result[key] = {
                    "left_model": left,
                    "right_model": right,
                    "micro_cer": paired_bootstrap_delta(
                        by_model[left],
                        by_model[right],
                        metric="cer",
                        samples=samples,
                        seed=seed + offset,
                        block_size=block_size,
                    ),
                    "micro_wer_like": paired_bootstrap_delta(
                        by_model[left],
                        by_model[right],
                        metric="wer_like",
                        samples=samples,
                        seed=seed + 10_000 + offset,
                        block_size=block_size,
                    ),
                }
                offset += 1
        result[strategy] = strategy_result
    return result


def grouped_aggregates(
    rows: list[dict[str, Any]],
    field: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    bootstrap_block_size: int,
    include_uncertainty: bool = False,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(field)
        key = str(value) if value not in (None, "") else "unknown"
        groups.setdefault(key, []).append(row)
    result = {}
    for offset, key in enumerate(sorted(groups)):
        aggregate = aggregate_metric_rows(groups[key])
        if include_uncertainty:
            aggregate = add_uncertainty(
                aggregate,
                groups[key],
                samples=bootstrap_samples,
                seed=bootstrap_seed + offset,
                block_size=bootstrap_block_size,
            )
        result[key] = aggregate
    return result


def parse_strategies(args: argparse.Namespace) -> list[str]:
    if not args.benchmark_strategies:
        return [args.strategy]

    strategies = []
    for raw_strategy in args.benchmark_strategies.split(","):
        strategy = raw_strategy.strip()
        if not strategy:
            continue
        if strategy not in STRATEGIES:
            valid = ", ".join(sorted(STRATEGIES))
            raise ValueError(
                f"Unknown benchmark strategy {strategy!r}; expected one of: {valid}"
            )
        if strategy not in strategies:
            strategies.append(strategy)
    if not strategies:
        raise ValueError("--benchmark-strategies did not contain any strategy names")
    return strategies


def build_run_metadata(
    *,
    args: argparse.Namespace,
    strategies: list[str],
    metadata: list[dict[str, Any]],
    requested_sample_ids: list[str] | None,
    selection_mode: str,
    manifest_source: str | None,
    split_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_ids = [str(record.get("_sample_id")) for record in metadata]
    manifest_hash = (
        sha256_file(Path(manifest_source))
        if manifest_source
        else (
            sample_set_hash(requested_sample_ids)
            if requested_sample_ids is not None
            else None
        )
    )
    decode_config = decode_config_payload(args, strategies)
    payload = {
        "dataset_hash": dataset_hash(args.dataset_dir),
        "manifest_hash": manifest_hash,
        "sample_set_hash": sample_set_hash(sample_ids),
        "decode_config_hash": sha256_json(decode_config),
        "selection_mode": selection_mode,
        "split_manifest_hash": split_identity.get("sha256") if split_identity else None,
        "required_split": split_identity.get("required_split")
        if split_identity
        else None,
        "split_assignment_hash": split_identity.get("assignment_sha256")
        if split_identity
        else None,
    }
    return {
        **payload,
        "manifest_source": manifest_source,
        "sample_ids": sample_ids,
        "decode_config": decode_config,
        "split_manifest": split_identity,
        "benchmark_fingerprint": sha256_json(payload),
        "primary_run": selection_mode in {"manifest", "sample_ids"}
        and split_identity is not None,
        "seed": args.bootstrap_seed,
    }


def compatibility_from_row(row: dict[str, Any]) -> dict[str, Any]:
    run = row.get("run") or {}
    return {
        "dataset_hash": run.get("dataset_hash"),
        "manifest_hash": run.get("manifest_hash"),
        "sample_set_hash": run.get("sample_set_hash"),
        "decode_config_hash": run.get("decode_config_hash"),
        "benchmark_fingerprint": run.get("benchmark_fingerprint"),
    }


def assert_compatible_raw_outputs(
    rows_by_path: dict[Path, list[dict[str, Any]]],
) -> dict[str, Any]:
    baseline: dict[str, Any] | None = None
    for path, rows in rows_by_path.items():
        if not rows:
            raise ValueError(f"Raw output is empty: {path}")
        row_compatibilities = {
            sha256_json(compatibility_from_row(row)): compatibility_from_row(row)
            for row in rows
        }
        if len(row_compatibilities) != 1:
            raise ValueError(
                f"Raw output mixes incompatible fingerprints within one file: {path}"
            )
        current = next(iter(row_compatibilities.values()))
        if baseline is None:
            baseline = current
            continue
        for key in (
            "dataset_hash",
            "manifest_hash",
            "sample_set_hash",
            "decode_config_hash",
            "benchmark_fingerprint",
        ):
            if baseline.get(key) != current.get(key):
                raise ValueError(
                    f"Incompatible raw outputs for {path}: {key} differs "
                    f"({current.get(key)!r} != {baseline.get(key)!r})"
                )
    return baseline or {}


def compare_raw_outputs(
    paths: list[Path], args: argparse.Namespace
) -> tuple[Path, dict[str, Any]]:
    rows_by_path = {path.resolve(): read_jsonl(path) for path in paths}
    compatibility = assert_compatible_raw_outputs(rows_by_path)
    rows = [row for file_rows in rows_by_path.values() for row in file_rows]
    for row in rows:
        if "model_name" not in row:
            model = row.get("model")
            row["model_name"] = (
                model.get("name") if isinstance(model, dict) else str(model)
            )
    run_id = f"{utc_stamp()}_compare_raw"
    summary_path = args.output_dir.resolve() / f"summary_{run_id}.json"
    run_metadata = {
        **compatibility,
        "decode_config": (rows[0].get("run") or {}).get("decode_config")
        if rows
        else None,
        "primary_run": bool((rows[0].get("run") or {}).get("primary_run"))
        if rows
        else False,
    }
    summary = build_summary(
        args=args,
        status="ok",
        run_id=run_id,
        details_path=args.output_dir.resolve() / "compare_raw_inputs.jsonl",
        summary_path=summary_path,
        rows_seen=len(rows),
        details=rows,
        started_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=0.0,
        message=f"Compared {len(paths)} raw output file(s).",
        run_metadata=run_metadata,
    )
    summary["compared_raw_paths"] = [str(path.resolve()) for path in paths]
    write_json(summary_path, summary)
    return summary_path, summary


def is_short_detail(row: dict[str, Any], threshold: float) -> bool:
    if row.get("contains_short_utterance"):
        return True
    duration = row.get("speech_duration")
    return isinstance(duration, int | float) and duration <= threshold


def model_strategy_summaries(
    rows: list[dict[str, Any]],
    *,
    short_utterance_seconds: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    bootstrap_block_size: int,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    model_names = sorted({str(row.get("model_name", "unknown")) for row in rows})
    strategies = sorted({str(row.get("strategy", "unknown")) for row in rows})
    offset = 0
    for model_name in model_names:
        summaries[model_name] = {}
        for strategy in strategies:
            selected = [
                row
                for row in rows
                if row.get("model_name") == model_name
                and row.get("strategy") == strategy
            ]
            if not selected:
                continue
            summaries[model_name][strategy] = {
                "overall": add_uncertainty(
                    aggregate_metric_rows(selected),
                    selected,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + offset,
                    block_size=bootstrap_block_size,
                ),
                "short_utterance_subset": aggregate_metric_rows(
                    [
                        row
                        for row in selected
                        if is_short_detail(row, short_utterance_seconds)
                    ]
                ),
                "by_language": grouped_aggregates(
                    selected,
                    "language_label",
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed + 1000 + offset,
                    bootstrap_block_size=bootstrap_block_size,
                ),
                "by_mixed_bucket": grouped_aggregates(
                    selected,
                    "mixed_bucket",
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed + 2000 + offset,
                    bootstrap_block_size=bootstrap_block_size,
                ),
            }
            offset += 1
    return summaries


def build_summary(
    *,
    args: argparse.Namespace,
    status: str,
    run_id: str,
    details_path: Path,
    summary_path: Path,
    rows_seen: int,
    details: list[dict[str, Any]],
    started_at: str,
    elapsed_seconds: float,
    message: str | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scored = [row for row in details if row.get("status") == "ok"]
    failures = [row for row in details if row.get("status") != "ok"]
    script_counter: Counter[str] = Counter()
    for row in scored:
        script_counter.update(row.get("prediction_non_thai_english_scripts", {}))
    short_rows = [
        row for row in scored if is_short_detail(row, args.short_utterance_seconds)
    ]
    term_rows = [row for row in scored if row.get("english_terms")]
    term_examples = sorted(
        term_rows,
        key=lambda row: (
            row["metrics"]["cer"] if row["metrics"]["cer"] is not None else -1.0,
            str(row.get("index", "")),
        ),
        reverse=True,
    )[:10]
    strategies = sorted(
        {str(row.get("strategy", args.strategy)) for row in details}
    ) or list(getattr(args, "requested_strategies", [args.strategy]))
    bootstrap_samples = getattr(args, "bootstrap_samples", 1000)
    bootstrap_seed = getattr(args, "bootstrap_seed", 1337)
    bootstrap_block_size = getattr(args, "bootstrap_block_size", 8)
    overall_metrics = add_uncertainty(
        aggregate_metric_rows(scored),
        scored,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        block_size=bootstrap_block_size,
    )
    model_names = sorted(
        {
            str(
                row.get("model_name") or (row.get("model") or {}).get("name", "unknown")
            )
            for row in details
            if isinstance(row.get("model") or {}, dict)
        }
    )
    manifest_hash = None
    dataset_fingerprint = None
    benchmark_fingerprint = None
    sample_fingerprint = None
    decode_config_hash = None
    primary_run = False
    if run_metadata:
        manifest_hash = run_metadata.get("manifest_hash")
        dataset_fingerprint = run_metadata.get("dataset_hash")
        benchmark_fingerprint = run_metadata.get("benchmark_fingerprint")
        sample_fingerprint = run_metadata.get("sample_set_hash")
        decode_config_hash = run_metadata.get("decode_config_hash")
        primary_run = bool(run_metadata.get("primary_run"))

    summary: dict[str, Any] = {
        "status": status,
        "message": message,
        "run_id": run_id,
        "started_at": started_at,
        "elapsed_seconds": elapsed_seconds,
        "model": args.model,
        "models": model_names or getattr(args, "models_list", []),
        "device": args.device,
        "compute_type": args.compute_type,
        "language": args.language,
        "initial_prompt": args.initial_prompt,
        "include_context_technical_terms": getattr(
            args, "include_context_technical_terms", False
        ),
        "beam_size": args.beam_size,
        "condition_on_previous_text": not args.no_condition_on_previous_text,
        "strategy": args.strategy,
        "benchmark_strategies": strategies,
        "short_utterance_seconds": args.short_utterance_seconds,
        "rolling_prompt_turns": args.rolling_prompt_turns,
        "rolling_prompt_chars": args.rolling_prompt_chars,
        "left_audio_context_seconds": args.left_audio_context_seconds,
        "context_max_gap_seconds": args.context_max_gap_seconds,
        "merge_max_seconds": args.merge_max_seconds,
        "merge_max_chunks": args.merge_max_chunks,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "details_path": str(details_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "manifest_path": str(args.manifest.resolve())
        if getattr(args, "manifest", None)
        else None,
        "selection_mode": getattr(args, "selection_mode", None),
        "primary_run": primary_run,
        "primary_run_reason": (
            "fixed_allowlist_linked_to_held_out_split_manifest"
            if primary_run
            else "non_primary_without_fixed_allowlist_and_held_out_split_linkage"
        ),
        "split_manifest": run_metadata.get("split_manifest") if run_metadata else None,
        "dataset_hash": dataset_fingerprint,
        "manifest_hash": manifest_hash,
        "sample_set_hash": sample_fingerprint,
        "decode_config_hash": decode_config_hash,
        "benchmark_fingerprint": benchmark_fingerprint,
        "decode_config": run_metadata.get("decode_config") if run_metadata else None,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_block_size": bootstrap_block_size,
        "rows_seen": rows_seen,
        "scored_count": len(scored),
        "failure_count": len(failures),
        "mean_cer": overall_metrics["mean_cer"],
        "micro_cer": overall_metrics["micro_cer"],
        "mean_cer_no_space": overall_metrics["mean_cer_no_space"],
        "mean_wer_like": overall_metrics["mean_wer_like"],
        "micro_wer_like": overall_metrics["micro_wer_like"],
        "overall": overall_metrics,
        "short_utterance_subset": add_uncertainty(
            aggregate_metric_rows(short_rows),
            short_rows,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 10,
            block_size=bootstrap_block_size,
        ),
        "english_technical_term_subset": aggregate_metric_rows(term_rows),
        "by_language": grouped_aggregates(
            scored,
            "language_label",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 100,
            bootstrap_block_size=bootstrap_block_size,
        ),
        "by_mixed_bucket": grouped_aggregates(
            scored,
            "mixed_bucket",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 200,
            bootstrap_block_size=bootstrap_block_size,
            include_uncertainty=True,
        ),
        "by_source_file": grouped_aggregates(
            scored,
            "source_file",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 300,
            bootstrap_block_size=bootstrap_block_size,
        ),
        "by_source_group": grouped_aggregates(
            scored,
            "source_group",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 400,
            bootstrap_block_size=bootstrap_block_size,
        ),
        "by_model": grouped_aggregates(
            scored,
            "model_name",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 500,
            bootstrap_block_size=bootstrap_block_size,
            include_uncertainty=True,
        ),
        "by_model_strategy": model_strategy_summaries(
            scored,
            short_utterance_seconds=args.short_utterance_seconds,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 1500,
            bootstrap_block_size=bootstrap_block_size,
        ),
        "paired_model_deltas": paired_model_delta_summaries(
            scored,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 700,
            block_size=bootstrap_block_size,
        ),
        "english_technical_term_examples": [
            {
                "strategy": row.get("strategy"),
                "index": row.get("index"),
                "metadata_line_number": row.get("metadata_line_number"),
                "file_name": row.get("file_name"),
                "terms": row.get("english_terms"),
                "reference": row.get("reference"),
                "prediction": row.get("hypothesis", row.get("prediction")),
                "cer": row["metrics"].get("cer"),
                "wer_like": row["metrics"].get("wer_like"),
            }
            for row in term_examples
        ],
        "prediction_non_thai_english_scripts": dict(sorted(script_counter.items())),
    }
    if len(strategies) > 1:
        summary["strategy_summaries"] = {
            strategy: {
                "overall": add_uncertainty(
                    aggregate_metric_rows(
                        [row for row in scored if row.get("strategy") == strategy]
                    ),
                    [row for row in scored if row.get("strategy") == strategy],
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + 600 + offset,
                    block_size=bootstrap_block_size,
                ),
                "short_utterance_subset": aggregate_metric_rows(
                    [
                        row
                        for row in scored
                        if row.get("strategy") == strategy
                        and is_short_detail(row, args.short_utterance_seconds)
                    ]
                ),
                "english_technical_term_subset": aggregate_metric_rows(
                    [
                        row
                        for row in scored
                        if row.get("strategy") == strategy and row.get("english_terms")
                    ]
                ),
            }
            for offset, strategy in enumerate(strategies)
        }
    if failures:
        summary["failures_preview"] = failures[:10]
    return summary


def transcribe_one(
    model: Any,
    audio_path: Path,
    args: argparse.Namespace,
    *,
    initial_prompt: str | None = None,
    include_after_seconds: float | None = None,
) -> tuple[str, dict[str, Any]]:
    memory_before = memory_snapshot()
    started = time.perf_counter()
    segments, info = model.transcribe(
        str(audio_path),
        language=args.language,
        initial_prompt=args.initial_prompt
        if initial_prompt is None
        else initial_prompt,
        beam_size=args.beam_size,
        condition_on_previous_text=not args.no_condition_on_previous_text,
        word_timestamps=include_after_seconds is not None,
    )
    segment_rows = []
    text_parts = []
    for segment in segments:
        text = segment.text.strip()
        start = getattr(segment, "start", None)
        end = getattr(segment, "end", None)
        included = include_after_seconds is None or (
            isinstance(end, int | float) and end >= include_after_seconds
        )
        words = []
        word_text_parts = []
        for word in getattr(segment, "words", None) or []:
            word_start = getattr(word, "start", None)
            word_end = getattr(word, "end", None)
            word_included = include_after_seconds is None or (
                isinstance(word_end, int | float) and word_end >= include_after_seconds
            )
            word_text = str(getattr(word, "word", "")).strip()
            if word_text and word_included:
                word_text_parts.append(word_text)
            words.append(
                {
                    "word": word_text,
                    "start": word_start,
                    "end": word_end,
                    "probability": getattr(word, "probability", None),
                    "included_in_prediction": word_included,
                }
            )
        if word_text_parts:
            text_parts.append(" ".join(word_text_parts))
        elif text and included:
            text_parts.append(text)
        segment_rows.append(
            {
                "id": getattr(segment, "id", None),
                "start": start,
                "end": end,
                "text": segment.text,
                "included_in_prediction": included,
                "words": words,
            }
        )

    info_row = {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory": memory_peak(memory_before, memory_snapshot()),
        "detected_language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "duration_after_vad": getattr(info, "duration_after_vad", None),
        "segments": segment_rows,
    }
    return normalize_thai_spacing(" ".join(text_parts)), info_row


def source_gap_seconds(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    left_end = left.get("source_end")
    right_start = right.get("source_start")
    if not isinstance(left_end, int | float) or not isinstance(
        right_start, int | float
    ):
        return None
    return float(right_start) - float(left_end)


def same_source(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(left.get("source_file")) and left.get("source_file") == right.get(
        "source_file"
    )


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frame_rate = handle.getframerate()
        if frame_rate <= 0:
            return 0.0
        return handle.getnframes() / frame_rate


def concatenate_wavs(paths: list[Path], output_path: Path) -> list[float]:
    durations: list[float] = []
    params = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as output:
        for path in paths:
            with wave.open(str(path), "rb") as source:
                source_params = source.getparams()
                comparable_params = source_params[:3] + source_params[4:]
                if params is None:
                    params = comparable_params
                    output.setparams(source_params)
                elif comparable_params != params:
                    raise ValueError(f"WAV parameters differ for {path}")
                output.writeframes(source.readframes(source.getnframes()))
                durations.append(source.getnframes() / source.getframerate())
    return durations


def base_detail_for_record(
    *,
    index: int,
    record: dict[str, Any],
    audio_path: Path | None,
    reference: str,
    strategy: str,
    model_info: dict[str, Any],
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    context_text = " ".join(
        str(record.get(key, "")) for key in ("context_before", "context_after", "notes")
    )
    language_label, mixed_bucket = language_and_mixed_bucket(record, reference)
    source_group = source_group_for_record(record)
    duration = record.get("speech_duration")
    return {
        "schema_version": 2,
        "run": run_metadata,
        "model": model_info,
        "model_name": model_info.get("name"),
        "strategy": strategy,
        "sample_id": record.get("_sample_id"),
        "index": index,
        "metadata_line_number": record.get("_line_number"),
        "file_name": record.get("file_name"),
        "audio_path": str(audio_path) if audio_path else None,
        "reference": reference,
        "language_label": language_label,
        "mixed_bucket": mixed_bucket,
        "source_group": source_group,
        "source_file": record.get("source_file"),
        "source_start": record.get("source_start"),
        "source_end": record.get("source_end"),
        "speech_duration": duration,
        "duration_seconds": duration,
        "context_before": record.get("context_before"),
        "context_after": record.get("context_after"),
        "english_terms": english_terms_from_text(reference, context_text),
        "reference_non_thai_english_scripts": non_thai_english_scripts(reference),
        "prompt": None,
        "prompt_metadata": {
            "initial_prompt_present": bool(
                run_metadata.get("decode_config", {}).get("initial_prompt")
            ),
            "rolling_prompt_turns": run_metadata.get("decode_config", {}).get(
                "rolling_prompt_turns"
            ),
            "rolling_prompt_chars": run_metadata.get("decode_config", {}).get(
                "rolling_prompt_chars"
            ),
            "runtime_prompt_chars": 0,
            "runtime_prompt_hash": None,
        },
    }


def score_prediction(
    base_detail: dict[str, Any], prediction: str, transcribe_info: dict[str, Any]
) -> dict[str, Any]:
    latency = transcribe_info.get("elapsed_seconds")
    target_duration = base_detail.get("duration_seconds") or transcribe_info.get(
        "duration_after_vad"
    )
    processed_duration = (
        transcribe_info.get("duration")
        or transcribe_info.get("duration_after_vad")
        or target_duration
    )
    target_rtf = (
        latency / target_duration
        if isinstance(latency, int | float)
        and isinstance(target_duration, int | float)
        and target_duration > 0
        else None
    )
    processed_rtf = (
        latency / processed_duration
        if isinstance(latency, int | float)
        and isinstance(processed_duration, int | float)
        and processed_duration > 0
        else None
    )
    return {
        **base_detail,
        "status": "ok",
        "hypothesis": prediction,
        "prediction": prediction,
        "prediction_non_thai_english_scripts": non_thai_english_scripts(prediction),
        "metrics": text_metrics(str(base_detail.get("reference", "")), prediction),
        "latency_seconds": latency,
        "duration_seconds": target_duration,
        "target_audio_duration_seconds": target_duration,
        "processed_audio_duration_seconds": processed_duration,
        "rtf": target_rtf,
        "target_audio_rtf": target_rtf,
        "processed_audio_rtf": processed_rtf,
        "peak_memory": transcribe_info.get("peak_memory", {}),
        "transcribe": transcribe_info,
    }


def attach_prompt_metadata(base_detail: dict[str, Any], prompt: str | None) -> None:
    base_detail["prompt"] = prompt
    base_detail["prompt_metadata"] = {
        **base_detail.get("prompt_metadata", {}),
        "runtime_prompt_chars": len(prompt or ""),
        "runtime_prompt_hash": sha256_json(prompt) if prompt else None,
    }


def contextual_terms_prompt(
    args: argparse.Namespace, records: Iterable[dict[str, Any]]
) -> str | None:
    if not getattr(args, "include_context_technical_terms", False):
        return None
    terms = english_terms_from_text(
        *(str(record.get("context_before") or "") for record in records)
    )
    return f"Technical terms: {', '.join(terms)}" if terms else None


def build_common_prompt(
    args: argparse.Namespace, records: Iterable[dict[str, Any]]
) -> str | None:
    prompt_parts = []
    if args.initial_prompt:
        prompt_parts.append(args.initial_prompt.strip())
    terms_prompt = contextual_terms_prompt(args, records)
    if terms_prompt:
        prompt_parts.append(terms_prompt)
    return "\n".join(prompt_parts) if prompt_parts else None


def build_rolling_prompt(
    args: argparse.Namespace,
    previous_predictions: list[str],
    records: Iterable[dict[str, Any]] = (),
) -> str | None:
    prompt_parts = []
    common_prompt = build_common_prompt(args, records)
    if common_prompt:
        prompt_parts.append(common_prompt)
    recent_predictions = [
        text
        for text in previous_predictions[-args.rolling_prompt_turns :]
        if text.strip()
    ]
    if recent_predictions:
        rolling_text = " ".join(recent_predictions)[-args.rolling_prompt_chars :]
        prompt_parts.append(f"Previous transcript: {rolling_text}")
    return "\n".join(prompt_parts) if prompt_parts else None


def evaluate_single_chunk_strategy(
    *,
    model: Any,
    model_info: dict[str, Any],
    metadata: list[dict[str, Any]],
    args: argparse.Namespace,
    strategy: str,
    temp_dir: Path,
    run_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    previous_predictions: list[str] = []
    previous_group: str | None = None
    previous_record: dict[str, Any] | None = None
    for index, record in enumerate(metadata, start=1):
        current_group = portable_source_id(
            str(record.get("source_file") or source_group_for_record(record))
        )
        gap = (
            source_gap_seconds(previous_record, record)
            if previous_record is not None
            else None
        )
        if current_group != previous_group or (
            gap is not None and gap > args.context_max_gap_seconds
        ):
            previous_predictions = []
        previous_group = current_group
        previous_record = record
        audio_path = resolve_audio_path(args.dataset_dir, record)
        reference = str(record.get("text", ""))
        base_detail = base_detail_for_record(
            index=index,
            record=record,
            audio_path=audio_path,
            reference=reference,
            strategy=strategy,
            model_info=model_info,
            run_metadata=run_metadata,
        )

        if audio_path is None:
            details.append(
                {
                    **base_detail,
                    "status": "missing_audio_path",
                    "error": "No file_name/path/audio.path",
                }
            )
            continue
        if not audio_path.exists():
            details.append(
                {
                    **base_detail,
                    "status": "missing_audio_file",
                    "error": f"Not found: {audio_path}",
                }
            )
            continue

        try:
            prompt = build_common_prompt(args, [record])
            transcribe_path = audio_path
            include_after_seconds = None
            if strategy == "rolling_initial_prompt":
                prompt = build_rolling_prompt(args, previous_predictions, [record])
                base_detail["rolling_prompt"] = prompt
            elif strategy == "left_audio_context":
                context_records: list[tuple[int, dict[str, Any], Path]] = []
                context_seconds = 0.0
                cursor = index - 2
                while cursor >= 0:
                    candidate = metadata[cursor]
                    if not same_source(candidate, record):
                        break
                    gap = source_gap_seconds(candidate, metadata[cursor + 1])
                    if gap is not None and gap > args.context_max_gap_seconds:
                        break
                    candidate_path = resolve_audio_path(args.dataset_dir, candidate)
                    if candidate_path is None or not candidate_path.exists():
                        break
                    duration = wav_duration(candidate_path)
                    if (
                        context_seconds + duration > args.left_audio_context_seconds
                        and context_records
                    ):
                        break
                    context_records.append((cursor + 1, candidate, candidate_path))
                    context_seconds += duration
                    if context_seconds >= args.left_audio_context_seconds:
                        break
                    cursor -= 1

                if context_records:
                    context_records.reverse()
                    concat_path = temp_dir / f"left-context-{index:05d}.wav"
                    durations = concatenate_wavs(
                        [path for _row_index, _candidate, path in context_records]
                        + [audio_path],
                        concat_path,
                    )
                    include_after_seconds = sum(durations[:-1])
                    transcribe_path = concat_path
                    base_detail["left_audio_context"] = {
                        "metadata_line_numbers": [
                            candidate.get("_line_number")
                            for _row_index, candidate, _path in context_records
                        ],
                        "file_names": [
                            candidate.get("file_name")
                            for _row_index, candidate, _path in context_records
                        ],
                        "audio_seconds": include_after_seconds,
                    }

            attach_prompt_metadata(base_detail, prompt)
            prediction, transcribe_info = transcribe_one(
                model,
                transcribe_path,
                args,
                initial_prompt=prompt,
                include_after_seconds=include_after_seconds,
            )
            detail = score_prediction(base_detail, prediction, transcribe_info)
            previous_predictions.append(prediction)
        except Exception as exc:
            detail = {**base_detail, "status": "transcribe_failed", "error": repr(exc)}
        details.append(detail)

        if index == 1 or index % 10 == 0 or index == len(metadata):
            ok_count = sum(1 for row in details if row.get("status") == "ok")
            print(
                f"[{strategy}] Processed {index}/{len(metadata)} rows ({ok_count} scored)."
            )
    return details


def evaluate_merged_deferred_short(
    *,
    model: Any,
    model_info: dict[str, Any],
    metadata: list[dict[str, Any]],
    args: argparse.Namespace,
    temp_dir: Path,
    run_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    index = 0
    while index < len(metadata):
        record = metadata[index]
        group = [record]
        duration = float(record.get("speech_duration") or 0.0)
        if duration <= args.short_utterance_seconds:
            while len(group) < args.merge_max_chunks and index + len(group) < len(
                metadata
            ):
                candidate = metadata[index + len(group)]
                previous = group[-1]
                gap = source_gap_seconds(previous, candidate)
                if not same_source(previous, candidate):
                    break
                if gap is not None and gap > args.context_max_gap_seconds:
                    break
                candidate_duration = float(candidate.get("speech_duration") or 0.0)
                if (
                    duration + candidate_duration > args.merge_max_seconds
                    and len(group) > 1
                ):
                    break
                group.append(candidate)
                duration += candidate_duration
                if duration >= args.short_utterance_seconds:
                    break

        first_line = group[0].get("_line_number")
        last_line = group[-1].get("_line_number")
        audio_paths = [resolve_audio_path(args.dataset_dir, item) for item in group]
        reference = " ".join(str(item.get("text", "")) for item in group).strip()
        language_label, mixed_bucket = language_and_mixed_bucket(group[0], reference)
        source_group = source_group_for_record(group[0])
        base_detail = {
            "schema_version": 2,
            "run": run_metadata,
            "model": model_info,
            "model_name": model_info.get("name"),
            "strategy": "merged_deferred_short",
            "sample_id": group[0].get("_sample_id")
            if len(group) == 1
            else "+".join(str(item.get("_sample_id")) for item in group),
            "index": index + 1
            if len(group) == 1
            else f"{index + 1}-{index + len(group)}",
            "metadata_line_number": first_line,
            "metadata_line_numbers": [item.get("_line_number") for item in group],
            "file_name": group[0].get("file_name") if len(group) == 1 else None,
            "file_names": [item.get("file_name") for item in group],
            "audio_path": str(audio_paths[0])
            if len(group) == 1 and audio_paths[0]
            else None,
            "audio_paths": [str(path) if path else None for path in audio_paths],
            "reference": reference,
            "language_label": language_label,
            "mixed_bucket": mixed_bucket,
            "source_group": source_group,
            "source_file": group[0].get("source_file"),
            "source_start": group[0].get("source_start"),
            "source_end": group[-1].get("source_end"),
            "speech_duration": duration,
            "duration_seconds": duration,
            "contains_short_utterance": any(
                float(item.get("speech_duration") or 0.0)
                <= args.short_utterance_seconds
                for item in group
            ),
            "merged_chunk_count": len(group),
            "english_terms": english_terms_from_text(
                reference,
                " ".join(
                    str(item.get(key, ""))
                    for item in group
                    for key in ("context_before", "context_after", "notes")
                ),
            ),
            "reference_non_thai_english_scripts": non_thai_english_scripts(reference),
            "prompt": build_common_prompt(args, group),
            "prompt_metadata": {
                "initial_prompt_present": bool(args.initial_prompt),
                "rolling_prompt_turns": args.rolling_prompt_turns,
                "rolling_prompt_chars": args.rolling_prompt_chars,
                "runtime_prompt_chars": len(build_common_prompt(args, group) or ""),
                "runtime_prompt_hash": (
                    sha256_json(build_common_prompt(args, group))
                    if build_common_prompt(args, group)
                    else None
                ),
            },
        }

        has_missing_path = any(path is None for path in audio_paths)
        missing_file = next(
            (path for path in audio_paths if path is not None and not path.exists()),
            None,
        )
        if not has_missing_path and missing_file is None:
            try:
                transcribe_path = audio_paths[0]
                if len(group) > 1:
                    transcribe_path = temp_dir / f"merged-{first_line}-{last_line}.wav"
                    concatenate_wavs(
                        [path for path in audio_paths if path is not None],
                        transcribe_path,
                    )
                if transcribe_path is None:
                    raise ValueError("No audio path for merged group")
                prediction, transcribe_info = transcribe_one(
                    model,
                    transcribe_path,
                    args,
                    initial_prompt=build_common_prompt(args, group),
                )
                detail = score_prediction(base_detail, prediction, transcribe_info)
            except Exception as exc:
                detail = {
                    **base_detail,
                    "status": "transcribe_failed",
                    "error": repr(exc),
                }
        elif has_missing_path:
            detail = {
                **base_detail,
                "status": "missing_audio_path",
                "error": "No file_name/path/audio.path",
            }
        else:
            detail = {
                **base_detail,
                "status": "missing_audio_file",
                "error": f"Not found: {missing_file}",
            }
        details.append(detail)

        processed = index + len(group)
        if processed == 1 or processed % 10 == 0 or processed == len(metadata):
            ok_count = sum(1 for row in details if row.get("status") == "ok")
            print(
                f"[merged_deferred_short] Processed {processed}/{len(metadata)} rows ({ok_count} groups scored)."
            )
        index += len(group)
    return details


def output_completeness_error(
    details: list[dict[str, Any]],
    models: list[str],
    strategies: list[str],
    metadata: list[dict[str, Any]],
) -> str | None:
    expected = {
        (model, strategy, str(record.get("_sample_id")))
        for model in models
        for strategy in strategies
        for record in metadata
    }
    actual_keys = [
        (
            str(row.get("model_name")),
            str(row.get("strategy")),
            str(row.get("sample_id")),
        )
        for row in details
    ]
    counts = Counter(actual_keys)
    actual = set(actual_keys)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    non_ok = [
        key for key, row in zip(actual_keys, details) if row.get("status") != "ok"
    ]
    if (
        len(actual_keys) == len(expected)
        and not missing
        and not unexpected
        and not duplicates
        and not non_ok
    ):
        return None
    return (
        f"Incomplete benchmark output: expected exactly {len(expected)} model×strategy×sample rows, "
        f"got {len(actual_keys)}; missing={missing[:5]}, unexpected={unexpected[:5]}, "
        f"duplicate={duplicates[:5]}, non_ok={non_ok[:5]}."
    )


def main() -> int:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        strategies = parse_strategies(args)
        models = parse_models(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    args.requested_strategies = strategies
    args.models_list = models

    if args.compare_raw:
        try:
            paths: list[Path] = []
            for raw_path in args.compare_raw:
                for part in str(raw_path).split(","):
                    part = part.strip()
                    if part:
                        paths.append(Path(part))
            summary_path, summary = compare_raw_outputs(paths, args)
        except Exception as exc:
            print(f"ASR raw comparison failed: {exc}", file=sys.stderr)
            return 2
        print(f"Summary: {summary_path}")
        print(
            "Compared raw outputs: micro CER={micro_cer}; micro WER-like={micro_wer_like}".format(
                micro_cer=summary["micro_cer"],
                micro_wer_like=summary["micro_wer_like"],
            )
        )
        return 0

    strategy_suffix = (
        args.strategy if len(strategies) == 1 else "benchmark_" + "_".join(strategies)
    )
    model_suffix = model_set_suffix(models)
    run_id = f"{utc_stamp()}_{model_suffix}_{safe_name(strategy_suffix)}"
    details_path = args.output_dir / f"details_{run_id}.jsonl"
    summary_path = args.output_dir / f"summary_{run_id}.json"

    try:
        requested_sample_ids, selection_mode, manifest_source = (
            parse_requested_sample_ids(args)
        )
        args.selection_mode = selection_mode
        metadata_limit = None if requested_sample_ids is not None else args.limit
        metadata_all = read_metadata(args.dataset_dir, metadata_limit)
        metadata = select_metadata(
            metadata_all, requested_sample_ids, args.sample_id_field
        )
        split_identity = (
            validate_selected_split(metadata, args.split_manifest, args.required_split)
            if args.split_manifest is not None
            else None
        )
        if "rolling_initial_prompt" in strategies:
            validate_rolling_order(metadata)
        run_metadata = build_run_metadata(
            args=args,
            strategies=strategies,
            metadata=metadata,
            requested_sample_ids=requested_sample_ids,
            selection_mode=selection_mode,
            manifest_source=manifest_source,
            split_identity=split_identity,
        )
    except Exception as exc:
        empty_run_metadata = None
        summary = build_summary(
            args=args,
            status="failed",
            run_id=run_id,
            details_path=details_path,
            summary_path=summary_path,
            rows_seen=0,
            details=[],
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - started,
            message=str(exc),
            run_metadata=empty_run_metadata,
        )
        write_json(summary_path, summary)
        print(f"ASR eval failed before transcription: {exc}", file=sys.stderr)
        print(f"Summary: {summary_path}")
        return 1

    WhisperModel = load_faster_whisper()
    if WhisperModel is None:
        message = (
            "faster-whisper is not installed; install it to run transcription "
            "(for example: pip install faster-whisper)."
        )
        summary = build_summary(
            args=args,
            status="failed",
            run_id=run_id,
            details_path=details_path,
            summary_path=summary_path,
            rows_seen=len(metadata),
            details=[],
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - started,
            message=message,
            run_metadata=run_metadata,
        )
        write_json(summary_path, summary)
        print(message)
        print(f"Summary: {summary_path}")
        return 1

    details: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="daiya-asr-eval-") as temp_name:
        temp_dir = Path(temp_name)
        for model_name in models:
            model_info = model_identity(model_name)
            print(
                f"Loading faster-whisper model {model_name!r} on {args.device} ({args.compute_type})..."
            )
            try:
                model = WhisperModel(
                    model_name, device=args.device, compute_type=args.compute_type
                )
            except Exception as exc:
                for strategy in strategies:
                    for index, record in enumerate(metadata, start=1):
                        details.append(
                            {
                                **base_detail_for_record(
                                    index=index,
                                    record=record,
                                    audio_path=resolve_audio_path(
                                        args.dataset_dir, record
                                    ),
                                    reference=str(record.get("text", "")),
                                    strategy=strategy,
                                    model_info=model_info,
                                    run_metadata=run_metadata,
                                ),
                                "status": "model_load_failed",
                                "error": repr(exc),
                            }
                        )
                continue
            for strategy in strategies:
                if strategy == "merged_deferred_short":
                    strategy_details = evaluate_merged_deferred_short(
                        model=model,
                        model_info=model_info,
                        metadata=metadata,
                        args=args,
                        temp_dir=temp_dir,
                        run_metadata=run_metadata,
                    )
                else:
                    strategy_details = evaluate_single_chunk_strategy(
                        model=model,
                        model_info=model_info,
                        metadata=metadata,
                        args=args,
                        strategy=strategy,
                        temp_dir=temp_dir,
                        run_metadata=run_metadata,
                    )
                details.extend(strategy_details)
            del model
            gc.collect()

    write_jsonl(details_path, details)
    completeness_message = output_completeness_error(
        details, models, strategies, metadata
    )
    summary = build_summary(
        args=args,
        status="failed" if completeness_message else "ok",
        run_id=run_id,
        details_path=details_path,
        summary_path=summary_path,
        rows_seen=len(metadata),
        details=details,
        started_at=started_at,
        elapsed_seconds=time.perf_counter() - started,
        message=completeness_message,
        run_metadata=run_metadata,
    )
    write_json(summary_path, summary)
    print(f"Details: {details_path}")
    print(f"Summary: {summary_path}")
    print(
        "CER mean={mean_cer} micro={micro_cer}; WER-like mean={mean_wer_like} micro={micro_wer_like}".format(
            mean_cer=summary["mean_cer"],
            micro_cer=summary["micro_cer"],
            mean_wer_like=summary["mean_wer_like"],
            micro_wer_like=summary["micro_wer_like"],
        )
    )
    return 1 if completeness_message else 0


if __name__ == "__main__":
    raise SystemExit(main())
