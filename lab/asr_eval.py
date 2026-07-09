#!/usr/bin/env python
"""Lab-only faster-whisper ASR probe for the Daiya clean labeled dataset.

Example:
    python lab/asr_eval.py --model large-v3 --limit 20 --device cuda --compute-type float16
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DAIYA_SRC = REPO_ROOT / "daiya" / "src"
if str(DAIYA_SRC) not in sys.path:
    sys.path.insert(0, str(DAIYA_SRC))

from daiya.asr import DECODING_POLICIES, decoder_options_for_duration

DEFAULT_DATASET_DIR = REPO_ROOT / "training" / "dataset" / "hf_datasets" / "whisper"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "lab" / "artifacts" / "asr_eval"
DURATION_BUCKETS = (
    ("lt_2s", 0.0, 2.0),
    ("2_to_lt_3s", 2.0, 3.0),
    ("3_to_lt_5s", 3.0, 5.0),
    ("5_to_lt_10s", 5.0, 10.0),
    ("gte_10s", 10.0, math.inf),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate faster-whisper models against Daiya's clean labeled ASR chunks."
    )
    parser.add_argument("--model", required=True, help="faster-whisper model name or local model path.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum metadata rows to evaluate.")
    parser.add_argument("--device", default="auto", help="faster-whisper device, e.g. auto, cpu, cuda.")
    parser.add_argument(
        "--compute-type",
        default="default",
        help="faster-whisper compute type, e.g. default, int8, float16, int8_float16.",
    )
    parser.add_argument("--language", default=None, help="Optional language hint passed to transcribe().")
    parser.add_argument("--initial-prompt", default=None, help="Optional prompt passed to transcribe().")
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
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size passed to transcribe().")
    parser.add_argument(
        "--decoding-policy",
        choices=DECODING_POLICIES,
        default="baseline",
        help="Duration-aware decoder experiment; baseline matches runtime decoding.",
    )
    parser.add_argument(
        "--short-utterance-seconds",
        type=float,
        default=3.0,
        help="Inclusive duration threshold for policy selection and short metrics.",
    )
    parser.add_argument(
        "--no-condition-on-previous-text",
        action="store_true",
        help="Disable conditioning on previous text for independent chunk scoring.",
    )
    return parser.parse_args()


def applied_decoder_options(
    args: argparse.Namespace, duration_seconds: float | None
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "beam_size": args.beam_size,
        "condition_on_previous_text": not args.no_condition_on_previous_text,
    }
    options.update(
        decoder_options_for_duration(
            args.decoding_policy,
            math.inf if duration_seconds is None else float(duration_seconds),
            short_utterance_seconds=args.short_utterance_seconds,
        )
    )
    return options


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "model"


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
                raise ValueError(f"Invalid JSON on {metadata_path}:{line_number}: {exc}") from exc
            record["_line_number"] = line_number
            rows.append(record)
    return rows


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
            while index < len(normalized) and is_thai(normalized[index]) and is_combining_or_tone(normalized[index]):
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
        if char.isspace() or is_punctuation_or_symbol(char) or is_combining_or_tone(char):
            continue
        script = script_name(char)
        if script in {None, "Thai", "Latin", "Digit"}:
            continue
        counts[script] += 1
    return dict(sorted(counts.items()))


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def nan_safe_mean(values: list[float | None]) -> float | None:
    real_values = [value for value in values if value is not None and not math.isnan(value)]
    if not real_values:
        return None
    return sum(real_values) / len(real_values)


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audio_seconds = sum(float(row.get("speech_duration") or 0.0) for row in rows)
    inference_seconds = sum(
        float(row.get("transcribe", {}).get("elapsed_seconds") or 0.0) for row in rows
    )
    cer_distance = sum(row["metrics"]["cer_distance"] for row in rows)
    cer_length = sum(row["metrics"]["cer_reference_length"] for row in rows)
    word_distance = sum(row["metrics"]["wer_like_distance"] for row in rows)
    word_length = sum(row["metrics"]["wer_like_reference_length"] for row in rows)
    return {
        "count": len(rows),
        "audio_seconds": audio_seconds,
        "inference_seconds": inference_seconds,
        "real_time_factor": inference_seconds / audio_seconds if audio_seconds else None,
        "mean_cer": nan_safe_mean([row["metrics"]["cer"] for row in rows]),
        "micro_cer": error_rate(cer_distance, cer_length),
        "mean_cer_no_space": nan_safe_mean(
            [row["metrics"]["cer_no_space"] for row in rows]
        ),
        "mean_wer_like": nan_safe_mean([row["metrics"]["wer_like"] for row in rows]),
        "micro_wer_like": error_rate(word_distance, word_length),
    }


def duration_bucket_name(duration: float | None) -> str | None:
    if duration is None:
        return None
    value = float(duration)
    return next(
        (name for name, lower, upper in DURATION_BUCKETS if lower <= value < upper),
        None,
    )


def duration_bucket_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: aggregate_metric_rows(
            [
                row
                for row in rows
                if duration_bucket_name(row.get("speech_duration")) == name
            ]
        )
        for name, _lower, _upper in DURATION_BUCKETS
    }


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
) -> dict[str, Any]:
    scored = [row for row in details if row.get("status") == "ok"]
    failures = [row for row in details if row.get("status") != "ok"]
    overall = aggregate_metric_rows(scored)
    short_rows = [
        row
        for row in scored
        if float(row.get("speech_duration") or math.inf) <= args.short_utterance_seconds
    ]
    aggregate_distance = sum(row["metrics"]["cer_distance"] for row in scored)
    aggregate_length = sum(row["metrics"]["cer_reference_length"] for row in scored)
    aggregate_word_distance = sum(row["metrics"]["wer_like_distance"] for row in scored)
    aggregate_word_length = sum(row["metrics"]["wer_like_reference_length"] for row in scored)
    script_counter: Counter[str] = Counter()
    for row in scored:
        script_counter.update(row.get("prediction_non_thai_english_scripts", {}))

    summary: dict[str, Any] = {
        "status": status,
        "message": message,
        "run_id": run_id,
        "started_at": started_at,
        "elapsed_seconds": elapsed_seconds,
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "language": args.language,
        "initial_prompt": args.initial_prompt,
        "beam_size": args.beam_size,
        "decoding_policy": args.decoding_policy,
        "decoder_config": {
            "baseline": applied_decoder_options(args, None),
            "short": applied_decoder_options(args, args.short_utterance_seconds),
        },
        "condition_on_previous_text": not args.no_condition_on_previous_text,
        "short_utterance_seconds": args.short_utterance_seconds,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "limit": args.limit,
        "details_path": str(details_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "rows_seen": rows_seen,
        "scored_count": len(scored),
        "failure_count": len(failures),
        "mean_cer": nan_safe_mean([row["metrics"]["cer"] for row in scored]),
        "micro_cer": error_rate(aggregate_distance, aggregate_length),
        "mean_cer_no_space": nan_safe_mean([row["metrics"]["cer_no_space"] for row in scored]),
        "mean_wer_like": nan_safe_mean([row["metrics"]["wer_like"] for row in scored]),
        "micro_wer_like": error_rate(aggregate_word_distance, aggregate_word_length),
        "overall": overall,
        "duration_buckets": duration_bucket_summaries(scored),
        "short_utterance_subset": aggregate_metric_rows(short_rows),
        "prediction_non_thai_english_scripts": dict(sorted(script_counter.items())),
    }
    if failures:
        summary["failures_preview"] = failures[:10]
    return summary


def transcribe_one(
    model: Any,
    audio_path: Path,
    args: argparse.Namespace,
    *,
    policy_duration_seconds: float | None = None,
) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    decoder_options = applied_decoder_options(args, policy_duration_seconds)
    segments, info = model.transcribe(
        str(audio_path),
        language=args.language,
        initial_prompt=args.initial_prompt,
        **decoder_options,
    )
    segment_rows = []
    text_parts = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            text_parts.append(text)
        segment_rows.append(
            {
                "id": getattr(segment, "id", None),
                "start": getattr(segment, "start", None),
                "end": getattr(segment, "end", None),
                "text": segment.text,
            }
        )

    info_row = {
        "elapsed_seconds": time.perf_counter() - started,
        "detected_language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "duration_after_vad": getattr(info, "duration_after_vad", None),
        "decoder_options": decoder_options,
        "segments": segment_rows,
    }
    return " ".join(text_parts), info_row


def main() -> int:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    run_id = f"{utc_stamp()}_{safe_name(args.model)}_{safe_name(args.decoding_policy)}"
    details_path = args.output_dir / f"details_{run_id}.jsonl"
    summary_path = args.output_dir / f"summary_{run_id}.json"

    try:
        metadata = read_metadata(args.dataset_dir, args.limit)
    except Exception as exc:
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
            status="skipped",
            run_id=run_id,
            details_path=details_path,
            summary_path=summary_path,
            rows_seen=len(metadata),
            details=[],
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - started,
            message=message,
        )
        write_json(summary_path, summary)
        print(message)
        print(f"Summary: {summary_path}")
        return 0

    print(f"Loading faster-whisper model {args.model!r} on {args.device} ({args.compute_type})...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    details: list[dict[str, Any]] = []
    for index, record in enumerate(metadata, start=1):
        audio_path = resolve_audio_path(args.dataset_dir, record)
        reference = str(record.get("text", ""))
        base_detail: dict[str, Any] = {
            "index": index,
            "metadata_line_number": record.get("_line_number"),
            "file_name": record.get("file_name"),
            "audio_path": str(audio_path) if audio_path else None,
            "reference": reference,
            "language_label": record.get("language"),
            "source_start": record.get("source_start"),
            "source_end": record.get("source_end"),
            "speech_duration": record.get("speech_duration"),
            "reference_non_thai_english_scripts": non_thai_english_scripts(reference),
        }

        if audio_path is None:
            details.append({**base_detail, "status": "missing_audio_path", "error": "No file_name/path/audio.path"})
            continue
        if not audio_path.exists():
            details.append({**base_detail, "status": "missing_audio_file", "error": f"Not found: {audio_path}"})
            continue

        try:
            prediction, transcribe_info = transcribe_one(
                model,
                audio_path,
                args,
                policy_duration_seconds=record.get("speech_duration"),
            )
            detail = {
                **base_detail,
                "status": "ok",
                "prediction": prediction,
                "prediction_non_thai_english_scripts": non_thai_english_scripts(prediction),
                "metrics": text_metrics(reference, prediction),
                "transcribe": transcribe_info,
            }
        except Exception as exc:
            detail = {**base_detail, "status": "transcribe_failed", "error": repr(exc)}
        details.append(detail)

        if index == 1 or index % 10 == 0 or index == len(metadata):
            ok_count = sum(1 for row in details if row.get("status") == "ok")
            print(f"Processed {index}/{len(metadata)} rows ({ok_count} scored).")

    write_jsonl(details_path, details)
    summary = build_summary(
        args=args,
        status="ok",
        run_id=run_id,
        details_path=details_path,
        summary_path=summary_path,
        rows_seen=len(metadata),
        details=details,
        started_at=started_at,
        elapsed_seconds=time.perf_counter() - started,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
