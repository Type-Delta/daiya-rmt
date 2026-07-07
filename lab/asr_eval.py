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
    return parser.parse_args()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default))
            handle.write("\n")


def json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()  # type: ignore[no-any-return]
    return str(value)


def nan_safe_mean(values: list[float | None]) -> float | None:
    real_values = [value for value in values if value is not None and not math.isnan(value)]
    if not real_values:
        return None
    return sum(real_values) / len(real_values)


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_distance = sum(row["metrics"]["cer_distance"] for row in rows)
    aggregate_length = sum(row["metrics"]["cer_reference_length"] for row in rows)
    aggregate_no_space_distance = sum(row["metrics"]["cer_no_space_distance"] for row in rows)
    aggregate_no_space_length = sum(row["metrics"]["cer_no_space_reference_length"] for row in rows)
    aggregate_word_distance = sum(row["metrics"]["wer_like_distance"] for row in rows)
    aggregate_word_length = sum(row["metrics"]["wer_like_reference_length"] for row in rows)
    return {
        "count": len(rows),
        "mean_cer": nan_safe_mean([row["metrics"]["cer"] for row in rows]),
        "micro_cer": error_rate(aggregate_distance, aggregate_length),
        "mean_cer_no_space": nan_safe_mean([row["metrics"]["cer_no_space"] for row in rows]),
        "micro_cer_no_space": error_rate(aggregate_no_space_distance, aggregate_no_space_length),
        "mean_wer_like": nan_safe_mean([row["metrics"]["wer_like"] for row in rows]),
        "micro_wer_like": error_rate(aggregate_word_distance, aggregate_word_length),
    }


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
            raise ValueError(f"Unknown benchmark strategy {strategy!r}; expected one of: {valid}")
        if strategy not in strategies:
            strategies.append(strategy)
    if not strategies:
        raise ValueError("--benchmark-strategies did not contain any strategy names")
    return strategies


def is_short_detail(row: dict[str, Any], threshold: float) -> bool:
    if row.get("contains_short_utterance"):
        return True
    duration = row.get("speech_duration")
    return isinstance(duration, int | float) and duration <= threshold


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
    script_counter: Counter[str] = Counter()
    for row in scored:
        script_counter.update(row.get("prediction_non_thai_english_scripts", {}))
    short_rows = [row for row in scored if is_short_detail(row, args.short_utterance_seconds)]
    term_rows = [row for row in scored if row.get("english_terms")]
    term_examples = sorted(
        term_rows,
        key=lambda row: (
            row["metrics"]["cer"] if row["metrics"]["cer"] is not None else -1.0,
            str(row.get("index", "")),
        ),
        reverse=True,
    )[:10]
    strategies = sorted({str(row.get("strategy", args.strategy)) for row in details}) or list(
        getattr(args, "requested_strategies", [args.strategy])
    )
    overall_metrics = aggregate_metric_rows(scored)

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
        "rows_seen": rows_seen,
        "scored_count": len(scored),
        "failure_count": len(failures),
        "mean_cer": overall_metrics["mean_cer"],
        "micro_cer": overall_metrics["micro_cer"],
        "mean_cer_no_space": overall_metrics["mean_cer_no_space"],
        "mean_wer_like": overall_metrics["mean_wer_like"],
        "micro_wer_like": overall_metrics["micro_wer_like"],
        "overall": overall_metrics,
        "short_utterance_subset": aggregate_metric_rows(short_rows),
        "english_technical_term_subset": aggregate_metric_rows(term_rows),
        "english_technical_term_examples": [
            {
                "strategy": row.get("strategy"),
                "index": row.get("index"),
                "metadata_line_number": row.get("metadata_line_number"),
                "file_name": row.get("file_name"),
                "terms": row.get("english_terms"),
                "reference": row.get("reference"),
                "prediction": row.get("prediction"),
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
                "overall": aggregate_metric_rows(
                    [row for row in scored if row.get("strategy") == strategy]
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
                    [row for row in scored if row.get("strategy") == strategy and row.get("english_terms")]
                ),
            }
            for strategy in strategies
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
    started = time.perf_counter()
    segments, info = model.transcribe(
        str(audio_path),
        language=args.language,
        initial_prompt=args.initial_prompt if initial_prompt is None else initial_prompt,
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
        included = include_after_seconds is None or (isinstance(end, int | float) and end >= include_after_seconds)
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
    if not isinstance(left_end, int | float) or not isinstance(right_start, int | float):
        return None
    return float(right_start) - float(left_end)


def same_source(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(left.get("source_file")) and left.get("source_file") == right.get("source_file")


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
) -> dict[str, Any]:
    context_text = " ".join(
        str(record.get(key, "")) for key in ("context_before", "context_after", "notes")
    )
    return {
        "strategy": strategy,
        "index": index,
        "metadata_line_number": record.get("_line_number"),
        "file_name": record.get("file_name"),
        "audio_path": str(audio_path) if audio_path else None,
        "reference": reference,
        "language_label": record.get("language"),
        "source_file": record.get("source_file"),
        "source_start": record.get("source_start"),
        "source_end": record.get("source_end"),
        "speech_duration": record.get("speech_duration"),
        "context_before": record.get("context_before"),
        "context_after": record.get("context_after"),
        "english_terms": english_terms_from_text(reference, context_text),
        "reference_non_thai_english_scripts": non_thai_english_scripts(reference),
    }


def score_prediction(base_detail: dict[str, Any], prediction: str, transcribe_info: dict[str, Any]) -> dict[str, Any]:
    return {
        **base_detail,
        "status": "ok",
        "prediction": prediction,
        "prediction_non_thai_english_scripts": non_thai_english_scripts(prediction),
        "metrics": text_metrics(str(base_detail.get("reference", "")), prediction),
        "transcribe": transcribe_info,
    }


def build_rolling_prompt(args: argparse.Namespace, previous_predictions: list[str]) -> str | None:
    prompt_parts = []
    if args.initial_prompt:
        prompt_parts.append(args.initial_prompt.strip())
    recent_predictions = [text for text in previous_predictions[-args.rolling_prompt_turns :] if text.strip()]
    if recent_predictions:
        rolling_text = " ".join(recent_predictions)[-args.rolling_prompt_chars :]
        prompt_parts.append(f"Previous transcript: {rolling_text}")
    return "\n".join(prompt_parts) if prompt_parts else None


def evaluate_single_chunk_strategy(
    *,
    model: Any,
    metadata: list[dict[str, Any]],
    args: argparse.Namespace,
    strategy: str,
    temp_dir: Path,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    previous_predictions: list[str] = []
    for index, record in enumerate(metadata, start=1):
        audio_path = resolve_audio_path(args.dataset_dir, record)
        reference = str(record.get("text", ""))
        base_detail = base_detail_for_record(
            index=index,
            record=record,
            audio_path=audio_path,
            reference=reference,
            strategy=strategy,
        )

        if audio_path is None:
            details.append({**base_detail, "status": "missing_audio_path", "error": "No file_name/path/audio.path"})
            continue
        if not audio_path.exists():
            details.append({**base_detail, "status": "missing_audio_file", "error": f"Not found: {audio_path}"})
            continue

        try:
            prompt = args.initial_prompt
            transcribe_path = audio_path
            include_after_seconds = None
            if strategy == "rolling_initial_prompt":
                prompt = build_rolling_prompt(args, previous_predictions)
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
                    if context_seconds + duration > args.left_audio_context_seconds and context_records:
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
                        [path for _row_index, _candidate, path in context_records] + [audio_path],
                        concat_path,
                    )
                    include_after_seconds = sum(durations[:-1])
                    transcribe_path = concat_path
                    base_detail["left_audio_context"] = {
                        "metadata_line_numbers": [
                            candidate.get("_line_number") for _row_index, candidate, _path in context_records
                        ],
                        "file_names": [candidate.get("file_name") for _row_index, candidate, _path in context_records],
                        "audio_seconds": include_after_seconds,
                    }

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
            print(f"[{strategy}] Processed {index}/{len(metadata)} rows ({ok_count} scored).")
    return details


def evaluate_merged_deferred_short(
    *,
    model: Any,
    metadata: list[dict[str, Any]],
    args: argparse.Namespace,
    temp_dir: Path,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    index = 0
    while index < len(metadata):
        record = metadata[index]
        group = [record]
        duration = float(record.get("speech_duration") or 0.0)
        if duration <= args.short_utterance_seconds:
            while len(group) < args.merge_max_chunks and index + len(group) < len(metadata):
                candidate = metadata[index + len(group)]
                previous = group[-1]
                gap = source_gap_seconds(previous, candidate)
                if not same_source(previous, candidate):
                    break
                if gap is not None and gap > args.context_max_gap_seconds:
                    break
                candidate_duration = float(candidate.get("speech_duration") or 0.0)
                if duration + candidate_duration > args.merge_max_seconds and len(group) > 1:
                    break
                group.append(candidate)
                duration += candidate_duration
                if duration >= args.short_utterance_seconds:
                    break

        first_line = group[0].get("_line_number")
        last_line = group[-1].get("_line_number")
        audio_paths = [resolve_audio_path(args.dataset_dir, item) for item in group]
        reference = " ".join(str(item.get("text", "")) for item in group).strip()
        base_detail = {
            "strategy": "merged_deferred_short",
            "index": index + 1 if len(group) == 1 else f"{index + 1}-{index + len(group)}",
            "metadata_line_number": first_line,
            "metadata_line_numbers": [item.get("_line_number") for item in group],
            "file_name": group[0].get("file_name") if len(group) == 1 else None,
            "file_names": [item.get("file_name") for item in group],
            "audio_path": str(audio_paths[0]) if len(group) == 1 and audio_paths[0] else None,
            "audio_paths": [str(path) if path else None for path in audio_paths],
            "reference": reference,
            "language_label": group[0].get("language"),
            "source_file": group[0].get("source_file"),
            "source_start": group[0].get("source_start"),
            "source_end": group[-1].get("source_end"),
            "speech_duration": duration,
            "contains_short_utterance": any(
                float(item.get("speech_duration") or 0.0) <= args.short_utterance_seconds for item in group
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
        }

        has_missing_path = any(path is None for path in audio_paths)
        missing_file = next((path for path in audio_paths if path is not None and not path.exists()), None)
        if not has_missing_path and missing_file is None:
            try:
                transcribe_path = audio_paths[0]
                if len(group) > 1:
                    transcribe_path = temp_dir / f"merged-{first_line}-{last_line}.wav"
                    concatenate_wavs([path for path in audio_paths if path is not None], transcribe_path)
                if transcribe_path is None:
                    raise ValueError("No audio path for merged group")
                prediction, transcribe_info = transcribe_one(model, transcribe_path, args)
                detail = score_prediction(base_detail, prediction, transcribe_info)
            except Exception as exc:
                detail = {**base_detail, "status": "transcribe_failed", "error": repr(exc)}
        elif has_missing_path:
            detail = {**base_detail, "status": "missing_audio_path", "error": "No file_name/path/audio.path"}
        else:
            detail = {**base_detail, "status": "missing_audio_file", "error": f"Not found: {missing_file}"}
        details.append(detail)

        processed = index + len(group)
        if processed == 1 or processed % 10 == 0 or processed == len(metadata):
            ok_count = sum(1 for row in details if row.get("status") == "ok")
            print(f"[merged_deferred_short] Processed {processed}/{len(metadata)} rows ({ok_count} groups scored).")
        index += len(group)
    return details


def main() -> int:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        strategies = parse_strategies(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    args.requested_strategies = strategies

    strategy_suffix = args.strategy if len(strategies) == 1 else "benchmark_" + "_".join(strategies)
    run_id = f"{utc_stamp()}_{safe_name(args.model)}_{safe_name(strategy_suffix)}"
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
    with tempfile.TemporaryDirectory(prefix="daiya-asr-eval-") as temp_name:
        temp_dir = Path(temp_name)
        for strategy in strategies:
            if strategy == "merged_deferred_short":
                strategy_details = evaluate_merged_deferred_short(
                    model=model,
                    metadata=metadata,
                    args=args,
                    temp_dir=temp_dir,
                )
            else:
                strategy_details = evaluate_single_chunk_strategy(
                    model=model,
                    metadata=metadata,
                    args=args,
                    strategy=strategy,
                    temp_dir=temp_dir,
                )
            details.extend(strategy_details)

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
