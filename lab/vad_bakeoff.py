#!/usr/bin/env python
"""Compare utterance segmentation settings without ASR/diarization downloads.

Example:
    python lab/vad_bakeoff.py audio.wav --backend energy --threshold 0.008,0.012
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.metadata
import itertools
import json
import math
import sys
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ImportError as exc:
    np = None  # type: ignore[assignment]
    NUMPY_IMPORT_ERROR: ImportError | None = exc
else:
    NUMPY_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DAIYA_SRC = REPO_ROOT / "daiya" / "src"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "lab" / "artifacts" / "vad_bakeoff"
FIELDNAMES = [
    "status",
    "backend",
    "segmenter_backend",
    "threshold",
    "min_speech_seconds",
    "min_silence_seconds",
    "speech_padding_seconds",
    "max_utterance_seconds",
    "audio_files",
    "audio_paths",
    "audio_seconds",
    "utterance_count",
    "duration_lt_1_count",
    "duration_1_2_count",
    "duration_2_5_count",
    "duration_5_8_count",
    "duration_8_plus_count",
    "total_speech_seconds",
    "unique_predicted_seconds",
    "overlap_duplicated_seconds",
    "mean_duration_seconds",
    "p50_duration_seconds",
    "p95_duration_seconds",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "scored_seconds",
    "reference_speech_seconds",
    "reference_speech_covered_seconds",
    "reference_speech_missed_seconds",
    "predicted_non_reference_seconds",
    "asr_cer",
    "asr_wer_like",
    "asr_status",
    "segmentation_seconds",
    "segmentation_rtf",
    "asr_seconds",
    "asr_rtf",
    "model_path",
    "python_version",
    "numpy_version",
    "silero_vad_version",
    "torch_version",
    "faster_whisper_version",
    "elapsed_seconds",
    "notes",
]


@dataclass(frozen=True)
class Setting:
    backend: str
    threshold: float
    min_speech_seconds: float
    min_silence_seconds: float
    speech_padding_seconds: float
    max_utterance_seconds: float


@dataclass(frozen=True)
class AudioInput:
    path: Path
    samples: Any
    sample_rate: int
    samples_16k: Any
    seconds: float


@dataclass(frozen=True)
class Segment:
    audio_path: Path
    start: float
    end: float
    samples: Any

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class BakeoffSkip(RuntimeError):
    pass


def main() -> int:
    args = parse_args()
    if NUMPY_IMPORT_ERROR is not None:
        print(
            "ERROR: numpy is required for segmentation bake-offs. "
            "Run through the project environment, for example: uv run python lab/vad_bakeoff.py ...",
            file=sys.stderr,
        )
        return 2

    ensure_daiya_on_path()
    try:
        audio_inputs = load_audio_inputs(args)
    except BakeoffSkip as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    boundary_refs = load_boundary_references(args)
    text_refs = load_text_references(args.reference_text_jsonl)
    settings = list(iter_settings(args))
    asr_runner = build_asr_runner(args, text_refs)

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for setting in settings:
        row, setting_details = run_setting(setting, audio_inputs, args, boundary_refs, asr_runner, text_refs)
        rows.append(row)
        details.extend(setting_details)
        print(format_compact_row(row))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = args.csv_output or output_dir / f"vad_bakeoff_{stamp}.csv"
    jsonl_path = args.jsonl_output or output_dir / f"vad_bakeoff_{stamp}.jsonl"
    details_path = args.details_jsonl or output_dir / f"vad_bakeoff_details_{stamp}.jsonl"
    write_csv(csv_path, rows)
    write_jsonl(jsonl_path, rows)
    if args.write_details:
        write_jsonl(details_path, details)

    print(f"CSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")
    if args.write_details:
        print(f"Details: {details_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Daiya utterance segmentation settings and emit comparable summary rows. "
            "ASR and diarization scoring are optional and skipped unless explicitly configured."
        )
    )
    parser.add_argument("audio", nargs="*", type=Path, help="Audio files to segment.")
    parser.add_argument(
        "--audio-glob",
        action="append",
        default=[],
        help="Glob for audio files. Can be repeated, e.g. --audio-glob 'data/*.wav'.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum audio files after glob expansion.")
    parser.add_argument("--chunk-seconds", type=positive_float, default=0.25)
    parser.add_argument("--backend", default="energy", help="Comma-separated: energy, auto, silero.")
    parser.add_argument("--prefer-silero", action="store_true", help="Prefer Silero when backend=auto.")
    parser.add_argument(
        "--threshold",
        default=None,
        help="Comma-separated VAD thresholds. Defaults to 0.012 for energy and 0.5 for Silero.",
    )
    parser.add_argument("--min-speech-seconds", default="0.20,0.35", help="Comma-separated values.")
    parser.add_argument(
        "--min-silence-seconds",
        "--trailing-silence-seconds",
        dest="min_silence_seconds",
        default="0.30,0.50,0.80",
        help="Comma-separated silence/trailing-silence values.",
    )
    parser.add_argument("--speech-padding-seconds", default="0.00,0.10,0.20", help="Comma-separated values.")
    parser.add_argument("--max-utterance-seconds", default="8.0,12.0", help="Comma-separated values.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--jsonl-output", type=Path, default=None)
    parser.add_argument("--details-jsonl", type=Path, default=None)
    parser.add_argument("--write-details", action="store_true", help="Write per-utterance detail JSONL.")
    parser.add_argument(
        "--reference-boundaries-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL with audio_path/file_name/source_file, start, end for boundary scoring.",
    )
    parser.add_argument(
        "--reference-rttm",
        type=Path,
        default=None,
        help="Optional RTTM file. SPEAKER rows are used as reference speech boundaries.",
    )
    parser.add_argument("--boundary-collar-seconds", type=non_negative_float, default=0.25)
    parser.add_argument(
        "--asr-model",
        default=None,
        help="Optional faster-whisper model path/name for CER hook. Local paths are used without downloads.",
    )
    parser.add_argument("--asr-device", default="auto")
    parser.add_argument("--asr-compute-type", default="int8_float16")
    parser.add_argument("--asr-language", default=None)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow faster-whisper to resolve non-local model names. Disabled by default for lightweight runs.",
    )
    parser.add_argument(
        "--reference-text-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL with audio_path/file_name/source_file and full reference text for ASR CER.",
    )
    return parser.parse_args()


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def ensure_daiya_on_path() -> None:
    src = str(DAIYA_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def load_audio_inputs(args: argparse.Namespace) -> list[AudioInput]:
    paths = collect_audio_paths(args)
    if not paths:
        raise BakeoffSkip("no audio files provided")

    from daiya.audio import SAMPLE_RATE, resample_linear

    audio_inputs: list[AudioInput] = []
    for path in paths:
        try:
            samples, sample_rate = read_audio(path)
        except BakeoffSkip as exc:
            print(f"WARNING: skipping unreadable audio {path}: {exc}", file=sys.stderr)
            continue
        samples_16k = resample_linear(samples, sample_rate, SAMPLE_RATE)
        seconds = float(samples_16k.size) / SAMPLE_RATE if samples_16k.size else 0.0
        audio_inputs.append(
            AudioInput(
                path=path,
                samples=samples,
                sample_rate=sample_rate,
                samples_16k=samples_16k,
                seconds=seconds,
            )
        )
    if not audio_inputs:
        raise BakeoffSkip("no readable audio files")
    return audio_inputs


def collect_audio_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for audio in args.audio:
        paths.append(audio)
    for pattern in args.audio_glob:
        paths.extend(Path(match) for match in glob.glob(pattern, recursive=True))

    unique: dict[str, Path] = {}
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.is_file():
            unique[str(resolved)] = resolved
    ordered = list(unique.values())
    if args.limit is not None:
        ordered = ordered[: args.limit]
    return ordered


def read_audio(path: Path) -> tuple[Any, int]:
    if path.suffix.lower() == ".wav":
        try:
            return read_wav_stdlib(path)
        except (wave.Error, BakeoffSkip):
            pass

    try:
        from daiya.audio import read_audio_file

        return read_audio_file(path)
    except Exception as exc:
        raise BakeoffSkip(f"cannot read {path}: {type(exc).__name__}: {exc}") from exc


def read_wav_stdlib(path: Path) -> tuple[Any, int]:
    if np is None:
        raise BakeoffSkip("numpy is not installed")
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise BakeoffSkip(f"unsupported WAV sample width: {sample_width} bytes")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.nan_to_num(audio.astype(np.float32, copy=False)), sample_rate


def iter_settings(args: argparse.Namespace) -> Iterable[Setting]:
    backends = parse_str_list(args.backend, allowed={"energy", "auto", "silero"})
    min_speech_values = parse_float_list(args.min_speech_seconds)
    min_silence_values = parse_float_list(args.min_silence_seconds)
    padding_values = parse_float_list(args.speech_padding_seconds)
    max_values = parse_float_list(args.max_utterance_seconds)
    for backend in backends:
        thresholds = (
            parse_float_list(args.threshold)
            if args.threshold is not None
            else [0.012 if backend == "energy" else 0.5]
        )
        for threshold, min_speech, min_silence, padding, max_utterance in itertools.product(
            thresholds,
            min_speech_values,
            min_silence_values,
            padding_values,
            max_values,
        ):
            yield Setting(
                backend=backend,
                threshold=threshold,
                min_speech_seconds=min_speech,
                min_silence_seconds=min_silence,
                speech_padding_seconds=padding,
                max_utterance_seconds=max_utterance,
            )


def parse_str_list(raw: str, *, allowed: set[str]) -> list[str]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise SystemExit(f"invalid value(s): {', '.join(invalid)}; expected one of {', '.join(sorted(allowed))}")
    return values


def parse_float_list(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise SystemExit("empty numeric sweep list")
    return values


def run_setting(
    setting: Setting,
    audio_inputs: list[AudioInput],
    args: argparse.Namespace,
    boundary_refs: dict[str, list[tuple[float, float]]],
    asr_runner: Any | None,
    text_refs: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_started = time.perf_counter()
    segmentation_started = total_started
    notes: list[str] = []
    segments: list[Segment] = []
    details: list[dict[str, Any]] = []
    segmenter_backend = ""
    status = "ok"

    try:
        segmenter, used_kwargs = create_segmenter(setting, args)
        segmenter_backend = backend_name(segmenter)
        silero_required = setting.backend == "silero" or (
            setting.backend == "auto" and args.prefer_silero
        )
        if silero_required and segmenter_backend != "silero":
            raise BakeoffSkip(
                f"requested Silero but factory returned {segmenter_backend}; "
                "install/configure the optional Silero dependency"
            )
        notes.extend(dropped_setting_notes(setting, used_kwargs, segmenter_backend))
        for audio in audio_inputs:
            audio_segments = segment_audio(segmenter, audio, args.chunk_seconds)
            segments.extend(audio_segments)
            for index, segment in enumerate(audio_segments, start=1):
                details.append(
                    {
                        **setting_to_dict(setting),
                        "segmenter_backend": segmenter_backend,
                        "audio_path": str(audio.path),
                        "utterance_index": index,
                        "start": segment.start,
                        "end": segment.end,
                        "duration": segment.duration,
                    }
                )
    except BakeoffSkip as exc:
        status = "skipped"
        notes.append(str(exc))
        segments.clear()
        details.clear()
    except Exception as exc:
        status = "failed"
        notes.append(f"{type(exc).__name__}: {exc}")

    segmentation_seconds = time.perf_counter() - segmentation_started
    durations = [segment.duration for segment in segments]
    audio_paths = [audio.path for audio in audio_inputs]
    boundary_metrics = score_boundaries(
        segments, boundary_refs, args.boundary_collar_seconds, audio_paths=audio_paths
    )
    coverage_metrics = score_coverage(segments, boundary_refs, audio_paths=audio_paths)
    asr_started = time.perf_counter()
    asr_cer, asr_wer_like, asr_status = score_asr(
        segments, asr_runner, text_refs, audio_paths=audio_paths
    )
    asr_seconds = time.perf_counter() - asr_started if isinstance(asr_runner, ASRRunner) else 0.0
    if asr_status:
        notes.append(asr_status)

    audio_seconds = sum(audio.seconds for audio in audio_inputs)
    predicted_audio_seconds = sum(durations)
    buckets = duration_bucket_counts(durations)
    identities = reproducibility_identities(args)

    row = {
        "status": status,
        "backend": setting.backend,
        "segmenter_backend": segmenter_backend,
        "threshold": setting.threshold,
        "min_speech_seconds": setting.min_speech_seconds,
        "min_silence_seconds": setting.min_silence_seconds,
        "speech_padding_seconds": setting.speech_padding_seconds,
        "max_utterance_seconds": setting.max_utterance_seconds,
        "audio_files": len(audio_inputs),
        "audio_paths": json.dumps([str(audio.path.resolve()) for audio in audio_inputs], ensure_ascii=False),
        "audio_seconds": round_float(audio_seconds),
        "utterance_count": len(segments),
        **buckets,
        "total_speech_seconds": round_float(sum(durations)),
        "unique_predicted_seconds": round_float(coverage_metrics["unique_predicted_seconds"]),
        "overlap_duplicated_seconds": round_float(coverage_metrics["overlap_duplicated_seconds"]),
        "mean_duration_seconds": round_float(mean(durations)),
        "p50_duration_seconds": round_float(percentile(durations, 50)),
        "p95_duration_seconds": round_float(percentile(durations, 95)),
        "boundary_precision": round_float(boundary_metrics.get("precision")),
        "boundary_recall": round_float(boundary_metrics.get("recall")),
        "boundary_f1": round_float(boundary_metrics.get("f1")),
        "scored_seconds": round_float(coverage_metrics["scored_seconds"]),
        "reference_speech_seconds": round_float(coverage_metrics["reference_speech_seconds"]),
        "reference_speech_covered_seconds": round_float(
            coverage_metrics["reference_speech_covered_seconds"]
        ),
        "reference_speech_missed_seconds": round_float(
            coverage_metrics["reference_speech_missed_seconds"]
        ),
        "predicted_non_reference_seconds": round_float(
            coverage_metrics["predicted_non_reference_seconds"]
        ),
        "asr_cer": round_float(asr_cer),
        "asr_wer_like": round_float(asr_wer_like),
        "asr_status": asr_status,
        "segmentation_seconds": round_float(segmentation_seconds),
        "segmentation_rtf": round_float(segmentation_seconds / audio_seconds if audio_seconds else None),
        "asr_seconds": round_float(asr_seconds),
        "asr_rtf": round_float(asr_seconds / predicted_audio_seconds if predicted_audio_seconds else None),
        **identities,
        "elapsed_seconds": round_float(time.perf_counter() - total_started),
        "notes": "; ".join(sorted(set(notes))),
    }
    return row, details


def create_segmenter(setting: Setting, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    from daiya.asr import create_utterance_segmenter

    prefer_silero = setting.backend == "silero" or (setting.backend == "auto" and args.prefer_silero)
    kwargs = {
        "backend": setting.backend,
        "prefer_silero": prefer_silero,
        "threshold": setting.threshold,
        "min_speech_seconds": setting.min_speech_seconds,
        "min_silence_seconds": setting.min_silence_seconds,
        "speech_padding_seconds": setting.speech_padding_seconds,
        "max_utterance_seconds": setting.max_utterance_seconds,
    }
    return create_utterance_segmenter(**kwargs), kwargs


def dropped_setting_notes(setting: Setting, used_kwargs: dict[str, Any], actual_backend: str = "") -> list[str]:
    notes = []
    if "backend" not in used_kwargs:
        notes.append("factory did not accept backend kwarg")
    if "min_silence_seconds" not in used_kwargs and "trailing_silence_seconds" in used_kwargs:
        notes.append("used trailing_silence_seconds for min_silence_seconds")
    if setting.speech_padding_seconds and "speech_padding_seconds" not in used_kwargs:
        notes.append("speech padding unsupported by active segmenter")
    if setting.speech_padding_seconds and actual_backend == "energy":
        notes.append("energy segmenter ignores speech padding")
    return notes


def backend_name(segmenter: Any) -> str:
    name = type(segmenter).__name__.lower()
    if "silero" in name:
        return "silero"
    if "energy" in name:
        return "energy"
    return type(segmenter).__name__


def segment_audio(segmenter: Any, audio: AudioInput, chunk_seconds: float) -> list[Segment]:
    from daiya.audio import iter_chunks_from_samples

    output: list[Segment] = []
    for chunk in iter_chunks_from_samples(audio.samples_16k, chunk_seconds=chunk_seconds, source=f"bakeoff:{audio.path}"):
        output.extend(segment_from_utterance(audio.path, utterance) for utterance in segmenter.accept(chunk))
    output.extend(segment_from_utterance(audio.path, utterance) for utterance in segmenter.flush())
    return output


def segment_from_utterance(audio_path: Path, utterance: Any) -> Segment:
    return Segment(
        audio_path=audio_path,
        start=float(getattr(utterance, "start", 0.0)),
        end=float(getattr(utterance, "end", 0.0)),
        samples=getattr(utterance, "samples", None),
    )


def load_boundary_references(args: argparse.Namespace) -> dict[str, list[tuple[float, float]]]:
    refs: dict[str, list[tuple[float, float]]] = {}
    if args.reference_boundaries_jsonl:
        for row in read_jsonl(args.reference_boundaries_jsonl):
            key = reference_key(row)
            start = as_float(row.get("start"))
            end = as_float(row.get("end"))
            if key and start is not None and end is not None and end > start:
                refs.setdefault(key, []).append((start, end))
    if args.reference_rttm:
        for key, start, end in read_rttm(args.reference_rttm):
            refs.setdefault(key, []).append((start, end))
    return {key: sorted(value) for key, value in refs.items()}


def load_text_references(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    parts: dict[str, list[tuple[tuple[float, float, int], str]]] = {}
    for row_index, row in enumerate(read_jsonl(path)):
        key = reference_key(row)
        text = row.get("text") or row.get("reference")
        if key and text:
            start = as_float(row.get("start"))
            end = as_float(row.get("end"))
            order = (
                start if start is not None else math.inf,
                end if end is not None else math.inf,
                row_index,
            )
            parts.setdefault(key, []).append((order, str(text).strip()))
    return {
        key: " ".join(text for _order, text in sorted(values) if text)
        for key, values in parts.items()
    }


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BakeoffSkip(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def read_rttm(path: Path) -> Iterable[tuple[str, float, float]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 5 or parts[0] != "SPEAKER":
                continue
            key = parts[1]
            start = float(parts[3])
            duration = float(parts[4])
            yield key, start, start + duration


def reference_key(row: dict[str, Any]) -> str:
    raw = row.get("audio_path") or row.get("file_name") or row.get("source_file") or row.get("uri")
    if raw is None:
        return ""
    value = str(raw)
    path = Path(value)
    if path.exists():
        return str(path.resolve())
    return value


def matching_reference_keys(path: Path) -> list[str]:
    return [str(path.resolve()), str(path), path.name, path.stem]


def score_boundaries(
    segments: list[Segment],
    refs: dict[str, list[tuple[float, float]]],
    collar: float,
    audio_paths: Iterable[Path] | None = None,
) -> dict[str, float | None]:
    if not refs:
        return {"precision": None, "recall": None, "f1": None}

    predicted_by_audio: dict[str, list[tuple[float, float]]] = {}
    for segment in segments:
        key = str(segment.audio_path.resolve())
        predicted_by_audio.setdefault(key, []).append((segment.start, segment.end))

    matched = 0
    predicted_count = 0
    reference_count = 0
    paths = {Path(key) for key in predicted_by_audio}
    paths.update(path.resolve() for path in audio_paths or [])
    for path in paths:
        predicted_segments = predicted_by_audio.get(str(path.resolve()), [])
        reference_segments = references_for_path(path, refs)
        if not reference_segments:
            continue
        scope_start = min(start for start, _end in reference_segments)
        scope_end = max(end for _start, end in reference_segments)
        pred_boundaries = [
            boundary
            for boundary in boundaries_from_segments(predicted_segments)
            if scope_start <= boundary <= scope_end
        ]
        ref_boundaries = boundaries_from_segments(reference_segments)
        matched += count_boundary_matches(pred_boundaries, ref_boundaries, collar)
        predicted_count += len(pred_boundaries)
        reference_count += len(ref_boundaries)

    if not predicted_count and not reference_count:
        return {"precision": None, "recall": None, "f1": None}
    precision = matched / predicted_count if predicted_count else 0.0
    recall = matched / reference_count if reference_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score_coverage(
    segments: list[Segment],
    refs: dict[str, list[tuple[float, float]]],
    audio_paths: Iterable[Path] | None = None,
) -> dict[str, float | None]:
    predicted_by_audio: dict[Path, list[tuple[float, float]]] = {}
    for segment in segments:
        predicted_by_audio.setdefault(segment.audio_path.resolve(), []).append(
            (segment.start, segment.end)
        )

    raw_predicted = sum(segment.duration for segment in segments)
    unique_predicted = sum(
        intervals_duration(merge_intervals(intervals))
        for intervals in predicted_by_audio.values()
    )
    result: dict[str, float | None] = {
        "unique_predicted_seconds": unique_predicted,
        "overlap_duplicated_seconds": max(0.0, raw_predicted - unique_predicted),
        "scored_seconds": None,
        "reference_speech_seconds": None,
        "reference_speech_covered_seconds": None,
        "reference_speech_missed_seconds": None,
        "predicted_non_reference_seconds": None,
    }
    if not refs:
        return result

    scored = reference_speech = covered = non_reference = 0.0
    matched_files = 0
    all_paths = set(predicted_by_audio)
    all_paths.update(path.resolve() for path in audio_paths or [])
    for path in all_paths:
        reference_intervals = references_for_path(path, refs)
        if not reference_intervals:
            continue
        matched_files += 1
        reference_union = merge_intervals(reference_intervals)
        scope_start = min(start for start, _end in reference_union)
        scope_end = max(end for _start, end in reference_union)
        predicted_union = merge_intervals(
            clip_intervals(predicted_by_audio.get(path, []), scope_start, scope_end)
        )
        scored += scope_end - scope_start
        reference_duration = intervals_duration(reference_union)
        intersection = intersection_duration(predicted_union, reference_union)
        reference_speech += reference_duration
        covered += intersection
        non_reference += max(0.0, intervals_duration(predicted_union) - intersection)

    if matched_files:
        result.update(
            {
                "scored_seconds": scored,
                "reference_speech_seconds": reference_speech,
                "reference_speech_covered_seconds": covered,
                "reference_speech_missed_seconds": max(0.0, reference_speech - covered),
                "predicted_non_reference_seconds": non_reference,
            }
        )
    return result


def references_for_path(
    path: Path,
    refs: dict[str, list[tuple[float, float]]],
) -> list[tuple[float, float]]:
    found: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for key in matching_reference_keys(path):
        for interval in refs.get(key, []):
            if interval not in seen:
                found.append(interval)
                seen.add(interval)
    return sorted(found)


def clip_intervals(
    intervals: Iterable[tuple[float, float]],
    scope_start: float,
    scope_end: float,
) -> list[tuple[float, float]]:
    return [
        (max(start, scope_start), min(end, scope_end))
        for start, end in intervals
        if min(end, scope_end) > max(start, scope_start)
    ]


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def intervals_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(max(0.0, end - start) for start, end in intervals)


def intersection_duration(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> float:
    left_index = right_index = 0
    total = 0.0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            total += end - start
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def boundaries_from_segments(segments: list[tuple[float, float]]) -> list[float]:
    boundaries: list[float] = []
    for start, end in segments:
        boundaries.extend([float(start), float(end)])
    return sorted(boundaries)


def count_boundary_matches(predicted: list[float], reference: list[float], collar: float) -> int:
    unmatched = list(reference)
    matches = 0
    for boundary in sorted(predicted):
        best_index = None
        best_distance = math.inf
        for index, ref_boundary in enumerate(unmatched):
            distance = abs(boundary - ref_boundary)
            if distance <= collar and distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is not None:
            matches += 1
            unmatched.pop(best_index)
    return matches


def build_asr_runner(args: argparse.Namespace, text_refs: dict[str, str]) -> Any | None:
    if not args.asr_model:
        return None
    if not text_refs:
        return ASRSkip("asr skipped: no reference text JSONL")
    model_path = Path(args.asr_model)
    if not model_path.exists() and not args.allow_model_download:
        return ASRSkip("asr skipped: model path not found and downloads disabled")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return ASRSkip("asr skipped: faster-whisper is not installed")
    try:
        model = WhisperModel(args.asr_model, device=args.asr_device, compute_type=args.asr_compute_type)
        return ASRRunner(model=model, language=args.asr_language)
    except Exception as exc:
        return ASRSkip(f"asr skipped: {type(exc).__name__}: {exc}")


class ASRSkip:
    def __init__(self, reason: str) -> None:
        self.reason = reason


@dataclass(frozen=True)
class ASRRunner:
    model: Any
    language: str | None


def score_asr(
    segments: list[Segment],
    asr_runner: Any | None,
    text_refs: dict[str, str],
    audio_paths: Iterable[Path] | None = None,
) -> tuple[float | None, float | None, str]:
    if asr_runner is None:
        return None, None, ""
    if isinstance(asr_runner, ASRSkip):
        return None, None, asr_runner.reason
    if np is None:
        return None, None, "asr skipped: numpy is not installed"

    predictions: dict[str, list[str]] = {}
    for path in audio_paths or []:
        if any(text_refs.get(key) for key in matching_reference_keys(path)):
            predictions[str(path.resolve())] = []
    for segment in sorted(segments, key=lambda item: (str(item.audio_path.resolve()), item.start, item.end)):
        refs = [text_refs.get(key) for key in matching_reference_keys(segment.audio_path)]
        if not any(refs):
            continue
        try:
            fw_segments, _info = asr_runner.model.transcribe(
                segment.samples,
                language=asr_runner.language,
                vad_filter=False,
            )
            text = " ".join(str(getattr(item, "text", "")).strip() for item in fw_segments).strip()
        except Exception as exc:
            return None, None, f"asr failed: {type(exc).__name__}: {exc}"
        predictions.setdefault(str(segment.audio_path.resolve()), []).append(text)

    char_distances = char_lengths = 0
    word_distances = word_lengths = 0
    for audio_key, parts in predictions.items():
        reference = None
        path = Path(audio_key)
        for key in matching_reference_keys(path):
            if key in text_refs:
                reference = text_refs[key]
                break
        if reference is None:
            continue
        ref = normalize_for_cer(reference)
        hyp = normalize_for_cer(" ".join(parts))
        char_distances += levenshtein(ref, hyp)
        char_lengths += len(ref)
        ref_words = normalize_for_wer(reference)
        hyp_words = normalize_for_wer(" ".join(parts))
        word_distances += levenshtein(ref_words, hyp_words)
        word_lengths += len(ref_words)
    if char_lengths == 0:
        return None, None, "asr skipped: no matching reference text"
    cer = char_distances / char_lengths
    wer_like = word_distances / word_lengths if word_lengths else None
    return cer, wer_like, "asr ok"


def normalize_for_cer(text: str) -> str:
    return " ".join(text.lower().split())


def normalize_for_wer(text: str) -> list[str]:
    return text.lower().split()


def levenshtein(reference: Any, hypothesis: Any) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_char in enumerate(reference, start=1):
        current = [row_index]
        for col_index, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[col_index - 1] + 1,
                    previous[col_index] + 1,
                    previous[col_index - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def setting_to_dict(setting: Setting) -> dict[str, Any]:
    return {
        "backend": setting.backend,
        "threshold": setting.threshold,
        "min_speech_seconds": setting.min_speech_seconds,
        "min_silence_seconds": setting.min_silence_seconds,
        "speech_padding_seconds": setting.speech_padding_seconds,
        "max_utterance_seconds": setting.max_utterance_seconds,
    }


def duration_bucket_counts(durations: Iterable[float]) -> dict[str, int]:
    result = {
        "duration_lt_1_count": 0,
        "duration_1_2_count": 0,
        "duration_2_5_count": 0,
        "duration_5_8_count": 0,
        "duration_8_plus_count": 0,
    }
    for duration in durations:
        if duration < 1.0:
            result["duration_lt_1_count"] += 1
        elif duration < 2.0:
            result["duration_1_2_count"] += 1
        elif duration < 5.0:
            result["duration_2_5_count"] += 1
        elif duration < 8.0:
            result["duration_5_8_count"] += 1
        else:
            result["duration_8_plus_count"] += 1
    return result


def reproducibility_identities(args: argparse.Namespace) -> dict[str, str | None]:
    model_path = None
    if args.asr_model:
        candidate = Path(args.asr_model).expanduser()
        model_path = str(candidate.resolve()) if candidate.exists() else str(args.asr_model)
    return {
        "model_path": model_path,
        "python_version": sys.version.split()[0],
        "numpy_version": package_version("numpy"),
        "silero_vad_version": package_version("silero-vad"),
        "torch_version": package_version("torch"),
        "faster_whisper_version": package_version("faster-whisper"),
    }


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def format_compact_row(row: dict[str, Any]) -> str:
    return (
        "{backend:<6} actual={actual:<6} thr={threshold:<6} silence={silence:<5} pad={pad:<4} "
        "n={count:<4} speech={speech:<8} mean={mean_duration} p95={p95} status={status}"
    ).format(
        backend=row["backend"],
        actual=row["segmenter_backend"],
        threshold=row["threshold"],
        silence=row["min_silence_seconds"],
        pad=row["speech_padding_seconds"],
        count=row["utterance_count"],
        speech=row["total_speech_seconds"],
        mean_duration=row["mean_duration_seconds"],
        p95=row["p95_duration_seconds"],
        status=row["status"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
