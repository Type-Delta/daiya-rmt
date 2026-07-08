#!/usr/bin/env python
"""Compare utterance segmentation settings without ASR/diarization downloads.

Example:
    python lab/vad_bakeoff.py audio.wav --backend energy --threshold 0.008,0.012
"""

from __future__ import annotations

import argparse
import csv
import glob
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
    "audio_seconds",
    "utterance_count",
    "total_speech_seconds",
    "mean_duration_seconds",
    "p50_duration_seconds",
    "p95_duration_seconds",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "asr_cer",
    "asr_status",
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
    parser.add_argument("--threshold", default="0.008,0.012,0.02", help="Comma-separated VAD thresholds.")
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
    thresholds = parse_float_list(args.threshold)
    min_speech_values = parse_float_list(args.min_speech_seconds)
    min_silence_values = parse_float_list(args.min_silence_seconds)
    padding_values = parse_float_list(args.speech_padding_seconds)
    max_values = parse_float_list(args.max_utterance_seconds)
    for backend, threshold, min_speech, min_silence, padding, max_utterance in itertools.product(
        backends,
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
    started = time.perf_counter()
    notes: list[str] = []
    segments: list[Segment] = []
    details: list[dict[str, Any]] = []
    segmenter_backend = ""
    status = "ok"

    try:
        for audio in audio_inputs:
            segmenter, used_kwargs = create_segmenter(setting, args)
            actual_backend = backend_name(segmenter)
            if segmenter_backend and segmenter_backend != actual_backend:
                segmenter_backend = "mixed"
            else:
                segmenter_backend = actual_backend
            setting_notes = dropped_setting_notes(setting, used_kwargs, actual_backend)
            for note in setting_notes:
                if note not in notes:
                    notes.append(note)

            audio_segments = segment_audio(segmenter, audio, args.chunk_seconds)
            segments.extend(audio_segments)
            for index, segment in enumerate(audio_segments, start=1):
                details.append(
                    {
                        **setting_to_dict(setting),
                        "segmenter_backend": actual_backend,
                        "audio_path": str(audio.path),
                        "utterance_index": index,
                        "start": segment.start,
                        "end": segment.end,
                        "duration": segment.duration,
                    }
                )
    except Exception as exc:
        status = "failed"
        notes.append(f"{type(exc).__name__}: {exc}")

    durations = [segment.duration for segment in segments]
    boundary_metrics = score_boundaries(segments, boundary_refs, args.boundary_collar_seconds)
    asr_cer, asr_status = score_asr(segments, asr_runner, text_refs)
    if asr_status:
        notes.append(asr_status)

    if setting.backend == "silero" and segmenter_backend == "energy":
        notes.append("requested silero but factory fell back to energy")

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
        "audio_seconds": round_float(sum(audio.seconds for audio in audio_inputs)),
        "utterance_count": len(segments),
        "total_speech_seconds": round_float(sum(durations)),
        "mean_duration_seconds": round_float(mean(durations)),
        "p50_duration_seconds": round_float(percentile(durations, 50)),
        "p95_duration_seconds": round_float(percentile(durations, 95)),
        "boundary_precision": round_float(boundary_metrics.get("precision")),
        "boundary_recall": round_float(boundary_metrics.get("recall")),
        "boundary_f1": round_float(boundary_metrics.get("f1")),
        "asr_cer": round_float(asr_cer),
        "asr_status": asr_status,
        "elapsed_seconds": round_float(time.perf_counter() - started),
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
    refs: dict[str, str] = {}
    for row in read_jsonl(path):
        key = reference_key(row)
        text = row.get("text") or row.get("reference")
        if key and text:
            refs[key] = str(text)
    return refs


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
) -> dict[str, float | None]:
    if not refs:
        return {"precision": None, "recall": None, "f1": None}

    predicted_by_audio: dict[str, list[tuple[float, float]]] = {}
    for segment in segments:
        key = str(segment.audio_path.resolve())
        predicted_by_audio.setdefault(key, []).append((segment.start, segment.end))

    pred_boundaries: list[float] = []
    ref_boundaries: list[float] = []
    for audio_key, predicted_segments in predicted_by_audio.items():
        path = Path(audio_key)
        reference_segments: list[tuple[float, float]] = []
        for key in matching_reference_keys(path):
            reference_segments.extend(refs.get(key, []))
        if not reference_segments:
            continue
        pred_boundaries.extend(boundaries_from_segments(predicted_segments))
        ref_boundaries.extend(boundaries_from_segments(reference_segments))

    if not pred_boundaries and not ref_boundaries:
        return {"precision": None, "recall": None, "f1": None}
    matched = count_boundary_matches(pred_boundaries, ref_boundaries, collar)
    precision = matched / len(pred_boundaries) if pred_boundaries else 0.0
    recall = matched / len(ref_boundaries) if ref_boundaries else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


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
) -> tuple[float | None, str]:
    if asr_runner is None:
        return None, ""
    if isinstance(asr_runner, ASRSkip):
        return None, asr_runner.reason
    if np is None:
        return None, "asr skipped: numpy is not installed"

    predictions: dict[str, list[str]] = {}
    for segment in segments:
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
            return None, f"asr failed: {type(exc).__name__}: {exc}"
        predictions.setdefault(str(segment.audio_path.resolve()), []).append(text)

    distances = 0
    lengths = 0
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
        distances += levenshtein(ref, hyp)
        lengths += len(ref)
    if lengths == 0:
        return None, "asr skipped: no matching reference text"
    return distances / lengths, "asr ok"


def normalize_for_cer(text: str) -> str:
    return " ".join(text.lower().split())


def levenshtein(reference: str, hypothesis: str) -> int:
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
