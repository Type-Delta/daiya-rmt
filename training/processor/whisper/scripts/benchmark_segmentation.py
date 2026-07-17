"""Offline old-vs-new Whisper segmentation benchmark (no transcription required)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from statistics import mean, median
from types import SimpleNamespace
from typing import Iterable

import soundfile as sf

from daiya_whisper_pipeline.segmentation import build_chunks
from daiya_whisper_pipeline.types import Interval, NormalizedAudio
from daiya_whisper_pipeline.vad import SileroVad


PROFILE_DEFAULTS = {
    # Current production release before this change.
    "legacy": (0.50, 250, 150, 80),
    # Evidence-oriented profiles reproduced from PR #10's VAD bake-off.
    "pr10-sensitive": (0.40, 200, 300, 50),
    "pr10-balanced": (0.50, 200, 500, 100),
    # New offline default: this profile had zero missed reference speech and
    # the fewest bounded-context fallbacks on the available human reference.
    "offline-wall-clock": (0.50, 250, 150, 80),
}
_HEADER = re.compile(r"^#\d+\s*\|\s*([0-9.]+)-([0-9.]+)s\s*$")


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speech_duration: float
    fallback: bool = False
    context_after_seconds: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


def _read_reference_labels(path: Path) -> list[Interval]:
    refs: list[Interval] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _HEADER.match(line.strip())
        if match:
            refs.append(Interval(float(match.group(1)), float(match.group(2))))
    if not refs:
        raise ValueError(f"No '#NNN | start-ends' rows found in {path}")
    return refs


def _union_seconds(intervals: Iterable[Interval]) -> float:
    ordered = sorted((item for item in intervals if item.duration > 0), key=lambda item: item.start)
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0].start, ordered[0].end
    for interval in ordered[1:]:
        if interval.start <= end:
            end = max(end, interval.end)
        else:
            total += end - start
            start, end = interval.start, interval.end
    return total + end - start


def _intersection_seconds(left: Iterable[Interval], right: Iterable[Interval]) -> float:
    left_ordered = sorted(left, key=lambda item: item.start)
    right_ordered = sorted(right, key=lambda item: item.start)
    total = 0.0
    left_index = right_index = 0
    while left_index < len(left_ordered) and right_index < len(right_ordered):
        first = left_ordered[left_index]
        second = right_ordered[right_index]
        total += max(0.0, min(first.end, second.end) - max(first.start, second.start))
        if first.end <= second.end:
            left_index += 1
        else:
            right_index += 1
    return total


def _legacy_segments(speech: list[Interval], max_seconds: float, target_seconds: float, merge_gap: float) -> list[Segment]:
    """Faithful pre-wall-clock grouping (without optional pyannote deletion)."""
    pieces: list[Interval] = []
    for interval in speech:
        cursor = interval.start
        while cursor < interval.end:
            end = min(cursor + max_seconds, interval.end)
            pieces.append(Interval(cursor, end))
            cursor = end
    groups: list[list[Interval]] = []
    current: list[Interval] = []
    current_speech = 0.0
    for interval in pieces:
        if not current:
            current, current_speech = [interval], interval.duration
            continue
        gap = interval.start - current[-1].end
        proposed_speech = current_speech + interval.duration
        proposed_wall = interval.end - current[0].start
        if gap <= merge_gap and proposed_speech <= target_seconds and proposed_wall <= max_seconds:
            current.append(interval)
            current_speech = proposed_speech
        else:
            groups.append(current)
            current, current_speech = [interval], interval.duration
    if current:
        groups.append(current)
    return [Segment(group[0].start, group[-1].end, sum(item.duration for item in group)) for group in groups]


def _profile_config(
    work_dir: Path, threshold: float, min_speech: int, min_silence: int, pad: int, device: str
) -> SimpleNamespace:
    return SimpleNamespace(
        work_dir=work_dir,
        sample_rate=16000,
        torch_device=device,
        vad_threshold=threshold,
        vad_min_speech_ms=min_speech,
        vad_min_silence_ms=min_silence,
        vad_speech_pad_ms=pad,
        overlap_mode="preserve",
        overlap_pad_seconds=0.15,
        min_chunk_seconds=1.0,
        target_chunk_seconds=18.0,
        max_chunk_seconds=25.0,
        merge_gap_seconds=0.8,
        boundary_min_silence_seconds=0.5,
        boundary_search_seconds=4.0,
        fallback_context_seconds=1.0,
    )


def _duration_summary(segments: list[Segment]) -> dict[str, float | int]:
    durations = sorted(segment.duration for segment in segments)
    if not durations:
        return {"count": 0, "total": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    percentile = durations[min(len(durations) - 1, math.ceil(len(durations) * 0.95) - 1)]
    return {
        "count": len(durations),
        "total": round(sum(durations), 3),
        "mean": round(mean(durations), 3),
        "p50": round(median(durations), 3),
        "p95": round(percentile, 3),
        "max": round(max(durations), 3),
    }


def _boundary_metrics(segments: list[Segment], references: list[Interval], collar: float) -> dict[str, float | int]:
    predicted = [point for segment in segments for point in (segment.start, segment.end)]
    actual = [point for interval in references for point in (interval.start, interval.end)]
    matched_actual: set[int] = set()
    matches = 0
    for point in predicted:
        eligible = [(abs(point - target), index) for index, target in enumerate(actual) if index not in matched_actual]
        if eligible and min(eligible)[0] <= collar:
            _, index = min(eligible)
            matched_actual.add(index)
            matches += 1
    precision = matches / len(predicted) if predicted else 0.0
    recall = matches / len(actual) if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matches": matches,
        "predicted": len(predicted),
        "reference": len(actual),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _boundary_speech_risk(segments: list[Segment], references: list[Interval], collar: float) -> dict[str, float | int]:
    # An internal boundary that falls in a hand-labelled speech interval is a
    # conservative proxy for cutting through a word/sentence. A no-silence
    # fallback carries source context across its handoff, so report it
    # separately instead of falsely counting it as missing-word risk.
    internal = segments[:-1]
    # Only a window's trailing context protects its own right boundary. A
    # successor with context-before is review-only but cannot protect the next
    # boundary it emits.
    protected = [segment for segment in internal if segment.context_after_seconds > 0]
    unprotected = [segment for segment in internal if segment.context_after_seconds <= 0]
    risky = sum(
        any(reference.start + collar < segment.end < reference.end - collar for reference in references)
        for segment in unprotected
    )
    protected_speech_adjacent = sum(
        any(reference.start + collar < segment.end < reference.end - collar for reference in references)
        for segment in protected
    )
    return {
        "internal_boundaries": len(internal),
        "context_protected_boundaries": len(protected),
        "context_protected_speech_adjacent": protected_speech_adjacent,
        "unprotected_speech_adjacent": risky,
        "unprotected_rate": round(risky / len(unprotected), 4) if unprotected else 0.0,
    }


def _metrics(segments: list[Segment], references: list[Interval], collar: float) -> dict[str, object]:
    windows = [Interval(segment.start, segment.end) for segment in segments]
    reference_seconds = _union_seconds(references)
    retained = _intersection_seconds(windows, references)
    total = sum(segment.duration for segment in segments)
    unique = _union_seconds(windows)
    return {
        "duration": _duration_summary(segments),
        "reference_speech_seconds": round(reference_seconds, 3),
        "retained_reference_speech_seconds": round(retained, 3),
        "missed_reference_speech_seconds": round(max(0.0, reference_seconds - retained), 3),
        "duplicated_window_seconds": round(max(0.0, total - unique), 3),
        "fallback_context_chunks": sum(segment.fallback for segment in segments),
        "fallback_handoffs": sum(segment.context_after_seconds > 0 for segment in segments),
        "long_chunks_over_target": sum(segment.duration > 18.0 for segment in segments),
        "boundary": _boundary_metrics(segments, references, collar),
        "boundary_word_proxy": _boundary_speech_risk(segments, references, collar),
    }


def _examples(legacy: list[Segment], wall_clock: list[Segment], references: list[Interval], limit: int) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for index, segment in enumerate(wall_clock):
        if len(examples) >= limit:
            break
        overlapping_refs = [reference for reference in references if reference.start < segment.end and reference.end > segment.start]
        if segment.fallback or overlapping_refs:
            examples.append(
                {
                    "new_index": index,
                    "new_window": [round(segment.start, 3), round(segment.end, 3)],
                    "fallback_context": segment.fallback,
                    "reference_spans": [[round(item.start, 3), round(item.end, 3)] for item in overlapping_refs[:4]],
                    "legacy_windows_nearby": [
                        [round(item.start, 3), round(item.end, 3)]
                        for item in legacy
                        if item.start < segment.end and item.end > segment.start
                    ][:4],
                }
            )
    return examples


def _load_old_rows(path: Path | None, source_name: str, start: float, end: float) -> list[dict[str, object]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if Path(str(row.get("source_file", ""))).name != source_name:
            continue
        row_start, row_end = float(row.get("source_start", -1)), float(row.get("source_end", -1))
        if row_start < end and row_end > start:
            rows.append(row)
    return rows


def _review_coverage(path: Path | None, old_rows: list[dict[str, object]]) -> dict[str, object]:
    if path is None or not path.is_file():
        return {"available": False, "total_reviews": 0, "reviews_in_benchmark_window": 0, "source_uris": []}
    reviewed_uris: set[str] = set()
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        chunk = row.get("chunk", {})
        if isinstance(chunk, dict) and chunk.get("source_uri"):
            reviewed_uris.add(str(chunk["source_uri"]).replace("\\", "/"))
    window_uris = sorted(
        str(row.get("file_name"))
        for row in old_rows
        if str(row.get("file_name", "")).replace("\\", "/") in reviewed_uris
    )
    return {
        "available": True,
        "total_reviews": total,
        "reviews_in_benchmark_window": len(window_uris),
        "source_uris": window_uris[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        type=Path,
        nargs="+",
        help="One or more contiguous normalized 16 kHz mono WAV benchmark windows.",
    )
    parser.add_argument("--reference-labels", required=True, type=Path)
    parser.add_argument("--source-offset-seconds", type=float, default=0.0, help="Raw-source timestamp of audio time 0.")
    parser.add_argument("--raw-source-name", help="Raw source basename used by --old-metadata, e.g. Th-En_sample_11.m4a.")
    parser.add_argument("--old-metadata", type=Path)
    parser.add_argument("--old-reviews", type=Path, help="Optional append-only human review JSONL for risk coverage.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--collar-seconds", type=float, default=0.25)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--device", default="cpu", help="Torch device; CPU is the reproducible default for this diagnostic.")
    args = parser.parse_args()

    references = _read_reference_labels(args.reference_labels)
    audio_durations = [float(sf.info(path).duration) for path in args.audio]
    duration = sum(audio_durations)
    if any(interval.start < 0 or interval.end > duration + 0.001 for interval in references):
        raise ValueError("Reference labels must use benchmark-local timestamps within the supplied audio duration")
    source = NormalizedAudio(args.audio[0], args.audio[0], args.audio[0].stem, duration_seconds=duration)
    profiles: dict[str, object] = {}
    vad: SileroVad | None = None
    for name, values in PROFILE_DEFAULTS.items():
        config = _profile_config(args.output.parent / "benchmark-work", *values, args.device)
        if vad is None:
            vad = SileroVad(config)
        else:
            # Silero's model state is independent of these timestamp options;
            # reuse it so the profile comparison changes only VAD parameters.
            vad.config = config
        speech: list[Interval] = []
        offset = 0.0
        for audio_path, audio_duration in zip(args.audio, audio_durations):
            speech.extend(
                Interval(interval.start + offset, interval.end + offset)
                for interval in vad.detect(audio_path)
            )
            offset += audio_duration
        legacy = _legacy_segments(speech, config.max_chunk_seconds, config.target_chunk_seconds, config.merge_gap_seconds)
        wall_clock_chunks = build_chunks(source, speech, [], config)
        wall_clock = [
            Segment(
                chunk.start,
                chunk.end,
                chunk.speech_duration,
                fallback=not chunk.training_eligible,
                context_after_seconds=chunk.context_overlap_after_seconds,
            )
            for chunk in wall_clock_chunks
        ]
        profiles[name] = {
            "vad": {"threshold": values[0], "min_speech_ms": values[1], "min_silence_ms": values[2], "speech_pad_ms": values[3], "detected_intervals": len(speech)},
            "legacy_spliced": _metrics(legacy, references, args.collar_seconds),
            "wall_clock": _metrics(wall_clock, references, args.collar_seconds),
            "examples": _examples(legacy, wall_clock, references, args.examples),
        }

    old_rows = _load_old_rows(
        args.old_metadata,
        args.raw_source_name or args.audio[0].name,
        args.source_offset_seconds,
        args.source_offset_seconds + duration,
    )
    risky_old = sorted(
        (
            {
                "file_name": row.get("file_name"),
                "source_start": row.get("source_start"),
                "source_end": row.get("source_end"),
                "text_density_chars_per_s": round(len(str(row.get("text", ""))) / max(float(row.get("speech_duration", 0.001)), 0.001), 2),
                "timeline_removed_seconds_proxy": round(
                    max(0.0, float(row.get("source_end", 0)) - float(row.get("source_start", 0)) - float(row.get("speech_duration", 0))), 3
                ),
            }
            for row in old_rows
        ),
        key=lambda item: (item["timeline_removed_seconds_proxy"], item["text_density_chars_per_s"]),
        reverse=True,
    )[:10]
    report = {
        "benchmark": "offline-whisper-wall-clock-segmentation-v1",
        "audio": [str(path) for path in args.audio],
        "reference_labels": str(args.reference_labels),
        "source_offset_seconds": args.source_offset_seconds,
        "reference_interval_count": len(references),
        "method_notes": [
            "Reference intervals are human-labelled speech spans; boundary F1 uses a 250 ms collar and is a proxy, not sentence truth.",
            "Legacy metrics model pre-change VAD grouping and speech-only export. pyannote overlap deletion is reported separately because no portable overlap truth was available for this window.",
            "The boundary-word proxy counts only unprotected internal boundaries inside a human reference interval after a collar; context-fallback handoffs are reported separately.",
        ],
        "profiles": profiles,
        "existing_dataset_risky_rows": risky_old,
        "review_coverage": _review_coverage(args.old_reviews, old_rows),
    }
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
