"""Compare wall-clock-v2 against timestamp-informed ownership segmentation.

Input must be a normalized mono 16 kHz WAV.  This diagnostic runs the local
Faster-Whisper model for *boundary evidence only*; it never invokes the audio
labeling LLM and never claims transcript timestamps from that model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace
import json
import math
import re

import soundfile as sf

from daiya_whisper_pipeline.evidence import TimestampEvidenceStage
from daiya_whisper_pipeline.segmentation import build_chunks, merge_intervals
from daiya_whisper_pipeline.types import Interval, NormalizedAudio


_HEADER = re.compile(r"^#\d+\s*\|\s*([0-9.]+)-([0-9.]+)s\s*$")


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    labeling_start: float
    training_eligible: bool
    fallback: bool
    boundary_method: str
    boundary_confidence: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def labeling_duration(self) -> float:
        return self.end - self.labeling_start


def _references(path: Path | None) -> list[Interval]:
    if path is None:
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := _HEADER.match(line.strip()):
            values.append(Interval(float(match.group(1)), float(match.group(2))))
    if not values:
        raise ValueError(f"No reference intervals in {path}")
    return values


def _union(intervals: list[Interval]) -> float:
    ordered = sorted((item for item in intervals if item.duration > 0), key=lambda item: item.start)
    if not ordered:
        return 0.0
    total, start, end = 0.0, ordered[0].start, ordered[0].end
    for item in ordered[1:]:
        if item.start <= end:
            end = max(end, item.end)
        else:
            total += end - start
            start, end = item.start, item.end
    return total + end - start


def _intersect(left: list[Interval], right: list[Interval]) -> float:
    intersections: list[Interval] = []
    for first in left:
        for second in right:
            start, end = max(first.start, second.start), min(first.end, second.end)
            if end > start:
                intersections.append(Interval(start, end))
    return _union(intersections)


def _v2_segments(speech: list[Interval], *, target: float, maximum: float, minimum_silence: float, search: float, context: float) -> list[Segment]:
    """Faithful wall-clock-v2 fallback behavior for the comparison baseline."""
    groups: list[list[Interval]] = []
    for interval in merge_intervals(speech):
        if groups and interval.start - groups[-1][-1].end <= 0.8:
            groups[-1].append(interval)
        else:
            groups.append([interval])
    output: list[Segment] = []
    for group in groups:
        cursor, end, inherited_context = group[0].start, group[-1].end, 0.0
        while end - cursor > maximum:
            hard_end, ideal = cursor + maximum, cursor + target
            candidates = []
            for left, right in zip(group, group[1:]):
                gap_start, gap_end = max(cursor, left.end), min(hard_end, right.start)
                midpoint = (gap_start + gap_end) / 2
                if gap_end - gap_start >= minimum_silence and ideal - search <= midpoint < hard_end and end - midpoint >= 1.0:
                    candidates.append((midpoint, gap_end - gap_start))
            if candidates:
                cut, _ = min(candidates, key=lambda item: (abs(item[0] - ideal), -item[1], item[0]))
                output.append(
                    Segment(
                        cursor,
                        cut,
                        cursor,
                        inherited_context == 0,
                        inherited_context > 0,
                        "silero_vad_gap",
                        1.0,
                    )
                )
                cursor, inherited_context = cut, 0.0
                continue
            next_cursor = hard_end - context
            output.append(Segment(cursor, hard_end, cursor, False, True, "no_silence_boundary_fallback", 0.0))
            cursor, inherited_context = next_cursor, context
        output.append(Segment(cursor, end, cursor, inherited_context == 0, inherited_context > 0, "source_end", 1.0))
    return output


def _summary(segments: list[Segment], references: list[Interval], collar: float) -> dict[str, object]:
    owned = [Interval(item.start, item.end) for item in segments]
    labeling = [Interval(item.labeling_start, item.end) for item in segments]
    eligible = [Interval(item.start, item.end) for item in segments if item.training_eligible]
    durations = sorted(item.duration for item in segments)
    reference_seconds = _union(references)
    retained = _intersect(owned, references) if references else None
    risky = sum(
        any(reference.start + collar < item.end < reference.end - collar for reference in references)
        for item in segments[:-1]
    )
    return {
        "chunk_duration_seconds": {
            "count": len(durations),
            "mean": round(mean(durations), 3) if durations else 0.0,
            "p50": round(median(durations), 3) if durations else 0.0,
            "p95": round(durations[min(len(durations) - 1, math.ceil(len(durations) * 0.95) - 1)], 3) if durations else 0.0,
            "max": round(max(durations), 3) if durations else 0.0,
        },
        "reference_speech_seconds": round(reference_seconds, 3) if references else None,
        "retained_reference_speech_seconds": round(retained, 3) if retained is not None else None,
        "missed_reference_speech_seconds": round(max(0.0, reference_seconds - retained), 3) if retained is not None else None,
        "unprotected_boundaries_inside_reference_speech": risky if references else None,
        "fallback_handoffs": sum(
            item.boundary_method in {"no_silence_boundary_fallback", "continuous_speech_fallback"}
            for item in segments
        ),
        "fallback_rows": sum(item.fallback for item in segments),
        "duplicated_labeling_audio_seconds": round(max(0.0, sum(item.labeling_duration for item in segments) - _union(labeling)), 3),
        "duplicated_eligible_training_seconds": round(max(0.0, sum(item.duration for item in segments if item.training_eligible) - _union(eligible)), 3),
    }


def main() -> None:
    from daiya_whisper_pipeline.vad import SileroVad

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--reference-labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamp-model", required=True, help="Local current Faster-Whisper CTranslate2 model path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--collar-seconds", type=float, default=0.25)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    duration = float(sf.info(args.audio).duration)
    references = _references(args.reference_labels)
    if any(item.start < 0 or item.end > duration + 0.001 for item in references):
        raise ValueError("Reference intervals must be local to the supplied WAV")
    config = SimpleNamespace(
        work_dir=args.output.parent / "benchmark-work",
        timestamp_evidence_cache_dir=args.cache_dir or args.output.parent / "timestamp-evidence-cache",
        timestamp_model=args.timestamp_model,
        timestamp_device=args.device,
        timestamp_compute_type=args.compute_type,
        timestamp_beam_size=5,
        timestamp_language="",
        timestamp_condition_on_previous_text=False,
        energy_window_seconds=0.05,
        energy_low_percentile=20.0,
        energy_min_gap_seconds=0.20,
        sample_rate=16_000,
        torch_device=args.device,
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        vad_min_silence_ms=150,
        vad_speech_pad_ms=80,
        overlap_mode="preserve",
        overlap_pad_seconds=0.15,
        min_chunk_seconds=1.0,
        target_chunk_seconds=18.0,
        max_chunk_seconds=25.0,
        hard_max_chunk_seconds=30.0,
        merge_gap_seconds=0.8,
        boundary_min_silence_seconds=0.5,
        boundary_search_seconds=4.0,
        fallback_context_seconds=1.0,
        boundary_candidate_tolerance_seconds=0.35,
        boundary_min_confidence=0.55,
        label_alignment_min_similarity=0.45,
    )
    source = NormalizedAudio(args.audio, args.audio, args.audio.stem, duration_seconds=duration)
    speech = SileroVad(config).detect(args.audio)
    evidence = TimestampEvidenceStage(config).collect(source)
    v2 = _v2_segments(speech, target=18.0, maximum=25.0, minimum_silence=0.5, search=4.0, context=1.0)
    timestamp = [
        Segment(chunk.start, chunk.end, chunk.labeling_start, chunk.training_eligible, chunk.has_labeling_preroll, chunk.boundary_method, chunk.boundary_confidence)
        for chunk in build_chunks(source, speech, [], config, evidence)
    ]
    changed = sum(
        1
        for index in range(max(len(v2), len(timestamp)))
        if index >= len(v2) or index >= len(timestamp) or (v2[index].start, v2[index].end) != (timestamp[index].start, timestamp[index].end)
    )
    support_counts = [len(chunk.boundary_evidence.get("methods", [])) for chunk in build_chunks(source, speech, [], config, evidence)[:-1]]
    report = {
        "benchmark": "offline-whisper-timestamp-ownership-v1",
        "audio": str(args.audio),
        "reference_labels": str(args.reference_labels) if args.reference_labels else None,
        "method_notes": [
            "Faster-Whisper word/phrase timestamps are boundary and validation evidence only; no audio-labeling LLM timestamps are requested or trusted.",
            "Reference metrics apply only to this supplied human-labelled window and do not establish universal zero missed speech.",
            "Timestamp ownership ranges are disjoint. Labeler pre-roll is counted separately from eligible training audio.",
        ],
        "wall_clock_v2": _summary(v2, references, args.collar_seconds),
        "timestamp_ownership": _summary(timestamp, references, args.collar_seconds),
        "changed_row_count": changed,
        "timestamp_evidence": {
            **evidence.provenance(),
            "boundary_evidence_coverage": round(sum(1 for count in support_counts if count > 0) / len(support_counts), 4) if support_counts else 1.0,
            "multi_signal_boundaries": sum(1 for count in support_counts if count >= 2),
            "single_signal_boundaries": sum(1 for count in support_counts if count == 1),
            "disagreement_or_no_signal_boundaries": sum(1 for count in support_counts if count == 0),
        },
        "boundary_examples": [
            {
                "index": index,
                "wall_clock_v2": [round(old.start, 3), round(old.end, 3)],
                "timestamp_ownership": [round(new.start, 3), round(new.end, 3)],
                "labeling_range": [round(new.labeling_start, 3), round(new.end, 3)],
                "method": new.boundary_method,
                "confidence": round(new.boundary_confidence, 3),
            }
            for index, (old, new) in enumerate(zip(v2, timestamp))
            if (old.start, old.end) != (new.start, new.end) or new.fallback
        ][:12],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
