from __future__ import annotations

from dataclasses import dataclass

from .config import PipelineConfig, segmentation_config_id
from .types import Chunk, Interval, NormalizedAudio


SEGMENTATION_VERSION = "wall-clock-v2"


def merge_intervals(intervals: list[Interval], max_gap: float = 0.0) -> list[Interval]:
    """Merge intervals without changing the source timeline."""
    if not intervals:
        return []
    ordered = sorted((item for item in intervals if item.duration > 0), key=lambda item: item.start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for interval in ordered[1:]:
        last = merged[-1]
        if interval.start <= last.end + max_gap:
            merged[-1] = Interval(last.start, max(last.end, interval.end))
        else:
            merged.append(interval)
    return merged


def subtract_intervals(speech: list[Interval], dirty: list[Interval]) -> list[Interval]:
    """Legacy-only destructive overlap handling.

    Kept for an explicit migration escape hatch.  Normal operation preserves
    overlap in the audio and records it as review evidence instead.
    """
    dirty = merge_intervals(dirty)
    clean: list[Interval] = []
    for speech_interval in speech:
        remaining = [speech_interval]
        for dirty_interval in dirty:
            next_remaining: list[Interval] = []
            for interval in remaining:
                if dirty_interval.end <= interval.start or dirty_interval.start >= interval.end:
                    next_remaining.append(interval)
                    continue
                if dirty_interval.start > interval.start:
                    next_remaining.append(Interval(interval.start, min(dirty_interval.start, interval.end)))
                if dirty_interval.end < interval.end:
                    next_remaining.append(Interval(max(dirty_interval.end, interval.start), interval.end))
            remaining = next_remaining
        clean.extend(remaining)
    return [interval for interval in clean if interval.duration > 0]


def _config_float(config: PipelineConfig, name: str) -> float:
    value = float(getattr(config, name))
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _source_duration(source: NormalizedAudio, intervals: list[Interval]) -> float:
    if source.duration_seconds is not None:
        if source.duration_seconds < 0:
            raise ValueError(f"Negative source duration for {source.normalized_path}")
        return source.duration_seconds
    # Direct unit-test callers from older releases did not provide duration.
    # Their known VAD evidence is still a safe upper bound, but production
    # normalization always provides the actual FFmpeg-normalized duration.
    return max((interval.end for interval in intervals), default=0.0)


def _clip(interval: Interval, duration: float) -> Interval | None:
    start = max(0.0, interval.start)
    end = min(duration, interval.end)
    if end <= start:
        return None
    return Interval(start, end)


def _clip_all(intervals: list[Interval], duration: float) -> list[Interval]:
    return [clipped for interval in intervals if (clipped := _clip(interval, duration)) is not None]


def _intersections(intervals: list[Interval], window: Interval) -> tuple[Interval, ...]:
    result = []
    for interval in intervals:
        start = max(interval.start, window.start)
        end = min(interval.end, window.end)
        if end > start:
            result.append(Interval(start, end))
    return tuple(result)


def _bridged_groups(intervals: list[Interval], max_gap: float) -> list[list[Interval]]:
    """Return VAD islands joined by a short, preserved source-timeline gap."""
    if not intervals:
        return []
    groups: list[list[Interval]] = [[intervals[0]]]
    for interval in intervals[1:]:
        group = groups[-1]
        if interval.start - group[-1].end <= max_gap:
            group.append(interval)
        else:
            groups.append([interval])
    return groups


@dataclass(frozen=True)
class _Window:
    start: float
    end: float
    context_before_seconds: float = 0.0
    context_after_seconds: float = 0.0

    @property
    def needs_boundary_review(self) -> bool:
        return self.context_before_seconds > 0 or self.context_after_seconds > 0


def _silence_candidates(
    speech: list[Interval],
    start: float,
    end: float,
    minimum_silence: float,
) -> list[tuple[float, float]]:
    """Return (silence midpoint, duration) candidates strictly inside a window."""
    candidates: list[tuple[float, float]] = []
    for left, right in zip(speech, speech[1:]):
        gap_start = max(start, left.end)
        gap_end = min(end, right.start)
        gap = gap_end - gap_start
        if gap >= minimum_silence:
            candidates.append(((gap_start + gap_end) / 2, gap))
    return candidates


def _choose_silence_cut(
    speech: list[Interval],
    start: float,
    hard_end: float,
    target_seconds: float,
    min_chunk_seconds: float,
    search_seconds: float,
    min_silence_seconds: float,
    group_end: float,
) -> float | None:
    ideal = min(start + target_seconds, hard_end)
    earliest = max(start + min_chunk_seconds, ideal - search_seconds)
    candidates = [
        candidate
        for candidate in _silence_candidates(speech, start, hard_end, min_silence_seconds)
        if earliest <= candidate[0] < hard_end and group_end - candidate[0] >= min_chunk_seconds
    ]
    if not candidates:
        return None
    # Prefer the target duration; for a tie, prefer the longer actual silence
    # and then the earlier deterministic boundary.
    return min(candidates, key=lambda item: (abs(item[0] - ideal), -item[1], item[0]))[0]


def _split_group(
    speech: list[Interval],
    config: PipelineConfig,
) -> list[_Window]:
    """Split a bridged VAD group while retaining every source-time sample.

    When no real silence exists before the hard maximum, adjacent chunks share
    bounded context.  Both rows are flagged for review/training filtering so
    the overlap cannot become an invisible duplicate training target.
    """
    if not speech:
        return []
    max_seconds = _config_float(config, "max_chunk_seconds")
    target_seconds = _config_float(config, "target_chunk_seconds")
    min_chunk_seconds = _config_float(config, "min_chunk_seconds")
    search_seconds = _config_float(config, "boundary_search_seconds")
    min_silence_seconds = _config_float(config, "boundary_min_silence_seconds")
    context_seconds = _config_float(config, "fallback_context_seconds")
    if not 0 < target_seconds <= max_seconds:
        raise ValueError("target_chunk_seconds must be positive and no greater than max_chunk_seconds")
    if context_seconds >= max_seconds:
        raise ValueError("fallback_context_seconds must be less than max_chunk_seconds")

    windows: list[_Window] = []
    cursor = speech[0].start
    group_end = speech[-1].end
    context_before = 0.0
    while group_end - cursor > max_seconds:
        hard_end = cursor + max_seconds
        cut = _choose_silence_cut(
            speech,
            cursor,
            hard_end,
            target_seconds,
            min_chunk_seconds,
            search_seconds,
            min_silence_seconds,
            group_end,
        )
        if cut is not None:
            windows.append(_Window(cursor, cut, context_before_seconds=context_before))
            cursor = cut
            context_before = 0.0
            continue

        # No trustworthy silence exists.  Keep the hard bound for predictable
        # labeler cost, but give the adjacent clip enough source context to
        # avoid silently amputating a word at the handoff.
        overlap = min(context_seconds, max_seconds / 2)
        next_cursor = hard_end - overlap
        if next_cursor <= cursor:
            raise RuntimeError("Fallback context did not advance segmentation")
        windows.append(
            _Window(
                cursor,
                hard_end,
                context_before_seconds=context_before,
                context_after_seconds=overlap,
            )
        )
        cursor = next_cursor
        context_before = overlap

    windows.append(_Window(cursor, group_end, context_before_seconds=context_before))
    return windows


def build_chunks(
    source: NormalizedAudio,
    speech: list[Interval],
    dirty: list[Interval],
    config: PipelineConfig,
) -> list[Chunk]:
    """Build contiguous, source-bounded windows for offline labeling.

    VAD is evidence for selecting a window, never a request to concatenate
    speech-only audio.  Overlap remains audible by default and is attached to
    metadata so a reviewer can make the appropriate decision.
    """
    source_duration = _source_duration(source, [*speech, *dirty])
    # Padding can make VAD evidence overlap. Normalize it before using the
    # final interval as a group boundary; otherwise nested evidence could make
    # a later source-time span disappear from a window.
    detected_speech = merge_intervals(_clip_all(speech, source_duration))
    overlap_evidence = merge_intervals(_clip_all(dirty, source_duration))
    if not detected_speech:
        return []

    overlap_mode = str(config.overlap_mode).lower()
    if overlap_mode not in {"preserve", "legacy-exclude"}:
        raise ValueError("overlap_mode must be 'preserve' or 'legacy-exclude'")
    speech_for_windows = (
        subtract_intervals(detected_speech, overlap_evidence)
        if overlap_mode == "legacy-exclude"
        else detected_speech
    )
    if not speech_for_windows:
        return []

    bridge_gap = _config_float(config, "merge_gap_seconds")
    windows = [window for group in _bridged_groups(speech_for_windows, bridge_gap) for window in _split_group(group, config)]

    chunk_dir = config.work_dir / "chunks" / source.source_id
    chunks: list[Chunk] = []
    for index, item in enumerate(windows):
        window = Interval(item.start, item.end)
        evidence = _intersections(detected_speech, window)
        overlaps = _intersections(overlap_evidence, window)
        chunks.append(
            Chunk(
                source=source,
                # A single interval is an explicit compatibility guard: no
                # exporter can interpret this as a concat instruction.
                intervals=(window,),
                speech_intervals=evidence,
                overlap_intervals=overlaps,
                context_overlap_before_seconds=item.context_before_seconds,
                context_overlap_after_seconds=item.context_after_seconds,
                training_eligible=not item.needs_boundary_review,
                segmentation_version=SEGMENTATION_VERSION,
                segmentation_config_id=segmentation_config_id(config),
                chunk_path=chunk_dir / f"{source.source_id}_{index:05d}.wav",
                index=index,
            )
        )
    return chunks
