from __future__ import annotations

from pathlib import Path

from .types import Chunk, Interval, NormalizedAudio
from .config import PipelineConfig


def merge_intervals(intervals: list[Interval], max_gap: float = 0.0) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: item.start)
    merged = [ordered[0]]
    for interval in ordered[1:]:
        last = merged[-1]
        if interval.start <= last.end + max_gap:
            merged[-1] = Interval(last.start, max(last.end, interval.end))
        else:
            merged.append(interval)
    return merged


def subtract_intervals(speech: list[Interval], dirty: list[Interval]) -> list[Interval]:
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


def split_long_interval(interval: Interval, max_seconds: float) -> list[Interval]:
    if interval.duration <= max_seconds:
        return [interval]
    parts: list[Interval] = []
    cursor = interval.start
    while cursor < interval.end:
        end = min(cursor + max_seconds, interval.end)
        parts.append(Interval(cursor, end))
        cursor = end
    return parts


def build_chunks(
    source: NormalizedAudio,
    speech: list[Interval],
    dirty: list[Interval],
    config: PipelineConfig,
) -> list[Chunk]:
    clean = subtract_intervals(speech, dirty)
    clean = [part for interval in clean for part in split_long_interval(interval, config.max_chunk_seconds)]
    clean = [interval for interval in clean if interval.duration >= config.min_chunk_seconds]

    groups: list[list[Interval]] = []
    current: list[Interval] = []
    current_duration = 0.0

    for interval in clean:
        if not current:
            current = [interval]
            current_duration = interval.duration
            continue

        gap = interval.start - current[-1].end
        proposed_duration = current_duration + interval.duration
        proposed_wall = interval.end - current[0].start
        can_merge = (
            gap <= config.merge_gap_seconds
            and proposed_duration <= config.target_chunk_seconds
            and proposed_wall <= config.max_chunk_seconds
        )

        if can_merge:
            current.append(interval)
            current_duration = proposed_duration
        else:
            groups.append(current)
            current = [interval]
            current_duration = interval.duration

    if current:
        groups.append(current)

    chunk_dir = config.work_dir / "chunks" / source.source_id
    chunks: list[Chunk] = []
    for idx, group in enumerate(groups):
        speech_duration = sum(interval.duration for interval in group)
        if speech_duration < config.min_chunk_seconds:
            continue
        chunk_path = chunk_dir / f"{source.source_id}_{idx:05d}.wav"
        chunks.append(Chunk(source=source, intervals=tuple(group), chunk_path=chunk_path, index=idx))
    return chunks
