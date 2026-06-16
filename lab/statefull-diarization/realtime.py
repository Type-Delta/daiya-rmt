from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from backends import (
    AudioWindow,
    DiarizationBackend,
    EmitRegion,
    crop_segments,
)
from metrics import HopMetrics, MetricsRecorder
from speaker_memory import CommittedSpeakerEvidence, SpeakerMemory
from timeline import SpeakerSegment, TimelineEvent, TimelineStore


@dataclass(frozen=True)
class RealtimeDiarizationConfig:
    window_seconds: float = 10.0
    hop_seconds: float = 1.0
    latency_seconds: float = 3.0
    commit_delay_seconds: float = 5.0

    @classmethod
    def for_profile(cls, name: str) -> "RealtimeDiarizationConfig":
        profiles = {
            "balanced": cls(10.0, 1.0, 3.0, 5.0),
            "fast": cls(5.0, 0.5, 1.5, 3.0),
            "accuracy": cls(18.0, 2.0, 5.0, 8.0),
        }
        try:
            return profiles[name]
        except KeyError as exc:
            known = ", ".join(sorted(profiles))
            raise ValueError(f"unknown realtime profile {name!r}; expected one of {known}") from exc


@dataclass(frozen=True)
class RealtimeHopResult:
    window: AudioWindow
    emit_region: EmitRegion
    events: list[TimelineEvent]
    metrics: HopMetrics


class RealtimeDiarizationDriver:
    def __init__(
        self,
        backend: DiarizationBackend,
        memory: SpeakerMemory,
        config: RealtimeDiarizationConfig | None = None,
        timeline: TimelineStore | None = None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self.backend = backend
        self.memory = memory
        self.config = config or RealtimeDiarizationConfig()
        self.timeline = timeline or TimelineStore()
        self.metrics = metrics
        self._evidence_vectors: dict[str, tuple[np.ndarray, str, float, float, float]] = {}

    def process_window(self, window: AudioWindow) -> RealtimeHopResult:
        result = self.backend.process_window(window)
        emit_region = self.emit_region_for(window)

        assignments = self.memory.match(
            result.local_labels,
            result.local_centroids,
            speech_seconds=result.speech_seconds,
            segment_end=window.end,
        )
        centroid_by_label = _centroid_by_label(result.local_labels, result.local_centroids)
        mapped_segments = self._map_segments(
            crop_segments(result.segments, emit_region.start, emit_region.end),
            assignments,
            centroid_by_label,
            window.index,
        )
        timeline_events, update_stats = self.timeline.update_region(
            emit_region.start,
            emit_region.end,
            mapped_segments,
        )

        commit_horizon = window.end - self.config.commit_delay_seconds
        commit_events, commit_stats = self.timeline.commit_before(commit_horizon)
        correction_events = self._commit_evidence_from_events(commit_events)
        events = [*timeline_events, *commit_events, *correction_events]

        emit_wall_time = time.perf_counter()
        metrics = HopMetrics(
            window_index=window.index,
            audio_now=window.end,
            window_start=window.start,
            window_end=window.end,
            emit_start=emit_region.start,
            emit_end=emit_region.end,
            pipeline_started_at=result.pipeline_started_at,
            pipeline_finished_at=result.pipeline_finished_at,
            pipeline_runtime_seconds=result.pipeline_runtime_seconds,
            emit_wall_time=emit_wall_time,
            emit_latency_seconds=max(0.0, window.end - emit_region.end),
            num_local_speakers=len(result.local_labels),
            num_global_speakers=len(self.memory.profiles),
            num_candidates=len(self.memory.candidates),
            num_turns_created=update_stats.created,
            num_turns_updated=update_stats.updated + update_stats.deleted + len(correction_events),
            num_turns_committed=commit_stats.committed,
            num_speaker_flips=update_stats.speaker_flips + len(correction_events),
            memory_match_count=self.memory.match_count,
            memory_update_count=self.memory.update_count,
        )
        if self.metrics is not None:
            self.metrics.record(metrics)
        return RealtimeHopResult(window, emit_region, events, metrics)

    def emit_region_for(self, window: AudioWindow) -> EmitRegion:
        emit_start = max(window.start, window.end - self.config.latency_seconds)
        emit_end = min(window.end, emit_start + self.config.hop_seconds)
        return EmitRegion(emit_start, emit_end)

    def _map_segments(
        self,
        segments: Iterable[SpeakerSegment],
        assignments,
        centroid_by_label: dict[str, np.ndarray],
        window_index: int,
    ) -> list[SpeakerSegment]:
        mapped: list[SpeakerSegment] = []
        for segment in segments:
            if segment.local_label is None:
                continue
            assignment = assignments.get(segment.local_label)
            if assignment is None or assignment.decision in {"invalid", "overlap_only"}:
                continue
            evidence_id = (
                f"window:{window_index}:label:{segment.local_label}:"
                f"{segment.start:.3f}-{segment.end:.3f}"
            )
            vector = centroid_by_label.get(segment.local_label)
            if vector is not None:
                self._evidence_vectors[evidence_id] = (
                    vector,
                    segment.local_label,
                    assignment.confidence,
                    segment.start,
                    segment.end,
                )
            mapped.append(
                SpeakerSegment(
                    start=segment.start,
                    end=segment.end,
                    speaker_id=assignment.speaker_id,
                    speaker_confidence=assignment.confidence,
                    local_label=segment.local_label,
                    evidence_id=evidence_id,
                )
            )
        return mapped

    def _commit_evidence_from_events(self, events: Iterable[TimelineEvent]) -> list[TimelineEvent]:
        corrections: list[TimelineEvent] = []
        for event in events:
            if event.type != "turn.committed":
                continue

            evidence_items: list[CommittedSpeakerEvidence] = []
            for evidence_id in event.turn.evidence_ids:
                cached = self._evidence_vectors.get(evidence_id)
                if cached is None:
                    continue
                vector, local_label, confidence, evidence_start, evidence_end = cached
                evidence_items.append(
                    CommittedSpeakerEvidence(
                        evidence_id=evidence_id,
                        speaker_id=event.turn.speaker_id,
                        vector=vector,
                        start=evidence_start,
                        end=evidence_end,
                        clean_speech_seconds=max(0.0, evidence_end - evidence_start),
                        overlap_ratio=0.0,
                        confidence=max(confidence, event.turn.speaker_confidence),
                        source="realtime_timeline",
                        local_label=local_label,
                    )
                )

            committed = self.memory.commit_evidence(evidence_items)
            actual_ids = [speaker_id for speaker_id in committed.values() if speaker_id]
            if not actual_ids:
                continue
            actual_id = actual_ids[-1]
            correction = self.timeline.correct_speaker(event.turn.turn_id, actual_id)
            if correction is not None:
                corrections.append(correction)
        return corrections


class AudioRingBuffer:
    def __init__(self, sample_rate: int, max_seconds: float, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.max_samples = max(1, int(max_seconds * sample_rate))
        self.channels = channels
        self._samples = np.empty((channels, 0), dtype=np.float32)
        self.total_samples = 0

    @property
    def duration(self) -> float:
        return self._samples.shape[1] / self.sample_rate

    @property
    def audio_now(self) -> float:
        return self.total_samples / self.sample_rate

    def append(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block[None, :]
        elif block.ndim == 2 and block.shape[0] != self.channels and block.shape[1] == self.channels:
            block = block.T
        if block.ndim != 2 or block.shape[1] == 0:
            return
        block = np.nan_to_num(block[: self.channels], nan=0.0, posinf=0.0, neginf=0.0)
        self.total_samples += block.shape[1]
        self._samples = np.concatenate([self._samples, block], axis=1)
        if self._samples.shape[1] > self.max_samples:
            self._samples = self._samples[:, -self.max_samples :]

    def latest(self, seconds: float) -> torch.Tensor:
        samples = max(1, int(seconds * self.sample_rate))
        if self._samples.shape[1] < samples:
            raise ValueError("not enough audio in ring buffer")
        return torch.from_numpy(self._samples[:, -samples:].copy())


class RollingWindowScheduler:
    def __init__(self, sample_rate: int, config: RealtimeDiarizationConfig, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.config = config
        self.ring = AudioRingBuffer(
            sample_rate=sample_rate,
            max_seconds=config.window_seconds + config.hop_seconds * 2.0,
            channels=channels,
        )
        self._next_window_end = config.window_seconds
        self._next_index = 0

    def append(self, block: np.ndarray) -> list[AudioWindow]:
        self.ring.append(block)
        windows: list[AudioWindow] = []
        while self.ring.audio_now + 1e-9 >= self._next_window_end:
            if self.ring.duration + 1e-9 < self.config.window_seconds:
                break
            waveform = self.ring.latest(self.config.window_seconds)
            window = AudioWindow(
                index=self._next_index,
                waveform=waveform,
                sample_rate=self.sample_rate,
                start=self._next_window_end - self.config.window_seconds,
                end=self._next_window_end,
            )
            windows.append(window)
            self._next_index += 1
            self._next_window_end += self.config.hop_seconds
        return windows


def iter_replay_windows(
    waveform: torch.Tensor,
    sample_rate: int,
    config: RealtimeDiarizationConfig,
    warmup: str = "full",
) -> Iterable[AudioWindow]:
    if waveform.ndim != 2:
        raise ValueError("waveform must be shaped as (channels, samples)")
    if warmup != "full":
        raise ValueError("only full-window warmup is implemented")

    total_samples = waveform.shape[1]
    window_samples = max(1, int(config.window_seconds * sample_rate))
    hop_samples = max(1, int(config.hop_seconds * sample_rate))
    index = 0
    end = window_samples
    while end <= total_samples:
        start = end - window_samples
        yield AudioWindow(
            index=index,
            waveform=waveform[:, start:end],
            sample_rate=sample_rate,
            start=start / sample_rate,
            end=end / sample_rate,
        )
        index += 1
        end += hop_samples


def run_replay(
    waveform: torch.Tensor,
    sample_rate: int,
    backend: DiarizationBackend,
    memory: SpeakerMemory,
    config: RealtimeDiarizationConfig,
    metrics_path: Path | str | None = None,
) -> tuple[SpeakerMemory, TimelineStore, str]:
    with MetricsRecorder(metrics_path) as metrics:
        driver = RealtimeDiarizationDriver(
            backend=backend,
            memory=memory,
            config=config,
            metrics=metrics,
        )
        for window in iter_replay_windows(waveform, sample_rate, config):
            hop = driver.process_window(window)
            print_events(hop.events)
        summary = metrics.summary()
    return memory, driver.timeline, summary


def print_events(events: Iterable[TimelineEvent]) -> None:
    for event in events:
        turn = event.turn
        final = "final" if turn.final else "provisional"
        print(
            f"{event.type:<15} {turn.turn_id} v{turn.version:<2} "
            f"{turn.start:7.2f}-{turn.end:7.2f}s {turn.speaker_id:<16} {final}"
        )


def _centroid_by_label(
    local_labels: list[str],
    local_centroids: np.ndarray,
) -> dict[str, np.ndarray]:
    if local_centroids is None:
        return {}
    array = np.asarray(local_centroids)
    if array.ndim != 2:
        return {}
    return {
        label: _normalize_vector(vector)
        for label, vector in zip(local_labels, array)
    }


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.nan_to_num(np.asarray(vector), nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(vector)
    if norm == 0 or not np.isfinite(norm):
        return np.zeros_like(vector)
    return vector / norm
