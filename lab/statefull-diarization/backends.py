from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from timeline import SpeakerSegment


@dataclass(frozen=True)
class AudioWindow:
    index: int
    waveform: torch.Tensor
    sample_rate: int
    start: float
    end: float


@dataclass(frozen=True)
class EmitRegion:
    start: float
    end: float


@dataclass(frozen=True)
class DiarizationWindowResult:
    window: AudioWindow
    segments: list[SpeakerSegment]
    local_labels: list[str]
    local_centroids: np.ndarray
    speech_seconds: dict[str, float]
    pipeline_started_at: float
    pipeline_finished_at: float

    @property
    def pipeline_runtime_seconds(self) -> float:
        return self.pipeline_finished_at - self.pipeline_started_at


class DiarizationBackend(Protocol):
    def process_window(self, window: AudioWindow) -> DiarizationWindowResult:
        ...


class PyannotePipelineBackend:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def process_window(self, window: AudioWindow) -> DiarizationWindowResult:
        started = time.perf_counter()
        output = self.pipeline(
            {
                "waveform": window.waveform,
                "sample_rate": window.sample_rate,
                "uri": f"window-{window.index}",
            }
        )
        finished = time.perf_counter()

        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = output.speaker_diarization

        local_labels = list(output.speaker_diarization.labels())
        local_centroids = np.asarray(getattr(output, "speaker_embeddings", np.empty((0, 0))))
        segments = annotation_to_segments(annotation, offset=window.start)

        return DiarizationWindowResult(
            window=window,
            segments=segments,
            local_labels=local_labels,
            local_centroids=local_centroids,
            speech_seconds=speech_seconds(annotation),
            pipeline_started_at=started,
            pipeline_finished_at=finished,
        )


class DiartBackend:
    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(
            "DiartBackend is a benchmark placeholder. Install/wire diart before use."
        )


class DaiyaOnlineBackend:
    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("DaiyaOnlineBackend is reserved for a future custom backend.")


def speech_seconds(annotation) -> dict[str, float]:
    totals: dict[str, float] = {}
    for turn, _, label in annotation.itertracks(yield_label=True):
        totals[label] = totals.get(label, 0.0) + max(0.0, float(turn.duration))
    return totals


def annotation_to_segments(annotation, offset: float = 0.0) -> list[SpeakerSegment]:
    segments: list[SpeakerSegment] = []
    for turn, _, label in annotation.itertracks(yield_label=True):
        segments.append(
            SpeakerSegment(
                start=offset + float(turn.start),
                end=offset + float(turn.end),
                speaker_id=str(label),
                speaker_confidence=1.0,
                local_label=str(label),
            )
        )
    return segments


def crop_segments(
    segments: list[SpeakerSegment],
    start: float,
    end: float,
) -> list[SpeakerSegment]:
    cropped = []
    for segment in segments:
        clipped_start = max(start, segment.start)
        clipped_end = min(end, segment.end)
        if clipped_end <= clipped_start:
            continue
        cropped.append(
            SpeakerSegment(
                start=clipped_start,
                end=clipped_end,
                speaker_id=segment.speaker_id,
                speaker_confidence=segment.speaker_confidence,
                local_label=segment.local_label,
                evidence_id=segment.evidence_id,
            )
        )
    return cropped
