from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from .config import PipelineConfig
from .types import Interval


def _load_pipeline(model_id: str, token: str | None) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\s*torchcodec is not installed correctly.*",
            category=UserWarning,
        )
        from pyannote.audio import Pipeline

    parameters = inspect.signature(Pipeline.from_pretrained).parameters
    if "token" in parameters:
        return Pipeline.from_pretrained(model_id, token=token)
    return Pipeline.from_pretrained(model_id, use_auth_token=token)


def _load_waveform(audio_path: Path) -> dict[str, Any]:
    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T))
    return {"waveform": waveform, "sample_rate": int(sample_rate), "uri": audio_path.stem}


def _overlap_intervals(annotation: Any, pad_seconds: float) -> list[Interval]:
    events: list[tuple[float, int]] = []
    for segment, _, _ in annotation.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        if end <= start:
            continue
        events.append((start, 1))
        events.append((end, -1))

    intervals: list[Interval] = []
    active = 0
    previous_time: float | None = None
    for time, delta in sorted(events, key=lambda event: (event[0], event[1])):
        if previous_time is not None and time > previous_time and active > 1:
            start = max(0.0, previous_time - pad_seconds)
            end = time + pad_seconds
            if intervals and start <= intervals[-1].end:
                intervals[-1] = Interval(intervals[-1].start, max(intervals[-1].end, end))
            else:
                intervals.append(Interval(start, end))
        active += delta
        previous_time = time

    return intervals


class OverlapDetector:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.pipeline = None
        if config.enable_overlap_filter:
            token = config.pyannote_auth_token or None
            self.pipeline = _load_pipeline(config.pyannote_overlap_model, token)
            device = torch.device(config.torch_device if torch.cuda.is_available() else "cpu")
            self.pipeline.to(device)

    def detect(self, audio_path: Path) -> list[Interval]:
        if self.pipeline is None:
            return []

        output = self.pipeline(_load_waveform(audio_path))
        annotation = getattr(output, "speaker_diarization", output)
        return _overlap_intervals(annotation, self.config.overlap_pad_seconds)
