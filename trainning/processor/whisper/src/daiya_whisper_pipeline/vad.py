from __future__ import annotations

from pathlib import Path

import numpy as np
from silero_vad import get_speech_timestamps, load_silero_vad
import soundfile as sf
import torch

from .config import PipelineConfig
from .types import Interval


class SileroVad:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.device = torch.device(config.torch_device if torch.cuda.is_available() else "cpu")
        self.model = load_silero_vad()
        self.model.to(self.device)

    def detect(self, audio_path: Path) -> list[Interval]:
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        if int(sample_rate) != self.config.sample_rate:
            raise ValueError(f"Expected {self.config.sample_rate} Hz audio, got {sample_rate} Hz: {audio_path}")
        mono = np.ascontiguousarray(samples.mean(axis=1))
        wav = torch.from_numpy(mono).to(self.device)
        timestamps = get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=self.config.sample_rate,
            threshold=self.config.vad_threshold,
            min_speech_duration_ms=self.config.vad_min_speech_ms,
            min_silence_duration_ms=self.config.vad_min_silence_ms,
            speech_pad_ms=self.config.vad_speech_pad_ms,
            return_seconds=True,
        )
        return [Interval(float(item["start"]), float(item["end"])) for item in timestamps]
