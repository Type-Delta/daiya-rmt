from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .asr import ASRUnavailableError, FasterWhisperASR, NullASR, create_utterance_segmenter
from .audio import PCMChunk
from .correct import NoOpCorrectionStage
from .diarizer import DiarizerConfig, create_diarizer
from .mux import TranscriptEvent, TranscriptMultiplexer


@dataclass(frozen=True)
class PipelineConfig:
    asr_model: str | None = None
    asr_device: str = "auto"
    asr_compute_type: str = "int8_float16"
    language: str | None = None
    initial_prompt: str | None = None
    vad_threshold: float = 0.012
    utterance_cap_seconds: float = 8.0
    diarization_profile: str = "balanced"
    diarization_commit_delay_seconds: float = 0.0
    window_seconds: float | None = None
    hop_seconds: float | None = None
    latency_seconds: float | None = None
    commit_delay_seconds: float | None = None
    match_threshold: float | None = None


class StreamingPipeline:
    """Shared chunk-to-transcript pipeline for CLI, replay, and live server paths."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = _with_runtime_defaults(config or PipelineConfig())
        self.mux = TranscriptMultiplexer()
        self.segmenter = create_utterance_segmenter(
            threshold=self.config.vad_threshold,
            max_utterance_seconds=self.config.utterance_cap_seconds,
        )
        self.diarizer = create_diarizer(
            config=DiarizerConfig(
                profile=self.config.diarization_profile,
                window_seconds=self.config.window_seconds,
                hop_seconds=self.config.hop_seconds,
                latency_seconds=self.config.latency_seconds,
                commit_delay_seconds=(
                    self.config.commit_delay_seconds
                    if self.config.commit_delay_seconds is not None
                    else self.config.diarization_commit_delay_seconds
                ),
            )
        )
        self.corrector = NoOpCorrectionStage()
        self.asr = self._create_asr()

    def accept_chunk(self, chunk: PCMChunk) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for event in self.mux.ingest_diarization_many(self.diarizer.accept(chunk)):
            payloads.extend(self._serialize_transcript_event(event))

        for utterance in self.segmenter.accept(chunk):
            payloads.extend(self._transcribe_utterance(utterance))
        return payloads

    def flush(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for utterance in self.segmenter.flush():
            payloads.extend(self._transcribe_utterance(utterance))
        for event in self.mux.ingest_diarization_many(self.diarizer.flush()):
            payloads.extend(self._serialize_transcript_event(event))
        return payloads

    def _create_asr(self) -> FasterWhisperASR | NullASR:
        if not self.config.asr_model:
            return NullASR("ASR model is not configured; pass --asr-model or send asr_model")
        try:
            return FasterWhisperASR(
                self.config.asr_model,
                device=self.config.asr_device,
                compute_type=self.config.asr_compute_type,
                language=self.config.language,
                initial_prompt=self.config.initial_prompt,
            )
        except ASRUnavailableError as exc:
            return NullASR(str(exc))

    def _transcribe_utterance(self, utterance: object) -> list[dict[str, Any]]:
        try:
            asr_segments = self.asr.transcribe_utterance(
                utterance,  # type: ignore[arg-type]
                language=self.config.language,
                initial_prompt=self.config.initial_prompt,
            )
        except ASRUnavailableError as exc:
            return [
                {
                    "type": "error",
                    "source": "asr",
                    "message": str(exc),
                    "utterance_start": getattr(utterance, "start", None),
                    "utterance_end": getattr(utterance, "end", None),
                }
            ]

        payloads: list[dict[str, Any]] = []
        for segment in asr_segments:
            if not segment.text:
                continue
            for event in self.mux.ingest_asr(segment):
                payloads.extend(self._serialize_transcript_event(event))
                payloads.extend(self._apply_corrections(event))
        return payloads

    def _serialize_transcript_event(self, event: TranscriptEvent) -> list[dict[str, Any]]:
        return [event.to_dict()]

    def _apply_corrections(self, event: TranscriptEvent) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for correction in self.corrector.review(event.segment):
            for update in self.mux.apply_correction(correction):
                payloads.append(update.to_dict())
        return payloads


def _with_runtime_defaults(config: PipelineConfig) -> PipelineConfig:
    updates: dict[str, object] = {}
    if not config.asr_model:
        updates["asr_model"] = default_asr_model()
    if config.asr_device == "auto":
        updates["asr_device"] = os.getenv("DAIYA_ASR_DEVICE", config.asr_device)
    if config.asr_compute_type == "int8_float16":
        updates["asr_compute_type"] = os.getenv("DAIYA_ASR_COMPUTE_TYPE", config.asr_compute_type)
    if config.language is None:
        updates["language"] = os.getenv("DAIYA_ASR_LANGUAGE") or None
    if config.initial_prompt is None:
        updates["initial_prompt"] = os.getenv("DAIYA_ASR_INITIAL_PROMPT") or None
    return replace(config, **updates) if updates else config


def default_asr_model() -> str:
    configured = os.getenv("DAIYA_ASR_MODEL")
    if configured:
        return configured

    local_ct2 = _repo_root() / "trainning" / "whisper" / "runs" / "medium-real-iter4-ct2-int8_float16"
    if (local_ct2 / "model.bin").exists():
        return str(local_ct2)

    return "medium"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
