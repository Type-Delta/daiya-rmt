from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np

from .audio import PCMChunk, SAMPLE_RATE, ensure_mono_float32
from .mux import ASRSegment, WordTimestamp

LOGGER = logging.getLogger(__name__)


class ASRUnavailableError(RuntimeError):
    """Raised when an optional ASR backend is requested but not installed/configured."""


@dataclass(frozen=True)
class Utterance:
    samples: np.ndarray
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class UtteranceSegmenter(Protocol):
    def accept(self, chunk: PCMChunk) -> list[Utterance]:
        ...

    def flush(self) -> list[Utterance]:
        ...


SpeechTimestamp = dict[str, int | float]
SpeechTimestampFn = Callable[..., Sequence[SpeechTimestamp]]


class EnergyUtteranceSegmenter:
    """Small dependency-free VAD fallback for tests and no-torch environments."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = 0.012,
        min_speech_seconds: float = 0.25,
        trailing_silence_seconds: float = 0.45,
        min_silence_seconds: float | None = None,
        speech_padding_seconds: float = 0.0,
        max_utterance_seconds: float = 8.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_seconds = min_speech_seconds
        self.trailing_silence_seconds = (
            trailing_silence_seconds if min_silence_seconds is None else min_silence_seconds
        )
        self.speech_padding_seconds = speech_padding_seconds
        self.max_utterance_seconds = max_utterance_seconds
        self._buffer: list[np.ndarray] = []
        self._start: float | None = None
        self._last_voice_end: float | None = None

    def accept(self, chunk: PCMChunk) -> list[Utterance]:
        samples = ensure_mono_float32(chunk.samples)
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        voiced = rms >= self.threshold
        completed: list[Utterance] = []

        if voiced and self._start is None:
            self._start = chunk.start_time
        if self._start is not None:
            self._buffer.append(samples.copy())
        if voiced:
            self._last_voice_end = chunk.end_time

        if self._start is None:
            return completed

        utterance_duration = chunk.end_time - self._start
        silence_duration = (
            0.0
            if self._last_voice_end is None
            else max(0.0, chunk.end_time - self._last_voice_end)
        )
        hit_silence = silence_duration >= self.trailing_silence_seconds
        hit_cap = utterance_duration >= self.max_utterance_seconds
        if hit_silence or hit_cap:
            utterance = self._drain(end=self._last_voice_end or chunk.end_time)
            if utterance.duration >= self.min_speech_seconds:
                completed.append(utterance)
        return completed

    def flush(self) -> list[Utterance]:
        if self._start is None:
            return []
        utterance = self._drain(end=self._last_voice_end or self._start)
        if utterance.duration < self.min_speech_seconds:
            return []
        return [utterance]

    def _drain(self, *, end: float) -> Utterance:
        start = self._start or 0.0
        samples = np.concatenate(self._buffer) if self._buffer else np.empty(0, dtype=np.float32)
        wanted = max(0, int(round((end - start) * self.sample_rate)))
        samples = samples[:wanted] if wanted else samples
        self._buffer = []
        self._start = None
        self._last_voice_end = None
        return Utterance(samples=samples, start=start, end=end)


class SileroUtteranceSegmenter:
    """Silero VAD utterance segmenter with lazy optional dependency loading."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = 0.5,
        min_speech_seconds: float = 0.25,
        min_silence_seconds: float | None = None,
        trailing_silence_seconds: float = 0.45,
        speech_padding_seconds: float = 0.1,
        max_utterance_seconds: float = 8.0,
        model: object | None = None,
        get_speech_timestamps: SpeechTimestampFn | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_seconds = min_speech_seconds
        self.min_silence_seconds = (
            trailing_silence_seconds if min_silence_seconds is None else min_silence_seconds
        )
        self.speech_padding_seconds = max(0.0, speech_padding_seconds)
        self.max_utterance_seconds = max_utterance_seconds
        self._model = model
        self._get_speech_timestamps = get_speech_timestamps
        if self._get_speech_timestamps is None:
            self._model, self._get_speech_timestamps = _load_silero_vad()
        self._buffer: list[np.ndarray] = []
        self._pre_roll = np.empty(0, dtype=np.float32)
        self._start: float | None = None
        self._last_voice_end: float | None = None

    def accept(self, chunk: PCMChunk) -> list[Utterance]:
        samples = ensure_mono_float32(chunk.samples)
        timestamps = self._detect_speech(samples)
        completed: list[Utterance] = []

        if timestamps:
            first_start = min(_timestamp_sample(item, "start", self.sample_rate) for item in timestamps)
            last_end = max(_timestamp_sample(item, "end", self.sample_rate) for item in timestamps)
            first_start = max(0, min(first_start, samples.size))
            last_end = max(first_start, min(last_end, samples.size))
            if self._start is None:
                self._start_utterance(chunk, samples, first_start)
            elif samples.size:
                self._buffer.append(samples.copy())
            self._last_voice_end = chunk.start_time + (last_end / self.sample_rate)
        elif self._start is not None and samples.size:
            self._buffer.append(samples.copy())

        if self._start is not None:
            utterance_duration = chunk.end_time - self._start
            silence_duration = (
                0.0
                if self._last_voice_end is None
                else max(0.0, chunk.end_time - self._last_voice_end)
            )
            hit_silence = silence_duration >= self.min_silence_seconds
            hit_cap = utterance_duration >= self.max_utterance_seconds
            if hit_silence or hit_cap:
                end = (
                    min(chunk.end_time, (self._last_voice_end or chunk.end_time) + self.speech_padding_seconds)
                    if hit_silence
                    else min(chunk.end_time, self._start + self.max_utterance_seconds)
                )
                utterance = self._drain(end=end)
                if utterance.duration >= self.min_speech_seconds:
                    completed.append(utterance)

        if self._start is None:
            self._remember_pre_roll(samples)
        return completed

    def flush(self) -> list[Utterance]:
        if self._start is None:
            self._pre_roll = np.empty(0, dtype=np.float32)
            return []
        utterance = self._drain(end=self._last_voice_end or self._start)
        if utterance.duration < self.min_speech_seconds:
            return []
        return [utterance]

    def _detect_speech(self, samples: np.ndarray) -> Sequence[SpeechTimestamp]:
        if samples.size == 0:
            return ()
        get_speech_timestamps = self._get_speech_timestamps
        if get_speech_timestamps is None:
            raise ASRUnavailableError("Silero VAD detector is not configured")
        return get_speech_timestamps(
            samples,
            self._model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=int(round(self.min_speech_seconds * 1000)),
            min_silence_duration_ms=int(round(self.min_silence_seconds * 1000)),
            speech_pad_ms=0,
            return_seconds=False,
        )

    def _start_utterance(self, chunk: PCMChunk, samples: np.ndarray, first_start: int) -> None:
        pad_samples = int(round(self.speech_padding_seconds * self.sample_rate))
        start_index = max(0, first_start - pad_samples)
        missing_pad = max(0, pad_samples - first_start)
        prefix = self._pre_roll[-missing_pad:] if missing_pad else np.empty(0, dtype=np.float32)
        current = samples[start_index:].copy() if samples.size else np.empty(0, dtype=np.float32)
        self._buffer = [part for part in (prefix.copy(), current) if part.size]
        self._start = chunk.start_time + (start_index / self.sample_rate) - (prefix.size / self.sample_rate)
        self._pre_roll = np.empty(0, dtype=np.float32)

    def _remember_pre_roll(self, samples: np.ndarray) -> None:
        pad_samples = int(round(self.speech_padding_seconds * self.sample_rate))
        if pad_samples <= 0 or samples.size == 0:
            self._pre_roll = np.empty(0, dtype=np.float32)
            return
        combined = np.concatenate([self._pre_roll, samples])
        self._pre_roll = combined[-pad_samples:].astype(np.float32, copy=False)

    def _drain(self, *, end: float) -> Utterance:
        start = self._start or 0.0
        samples = np.concatenate(self._buffer) if self._buffer else np.empty(0, dtype=np.float32)
        wanted = max(0, int(round((end - start) * self.sample_rate)))
        samples = samples[:wanted] if wanted else samples
        self._buffer = []
        self._pre_roll = np.empty(0, dtype=np.float32)
        self._start = None
        self._last_voice_end = None
        return Utterance(samples=samples, start=start, end=end)


def create_utterance_segmenter(
    *,
    backend: str = "energy",
    prefer_silero: bool = False,
    **kwargs: object,
) -> UtteranceSegmenter:
    selected = backend.lower()
    if selected not in {"energy", "silero", "auto"}:
        raise ValueError(f"unknown utterance segmenter backend: {backend}")
    if prefer_silero and selected == "energy":
        selected = "auto"
    if selected in {"silero", "auto"}:
        try:
            return SileroUtteranceSegmenter(**kwargs)
        except ASRUnavailableError as exc:
            if selected == "silero":
                LOGGER.warning("Silero VAD unavailable; falling back to energy segmenter: %s", exc)
            pass
    return EnergyUtteranceSegmenter(**kwargs)


def _load_silero_vad() -> tuple[object, SpeechTimestampFn]:
    try:
        import torch
    except ImportError as exc:
        raise ASRUnavailableError("Silero VAD requires the optional 'vad' extra") from exc
    try:
        model, utils = torch.hub.load(  # type: ignore[attr-defined]
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
    except Exception as exc:
        raise ASRUnavailableError(f"failed to load Silero VAD: {exc}") from exc
    raw_get_speech_timestamps = utils[0]

    def get_speech_timestamps(audio: np.ndarray, vad_model: object, **kwargs: object) -> Sequence[SpeechTimestamp]:
        tensor = torch.as_tensor(audio, dtype=torch.float32)
        return raw_get_speech_timestamps(tensor, vad_model, **kwargs)

    return model, get_speech_timestamps


def _timestamp_sample(timestamp: SpeechTimestamp, key: str, sample_rate: int) -> int:
    value = timestamp.get(key, 0)
    if isinstance(value, float):
        return int(round(value * sample_rate))
    return int(value)


class FasterWhisperASR:
    """Runtime wrapper around faster-whisper, imported only when configured."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        compute_type: str = "int8_float16",
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> None:
        if not model_path:
            raise ASRUnavailableError("faster-whisper model_path is required")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRUnavailableError(
                "faster-whisper is not installed; sync the daiya package dependencies or disable ASR"
            ) from exc

        self.model_path = model_path
        self.language = language
        self.initial_prompt = initial_prompt
        try:
            self._model = WhisperModel(model_path, device=device, compute_type=compute_type)
        except Exception as exc:
            raise ASRUnavailableError(f"failed to load faster-whisper model {model_path!r}: {exc}") from exc

    def transcribe_utterance(
        self,
        utterance: Utterance,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> list[ASRSegment]:
        audio = ensure_mono_float32(utterance.samples)
        if audio.size == 0:
            return []
        segments, _info = self._model.transcribe(
            audio,
            language=language or self.language,
            initial_prompt=initial_prompt or self.initial_prompt,
            word_timestamps=True,
            vad_filter=False,
            # ponytail: sparse temperature ladder instead of the default 6-rung one.
            # Degenerate greedy loops on short utterances need temp ~0.8 to recover;
            # the intermediate rungs (0.2-0.6) never rescue, they just add full decode
            # passes (~16s stall on a 1s clip). Normal clips never leave temp 0.
            temperature=[0.0, 0.8, 1.0],
        )
        return list(_convert_fw_segments(segments, offset=utterance.start))


class NullASR:
    """No-model ASR backend that fails at runtime with a useful message."""

    def __init__(self, reason: str = "ASR model is not configured") -> None:
        self.reason = reason

    def transcribe_utterance(self, utterance: Utterance, **_kwargs: object) -> list[ASRSegment]:
        del utterance
        raise ASRUnavailableError(self.reason)


_THAI_GAP = re.compile(r"(?<=[฀-๿]) +(?=[฀-๿])")


def normalize_thai_spacing(text: str) -> str:
    """Collapse word-by-word spaced Thai (a fine-tune label artifact); keep normal phrase spacing."""
    thai_chars = sum("฀" <= ch <= "๿" for ch in text)
    if thai_chars < 10:
        return text
    gaps = len(_THAI_GAP.findall(text))
    if gaps / thai_chars <= 0.12:
        return text
    return _THAI_GAP.sub("", text)


def _convert_fw_segments(segments: Iterable[object], *, offset: float) -> Iterable[ASRSegment]:
    for segment in segments:
        start = offset + float(getattr(segment, "start", 0.0))
        end = offset + float(getattr(segment, "end", 0.0))
        text = normalize_thai_spacing(str(getattr(segment, "text", "")).strip())
        words = []
        for word in getattr(segment, "words", None) or []:
            words.append(
                WordTimestamp(
                    word=str(getattr(word, "word", "")).strip(),
                    start=offset + float(getattr(word, "start", 0.0)),
                    end=offset + float(getattr(word, "end", 0.0)),
                    probability=getattr(word, "probability", None),
                )
            )
        yield ASRSegment(
            start=start,
            end=end,
            text=text,
            words=tuple(words),
            language=getattr(segment, "language", None),
            confidence=getattr(segment, "avg_logprob", None),
        )
