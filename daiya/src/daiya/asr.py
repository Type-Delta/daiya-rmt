from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .audio import PCMChunk, SAMPLE_RATE, ensure_mono_float32
from .mux import ASRSegment, WordTimestamp


class ASRUnavailableError(RuntimeError):
    """Raised when an optional ASR backend is requested but not installed/configured."""


DECODING_POLICIES = ("baseline", "short_beam", "short_greedy")
DEFAULT_SHORT_UTTERANCE_SECONDS = 3.0


def validate_decoding_policy(policy: str, short_utterance_seconds: float) -> None:
    if policy not in DECODING_POLICIES:
        choices = ", ".join(DECODING_POLICIES)
        raise ValueError(f"unknown ASR decoding policy {policy!r}; expected one of: {choices}")
    if not math.isfinite(short_utterance_seconds) or short_utterance_seconds <= 0:
        raise ValueError("ASR short-utterance threshold must be finite and greater than zero")


def decoder_options_for_duration(
    policy: str,
    duration: float,
    *,
    short_utterance_seconds: float = DEFAULT_SHORT_UTTERANCE_SECONDS,
) -> dict[str, object]:
    """Return fresh faster-whisper options for one original utterance."""
    validate_decoding_policy(policy, short_utterance_seconds)
    # Sparse fallback ladder: short-loop failures need ~0.8 to recover, while
    # intermediate rungs add full decode passes without helping observed clips.
    options: dict[str, object] = {"temperature": [0.0, 0.8, 1.0]}
    if duration <= short_utterance_seconds:
        if policy == "short_beam":
            options.update(beam_size=8, patience=1.2)
        elif policy == "short_greedy":
            options.update(beam_size=1, best_of=1)
    return options


@dataclass(frozen=True)
class Utterance:
    samples: np.ndarray
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class EnergyUtteranceSegmenter:
    """Small dependency-free VAD fallback for tests and no-torch environments."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = 0.012,
        min_speech_seconds: float = 0.25,
        trailing_silence_seconds: float = 0.45,
        max_utterance_seconds: float = 8.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_seconds = min_speech_seconds
        self.trailing_silence_seconds = trailing_silence_seconds
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
    """Placeholder hook for Silero VAD without making torch a hard dependency."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ASRUnavailableError("Silero VAD requires the optional 'vad' extra") from exc
        raise NotImplementedError("Silero VAD integration is not wired in this prototype yet")


def create_utterance_segmenter(*, prefer_silero: bool = False, **kwargs: object) -> EnergyUtteranceSegmenter:
    if prefer_silero:
        try:
            return SileroUtteranceSegmenter(**kwargs)  # type: ignore[return-value]
        except (ASRUnavailableError, NotImplementedError):
            pass
    return EnergyUtteranceSegmenter(**kwargs)


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
        decoding_policy: str = "baseline",
        short_utterance_seconds: float = DEFAULT_SHORT_UTTERANCE_SECONDS,
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
        validate_decoding_policy(decoding_policy, short_utterance_seconds)
        self.decoding_policy = decoding_policy
        self.short_utterance_seconds = short_utterance_seconds
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
        decoder_options = decoder_options_for_duration(
            self.decoding_policy,
            utterance.duration,
            short_utterance_seconds=self.short_utterance_seconds,
        )
        segments, _info = self._model.transcribe(
            audio,
            language=language or self.language,
            initial_prompt=initial_prompt or self.initial_prompt,
            word_timestamps=True,
            vad_filter=False,
            **decoder_options,
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
