from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol

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

    def reset(self) -> None:
        ...


class StreamingVADIterator(Protocol):
    def __call__(
        self,
        samples: np.ndarray,
        *,
        return_seconds: bool = False,
    ) -> Mapping[str, int | float] | None:
        ...

    def reset_states(self) -> None:
        ...


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

    def reset(self) -> None:
        self._buffer = []
        self._start = None
        self._last_voice_end = None

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
    """Stateful streaming Silero VAD over fixed 512-sample inference windows."""

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
        vad_iterator: StreamingVADIterator | None = None,
    ) -> None:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Silero VAD requires {SAMPLE_RATE} Hz audio")
        if min_speech_seconds < 0:
            raise ValueError("min_speech_seconds must be non-negative")
        if max_utterance_seconds <= 0:
            raise ValueError("max_utterance_seconds must be positive")
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_seconds = min_speech_seconds
        self.min_silence_seconds = (
            trailing_silence_seconds if min_silence_seconds is None else min_silence_seconds
        )
        self.speech_padding_seconds = max(0.0, speech_padding_seconds)
        self.max_utterance_seconds = max_utterance_seconds
        self._window_samples = 512
        padding_ms = int(round(self.speech_padding_seconds * 1000))
        self._padding_samples = (self.sample_rate * padding_ms) // 1000
        self._min_speech_samples = int(round(self.min_speech_seconds * self.sample_rate))
        self._max_utterance_samples = int(round(self.max_utterance_seconds * self.sample_rate))
        self._vad_iterator = vad_iterator or _load_silero_vad_iterator(
            threshold=self.threshold,
            sample_rate=self.sample_rate,
            min_silence_seconds=self.min_silence_seconds,
            speech_padding_ms=padding_ms,
        )
        self._origin_time: float | None = None
        self._total_samples = 0
        self._pending = np.empty(0, dtype=np.float32)
        self._pending_start = 0
        self._audio = np.empty(0, dtype=np.float32)
        self._audio_start = 0
        self._speech_start: int | None = None
        self._raw_speech_start: int | None = None
        self._emitted_for_current_speech = False

    def accept(self, chunk: PCMChunk) -> list[Utterance]:
        samples = ensure_mono_float32(chunk.samples)
        completed: list[Utterance] = []
        if self._origin_time is not None and not self._is_contiguous(chunk.start_time):
            completed.extend(self.flush())
        if self._origin_time is None:
            self._origin_time = chunk.start_time
        if samples.size == 0:
            return completed

        self._append_audio(samples)
        self._pending = np.concatenate([self._pending, samples])
        while self._pending.size >= self._window_samples:
            frame = self._pending[: self._window_samples]
            self._pending = self._pending[self._window_samples :]
            frame_end = self._pending_start + self._window_samples
            self._pending_start = frame_end
            event = self._vad_iterator(frame, return_seconds=False)
            completed.extend(self._handle_event(event, limit=frame_end))
            completed.extend(self._split_at_max_duration(limit=frame_end))
        self._prune_audio()
        return completed

    def flush(self) -> list[Utterance]:
        completed: list[Utterance] = []
        actual_end = self._total_samples
        if self._pending.size:
            padded = np.pad(self._pending, (0, self._window_samples - self._pending.size))
            event = self._vad_iterator(padded, return_seconds=False)
            completed.extend(self._handle_event(event, limit=actual_end))
            completed.extend(self._split_at_max_duration(limit=actual_end))
        if self._speech_start is not None:
            raw_duration = actual_end - (self._raw_speech_start or self._speech_start)
            if raw_duration >= self._min_speech_samples or self._emitted_for_current_speech:
                utterance = self._utterance(self._speech_start, actual_end)
                if utterance is not None:
                    completed.append(utterance)
        self.reset()
        return completed

    def reset(self) -> None:
        self._vad_iterator.reset_states()
        self._origin_time = None
        self._total_samples = 0
        self._pending = np.empty(0, dtype=np.float32)
        self._pending_start = 0
        self._audio = np.empty(0, dtype=np.float32)
        self._audio_start = 0
        self._speech_start = None
        self._raw_speech_start = None
        self._emitted_for_current_speech = False

    def _is_contiguous(self, start_time: float) -> bool:
        if self._origin_time is None:
            return True
        expected = self._origin_time + (self._total_samples / self.sample_rate)
        return abs(start_time - expected) <= (1.5 / self.sample_rate)

    def _append_audio(self, samples: np.ndarray) -> None:
        self._audio = np.concatenate([self._audio, samples])
        self._total_samples += int(samples.size)

    def _handle_event(
        self,
        event: Mapping[str, int | float] | None,
        *,
        limit: int,
    ) -> list[Utterance]:
        if not event:
            return []
        completed: list[Utterance] = []
        if "start" in event and self._speech_start is None:
            start = max(0, min(_event_sample(event["start"], self.sample_rate), limit))
            self._speech_start = start
            self._raw_speech_start = max(start, limit - self._window_samples)
            self._emitted_for_current_speech = False
        if "end" in event and self._speech_start is not None:
            end = max(self._speech_start, min(_event_sample(event["end"], self.sample_rate), limit))
            raw_end = max(self._raw_speech_start or self._speech_start, end - self._padding_samples)
            raw_duration = raw_end - (self._raw_speech_start or self._speech_start)
            if raw_duration >= self._min_speech_samples or self._emitted_for_current_speech:
                utterance = self._utterance(self._speech_start, end)
                if utterance is not None:
                    completed.append(utterance)
            self._speech_start = None
            self._raw_speech_start = None
            self._emitted_for_current_speech = False
        return completed

    def _split_at_max_duration(self, *, limit: int) -> list[Utterance]:
        completed: list[Utterance] = []
        while (
            self._speech_start is not None
            and limit - self._speech_start >= self._max_utterance_samples
        ):
            end = self._speech_start + self._max_utterance_samples
            utterance = self._utterance(self._speech_start, end)
            if utterance is not None:
                completed.append(utterance)
            self._speech_start = end
            self._raw_speech_start = end
            self._emitted_for_current_speech = True
        return completed

    def _utterance(self, start_sample: int, end_sample: int) -> Utterance | None:
        if end_sample <= start_sample or self._origin_time is None:
            return None
        relative_start = start_sample - self._audio_start
        relative_end = end_sample - self._audio_start
        if relative_start < 0 or relative_end > self._audio.size:
            raise RuntimeError("Silero audio buffer no longer contains an emitted event range")
        samples = self._audio[relative_start:relative_end].copy()
        return Utterance(
            samples=samples,
            start=self._origin_time + (start_sample / self.sample_rate),
            end=self._origin_time + (end_sample / self.sample_rate),
        )

    def _prune_audio(self) -> None:
        if self._speech_start is not None:
            keep_from = self._speech_start
        else:
            keep_from = max(0, self._pending_start - self._padding_samples)
        drop = min(max(0, keep_from - self._audio_start), int(self._audio.size))
        if drop:
            self._audio = self._audio[drop:].copy()
            self._audio_start += drop


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


def _load_silero_vad_iterator(
    *,
    threshold: float,
    sample_rate: int,
    min_silence_seconds: float,
    speech_padding_ms: int,
) -> StreamingVADIterator:
    try:
        from silero_vad import VADIterator, load_silero_vad
    except ImportError as exc:
        raise ASRUnavailableError(
            "Silero VAD requires the optional 'vad' extra (silero-vad==6.2.1)"
        ) from exc
    try:
        model = load_silero_vad(onnx=False)
        return VADIterator(
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=int(round(min_silence_seconds * 1000)),
            speech_pad_ms=speech_padding_ms,
        )
    except Exception as exc:
        raise ASRUnavailableError(f"failed to load Silero VAD: {exc}") from exc


def _event_sample(value: int | float, sample_rate: int) -> int:
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
        left_context_samples: np.ndarray | None = None,
    ) -> list[ASRSegment]:
        utterance_audio = ensure_mono_float32(utterance.samples)
        if utterance_audio.size == 0:
            return []
        audio = utterance_audio
        offset = utterance.start
        clip_start: float | None = None
        clip_end: float | None = None
        if left_context_samples is not None:
            left_context = ensure_mono_float32(left_context_samples)
            if left_context.size:
                audio = np.concatenate([left_context, utterance_audio])
                left_context_duration = left_context.size / SAMPLE_RATE
                offset = utterance.start - left_context_duration
                clip_start = utterance.start
                clip_end = utterance.end
        segments, _info = self._model.transcribe(
            audio,
            language=self.language if language is None else language,
            initial_prompt=self.initial_prompt if initial_prompt is None else initial_prompt,
            word_timestamps=True,
            vad_filter=False,
            # ponytail: sparse temperature ladder instead of the default 6-rung one.
            # Degenerate greedy loops on short utterances need temp ~0.8 to recover;
            # the intermediate rungs (0.2-0.6) never rescue, they just add full decode
            # passes (~16s stall on a 1s clip). Normal clips never leave temp 0.
            temperature=[0.0, 0.8, 1.0],
        )
        return list(_convert_fw_segments(segments, offset=offset, clip_start=clip_start, clip_end=clip_end))


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


def is_low_confidence_segment(
    segment: ASRSegment,
    *,
    min_avg_logprob: float = -1.0,
    min_word_probability: float = 0.5,
) -> bool:
    """Return True when a decoded segment is a likely candidate for correction."""
    if segment.confidence is not None and segment.confidence < min_avg_logprob:
        return True
    probabilities = [word.probability for word in segment.words if word.probability is not None]
    return bool(probabilities) and min(probabilities) < min_word_probability


def low_confidence_words(
    segment: ASRSegment,
    *,
    min_word_probability: float = 0.5,
) -> tuple[WordTimestamp, ...]:
    return tuple(
        word
        for word in segment.words
        if word.probability is not None and word.probability < min_word_probability
    )


def _convert_fw_segments(
    segments: Iterable[object],
    *,
    offset: float,
    clip_start: float | None = None,
    clip_end: float | None = None,
) -> Iterable[ASRSegment]:
    for segment in segments:
        start = offset + float(getattr(segment, "start", 0.0))
        end = offset + float(getattr(segment, "end", 0.0))
        if clip_start is not None and end <= clip_start:
            continue
        if clip_end is not None and start >= clip_end:
            continue
        start = max(start, clip_start) if clip_start is not None else start
        end = min(end, clip_end) if clip_end is not None else end
        text = normalize_thai_spacing(str(getattr(segment, "text", "")).strip())
        words = []
        for word in getattr(segment, "words", None) or []:
            word_start = offset + float(getattr(word, "start", 0.0))
            word_end = offset + float(getattr(word, "end", 0.0))
            if clip_start is not None and word_end <= clip_start:
                continue
            if clip_end is not None and word_start >= clip_end:
                continue
            word_start = max(word_start, clip_start) if clip_start is not None else word_start
            word_end = min(word_end, clip_end) if clip_end is not None else word_end
            words.append(
                WordTimestamp(
                    word=str(getattr(word, "word", "")).strip(),
                    start=word_start,
                    end=word_end,
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
