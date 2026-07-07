from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .asr import ASRUnavailableError, FasterWhisperASR, NullASR, Utterance, create_utterance_segmenter
from .audio import PCMChunk, SAMPLE_RATE, ensure_mono_float32
from .correct import NoOpCorrectionStage
from .diarizer import DiarizerConfig, create_diarizer
from .mux import ASRSegment, CorrectionUpdate, TranscriptEvent, TranscriptMultiplexer, WordTimestamp


@dataclass(frozen=True)
class PipelineConfig:
    enable_asr: bool = True
    enable_diarization: bool = True
    asr_model: str | None = None
    asr_device: str = "auto"
    asr_compute_type: str = "int8_float16"
    language: str | None = None
    initial_prompt: str | None = None
    vad_threshold: float = 0.012
    utterance_cap_seconds: float = 8.0
    diarization_profile: str = "balanced"
    diarization_backend: str = "auto"
    diarization_commit_delay_seconds: float | None = None
    window_seconds: float | None = None
    hop_seconds: float | None = None
    latency_seconds: float | None = None
    commit_delay_seconds: float | None = None
    match_threshold: float | None = None
    asr_prompt_memory_enabled: bool = True
    asr_prompt_tail_chars: int = 420
    asr_prompt_max_chars: int = 900
    asr_prompt_max_terms: int = 24
    asr_left_context_enabled: bool = False
    asr_left_context_seconds: float = 3.0
    asr_left_context_short_utterance_seconds: float = 1.4
    asr_low_confidence_threshold: float = -1.0
    asr_delayed_correction_enabled: bool = False
    asr_delayed_correction_window_seconds: float = 10.0
    asr_tiny_utterance_merge_enabled: bool = False
    asr_tiny_utterance_seconds: float = 0.55
    asr_tiny_utterance_max_gap_seconds: float = 0.35
    asr_tiny_utterance_max_delay_seconds: float = 0.7


class StreamingPipeline:
    """Shared chunk-to-transcript pipeline for CLI, replay, and live server paths."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = _with_runtime_defaults(config or PipelineConfig())
        self.mux = TranscriptMultiplexer()
        self.segmenter = (
            create_utterance_segmenter(
                threshold=self.config.vad_threshold,
                max_utterance_seconds=self.config.utterance_cap_seconds,
            )
            if self.config.enable_asr
            else None
        )
        self.diarizer = (
            create_diarizer(
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
                    match_threshold=self.config.match_threshold,
                ),
                prefer_real=self.config.diarization_backend != "null",
            )
            if self.config.enable_diarization
            else None
        )
        self.corrector = NoOpCorrectionStage()
        self.asr = self._create_asr() if self.config.enable_asr else NullASR("ASR is disabled")
        self._solo_segment_seq = 0
        self._prompt_memory = ASRPromptMemory(
            static_prompt=self.config.initial_prompt,
            tail_chars=self.config.asr_prompt_tail_chars,
            max_prompt_chars=self.config.asr_prompt_max_chars,
            max_terms=self.config.asr_prompt_max_terms,
        )
        self._audio_history: list[PCMChunk] = []
        self._pending_tiny_utterance: Utterance | None = None

    def accept_chunk(self, chunk: PCMChunk) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        if self.config.enable_asr:
            self._remember_audio(chunk)
        if not self.config.enable_asr:
            # Stream clock for the diarization-only view: lets the frontend
            # advance timestamps even when no speaker is being detected.
            payloads.append({"type": "tick", "time": chunk.end_time})
        if self.diarizer is not None:
            payloads.extend(self._handle_diarization_turns(self.diarizer.accept(chunk)))
        if self.segmenter is not None:
            for utterance in self.segmenter.accept(chunk):
                payloads.extend(self._handle_utterance(utterance))
            payloads.extend(self._release_stale_tiny_utterance(chunk.end_time))
        return payloads

    def flush(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        if self.segmenter is not None:
            for utterance in self.segmenter.flush():
                payloads.extend(self._handle_utterance(utterance))
            if self._pending_tiny_utterance is not None:
                payloads.extend(self._transcribe_utterance(self._drain_pending_tiny_utterance()))
        if self.diarizer is not None:
            payloads.extend(self._handle_diarization_turns(self.diarizer.flush()))
        return payloads

    def _handle_diarization_turns(self, turns: list[Any]) -> list[dict[str, Any]]:
        if self.config.enable_asr:
            payloads: list[dict[str, Any]] = []
            for event in self.mux.ingest_diarization_many(turns):
                payloads.extend(self._serialize_transcript_event(event))
            return payloads
        # ponytail: ASR off — bypass the mux and emit speaker turns as text-less
        # transcript events so the existing frontend renders the speaker timeline.
        return [
            {
                "type": "transcript.final" if turn.final else "transcript.partial",
                "source": "diarizer",
                "segment_id": turn.turn_id,
                "start": turn.start,
                "end": turn.end,
                "speaker": turn.speaker_id,
                "text": "",
                "final": turn.final,
            }
            for turn in turns
        ]

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

    def _handle_utterance(self, utterance: Utterance) -> list[dict[str, Any]]:
        if not self.config.asr_tiny_utterance_merge_enabled:
            return self._transcribe_utterance(utterance)

        pending = self._pending_tiny_utterance
        if pending is not None:
            gap = utterance.start - pending.end
            if 0.0 <= gap <= self.config.asr_tiny_utterance_max_gap_seconds:
                self._pending_tiny_utterance = None
                return self._transcribe_utterance(_merge_utterances(pending, utterance))
            payloads = self._transcribe_utterance(self._drain_pending_tiny_utterance())
            if utterance.duration < self.config.asr_tiny_utterance_seconds:
                self._pending_tiny_utterance = utterance
                return payloads
            payloads.extend(self._transcribe_utterance(utterance))
            return payloads

        if utterance.duration < self.config.asr_tiny_utterance_seconds:
            self._pending_tiny_utterance = utterance
            return []
        return self._transcribe_utterance(utterance)

    def _release_stale_tiny_utterance(self, now: float) -> list[dict[str, Any]]:
        pending = self._pending_tiny_utterance
        if pending is None:
            return []
        if now - pending.end < self.config.asr_tiny_utterance_max_delay_seconds:
            return []
        return self._transcribe_utterance(self._drain_pending_tiny_utterance())

    def _drain_pending_tiny_utterance(self) -> Utterance:
        pending = self._pending_tiny_utterance
        if pending is None:
            raise RuntimeError("no pending tiny utterance to drain")
        self._pending_tiny_utterance = None
        return pending

    def _transcribe_utterance(self, utterance: Utterance) -> list[dict[str, Any]]:
        prompt = self._build_asr_prompt()
        try:
            asr_segments = self.asr.transcribe_utterance(
                utterance,
                language=self.config.language,
                initial_prompt=prompt,
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

        if self._should_retry_with_left_context(utterance, asr_segments):
            contextual_segments = self._transcribe_with_left_context(utterance, prompt)
            if contextual_segments:
                asr_segments = contextual_segments

        payloads: list[dict[str, Any]] = []
        accepted_events: list[TranscriptEvent] = []
        for segment in asr_segments:
            if not segment.text:
                continue
            if not self.config.enable_diarization:
                # ponytail: diarization off — bypass the mux (it would never
                # commit without a diarization horizon) and finalize immediately.
                self._solo_segment_seq += 1
                payloads.append(
                    {
                        "type": "transcript.final",
                        "source": "asr",
                        "segment_id": f"seg_{self._solo_segment_seq:06d}",
                        "start": segment.start,
                        "end": segment.end,
                        "speaker": None,
                        "text": segment.text,
                        "final": True,
                        "words": [word.to_dict() for word in segment.words],
                        "language": segment.language,
                    }
                )
                self._prompt_memory.remember(segment.text)
                continue
            for event in self.mux.ingest_asr(segment):
                accepted_events.append(event)
                payloads.extend(self._serialize_transcript_event(event))
                payloads.extend(self._apply_corrections(event))
            self._prompt_memory.remember(segment.text)
        payloads.extend(self._apply_delayed_asr_correction(utterance, accepted_events, prompt))
        return payloads

    def _build_asr_prompt(self) -> str | None:
        if not self.config.asr_prompt_memory_enabled:
            return self.config.initial_prompt
        return self._prompt_memory.build_prompt()

    def _should_retry_with_left_context(self, utterance: Utterance, segments: list[ASRSegment]) -> bool:
        if not self.config.asr_left_context_enabled:
            return False
        utterance_start = float(getattr(utterance, "start", 0.0))
        utterance_duration = float(getattr(utterance, "duration", 0.0))
        if utterance_start <= 0.0 or self.config.asr_left_context_seconds <= 0.0:
            return False
        if utterance_duration <= self.config.asr_left_context_short_utterance_seconds:
            return True
        return _segments_are_low_confidence(segments, self.config.asr_low_confidence_threshold)

    def _transcribe_with_left_context(self, utterance: Utterance, prompt: str | None) -> list[ASRSegment]:
        context = self._audio_between(
            max(0.0, utterance.start - self.config.asr_left_context_seconds),
            utterance.start,
        )
        if context.size == 0:
            return []
        try:
            segments = self.asr.transcribe_utterance(
                utterance,
                language=self.config.language,
                initial_prompt=prompt,
                left_context_samples=context,
            )
        except ASRUnavailableError:
            return []
        return _clip_segments_to_window(segments, utterance.start, utterance.end)

    def _apply_delayed_asr_correction(
        self,
        utterance: Utterance,
        events: list[TranscriptEvent],
        prompt: str | None,
    ) -> list[dict[str, Any]]:
        if not self.config.asr_delayed_correction_enabled or not self.config.enable_diarization:
            return []
        current_events = [
            event
            for event in events
            if event.type == "transcript.partial"
            and event.segment.start >= utterance.start - 0.05
            and event.segment.end <= utterance.end + 0.05
        ]
        if not current_events or self.config.asr_delayed_correction_window_seconds <= utterance.duration:
            return []
        window_start = max(0.0, utterance.end - self.config.asr_delayed_correction_window_seconds)
        context = self._audio_between(window_start, utterance.start)
        if context.size == 0:
            return []
        window = Utterance(
            samples=np.concatenate([context, ensure_mono_float32(utterance.samples)]),
            start=utterance.start - (context.size / SAMPLE_RATE),
            end=utterance.end,
        )
        try:
            decoded = self.asr.transcribe_utterance(
                window,
                language=self.config.language,
                initial_prompt=prompt,
            )
        except ASRUnavailableError:
            return []
        replacements = _clip_segments_to_window(decoded, utterance.start, utterance.end)
        if not replacements:
            return []
        replacement_text = " ".join(segment.text for segment in replacements if segment.text).strip()
        if not replacement_text:
            return []

        payloads: list[dict[str, Any]] = []
        current_text = " ".join(event.segment.text for event in current_events).strip()
        if _too_similar_or_repetitive(current_text, replacement_text):
            return []
        replacement_words = tuple(word for segment in replacements for word in segment.words)
        for event in current_events[:1]:
            for update in self.mux.apply_correction(
                CorrectionUpdate(
                    segment_id=event.segment.segment_id,
                    text=replacement_text,
                    words=replacement_words or None,
                )
            ):
                self._prompt_memory.remember(update.segment.text)
                payloads.append(update.to_dict())
        return payloads

    def _remember_audio(self, chunk: PCMChunk) -> None:
        self._audio_history.append(chunk)
        keep_seconds = max(
            self.config.asr_left_context_seconds,
            self.config.asr_delayed_correction_window_seconds,
            self.config.utterance_cap_seconds,
        ) + 2.0
        cutoff = chunk.end_time - keep_seconds
        while self._audio_history and self._audio_history[0].end_time < cutoff:
            self._audio_history.pop(0)

    def _audio_between(self, start: float, end: float) -> np.ndarray:
        if end <= start:
            return np.empty(0, dtype=np.float32)
        pieces: list[np.ndarray] = []
        cursor = start
        for chunk in self._audio_history:
            if chunk.end_time <= start:
                continue
            if chunk.start_time >= end:
                break
            piece_start = max(start, chunk.start_time)
            piece_end = min(end, chunk.end_time)
            if piece_start > cursor:
                pieces.append(np.zeros(int(round((piece_start - cursor) * SAMPLE_RATE)), dtype=np.float32))
            offset = int(round((piece_start - chunk.start_time) * SAMPLE_RATE))
            count = int(round((piece_end - piece_start) * SAMPLE_RATE))
            if count > 0:
                pieces.append(chunk.samples[offset : offset + count])
            cursor = piece_end
        if cursor < end:
            pieces.append(np.zeros(int(round((end - cursor) * SAMPLE_RATE)), dtype=np.float32))
        return np.concatenate(pieces).astype(np.float32, copy=False) if pieces else np.empty(0, dtype=np.float32)

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
    configured_diarization_backend = os.getenv("DAIYA_DIARIZATION_BACKEND")
    if configured_diarization_backend and config.diarization_backend == "auto":
        updates["diarization_backend"] = configured_diarization_backend
    return replace(config, **updates) if updates else config


@dataclass
class ASRPromptMemory:
    static_prompt: str | None = None
    tail_chars: int = 420
    max_prompt_chars: int = 900
    max_terms: int = 24
    _tail: str = ""
    _terms: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self._terms is None:
            self._terms = {}
        if self.static_prompt:
            self._remember_terms(self.static_prompt)

    def remember(self, text: str) -> None:
        cleaned = _clean_prompt_text(text)
        if not cleaned:
            return
        self._tail = _bounded_tail(f"{self._tail} {cleaned}".strip(), self.tail_chars)
        self._remember_terms(cleaned)

    def build_prompt(self) -> str | None:
        terms = self.terms()
        terms_line = _format_terms_line(terms, self.max_prompt_chars)
        static_budget = self.max_prompt_chars - len(terms_line) - (1 if terms_line else 0)
        static_line = self._bounded_static_prompt(static_budget)
        prompt_without_tail = "\n".join(part for part in (static_line, terms_line) if part)
        transcript_line = self._bounded_transcript_line(prompt_without_tail)
        prompt = "\n".join(part for part in (static_line, terms_line, transcript_line) if part).strip()
        return prompt or None

    def terms(self) -> list[str]:
        terms = self._terms or {}
        return [
            term
            for term, _count in sorted(terms.items(), key=lambda item: (-item[1], item[0].lower()))[: self.max_terms]
        ]

    def _remember_terms(self, text: str) -> None:
        terms = self._terms
        if terms is None:
            return
        for term in _english_domain_terms(text):
            key = term.lower()
            canonical = next((known for known in terms if known.lower() == key), term)
            terms[canonical] = terms.get(canonical, 0) + 1

    def _bounded_static_prompt(self, remaining_after_terms: int) -> str:
        if not self.static_prompt or remaining_after_terms <= 0:
            return ""
        static_budget = min(200, self.max_prompt_chars // 4, remaining_after_terms)
        return _bounded_tail(_clean_prompt_text(self.static_prompt), static_budget)

    def _bounded_transcript_line(self, prefix: str) -> str:
        if not self._tail:
            return ""
        separator_cost = 1 if prefix else 0
        label = "Recent transcript: "
        transcript_budget = self.max_prompt_chars - len(prefix) - separator_cost - len(label)
        if transcript_budget <= 0:
            return ""
        transcript = _bounded_tail(self._tail, transcript_budget)
        return f"{label}{transcript}" if transcript else ""


_ENGLISH_TERM_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_+#.-]*\b")
_TERM_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "are",
    "and",
    "but",
    "before",
    "by",
    "context",
    "current",
    "do",
    "does",
    "first",
    "for",
    "from",
    "hint",
    "into",
    "is",
    "it",
    "label",
    "labels",
    "latest",
    "memory",
    "metadata",
    "note",
    "notes",
    "only",
    "previous",
    "prompt",
    "recent",
    "repeat",
    "said",
    "spoken",
    "static",
    "successful",
    "term",
    "terms",
    "text",
    "that",
    "the",
    "this",
    "to",
    "topic",
    "transcript",
    "unless",
    "use",
    "with",
    "word",
    "words",
    "you",
}


def _english_domain_terms(text: str) -> list[str]:
    found: dict[str, str] = {}
    for match in _ENGLISH_TERM_RE.finditer(text):
        term = match.group(0).strip("._-")
        if not _is_useful_english_term(term):
            continue
        found.setdefault(term.lower(), term)
    return sorted(found.values(), key=str.lower)


def _format_terms_line(terms: list[str], max_chars: int) -> str:
    if not terms or max_chars <= len("Terms: "):
        return ""
    selected: list[str] = []
    used = len("Terms: ")
    for term in terms:
        addition = len(term) + (2 if selected else 0)
        if used + addition > max_chars:
            break
        selected.append(term)
        used += addition
    return "Terms: " + ", ".join(selected) if selected else ""


def _is_useful_english_term(term: str) -> bool:
    normalized = term.lower()
    if len(term) < 2 or normalized in _TERM_STOPWORDS:
        return False
    if normalized.isdigit():
        return False
    if len(term) == 2 and not term.isupper():
        return False
    if "." in term or "_" in term or "+" in term or "#" in term:
        return True
    if term.isupper() and len(term) >= 2:
        return True
    return len(term) >= 4


def _merge_utterances(left: Utterance, right: Utterance) -> Utterance:
    gap_samples = max(0, int(round((right.start - left.end) * SAMPLE_RATE)))
    samples = np.concatenate(
        [
            ensure_mono_float32(left.samples),
            np.zeros(gap_samples, dtype=np.float32),
            ensure_mono_float32(right.samples),
        ]
    )
    return Utterance(samples=samples, start=left.start, end=right.end)


def _segments_are_low_confidence(segments: list[ASRSegment], threshold: float) -> bool:
    confidences = [segment.confidence for segment in segments if segment.confidence is not None]
    if not confidences:
        return False
    return sum(confidences) / len(confidences) < threshold


def _clip_segments_to_window(segments: list[ASRSegment], start: float, end: float) -> list[ASRSegment]:
    clipped: list[ASRSegment] = []
    for segment in segments:
        if segment.end <= start or segment.start >= end:
            continue
        words = tuple(word for word in segment.words if word.end > start and word.start < end)
        if words:
            text = " ".join(word.word for word in words if word.word).strip()
        elif segment.start >= start and segment.end <= end:
            text = segment.text
        else:
            text = segment.text
        text = text.strip()
        if not text:
            continue
        clipped.append(
            ASRSegment(
                start=max(segment.start, start),
                end=min(segment.end, end),
                text=text,
                words=tuple(_clip_word(word, start, end) for word in words),
                language=segment.language,
                confidence=segment.confidence,
            )
        )
    return clipped


def _clip_word(word: WordTimestamp, start: float, end: float) -> WordTimestamp:
    return WordTimestamp(
        word=word.word,
        start=max(word.start, start),
        end=min(word.end, end),
        probability=word.probability,
    )


def _clean_prompt_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _bounded_tail(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :] if first_space > 0 else tail


def _too_similar_or_repetitive(current: str, replacement: str) -> bool:
    current_norm = _clean_prompt_text(current).lower()
    replacement_norm = _clean_prompt_text(replacement).lower()
    if not replacement_norm or replacement_norm == current_norm:
        return True
    words = replacement_norm.split()
    if len(words) >= 4 and len(set(words)) <= max(1, len(words) // 3):
        return True
    if current_norm and replacement_norm in current_norm:
        return True
    return False


def default_asr_model() -> str:
    configured = os.getenv("DAIYA_ASR_MODEL")
    if configured:
        return configured

    local_ct2 = _repo_root() / "training" / "whisper" / "runs" / "largev3-m2-iter1-ct2-int8_float16"
    if (local_ct2 / "model.bin").exists():
        return str(local_ct2)

    return "medium"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
