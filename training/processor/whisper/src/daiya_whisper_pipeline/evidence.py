"""Grounded timestamp and acoustic evidence for ownership-safe segmentation.

The audio-capable labeling LLM deliberately never produces timing.  This
module obtains optional word/phrase timing from local Faster-Whisper and
low-energy spans from the waveform.  Whisper text is retained only in the
private cache for conservative post-label validation; it is not exported as a
dataset target and is never used to rewrite a label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import soundfile as sf

from .types import Interval, NormalizedAudio


EVIDENCE_SCHEMA_VERSION = "daiya-timestamp-evidence-1"
_UNSET = object()


def _audio_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _round(value: float) -> float:
    return round(float(value), 6)


@dataclass(frozen=True)
class TimestampWord:
    """Local-ASR timing. ``text`` is validation-only and stays out of exports."""

    start: float
    end: float
    text: str = ""
    probability: float | None = None

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2

    def to_cache(self) -> dict[str, object]:
        return {
            "start": _round(self.start),
            "end": _round(self.end),
            "text": self.text,
            "probability": _round(self.probability) if self.probability is not None else None,
        }

    @classmethod
    def from_cache(cls, row: object) -> "TimestampWord":
        value = row if isinstance(row, dict) else {}
        probability = value.get("probability")
        return cls(
            start=float(value.get("start", 0.0)),
            end=float(value.get("end", 0.0)),
            text=str(value.get("text") or ""),
            probability=float(probability) if isinstance(probability, (int, float)) else None,
        )


@dataclass(frozen=True)
class TimestampEvidence:
    source_id: str
    source_audio_sha256: str
    duration_seconds: float
    status: str
    model_identity: dict[str, object]
    decoding_settings: dict[str, object]
    acoustic_settings: dict[str, object] = field(default_factory=dict)
    words: tuple[TimestampWord, ...] = ()
    energy_gaps: tuple[Interval, ...] = ()
    failure: str = ""
    cache_path: Path | None = None

    @property
    def available(self) -> bool:
        return self.status == "ok" and bool(self.words)

    def whisper_gaps(self, minimum_seconds: float = 0.12) -> list[Interval]:
        gaps: list[Interval] = []
        words = sorted((word for word in self.words if word.end > word.start), key=lambda word: word.start)
        for left, right in zip(words, words[1:]):
            start, end = left.end, right.start
            if end - start >= minimum_seconds:
                gaps.append(Interval(start, end))
        return gaps

    def provenance(self) -> dict[str, object]:
        """Compact metadata safe to include with an exported row.

        No Whisper text/word list is included here: labels remain owned by the
        audio LLM and the timestamps are only independent evidence.
        """
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": self.status,
            "source_audio_sha256": self.source_audio_sha256,
            "model": self.model_identity,
            "decoding": self.decoding_settings,
            "acoustic": self.acoustic_settings,
            "word_timestamp_count": len(self.words),
            "energy_gap_count": len(self.energy_gaps),
            "failure": self.failure or None,
            # Do not export an absolute workstation-local cache path in dataset
            # provenance.  The source hash and settings identify its contents.
            "cache_available": self.cache_path is not None,
        }

    def to_cache(self) -> dict[str, object]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "source_id": self.source_id,
            "source_audio_sha256": self.source_audio_sha256,
            "duration_seconds": _round(self.duration_seconds),
            "status": self.status,
            "model_identity": self.model_identity,
            "decoding_settings": self.decoding_settings,
            "acoustic_settings": self.acoustic_settings,
            "words": [word.to_cache() for word in self.words],
            "energy_gaps": [{"start": _round(gap.start), "end": _round(gap.end)} for gap in self.energy_gaps],
            "failure": self.failure or None,
        }

    @classmethod
    def from_cache(cls, row: dict[str, object], cache_path: Path) -> "TimestampEvidence":
        raw_gaps = row.get("energy_gaps")
        gaps: list[Interval] = []
        if isinstance(raw_gaps, list):
            for item in raw_gaps:
                if not isinstance(item, dict):
                    continue
                start, end = float(item.get("start", 0)), float(item.get("end", 0))
                if end > start:
                    gaps.append(Interval(start, end))
        raw_words = row.get("words")
        words = tuple(
            word for item in raw_words if isinstance(item, dict)
            if (word := TimestampWord.from_cache(item)).end > word.start
        ) if isinstance(raw_words, list) else ()
        return cls(
            source_id=str(row.get("source_id") or ""),
            source_audio_sha256=str(row.get("source_audio_sha256") or ""),
            duration_seconds=float(row.get("duration_seconds") or 0),
            status=str(row.get("status") or "failed"),
            model_identity=dict(row.get("model_identity") or {}),
            decoding_settings=dict(row.get("decoding_settings") or {}),
            words=words,
            acoustic_settings=dict(row.get("acoustic_settings") or {}),
            energy_gaps=tuple(gaps),
            failure=str(row.get("failure") or ""),
            cache_path=cache_path,
        )


def low_energy_gaps(
    path: Path,
    *,
    window_seconds: float = 0.05,
    low_percentile: float = 20.0,
    minimum_seconds: float = 0.20,
) -> list[Interval]:
    """Return deterministic low-energy windows without treating them as silence.

    This deliberately produces acoustic *evidence*, rather than deleting or
    trimming audio.  The scorer can combine it with VAD and ASR timing.
    """
    if window_seconds <= 0 or minimum_seconds <= 0:
        raise ValueError("Energy window and minimum gap must be positive")
    if not 0 <= low_percentile <= 100:
        raise ValueError("Energy low percentile must be within 0..100")
    with sf.SoundFile(path) as audio:
        sample_rate = int(audio.samplerate)
        block_size = max(1, round(sample_rate * window_seconds))
        rms: list[float] = []
        for block in audio.blocks(blocksize=block_size, dtype="float32", always_2d=True):
            if len(block):
                rms.append(float(np.sqrt(np.mean(np.square(block), dtype=np.float64))))
    if not rms:
        return []
    values = np.asarray(rms, dtype=np.float64)
    # A floor avoids considering all quiet-but-audible recording noise as an
    # exact zero.  A percentile keeps this gain-agnostic across recordings.
    threshold = max(1e-5, float(np.percentile(values, low_percentile)))
    # Relative low energy is only supporting evidence.  A flat waveform (for
    # example, continuous constant-energy speech or a synthetic test tone) has
    # no local minimum and must not turn the entire source into a "gap".
    peak = float(values.max())
    if threshold >= peak - max(1e-8, peak * 0.02):
        return []
    quiet = values <= threshold
    gaps: list[Interval] = []
    index = 0
    while index < len(quiet):
        if not quiet[index]:
            index += 1
            continue
        end_index = index + 1
        while end_index < len(quiet) and quiet[end_index]:
            end_index += 1
        start = index * block_size / sample_rate
        end = min(end_index * block_size / sample_rate, len(values) * block_size / sample_rate)
        if end - start >= minimum_seconds:
            gaps.append(Interval(start, end))
        index = end_index
    return gaps


class TimestampEvidenceStage:
    """Caches continuous-source local-ASR/energy evidence outside the dataset."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self._model: Any | None = None
        self._model_version = "unknown"
        self._model_artifact_fingerprint: str | None | object = _UNSET

    def _settings(self) -> dict[str, object]:
        return {
            "word_timestamps": True,
            "beam_size": int(getattr(self.config, "timestamp_beam_size", 5)),
            "language": str(getattr(self.config, "timestamp_language", "") or "") or None,
            "condition_on_previous_text": bool(getattr(self.config, "timestamp_condition_on_previous_text", False)),
            "vad_filter": False,
        }

    def _acoustic_settings(self) -> dict[str, object]:
        return {
            "energy_window_seconds": float(getattr(self.config, "energy_window_seconds", 0.05)),
            "energy_low_percentile": float(getattr(self.config, "energy_low_percentile", 20.0)),
            "energy_min_gap_seconds": float(getattr(self.config, "energy_min_gap_seconds", 0.20)),
        }

    def _model_request_identity(self) -> dict[str, object]:
        """Identity available before model loading, suitable for cache reuse."""
        return {
            "family": "faster-whisper",
            "model": str(getattr(self.config, "timestamp_model", "large-v3")),
            "device": str(getattr(self.config, "timestamp_device", "auto")),
            "compute_type": str(getattr(self.config, "timestamp_compute_type", "auto")),
            "artifact_fingerprint": self._artifact_fingerprint(),
        }

    def _model_identity(self) -> dict[str, object]:
        return {
            **self._model_request_identity(),
            "library_version": self._faster_whisper_version(),
        }

    def _faster_whisper_version(self) -> str:
        if self._model_version != "unknown":
            return self._model_version
        try:
            import faster_whisper

            self._model_version = str(getattr(faster_whisper, "__version__", "unknown"))
        except Exception:  # pragma: no cover - optional dependency/model provenance
            pass
        return self._model_version

    def _artifact_fingerprint(self) -> str | None:
        """Content-address a local model once, without trusting a reused path."""
        if self._model_artifact_fingerprint is not _UNSET:
            return self._model_artifact_fingerprint  # type: ignore[return-value]
        model_path = Path(str(getattr(self.config, "timestamp_model", "large-v3")))
        if not model_path.exists():
            self._model_artifact_fingerprint = None
            return None
        try:
            files = [model_path] if model_path.is_file() else sorted(path for path in model_path.rglob("*") if path.is_file())
            digest = sha256()
            for path in files:
                relative = path.name if model_path.is_file() else str(path.relative_to(model_path)).replace("\\", "/")
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                with path.open("rb") as handle:
                    while block := handle.read(1024 * 1024):
                        digest.update(block)
            self._model_artifact_fingerprint = f"sha256:{digest.hexdigest()}"
        except OSError:
            # A remote model ID or inaccessible local artifact retains its
            # configured identity but cannot claim a content fingerprint.
            self._model_artifact_fingerprint = None
        return self._model_artifact_fingerprint  # type: ignore[return-value]

    def _cache_path(self, source: NormalizedAudio) -> Path:
        cache_root = getattr(self.config, "timestamp_evidence_cache_dir", None)
        if cache_root is None:
            cache_root = Path(getattr(self.config, "work_dir")) / "evidence"
        return Path(cache_root) / f"{source.source_id}.json"

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel  # imported lazily for CPU-only tests
        self._faster_whisper_version()
        self._model = WhisperModel(
            str(getattr(self.config, "timestamp_model", "large-v3")),
            device=str(getattr(self.config, "timestamp_device", "auto")),
            compute_type=str(getattr(self.config, "timestamp_compute_type", "auto")),
        )
        return self._model

    def _decode(self, path: Path) -> tuple[TimestampWord, ...]:
        model = self._load_model()
        settings = self._settings()
        segments, _ = model.transcribe(str(path), **settings)
        words: list[TimestampWord] = []
        for segment in segments:
            for word in getattr(segment, "words", None) or ():
                start, end = float(word.start), float(word.end)
                if end > start:
                    probability = getattr(word, "probability", None)
                    words.append(
                        TimestampWord(
                            start=start,
                            end=end,
                            text=str(getattr(word, "word", "") or ""),
                            probability=float(probability) if probability is not None else None,
                        )
                    )
        return tuple(sorted(words, key=lambda word: (word.start, word.end, word.text)))

    def collect(self, source: NormalizedAudio) -> TimestampEvidence:
        cache_path = self._cache_path(source)
        source_hash = _audio_sha256(source.normalized_path)
        settings = self._settings()
        acoustic_settings = self._acoustic_settings()
        requested_model = self._model_identity()
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if (
                    isinstance(cached, dict)
                    and cached.get("schema_version") == EVIDENCE_SCHEMA_VERSION
                    and cached.get("source_audio_sha256") == source_hash
                    and cached.get("decoding_settings") == settings
                    and cached.get("acoustic_settings") == acoustic_settings
                    and isinstance(cached.get("model_identity"), dict)
                    and all(cached["model_identity"].get(key) == value for key, value in requested_model.items())
                ):
                    return TimestampEvidence.from_cache(cached, cache_path)
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                # Cache corruption must not change segmentation semantics.
                pass

        duration = float(source.duration_seconds or sf.info(source.normalized_path).duration)
        try:
            energy_gaps = tuple(
                low_energy_gaps(
                    source.normalized_path,
                    window_seconds=float(getattr(self.config, "energy_window_seconds", 0.05)),
                    low_percentile=float(getattr(self.config, "energy_low_percentile", 20.0)),
                    minimum_seconds=float(getattr(self.config, "energy_min_gap_seconds", 0.20)),
                )
            )
        except Exception as error:
            energy_gaps = ()
            energy_failure = f"energy:{type(error).__name__}:{error}"
        else:
            energy_failure = ""

        try:
            words = self._decode(source.normalized_path)
            status = "ok" if words else "empty"
            failure = energy_failure
        except Exception as error:
            words = ()
            status = "failed"
            failure = "; ".join(value for value in (energy_failure, f"whisper:{type(error).__name__}:{error}") if value)

        evidence = TimestampEvidence(
            source_id=source.source_id,
            source_audio_sha256=source_hash,
            duration_seconds=duration,
            status=status,
            model_identity=self._model_identity(),
            decoding_settings=settings,
            acoustic_settings=acoustic_settings,
            words=words,
            energy_gaps=energy_gaps,
            failure=failure,
            cache_path=cache_path,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(evidence.to_cache(), ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(cache_path)
        return evidence


def words_in_interval(words: Iterable[TimestampWord], interval: Interval) -> list[TimestampWord]:
    """Select words wholly inside an owned source range.

    A timestamp can straddle a fixed fallback handoff.  Assigning it by its
    midpoint would silently make one owner appear to contain speech that is
    audible in both sides, so ambiguous words are deliberately excluded from
    the post-label gate.
    """
    return [word for word in words if interval.start <= word.start and word.end <= interval.end]
