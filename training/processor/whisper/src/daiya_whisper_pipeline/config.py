from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable
import os

from dotenv import load_dotenv


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _positive_int(name: str, default: int) -> int:
    value = _int(name, default)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _choice(name: str, default: str, choices: set[str]) -> str:
    value = _str(name, default).strip().lower()
    if value not in choices:
        choices_text = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {choices_text}")
    return value


def _path(name: str, default: str, base_dir: Path) -> Path:
    value = Path(_str(name, default)).expanduser()
    if value.is_absolute():
        return value
    return (base_dir / value).resolve()


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def segmentation_config_id(config: object) -> str:
    """Stable segmentation identity shared by production and diagnostics."""
    values = {
        "version": "timestamp-ownership-v1",
        "vad_threshold": getattr(config, "vad_threshold", None),
        "vad_min_speech_ms": getattr(config, "vad_min_speech_ms", None),
        "vad_min_silence_ms": getattr(config, "vad_min_silence_ms", None),
        "vad_speech_pad_ms": getattr(config, "vad_speech_pad_ms", None),
        "overlap_mode": getattr(config, "overlap_mode", None),
        "overlap_pad_seconds": getattr(config, "overlap_pad_seconds", None),
        "min_chunk_seconds": getattr(config, "min_chunk_seconds", None),
        "target_chunk_seconds": getattr(config, "target_chunk_seconds", None),
        "max_chunk_seconds": getattr(config, "max_chunk_seconds", None),
        "hard_max_chunk_seconds": getattr(config, "hard_max_chunk_seconds", None),
        "merge_gap_seconds": getattr(config, "merge_gap_seconds", None),
        "boundary_min_silence_seconds": getattr(config, "boundary_min_silence_seconds", None),
        "boundary_search_seconds": getattr(config, "boundary_search_seconds", None),
        "fallback_context_seconds": getattr(config, "fallback_context_seconds", None),
        "boundary_candidate_tolerance_seconds": getattr(config, "boundary_candidate_tolerance_seconds", None),
        "boundary_min_confidence": getattr(config, "boundary_min_confidence", None),
        "timestamp_model": getattr(config, "timestamp_model", None),
        "timestamp_device": getattr(config, "timestamp_device", None),
        "timestamp_compute_type": getattr(config, "timestamp_compute_type", None),
        "timestamp_beam_size": getattr(config, "timestamp_beam_size", None),
        "timestamp_language": getattr(config, "timestamp_language", None),
        "timestamp_condition_on_previous_text": getattr(config, "timestamp_condition_on_previous_text", None),
        "energy_window_seconds": getattr(config, "energy_window_seconds", None),
        "energy_low_percentile": getattr(config, "energy_low_percentile", None),
        "energy_min_gap_seconds": getattr(config, "energy_min_gap_seconds", None),
        "label_alignment_min_similarity": getattr(config, "label_alignment_min_similarity", None),
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PipelineConfig:
    base_dir: Path
    input_dir: Path
    output_dir: Path
    work_dir: Path
    dataset_split: str
    audio_extensions: tuple[str, ...]

    ffmpeg_bin: str
    ffmpeg_max_workers: int
    ffmpeg_max_in_flight: int
    sample_rate: int
    channels: int
    audio_codec: str

    vad_threshold: float
    vad_min_speech_ms: int
    vad_min_silence_ms: int
    vad_speech_pad_ms: int

    enable_overlap_filter: bool
    overlap_mode: str
    pyannote_auth_token: str
    pyannote_overlap_model: str
    overlap_pad_seconds: float

    min_chunk_seconds: float
    target_chunk_seconds: float
    max_chunk_seconds: float
    hard_max_chunk_seconds: float
    merge_gap_seconds: float
    boundary_min_silence_seconds: float
    boundary_search_seconds: float
    fallback_context_seconds: float
    boundary_candidate_tolerance_seconds: float
    boundary_min_confidence: float

    timestamp_model: str
    timestamp_evidence_cache_dir: Path
    timestamp_device: str
    timestamp_compute_type: str
    timestamp_beam_size: int
    timestamp_language: str
    timestamp_condition_on_previous_text: bool
    energy_window_seconds: float
    energy_low_percentile: float
    energy_min_gap_seconds: float
    label_alignment_min_similarity: float

    torch_device: str

    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_model: str
    openrouter_app_name: str
    openrouter_site_url: str
    llm_max_workers: int
    llm_max_in_flight: int
    llm_temperature: float
    llm_timeout_seconds: float
    llm_audio_format: str
    llm_context_max_chars: int
    llm_reasoning_effort: str

    language_hint: str
    text_column: str
    keep_intermediate: bool
    export_max_workers: int
    export_max_in_flight: int

    @classmethod
    def load(cls, env_file: Path | None = None) -> "PipelineConfig":
        base_dir = Path(__file__).resolve().parents[2]
        load_dotenv(env_file or base_dir / ".env")

        return cls(
            base_dir=base_dir,
            input_dir=_path("DAIYA_INPUT_DIR", "../../dataset/raw", base_dir),
            output_dir=_path("DAIYA_OUTPUT_DIR", "output/hf_dataset", base_dir),
            work_dir=_path("DAIYA_WORK_DIR", "work", base_dir),
            dataset_split=_str("DAIYA_DATASET_SPLIT", "train"),
            audio_extensions=_csv(_str("DAIYA_AUDIO_EXTENSIONS", ".wav,.mp3,.m4a,.flac,.ogg,.opus,.aac,.wma,.webm,.mp4,.mov,.mkv")),
            ffmpeg_bin=_str("DAIYA_FFMPEG_BIN", "ffmpeg"),
            ffmpeg_max_workers=_positive_int("DAIYA_FFMPEG_MAX_WORKERS", 4),
            ffmpeg_max_in_flight=_positive_int("DAIYA_FFMPEG_MAX_IN_FLIGHT", 4),
            sample_rate=_int("DAIYA_SAMPLE_RATE", 16000),
            channels=_int("DAIYA_CHANNELS", 1),
            audio_codec=_str("DAIYA_AUDIO_CODEC", "pcm_s16le"),
            # The offline profile favors speech recall and boundary padding.
            # PR #10's Silero evidence showed that its balanced/sensitive rows
            # are materially safer for missing speech than a latency-tuned
            # streaming profile.
            vad_threshold=_float("DAIYA_VAD_THRESHOLD", 0.5),
            vad_min_speech_ms=_int("DAIYA_VAD_MIN_SPEECH_MS", 250),
            vad_min_silence_ms=_int("DAIYA_VAD_MIN_SILENCE_MS", 150),
            vad_speech_pad_ms=_int("DAIYA_VAD_SPEECH_PAD_MS", 80),
            enable_overlap_filter=_bool(os.getenv("DAIYA_ENABLE_OVERLAP_FILTER"), True),
            overlap_mode=_choice("DAIYA_OVERLAP_MODE", "preserve", {"preserve", "legacy-exclude"}),
            pyannote_auth_token=_str("DAIYA_PYANNOTE_AUTH_TOKEN", ""),
            pyannote_overlap_model=_str("DAIYA_PYANNOTE_OVERLAP_MODEL", "pyannote/speaker-diarization-community-1"),
            overlap_pad_seconds=_float("DAIYA_OVERLAP_PAD_SECONDS", 0.15),
            min_chunk_seconds=_float("DAIYA_MIN_CHUNK_SECONDS", 1.0),
            target_chunk_seconds=_float("DAIYA_TARGET_CHUNK_SECONDS", 18.0),
            max_chunk_seconds=_float("DAIYA_MAX_CHUNK_SECONDS", 25.0),
            # Faster-Whisper's practical single-pass context is about 30 s.
            # 25 s remains a planning goal; this hard ceiling only gives a
            # high-confidence boundary a little room to avoid a bad cut.
            hard_max_chunk_seconds=_float("DAIYA_HARD_MAX_CHUNK_SECONDS", 30.0),
            # This is a bridge threshold, not a cue to remove the intervening
            # wall-clock audio.  The silence remains in the exported clip.
            merge_gap_seconds=_float("DAIYA_MERGE_GAP_SECONDS", 0.8),
            boundary_min_silence_seconds=_float("DAIYA_BOUNDARY_MIN_SILENCE_SECONDS", 0.5),
            boundary_search_seconds=_float("DAIYA_BOUNDARY_SEARCH_SECONDS", 4.0),
            fallback_context_seconds=_float("DAIYA_FALLBACK_CONTEXT_SECONDS", 1.0),
            boundary_candidate_tolerance_seconds=_float("DAIYA_BOUNDARY_CANDIDATE_TOLERANCE_SECONDS", 0.35),
            boundary_min_confidence=_float("DAIYA_BOUNDARY_MIN_CONFIDENCE", 0.55),
            # This is intentionally a local CTranslate2/Faster-Whisper model
            # path or a model id.  The evidence stage never uses the audio LLM.
            timestamp_model=_str("DAIYA_TIMESTAMP_MODEL", "large-v3"),
            timestamp_evidence_cache_dir=_path(
                "DAIYA_TIMESTAMP_EVIDENCE_CACHE_DIR", "cache/timestamp-evidence", base_dir
            ),
            timestamp_device=_str("DAIYA_TIMESTAMP_DEVICE", "auto"),
            timestamp_compute_type=_str("DAIYA_TIMESTAMP_COMPUTE_TYPE", "auto"),
            timestamp_beam_size=_positive_int("DAIYA_TIMESTAMP_BEAM_SIZE", 5),
            timestamp_language=_str("DAIYA_TIMESTAMP_LANGUAGE", ""),
            timestamp_condition_on_previous_text=_bool(os.getenv("DAIYA_TIMESTAMP_CONDITION_ON_PREVIOUS_TEXT"), False),
            energy_window_seconds=_float("DAIYA_ENERGY_WINDOW_SECONDS", 0.05),
            energy_low_percentile=_float("DAIYA_ENERGY_LOW_PERCENTILE", 20.0),
            energy_min_gap_seconds=_float("DAIYA_ENERGY_MIN_GAP_SECONDS", 0.20),
            label_alignment_min_similarity=_float("DAIYA_LABEL_ALIGNMENT_MIN_SIMILARITY", 0.45),
            torch_device=_str("DAIYA_TORCH_DEVICE", "cuda"),
            openrouter_api_key=_str("OPENROUTER_API_KEY", ""),
            openrouter_base_url=_str("DAIYA_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openrouter_model=_str("DAIYA_OPENROUTER_MODEL", "openai/gpt-4o-audio-preview"),
            openrouter_app_name=_str("DAIYA_OPENROUTER_APP_NAME", "Daiya-RMT"),
            openrouter_site_url=_str("DAIYA_OPENROUTER_SITE_URL", "http://localhost"),
            llm_max_workers=_positive_int("DAIYA_LLM_MAX_WORKERS", 2),
            llm_max_in_flight=_positive_int("DAIYA_LLM_MAX_IN_FLIGHT", 2),
            llm_temperature=_float("DAIYA_LLM_TEMPERATURE", 0.0),
            llm_timeout_seconds=_float("DAIYA_LLM_TIMEOUT_SECONDS", 120.0),
            llm_audio_format=_str("DAIYA_LLM_AUDIO_FORMAT", "wav"),
            llm_context_max_chars=_int("DAIYA_LLM_CONTEXT_MAX_CHARS", 3000),
            # reasoning-model thinking adds cost without ASR accuracy gains; "" = don't send
            llm_reasoning_effort=_str("DAIYA_LLM_REASONING_EFFORT", ""),
            language_hint=_str("DAIYA_LANGUAGE_HINT", "mixed"),
            text_column=_str("DAIYA_TEXT_COLUMN", "text"),
            keep_intermediate=_bool(os.getenv("DAIYA_KEEP_INTERMEDIATE"), False),
            export_max_workers=_positive_int("DAIYA_EXPORT_MAX_WORKERS", 4),
            export_max_in_flight=_positive_int("DAIYA_EXPORT_MAX_IN_FLIGHT", 4),
        )

    def find_audio_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self.input_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.audio_extensions:
                files.append(path)
        return sorted(files)

    @property
    def segmentation_config_id(self) -> str:
        """Stable ID recorded in every metadata row for regeneration audits."""
        return segmentation_config_id(self)


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
