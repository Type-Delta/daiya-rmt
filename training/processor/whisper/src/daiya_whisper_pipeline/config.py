from __future__ import annotations

from dataclasses import dataclass
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


def _path(name: str, default: str, base_dir: Path) -> Path:
    value = Path(_str(name, default)).expanduser()
    if value.is_absolute():
        return value
    return (base_dir / value).resolve()


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


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
    pyannote_auth_token: str
    pyannote_overlap_model: str
    overlap_pad_seconds: float

    min_chunk_seconds: float
    target_chunk_seconds: float
    max_chunk_seconds: float
    merge_gap_seconds: float

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
            vad_threshold=_float("DAIYA_VAD_THRESHOLD", 0.5),
            vad_min_speech_ms=_int("DAIYA_VAD_MIN_SPEECH_MS", 250),
            vad_min_silence_ms=_int("DAIYA_VAD_MIN_SILENCE_MS", 150),
            vad_speech_pad_ms=_int("DAIYA_VAD_SPEECH_PAD_MS", 80),
            enable_overlap_filter=_bool(os.getenv("DAIYA_ENABLE_OVERLAP_FILTER"), True),
            pyannote_auth_token=_str("DAIYA_PYANNOTE_AUTH_TOKEN", ""),
            pyannote_overlap_model=_str("DAIYA_PYANNOTE_OVERLAP_MODEL", "pyannote/speaker-diarization-community-1"),
            overlap_pad_seconds=_float("DAIYA_OVERLAP_PAD_SECONDS", 0.15),
            min_chunk_seconds=_float("DAIYA_MIN_CHUNK_SECONDS", 1.0),
            target_chunk_seconds=_float("DAIYA_TARGET_CHUNK_SECONDS", 18.0),
            max_chunk_seconds=_float("DAIYA_MAX_CHUNK_SECONDS", 25.0),
            merge_gap_seconds=_float("DAIYA_MERGE_GAP_SECONDS", 0.35),
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


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
