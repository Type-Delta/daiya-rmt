from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, AsyncIterator, Protocol

import numpy as np

SAMPLE_RATE = 16_000
CHANNELS = 1
PCM_DTYPE = np.float32


@dataclass(frozen=True)
class PCMChunk:
    """A normalized 16 kHz mono PCM chunk shared by live and replay inputs."""

    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE
    start_time: float = 0.0
    index: int = 0
    source: str = "unknown"

    def __post_init__(self) -> None:
        samples = ensure_mono_float32(self.samples)
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"PCMChunk sample_rate must be {SAMPLE_RATE}, got {self.sample_rate}")
        object.__setattr__(self, "samples", samples)

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def as_int16_bytes(self) -> bytes:
        return float32_to_pcm_int16_bytes(self.samples)

    def with_timing(self, *, start_time: float, index: int, source: str | None = None) -> "PCMChunk":
        return replace(
            self,
            start_time=start_time,
            index=index,
            source=self.source if source is None else source,
        )


class AudioSource(Protocol):
    def __aiter__(self) -> AsyncIterator[PCMChunk]:
        ...


def ensure_mono_float32(samples: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 2:
        if 1 in array.shape:
            array = array.reshape(-1)
        else:
            array = array.mean(axis=1)
    if array.ndim != 1:
        raise ValueError("audio samples must be mono or a 2-D channel array")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def pcm_int16_bytes_to_float32(data: bytes) -> np.ndarray:
    if len(data) % 2 != 0:
        raise ValueError("PCM int16 frames must have an even byte length")
    raw = np.frombuffer(data, dtype="<i2")
    return (raw.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def float32_to_pcm_int16_bytes(samples: np.ndarray | list[float]) -> bytes:
    array = ensure_mono_float32(samples)
    clipped = np.clip(array, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def chunk_from_pcm_bytes(
    data: bytes,
    *,
    start_time: float,
    index: int,
    source: str = "websocket",
) -> PCMChunk:
    return PCMChunk(
        samples=pcm_int16_bytes_to_float32(data),
        start_time=start_time,
        index=index,
        source=source,
    )


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    samples = ensure_mono_float32(samples)
    if source_rate == target_rate:
        return samples
    if samples.size == 0:
        return samples
    duration = samples.size / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def iter_chunks_from_samples(
    samples: np.ndarray,
    *,
    chunk_seconds: float = 0.5,
    source: str = "array",
    start_time: float = 0.0,
) -> list[PCMChunk]:
    samples = ensure_mono_float32(samples)
    chunk_samples = max(1, int(round(chunk_seconds * SAMPLE_RATE)))
    chunks: list[PCMChunk] = []
    for index, offset in enumerate(range(0, samples.size, chunk_samples)):
        block = samples[offset : offset + chunk_samples]
        chunks.append(
            PCMChunk(
                samples=block,
                start_time=start_time + offset / SAMPLE_RATE,
                index=index,
                source=source,
            )
        )
    return chunks


class FileReplayAudioSource:
    """Replay an audio file as paced 16 kHz mono PCM chunks."""

    def __init__(
        self,
        path: str | Path,
        *,
        chunk_seconds: float = 0.5,
        pace: bool = True,
    ) -> None:
        self.path = Path(path)
        self.chunk_seconds = chunk_seconds
        self.pace = pace

    async def __aiter__(self) -> AsyncIterator[PCMChunk]:
        chunks = await asyncio.to_thread(_read_replay_chunks, self.path, self.chunk_seconds)
        started = time.perf_counter()
        for chunk in chunks:
            if self.pace:
                target = started + chunk.start_time
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            yield chunk


class WebSocketPCMAudioSource:
    """Async source for raw little-endian int16 PCM frames from a WebSocket."""

    def __init__(self, websocket: Any, *, source: str = "websocket") -> None:
        self.websocket = websocket
        self.source = source
        self._next_index = 0
        self._next_start = 0.0

    async def __aiter__(self) -> AsyncIterator[PCMChunk]:
        while True:
            message = await self.websocket.receive()
            if "bytes" not in message or message["bytes"] is None:
                continue
            chunk = chunk_from_pcm_bytes(
                message["bytes"],
                start_time=self._next_start,
                index=self._next_index,
                source=self.source,
            )
            self._next_index += 1
            self._next_start = chunk.end_time
            yield chunk


class ServerCaptureAudioSource:
    """Placeholder for server-side mic/desktop capture."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        try:
            import sounddevice  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "server capture requires the optional 'capture' extra: sounddevice"
            ) from exc
        raise NotImplementedError("server-side capture is reserved for the next v0 iteration")


def read_audio_file(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("audio file replay requires the soundfile dependency") from exc

    data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return ensure_mono_float32(data), int(sample_rate)


def _read_replay_chunks(path: Path, chunk_seconds: float) -> list[PCMChunk]:
    samples, sample_rate = read_audio_file(path)
    samples = resample_linear(samples, sample_rate, SAMPLE_RATE)
    return iter_chunks_from_samples(
        samples,
        chunk_seconds=chunk_seconds,
        source=f"file:{path}",
    )
