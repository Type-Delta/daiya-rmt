from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
import os
from pathlib import Path
import subprocess
import tempfile

import soundfile as sf

from tqdm import tqdm

from .concurrency import bounded_ordered_map
from .config import PipelineConfig, ensure_dirs
from .types import Chunk, NormalizedAudio


def _source_id(path: Path) -> str:
    digest = sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}"


def _destination_temp_path(destination: Path) -> Path:
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        # FFmpeg chooses the muxer from the output suffix unless an explicit
        # ``-f`` is supplied.  Keep the temporary output a WAV as well as the
        # final destination so atomic publication cannot change the format.
        suffix=".wav",
        dir=destination.parent,
    )
    os.close(fd)
    return Path(temporary)


def _run_ffmpeg_atomic(command: list[str], destination: Path) -> None:
    temporary = _destination_temp_path(destination)
    try:
        subprocess.run([*command, str(temporary)], check=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_audio_file(path: Path, config: PipelineConfig) -> NormalizedAudio:
    out_dir = config.work_dir / "normalized"
    ensure_dirs([out_dir])
    source_id = _source_id(path)
    output = out_dir / f"{source_id}.wav"

    cmd = [
        config.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-vn",
        "-ac",
        str(config.channels),
        "-ar",
        str(config.sample_rate),
        "-c:a",
        config.audio_codec,
    ]
    _run_ffmpeg_atomic(cmd, output)
    try:
        duration_seconds: float | None = float(sf.info(output).duration)
    except RuntimeError:
        # FFmpeg has already succeeded.  This path mainly keeps lightweight
        # command-mocking callers compatible; real pipeline output is a valid
        # WAV and therefore always records its duration.
        duration_seconds = None
    return NormalizedAudio(
        source_path=path,
        normalized_path=output,
        source_id=source_id,
        duration_seconds=duration_seconds,
    )


def normalize_audio_files(paths: list[Path], config: PipelineConfig) -> list[NormalizedAudio]:
    if not paths:
        return []

    ordered_paths = sorted(paths, key=lambda path: str(path))
    normalized: list[NormalizedAudio] = []
    max_in_flight = config.ffmpeg_max_in_flight
    max_workers = min(config.ffmpeg_max_workers, max_in_flight)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = bounded_ordered_map(
            executor,
            lambda path: normalize_audio_file(path, config),
            ordered_paths,
            max_in_flight,
        )
        for item in tqdm(results, total=len(ordered_paths), desc="ffmpeg normalize"):
            normalized.append(item)
    return normalized


def _export_interval(source: Path, destination: Path, start: float, end: float, config: PipelineConfig) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Never concatenate VAD islands.  Doing so deleted short pauses and any
    # VAD false-negative gap from the audio seen by the labeler.  A Chunk is a
    # single wall-clock window and its VAD islands remain metadata only.
    filter_complex = (
        f"[0:a]atrim=start={start:.6f}:end={end:.6f},"
        "asetpts=PTS-STARTPTS[out]"
    )

    cmd = [
        config.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-ac",
        str(config.channels),
        "-ar",
        str(config.sample_rate),
        "-c:a",
        config.audio_codec,
    ]
    _run_ffmpeg_atomic(cmd, destination)


def export_chunk(chunk: Chunk, config: PipelineConfig) -> None:
    """Export a disjoint owned artifact and, only when needed, labeler pre-roll.

    The pre-roll WAV is a private labeling input.  The owned WAV at
    ``chunk_path`` is the only file the dataset exporter can publish.
    """
    _export_interval(chunk.source.normalized_path, chunk.chunk_path, chunk.start, chunk.end, config)
    if chunk.has_labeling_preroll:
        _export_interval(
            chunk.source.normalized_path,
            chunk.labeling_path,
            chunk.labeling_start,
            chunk.labeling_end,
            config,
        )
