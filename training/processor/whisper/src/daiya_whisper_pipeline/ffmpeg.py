from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1
from pathlib import Path
import subprocess

from tqdm import tqdm

from .config import PipelineConfig, ensure_dirs
from .types import Chunk, NormalizedAudio


def _source_id(path: Path) -> str:
    digest = sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}"


def normalize_audio_file(path: Path, config: PipelineConfig) -> NormalizedAudio:
    out_dir = config.work_dir / "normalized"
    ensure_dirs([out_dir])
    source_id = _source_id(path)
    output = out_dir / f"{source_id}.wav"
    if output.exists():
        return NormalizedAudio(source_path=path, normalized_path=output, source_id=source_id)

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
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return NormalizedAudio(source_path=path, normalized_path=output, source_id=source_id)


def normalize_audio_files(paths: list[Path], config: PipelineConfig) -> list[NormalizedAudio]:
    if not paths:
        return []

    normalized: list[NormalizedAudio] = []
    with ThreadPoolExecutor(max_workers=config.ffmpeg_max_workers) as executor:
        futures = [executor.submit(normalize_audio_file, path, config) for path in paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="ffmpeg normalize"):
            normalized.append(future.result())
    return sorted(normalized, key=lambda item: str(item.source_path))


def export_chunk(chunk: Chunk, config: PipelineConfig) -> None:
    chunk.chunk_path.parent.mkdir(parents=True, exist_ok=True)
    if chunk.chunk_path.exists():
        return

    filters: list[str] = []
    labels: list[str] = []
    for idx, interval in enumerate(chunk.intervals):
        label = f"a{idx}"
        filters.append(
            f"[0:a]atrim=start={interval.start:.3f}:end={interval.end:.3f},"
            f"asetpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")

    if len(labels) == 1:
        filter_complex = filters[0]
        map_label = labels[0]
    else:
        filter_complex = ";".join(filters) + ";" + "".join(labels) + f"concat=n={len(labels)}:v=0:a=1[out]"
        map_label = "[out]"

    cmd = [
        config.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(chunk.source.normalized_path),
        "-filter_complex",
        filter_complex,
        "-map",
        map_label,
        "-ac",
        str(config.channels),
        "-ar",
        str(config.sample_rate),
        "-c:a",
        config.audio_codec,
        str(chunk.chunk_path),
    ]
    subprocess.run(cmd, check=True)
