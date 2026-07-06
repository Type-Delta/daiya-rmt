from __future__ import annotations

import shutil

from rich.console import Console
from tqdm import tqdm

from .config import PipelineConfig, ensure_dirs
from .diarization import OverlapDetector
from .export import export_audiofolder
from .ffmpeg import export_chunk, normalize_audio_files
from .llm import OpenRouterAudioTranscriber, transcribe_chunks
from .segmentation import build_chunks
from .vad import SileroVad


def run_pipeline(config: PipelineConfig) -> None:
    console = Console()
    ensure_dirs([config.work_dir, config.output_dir])

    files = config.find_audio_files()
    if not files:
        raise FileNotFoundError(f"No audio files found in {config.input_dir}")

    console.print(f"[bold]Found {len(files)} raw audio files[/bold]")
    normalized = normalize_audio_files(files, config)

    vad = SileroVad(config)
    overlap = OverlapDetector(config)

    chunks = []
    for audio in tqdm(normalized, desc="segment"):
        speech = vad.detect(audio.normalized_path)
        dirty = overlap.detect(audio.normalized_path)
        audio_chunks = build_chunks(audio, speech, dirty, config)
        for chunk in audio_chunks:
            export_chunk(chunk, config)
        chunks.extend(audio_chunks)

    console.print(f"[bold]Built {len(chunks)} clean chunks[/bold]")
    if not chunks:
        raise RuntimeError("No chunks survived VAD/overlap filtering")

    transcriber = OpenRouterAudioTranscriber(config)
    labeled = transcribe_chunks(chunks, transcriber, config)
    metadata_path = export_audiofolder(labeled, config)

    if not config.keep_intermediate:
        shutil.rmtree(config.work_dir, ignore_errors=True)

    console.print(f"[green]Done.[/green] Wrote HuggingFace audiofolder dataset to {config.output_dir}")
    console.print(f"Metadata: {metadata_path}")
