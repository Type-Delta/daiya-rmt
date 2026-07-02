from __future__ import annotations

from pathlib import Path
import json
import shutil

from .config import PipelineConfig, ensure_dirs
from .types import LabeledChunk


def export_audiofolder(labeled_chunks: list[LabeledChunk], config: PipelineConfig) -> Path:
    split_dir = config.output_dir / config.dataset_split
    ensure_dirs([config.output_dir, split_dir])
    metadata_path = config.output_dir / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as handle:
        for item in labeled_chunks:
            target_name = f"{config.dataset_split}/{item.chunk.source.source_id}_{item.chunk.index:05d}.wav"
            target_path = config.output_dir / target_name
            shutil.copy2(item.chunk.chunk_path, target_path)

            row = {
                "file_name": target_name.replace("\\", "/"),
                config.text_column: item.transcript_text,
                "language": item.language or config.language_hint,
                "context_before": item.extra.get("context_before", ""),
                "context_after": item.extra.get("context_after", ""),
                "source_file": str(item.chunk.source.source_path),
                "source_start": round(item.chunk.start, 3),
                "source_end": round(item.chunk.end, 3),
                "speech_duration": round(item.chunk.speech_duration, 3),
                "notes": item.notes,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metadata_path
