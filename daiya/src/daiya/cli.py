from __future__ import annotations

import argparse
import asyncio
import errno
import json
from pathlib import Path
from typing import Any

from .audio import FileReplayAudioSource
from .pipeline import PipelineConfig, StreamingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daiya",
        description="Run the Daiya v0 offline file replay pipeline.",
    )
    parser.add_argument("audio_path", type=Path, help="Audio file to replay through the v0 backend.")
    parser.add_argument(
        "--asr-model",
        default=None,
        help="faster-whisper model name or CTranslate2 path. Defaults to DAIYA_ASR_MODEL, local CT2, then medium.",
    )
    parser.add_argument("--device", default="auto", help="faster-whisper device.")
    parser.add_argument("--compute-type", default="int8_float16", help="faster-whisper compute type.")
    parser.add_argument("--language", default=None, help="Optional ASR language hint.")
    parser.add_argument("--initial-prompt", default=None, help="Optional ASR prompt/context.")
    parser.add_argument(
        "--diarization-backend",
        choices=("auto", "null"),
        default="auto",
        help="Use lab pyannote when available, or force the UNKNOWN fallback.",
    )
    parser.add_argument("--chunk-seconds", type=float, default=0.5)
    parser.add_argument("--no-pace", action="store_true", help="Replay as fast as possible.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON events.")
    return parser


async def run(args: argparse.Namespace) -> int:
    pipeline = StreamingPipeline(
        PipelineConfig(
            asr_model=args.asr_model,
            asr_device=args.device,
            asr_compute_type=args.compute_type,
            language=args.language,
            initial_prompt=args.initial_prompt,
            diarization_backend=args.diarization_backend,
        )
    )
    source = FileReplayAudioSource(
        args.audio_path,
        chunk_seconds=args.chunk_seconds,
        pace=not args.no_pace,
    )

    async for chunk in source:
        for payload in pipeline.accept_chunk(chunk):
            _print_payload(payload, json_output=args.json)
    for payload in pipeline.flush():
        _print_payload(payload, json_output=args.json)
    return 0


def _print_payload(payload: dict[str, Any], *, json_output: bool) -> None:
    try:
        if json_output:
            print(json.dumps(payload, ensure_ascii=False))
            return
        event_type = payload.get("type")
        if event_type in {"transcript.partial", "transcript.final", "transcript.update"}:
            final = "final" if payload.get("final") else "partial"
            print(
                f"[{payload.get('start', 0.0):7.2f}-{payload.get('end', 0.0):7.2f}] "
                f"{payload.get('speaker', 'UNKNOWN')} {final}: {payload.get('text', '')}"
            )
        elif event_type == "error":
            print(f"error({payload.get('source', 'unknown')}): {payload.get('message', '')}")
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.EPIPE}:
            raise SystemExit(0) from exc
        raise


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
