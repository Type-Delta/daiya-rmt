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
        "--no-asr-prompt-memory",
        action="store_true",
        help="Disable rolling transcript/term context in the ASR initial_prompt.",
    )
    parser.add_argument(
        "--asr-left-context",
        action="store_true",
        help="Enable experimental left-audio context retries for short or low-confidence utterances.",
    )
    parser.add_argument(
        "--asr-delayed-correction",
        action="store_true",
        help="Enable experimental rolling-window ASR correction updates.",
    )
    parser.add_argument(
        "--asr-tiny-merge",
        action="store_true",
        help="Enable experimental tiny VAD utterance deferral/merge before ASR.",
    )
    parser.add_argument("--asr-left-context-seconds", type=float, default=3.0)
    parser.add_argument("--asr-tiny-utterance-seconds", type=float, default=0.55)
    parser.add_argument("--asr-delayed-correction-window-seconds", type=float, default=10.0)
        "--segmenter-backend",
        choices=("energy", "silero", "auto"),
        default="energy",
        help="Utterance segmentation backend. Defaults to dependency-free energy VAD.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="VAD threshold. Defaults to 0.012 for energy and 0.5 for Silero.",
    )
    parser.add_argument("--vad-min-speech-seconds", type=float, default=0.25)
    parser.add_argument("--vad-min-silence-seconds", type=float, default=0.45)
    parser.add_argument(
        "--vad-speech-padding-seconds",
        type=float,
        default=None,
        help="Speech padding. Defaults to 0 for energy and 0.1 seconds for Silero.",
    )
    parser.add_argument("--utterance-cap-seconds", type=float, default=8.0)
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
            segmenter_backend=args.segmenter_backend,
            vad_threshold=args.vad_threshold,
            vad_min_speech_seconds=args.vad_min_speech_seconds,
            vad_min_silence_seconds=args.vad_min_silence_seconds,
            vad_speech_padding_seconds=args.vad_speech_padding_seconds,
            utterance_cap_seconds=args.utterance_cap_seconds,
            diarization_backend=args.diarization_backend,
            asr_prompt_memory_enabled=not args.no_asr_prompt_memory,
            asr_left_context_enabled=args.asr_left_context,
            asr_delayed_correction_enabled=args.asr_delayed_correction,
            asr_tiny_utterance_merge_enabled=args.asr_tiny_merge,
            asr_left_context_seconds=args.asr_left_context_seconds,
            asr_tiny_utterance_seconds=args.asr_tiny_utterance_seconds,
            asr_delayed_correction_window_seconds=args.asr_delayed_correction_window_seconds,
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
