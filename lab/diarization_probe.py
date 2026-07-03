from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import wave
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

try:
    import numpy as np
except ImportError as exc:
    np = None  # type: ignore[assignment]
    NUMPY_IMPORT_ERROR: ImportError | None = exc
else:
    NUMPY_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DAIYA_SRC = REPO_ROOT / "daiya" / "src"
LAB_STATEFUL_ROOT = REPO_ROOT / "lab" / "statefull-diarization"


@dataclass
class ProbeSummary:
    name: str
    status: str
    message: str = ""
    audio_path: str = ""
    audio_seconds: float = 0.0
    chunks: int = 0
    windows: int = 0
    events: int = 0
    transcript_events: int = 0
    partial_events: int = 0
    final_events: int = 0
    update_events: int = 0
    tick_events: int = 0
    error_events: int = 0
    unique_segments: int = 0
    unique_speakers: list[str] | None = None
    speaker_event_counts: dict[str, int] | None = None
    speaker_final_counts: dict[str, int] | None = None
    first_turn_start: float | None = None
    last_turn_end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def main() -> int:
    load_lab_env_file()
    args = parse_args()
    if NUMPY_IMPORT_ERROR is not None:
        print(
            "ERROR: numpy is required for audio chunking. Run through the project environment, "
            "for example: uv run python lab/diarization_probe.py <audio>",
            file=sys.stderr,
        )
        return 2

    audio_path = args.audio.resolve()

    try:
        samples, sample_rate = load_audio_samples(audio_path)
    except ProbeSkip as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    production_summary: ProbeSummary | None = None
    lab_summary: ProbeSummary | None = None

    if args.mode in {"production", "compare"}:
        production_summary = run_production_probe(samples, sample_rate, audio_path, args)

    if args.mode in {"lab", "compare"}:
        lab_summary = run_lab_probe(samples, sample_rate, audio_path, args)

    summaries = [summary for summary in (production_summary, lab_summary) if summary is not None]
    if args.json:
        print(json.dumps([summary.to_dict() for summary in summaries], indent=2, sort_keys=True))
    else:
        for index, summary in enumerate(summaries):
            if index:
                print()
            print_summary(summary)
        if production_summary and lab_summary:
            print()
            print_comparison(production_summary, lab_summary)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Daiya diarization from an audio file. Production mode uses "
            "StreamingPipeline(enable_asr=False, enable_diarization=True) and can use either "
            "the real lab pyannote backend or the null fallback."
        )
    )
    parser.add_argument("audio", type=Path, help="Input audio file. WAV works without soundfile.")
    parser.add_argument(
        "--mode",
        choices=("production", "lab", "compare"),
        default="production",
        help="Run the production null-diarization probe, the lab pyannote path, or both.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=positive_float,
        default=0.5,
        help="Chunk size used when feeding the production StreamingPipeline.",
    )
    parser.add_argument(
        "--profile",
        default="balanced",
        help="Diarization profile passed to production and lab realtime configs.",
    )
    parser.add_argument(
        "--diarization-backend",
        choices=("auto", "null"),
        default="auto",
        help="Production backend selector. auto loads lab pyannote when available; null forces UNKNOWN.",
    )
    parser.add_argument("--window-seconds", type=positive_float, default=None)
    parser.add_argument("--hop-seconds", type=positive_float, default=None)
    parser.add_argument("--latency-seconds", type=positive_float, default=None)
    parser.add_argument("--commit-delay-seconds", type=non_negative_float, default=None)
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        help="Hugging Face token for gated pyannote models. Defaults to HF_TOKEN/HUGGINGFACE_TOKEN.",
    )
    parser.add_argument(
        "--pyannote-model",
        default=os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"),
        help="Pyannote model id or local path for the lab backend.",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("DEVICE", "cuda"),
        help="Lab pyannote device preference. Uses CUDA only when available.",
    )
    parser.add_argument(
        "--allow-no-token",
        action="store_true",
        help="Attempt lab pyannote loading even when no HF token is configured.",
    )
    parser.add_argument(
        "--events",
        action="store_true",
        help="Print compact event rows before the summary.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary JSON.")
    return parser.parse_args()


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def run_production_probe(
    samples: np.ndarray,
    sample_rate: int,
    audio_path: Path,
    args: argparse.Namespace,
) -> ProbeSummary:
    ensure_production_on_path()
    from daiya.audio import SAMPLE_RATE, iter_chunks_from_samples, resample_linear
    from daiya.pipeline import PipelineConfig, StreamingPipeline

    samples_16k = resample_linear(samples, sample_rate, SAMPLE_RATE)
    chunks = iter_chunks_from_samples(
        samples_16k,
        chunk_seconds=args.chunk_seconds,
        source=f"probe:{audio_path}",
    )
    pipeline = StreamingPipeline(
        PipelineConfig(
            enable_asr=False,
            enable_diarization=True,
            diarization_backend=args.diarization_backend,
            diarization_profile=args.profile,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            latency_seconds=args.latency_seconds,
            commit_delay_seconds=args.commit_delay_seconds,
        )
    )

    payloads: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_payloads = pipeline.accept_chunk(chunk)
        payloads.extend(chunk_payloads)
        if args.events and not args.json:
            print_payload_rows("production", chunk_payloads)
    flush_payloads = pipeline.flush()
    payloads.extend(flush_payloads)
    if args.events and not args.json:
        print_payload_rows("production", flush_payloads)

    summary = summarize_payloads("production", payloads)
    summary.audio_path = str(audio_path)
    summary.audio_seconds = len(samples_16k) / SAMPLE_RATE
    summary.chunks = len(chunks)
    if summary.unique_speakers == ["UNKNOWN"]:
        summary.message = (
            "Only UNKNOWN speaker IDs were emitted. This reproduces the prototype null diarizer path."
        )
    elif not summary.unique_speakers:
        summary.message = "No speaker events were emitted by the production pipeline."
    else:
        summary.message = "Production pipeline emitted non-UNKNOWN speakers."
    return summary


def run_lab_probe(
    samples: np.ndarray,
    sample_rate: int,
    audio_path: Path,
    args: argparse.Namespace,
) -> ProbeSummary:
    if not LAB_STATEFUL_ROOT.exists():
        return skipped("lab", f"lab stateful diarization path not found: {LAB_STATEFUL_ROOT}")

    model_is_local = Path(args.pyannote_model).exists()
    if not args.hf_token and not args.allow_no_token and not model_is_local:
        return skipped(
            "lab",
            "HF token is not configured; set HF_TOKEN/HUGGINGFACE_TOKEN or pass --allow-no-token.",
        )

    try:
        modules = load_lab_modules()
        demo = modules["demo"]
        realtime = modules["realtime"]
        backends = modules["backends"]
        speaker_memory = modules["speaker_memory"]
        torch = import_torch()
    except ProbeSkip as exc:
        return skipped("lab", str(exc))

    try:
        configure_demo_module(demo, args)
        pipeline = demo.load_pyannote_pipeline()
        backend = backends.PyannotePipelineBackend(pipeline)
        config = make_lab_realtime_config(realtime, args)
        memory = speaker_memory.SpeakerMemory()
        driver = realtime.RealtimeDiarizationDriver(backend=backend, memory=memory, config=config)
        waveform = torch.from_numpy(as_16k(samples, sample_rate).copy()).unsqueeze(0)
        windows = list(realtime.iter_replay_windows(waveform, 16000, config))
    except SystemExit as exc:
        return skipped("lab", str(exc))
    except Exception as exc:  # optional dependency errors vary a lot across pyannote installs
        return skipped("lab", f"{type(exc).__name__}: {exc}")

    if not windows:
        return skipped(
            "lab",
            (
                f"audio is shorter than the lab realtime window "
                f"({config.window_seconds:.2f}s); pass --window-seconds with a smaller value."
            ),
            audio_path=audio_path,
            audio_seconds=len(as_16k(samples, sample_rate)) / 16000,
        )

    events: list[Any] = []
    try:
        for window in windows:
            hop = driver.process_window(window)
            events.extend(hop.events)
            if args.events and not args.json:
                print_lab_event_rows("lab", hop.events)
    except Exception as exc:
        return skipped("lab", f"{type(exc).__name__}: {exc}")

    summary = summarize_lab_events("lab", events)
    summary.audio_path = str(audio_path)
    summary.audio_seconds = len(as_16k(samples, sample_rate)) / 16000
    summary.windows = len(windows)
    summary.message = "Lab pyannote realtime backend completed."
    return summary


def ensure_production_on_path() -> None:
    src = str(DAIYA_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def load_lab_env_file() -> None:
    env_path = LAB_STATEFUL_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_audio_samples(path: Path) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise ProbeSkip(f"audio file does not exist: {path}")
    if path.suffix.lower() == ".wav":
        try:
            return load_wav_stdlib(path)
        except wave.Error:
            pass

    ensure_production_on_path()
    try:
        from daiya.audio import read_audio_file
    except Exception as exc:
        raise ProbeSkip(f"cannot import production audio reader: {type(exc).__name__}: {exc}") from exc

    try:
        return read_audio_file(path)
    except RuntimeError as exc:
        raise ProbeSkip(str(exc)) from exc
    except Exception as exc:
        raise ProbeSkip(f"cannot read audio file: {type(exc).__name__}: {exc}") from exc


def load_wav_stdlib(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ProbeSkip(f"unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.nan_to_num(audio.astype(np.float32, copy=False)), sample_rate


def as_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    ensure_production_on_path()
    from daiya.audio import SAMPLE_RATE, resample_linear

    return resample_linear(samples, sample_rate, SAMPLE_RATE)


def summarize_payloads(name: str, payloads: list[dict[str, Any]]) -> ProbeSummary:
    speaker_counts: Counter[str] = Counter()
    speaker_final_counts: Counter[str] = Counter()
    segment_ids: set[str] = set()
    starts: list[float] = []
    ends: list[float] = []
    event_types = Counter(str(payload.get("type", "")) for payload in payloads)

    for payload in payloads:
        speaker = payload.get("speaker")
        event_type = str(payload.get("type", ""))
        if speaker is not None:
            speaker_id = str(speaker)
            speaker_counts[speaker_id] += 1
            if event_type == "transcript.final":
                speaker_final_counts[speaker_id] += 1
        segment_id = payload.get("segment_id")
        if segment_id is not None:
            segment_ids.add(str(segment_id))
        start = as_float_or_none(payload.get("start"))
        end = as_float_or_none(payload.get("end"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)

    return ProbeSummary(
        name=name,
        status="ok",
        events=len(payloads),
        transcript_events=sum(
            event_types[event_type]
            for event_type in ("transcript.partial", "transcript.final", "transcript.update")
        ),
        partial_events=event_types["transcript.partial"],
        final_events=event_types["transcript.final"],
        update_events=event_types["transcript.update"],
        tick_events=event_types["tick"],
        error_events=event_types["error"],
        unique_segments=len(segment_ids),
        unique_speakers=sorted(speaker_counts),
        speaker_event_counts=dict(sorted(speaker_counts.items())),
        speaker_final_counts=dict(sorted(speaker_final_counts.items())),
        first_turn_start=min(starts) if starts else None,
        last_turn_end=max(ends) if ends else None,
    )


def summarize_lab_events(name: str, events: list[Any]) -> ProbeSummary:
    speaker_counts: Counter[str] = Counter()
    speaker_final_counts: Counter[str] = Counter()
    segment_ids: set[str] = set()
    starts: list[float] = []
    ends: list[float] = []
    event_types: Counter[str] = Counter()

    for event in events:
        event_type = str(getattr(event, "type", ""))
        event_types[event_type] += 1
        turn = getattr(event, "turn", None)
        if turn is None:
            continue
        speaker_id = str(getattr(turn, "speaker_id", ""))
        if speaker_id:
            speaker_counts[speaker_id] += 1
            if bool(getattr(turn, "final", False)):
                speaker_final_counts[speaker_id] += 1
        turn_id = getattr(turn, "turn_id", None)
        if turn_id is not None:
            segment_ids.add(str(turn_id))
        start = as_float_or_none(getattr(turn, "start", None))
        end = as_float_or_none(getattr(turn, "end", None))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)

    return ProbeSummary(
        name=name,
        status="ok",
        events=len(events),
        transcript_events=len(events),
        partial_events=event_types["turn.created"] + event_types["turn.updated"],
        final_events=event_types["turn.committed"],
        update_events=event_types["turn.speaker_corrected"],
        unique_segments=len(segment_ids),
        unique_speakers=sorted(speaker_counts),
        speaker_event_counts=dict(sorted(speaker_counts.items())),
        speaker_final_counts=dict(sorted(speaker_final_counts.items())),
        first_turn_start=min(starts) if starts else None,
        last_turn_end=max(ends) if ends else None,
    )


def print_summary(summary: ProbeSummary) -> None:
    print(f"{summary.name}: {summary.status.upper()}")
    if summary.message:
        print(f"  message: {summary.message}")
    if summary.audio_path:
        print(f"  audio: {summary.audio_path}")
    if summary.audio_seconds:
        print(f"  audio_seconds: {summary.audio_seconds:.2f}")
    if summary.chunks:
        print(f"  chunks: {summary.chunks}")
    if summary.windows:
        print(f"  windows: {summary.windows}")
    print(f"  events: {summary.events}")
    print(f"  transcript_events: {summary.transcript_events}")
    print(f"  partial/final/update: {summary.partial_events}/{summary.final_events}/{summary.update_events}")
    if summary.tick_events:
        print(f"  tick_events: {summary.tick_events}")
    if summary.error_events:
        print(f"  error_events: {summary.error_events}")
    print(f"  unique_segments: {summary.unique_segments}")
    print(f"  unique_speakers: {summary.unique_speakers or []}")
    print(f"  speaker_event_counts: {summary.speaker_event_counts or {}}")
    print(f"  speaker_final_counts: {summary.speaker_final_counts or {}}")
    if summary.first_turn_start is not None and summary.last_turn_end is not None:
        print(f"  turn_span: {summary.first_turn_start:.2f}-{summary.last_turn_end:.2f}s")


def print_comparison(production: ProbeSummary, lab: ProbeSummary) -> None:
    print("comparison:")
    print(f"  production_speakers: {production.unique_speakers or []}")
    print(f"  lab_speakers: {lab.unique_speakers or []}")
    print(f"  production_final_events: {production.final_events}")
    print(f"  lab_final_events: {lab.final_events}")
    if production.unique_speakers == ["UNKNOWN"] and lab.status == "ok" and lab.unique_speakers:
        print("  result: production is on the null diarizer path while lab emitted speaker IDs.")
    elif production.unique_speakers == ["UNKNOWN"]:
        print("  result: production null diarizer reproduced; lab comparison did not complete with speakers.")
    else:
        print("  result: production did not look like the UNKNOWN-only null diarizer path.")


def print_payload_rows(label: str, payloads: Iterable[dict[str, Any]]) -> None:
    for payload in payloads:
        event_type = payload.get("type")
        if event_type == "tick":
            continue
        print(
            f"{label:<10} {str(event_type):<18} {str(payload.get('segment_id', '')):<12} "
            f"{float(payload.get('start', 0.0)):7.2f}-{float(payload.get('end', 0.0)):7.2f}s "
            f"{str(payload.get('speaker', '')):<16}"
        )


def print_lab_event_rows(label: str, events: Iterable[Any]) -> None:
    for event in events:
        turn = getattr(event, "turn", None)
        if turn is None:
            continue
        print(
            f"{label:<10} {str(getattr(event, 'type', '')):<18} {str(getattr(turn, 'turn_id', '')):<12} "
            f"{float(getattr(turn, 'start', 0.0)):7.2f}-{float(getattr(turn, 'end', 0.0)):7.2f}s "
            f"{str(getattr(turn, 'speaker_id', '')):<16}"
        )


def load_lab_modules() -> dict[str, ModuleType]:
    inserted = False
    root = str(LAB_STATEFUL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
        inserted = True
    try:
        return {
            name: importlib.import_module(name)
            for name in ("timeline", "backends", "metrics", "speaker_memory", "realtime", "demo")
        }
    except ImportError as exc:
        raise ProbeSkip(f"cannot import lab stateful diarization modules: {exc}") from exc
    finally:
        if inserted:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


def import_torch() -> ModuleType:
    try:
        import torch
    except ImportError as exc:
        raise ProbeSkip("torch is not installed; install lab/statefull-diarization deps") from exc
    return torch


def configure_demo_module(demo: ModuleType, args: argparse.Namespace) -> None:
    demo.HF_TOKEN = args.hf_token
    demo.MODEL_ID = args.pyannote_model
    demo.DEVICE = args.device


def make_lab_realtime_config(realtime: ModuleType, args: argparse.Namespace) -> Any:
    config = realtime.RealtimeDiarizationConfig.for_profile(args.profile)
    updates = {
        "window_seconds": args.window_seconds,
        "hop_seconds": args.hop_seconds,
        "latency_seconds": args.latency_seconds,
        "commit_delay_seconds": args.commit_delay_seconds,
    }
    for field, value in updates.items():
        if value is not None:
            config = replace(config, **{field: float(value)})
    return config


def skipped(
    name: str,
    message: str,
    *,
    audio_path: Path | None = None,
    audio_seconds: float = 0.0,
) -> ProbeSummary:
    return ProbeSummary(
        name=name,
        status="skipped",
        message=message,
        audio_path="" if audio_path is None else str(audio_path),
        audio_seconds=audio_seconds,
        unique_speakers=[],
        speaker_event_counts={},
        speaker_final_counts={},
    )


def as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ProbeSkip(RuntimeError):
    pass


if __name__ == "__main__":
    raise SystemExit(main())
