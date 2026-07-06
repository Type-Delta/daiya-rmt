from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

from backends import AudioWindow, DiarizationWindowResult, PyannotePipelineBackend
from demo import load_audio, load_pyannote_pipeline, make_memory
from metrics import MetricsRecorder
from realtime import RealtimeDiarizationConfig, RealtimeDiarizationDriver, iter_replay_windows
from timeline import SpeakerSegment, TimelineStore


@dataclass(frozen=True)
class BackendBenchmark:
    backend: str
    status: str
    error: str | None
    runtime_seconds: float
    audio_duration_seconds: float
    realtime_factor: float
    der: float | None
    num_reference_speakers: int
    num_hypothesis_speakers: int
    num_reference_turns: int
    num_hypothesis_turns: int
    speaker_flips: int = 0
    corrections: int = 0
    p50_pipeline_runtime_seconds: float | None = None
    p95_pipeline_runtime_seconds: float | None = None
    p50_emit_latency_seconds: float | None = None
    p95_emit_latency_seconds: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 6 diarization benchmark against offline pyannote reference."
    )
    parser.add_argument(
        "--audio",
        nargs="+",
        default=["../../training/resources/Th-En_sample_02.mp3"],
        help="Audio file(s) to benchmark.",
    )
    parser.add_argument(
        "--backends",
        default="daiya,diart",
        help="Comma-separated backends: daiya,diart.",
    )
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=["balanced", "fast", "accuracy"],
        help="Realtime profile for Daiya and diart.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=180.0,
        help="Clip each file to this many seconds. Use 0 for full audio.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/stage6-benchmark",
        help="Directory for JSON reports and metrics CSV files.",
    )
    parser.add_argument(
        "--reference-rttm",
        default="",
        help="Optional offline-reference RTTM to load instead of running pyannote.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = RealtimeDiarizationConfig.for_profile(args.profile)
    requested_backends = [backend.strip() for backend in args.backends.split(",") if backend.strip()]
    if "diart" in requested_backends:
        patch_torchaudio_for_diart()

    reports = []
    for audio_arg in args.audio:
        audio_path = Path(audio_arg)
        if not audio_path.is_absolute():
            audio_path = Path.cwd() / audio_path
        audio_path = audio_path.resolve()

        waveform, sample_rate = load_audio(audio_path)
        waveform = clip_waveform(waveform, sample_rate, args.max_duration)
        duration = waveform.shape[1] / sample_rate

        print(f"\n=== {audio_path.name} ({duration:.1f}s) ===")
        pipeline = None
        reference_rttm = resolve_reference_rttm(args.reference_rttm, output_dir, audio_path, args.profile)
        if args.reference_rttm:
            reference = load_reference_rttm(reference_rttm)
            offline_runtime = 0.0
        else:
            pipeline = load_pyannote_pipeline()
            reference, offline_runtime = run_offline_reference(
                pipeline,
                waveform,
                sample_rate,
                audio_path.stem,
            )
            write_rttm(reference, reference_rttm)
        print(
            f"offline-reference runtime={offline_runtime:.3f}s "
            f"speakers={len(reference.labels())} turns={count_turns(reference)}"
        )

        audio_report = {
            "audio": str(audio_path),
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "profile": args.profile,
            "config": asdict(config),
            "offline_reference_runtime_seconds": offline_runtime,
            "reference_rttm": str(reference_rttm),
            "backends": [],
        }

        for backend_name in requested_backends:
            if backend_name == "daiya":
                if pipeline is None:
                    pipeline = load_pyannote_pipeline()
                metrics_path = output_dir / f"{audio_path.stem}-daiya-{args.profile}.csv"
                result = benchmark_daiya(
                    pipeline,
                    waveform,
                    sample_rate,
                    reference,
                    config,
                    metrics_path,
                )
            elif backend_name == "diart":
                result = benchmark_diart(
                    waveform,
                    sample_rate,
                    reference,
                    config,
                )
            else:
                result = skipped_backend(backend_name, duration, reference, f"unknown backend {backend_name!r}")

            audio_report["backends"].append(asdict(result))
            print(format_backend_result(result))

        report_path = output_dir / f"{audio_path.stem}-{args.profile}.json"
        report_path.write_text(json.dumps(audio_report, indent=2), encoding="utf-8")
        print(f"report={report_path}")
        reports.append(audio_report)

    summary_path = output_dir / f"summary-{args.profile}.json"
    summary_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nsummary={summary_path}")


def run_offline_reference(
    pipeline,
    waveform: torch.Tensor,
    sample_rate: int,
    uri: str,
) -> tuple[Annotation, float]:
    started = time.perf_counter()
    output = pipeline({"waveform": waveform, "sample_rate": sample_rate, "uri": uri})
    finished = time.perf_counter()
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = output.speaker_diarization
    return annotation, finished - started


def resolve_reference_rttm(
    reference_arg: str,
    output_dir: Path,
    audio_path: Path,
    profile: str,
) -> Path:
    if reference_arg:
        path = Path(reference_arg)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()
    return output_dir / f"{audio_path.stem}-{profile}-offline-reference.rttm"


def write_rttm(annotation: Annotation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        annotation.write_rttm(handle)


def load_reference_rttm(path: Path) -> Annotation:
    loaded = load_rttm(path)
    if not loaded:
        raise ValueError(f"reference RTTM has no annotations: {path}")
    return next(iter(loaded.values()))


def benchmark_daiya(
    pipeline,
    waveform: torch.Tensor,
    sample_rate: int,
    reference: Annotation,
    config: RealtimeDiarizationConfig,
    metrics_path: Path,
) -> BackendBenchmark:
    started = time.perf_counter()
    metrics = MetricsRecorder(metrics_path)
    timeline = TimelineStore()
    driver = RealtimeDiarizationDriver(
        backend=PyannotePipelineBackend(pipeline),
        memory=make_memory(),
        config=config,
        timeline=timeline,
        metrics=metrics,
    )
    speaker_flips = 0
    corrections = 0
    for window in iter_replay_windows(waveform, sample_rate, config):
        hop = driver.process_window(window)
        speaker_flips += hop.metrics.num_speaker_flips
        corrections += sum(1 for event in hop.events if event.type == "turn.corrected")
    metrics.close()
    runtime = time.perf_counter() - started

    hypothesis = timeline_to_annotation(timeline)
    duration = waveform.shape[1] / sample_rate
    return BackendBenchmark(
        backend="daiya",
        status="ok",
        error=None,
        runtime_seconds=runtime,
        audio_duration_seconds=duration,
        realtime_factor=runtime / duration if duration else 0.0,
        der=score_der(reference, hypothesis),
        num_reference_speakers=len(reference.labels()),
        num_hypothesis_speakers=len(hypothesis.labels()),
        num_reference_turns=count_turns(reference),
        num_hypothesis_turns=count_turns(hypothesis),
        speaker_flips=speaker_flips,
        corrections=corrections,
        p50_pipeline_runtime_seconds=percentile(
            [row.pipeline_runtime_seconds for row in metrics.rows], 50
        ),
        p95_pipeline_runtime_seconds=percentile(
            [row.pipeline_runtime_seconds for row in metrics.rows], 95
        ),
        p50_emit_latency_seconds=percentile(
            [row.emit_latency_seconds for row in metrics.rows], 50
        ),
        p95_emit_latency_seconds=percentile(
            [row.emit_latency_seconds for row in metrics.rows], 95
        ),
    )


def benchmark_diart(
    waveform: torch.Tensor,
    sample_rate: int,
    reference: Annotation,
    config: RealtimeDiarizationConfig,
) -> BackendBenchmark:
    duration = waveform.shape[1] / sample_rate
    started = time.perf_counter()
    try:
        hypothesis = run_diart_direct(waveform, sample_rate, config)
    except Exception as exc:
        return skipped_backend("diart", duration, reference, f"{type(exc).__name__}: {exc}")

    runtime = time.perf_counter() - started
    return BackendBenchmark(
        backend="diart",
        status="ok",
        error=None,
        runtime_seconds=runtime,
        audio_duration_seconds=duration,
        realtime_factor=runtime / duration if duration else 0.0,
        der=score_der(reference, hypothesis),
        num_reference_speakers=len(reference.labels()),
        num_hypothesis_speakers=len(hypothesis.labels()),
        num_reference_turns=count_turns(reference),
        num_hypothesis_turns=count_turns(hypothesis),
    )


def run_diart_direct(
    waveform: torch.Tensor,
    sample_rate: int,
    config: RealtimeDiarizationConfig,
) -> Annotation:
    patch_torchaudio_for_diart()
    from diart import models
    from diart.blocks.diarization import SpeakerDiarization, SpeakerDiarizationConfig
    from pyannote.core import SlidingWindow, SlidingWindowFeature

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or True
    segmentation = models.SegmentationModel.from_pyannote(
        "pyannote/segmentation",
        use_hf_token=hf_token,
    )
    embedding = models.EmbeddingModel.from_pyannote(
        "pyannote/embedding",
        use_hf_token=hf_token,
    )
    pipeline = SpeakerDiarization(
        SpeakerDiarizationConfig(
            segmentation=segmentation,
            embedding=embedding,
            duration=config.window_seconds,
            step=config.hop_seconds,
            latency=config.latency_seconds,
            sample_rate=sample_rate,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    )

    hypothesis = Annotation(uri="diart")
    for window in iter_replay_windows(waveform, sample_rate, config):
        mono = window.waveform.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        data = mono[:, None]
        sw = SlidingWindow(
            start=window.start,
            duration=1.0 / sample_rate,
            step=1.0 / sample_rate,
        )
        feature = SlidingWindowFeature(data, sw)
        outputs = pipeline([feature])
        for annotation, _audio in outputs:
            append_annotation(hypothesis, annotation, prefix="D")
    return hypothesis


def patch_torchaudio_for_diart() -> None:
    import inspect
    import torch
    import torchaudio

    if not getattr(inspect.stack, "_daiya_diart_inspect_patch", False):
        def stack_without_lazy_module_scan(*_args, **_kwargs):
            return []

        stack_without_lazy_module_scan._daiya_diart_inspect_patch = True
        inspect.stack = stack_without_lazy_module_scan

    if not getattr(torch.load, "_daiya_diart_weights_patch", False):
        original_torch_load = torch.load

        def torch_load_with_legacy_checkpoint_default(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch_load_with_legacy_checkpoint_default._daiya_diart_weights_patch = True
        torch.load = torch_load_with_legacy_checkpoint_default

    if not hasattr(torchaudio, "AudioMetaData"):
        from dataclasses import dataclass

        @dataclass
        class AudioMetaData:
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int | None = None
            encoding: str | None = None

        torchaudio.AudioMetaData = AudioMetaData
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda _backend: None


def timeline_to_annotation(timeline: TimelineStore) -> Annotation:
    annotation = Annotation(uri="daiya")
    for index, turn in enumerate(timeline.turns):
        if turn.end <= turn.start:
            continue
        annotation[Segment(turn.start, turn.end), f"turn_{index}"] = turn.speaker_id
    return annotation


def append_annotation(target: Annotation, source: Annotation, prefix: str = "") -> None:
    for index, (segment, _track, label) in enumerate(source.itertracks(yield_label=True)):
        if segment.end <= segment.start:
            continue
        target[Segment(segment.start, segment.end), f"{prefix}{index}_{count_turns(target)}"] = str(label)


def score_der(reference: Annotation, hypothesis: Annotation) -> float:
    if count_turns(hypothesis) == 0:
        return 1.0
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    return float(metric(reference, hypothesis))


def skipped_backend(
    backend: str,
    duration: float,
    reference: Annotation,
    error: str,
) -> BackendBenchmark:
    return BackendBenchmark(
        backend=backend,
        status="skipped",
        error=error,
        runtime_seconds=0.0,
        audio_duration_seconds=duration,
        realtime_factor=0.0,
        der=None,
        num_reference_speakers=len(reference.labels()),
        num_hypothesis_speakers=0,
        num_reference_turns=count_turns(reference),
        num_hypothesis_turns=0,
    )


def clip_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    max_duration: float,
) -> torch.Tensor:
    if max_duration <= 0:
        return waveform
    max_samples = int(max_duration * sample_rate)
    return waveform[:, : min(waveform.shape[1], max_samples)]


def count_turns(annotation: Annotation) -> int:
    return sum(1 for _ in annotation.itertracks(yield_label=True))


def percentile(values: Iterable[float], percent: int) -> float | None:
    values = sorted(float(value) for value in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percent / 100
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def format_backend_result(result: BackendBenchmark) -> str:
    if result.status != "ok":
        return f"{result.backend}: skipped error={result.error}"
    latency = ""
    if result.p50_pipeline_runtime_seconds is not None:
        latency = (
            f" pipeline_p50={result.p50_pipeline_runtime_seconds:.3f}s"
            f" emit_latency_p50={result.p50_emit_latency_seconds:.3f}s"
        )
    return (
        f"{result.backend}: DER={result.der:.3f} runtime={result.runtime_seconds:.3f}s "
        f"RTF={result.realtime_factor:.3f} ref_speakers={result.num_reference_speakers} "
        f"hyp_speakers={result.num_hypothesis_speakers} flips={result.speaker_flips}"
        f"{latency}"
    )


if __name__ == "__main__":
    main()
