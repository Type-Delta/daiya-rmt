from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from backends import PyannotePipelineBackend
from metrics import MetricsRecorder
from realtime import (
    RealtimeDiarizationConfig,
    RealtimeDiarizationDriver,
    RollingWindowScheduler,
    print_events,
    run_replay,
)
from speaker_memory import SpeakerMemory


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env_file()


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# Keep this throwaway-friendly: edit these constants or use env vars.
AUDIO_PATH = os.getenv("AUDIO_PATH", "")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
MODEL_ID = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1")

CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "20"))
STRIDE_SECONDS = float(os.getenv("STRIDE_SECONDS", "12"))
DIARIZATION_PROFILE = os.getenv("DIARIZATION_PROFILE", "balanced")
DIARIZATION_WINDOW_SECONDS = os.getenv("DIARIZATION_WINDOW_SECONDS")
DIARIZATION_HOP_SECONDS = os.getenv("DIARIZATION_HOP_SECONDS")
DIARIZATION_LATENCY_SECONDS = os.getenv("DIARIZATION_LATENCY_SECONDS")
DIARIZATION_COMMIT_DELAY_SECONDS = os.getenv("DIARIZATION_COMMIT_DELAY_SECONDS")
METRICS_PATH = os.getenv("METRICS_PATH", "")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.38"))
MIN_NEW_PROFILE_SECONDS = float(os.getenv("MIN_NEW_PROFILE_SECONDS", "6.0"))
CANDIDATE_PROMOTE_SECONDS = float(os.getenv("CANDIDATE_PROMOTE_SECONDS", "3.0"))
CANDIDATE_PROMOTE_OBSERVATIONS = int(os.getenv("CANDIDATE_PROMOTE_OBSERVATIONS", "2"))
EMBEDDING_EXCLUDE_OVERLAP = env_bool("EMBEDDING_EXCLUDE_OVERLAP", True)
CLUSTER_FALLBACK_TO_ACTIVE_SPEECH = env_bool("CLUSTER_FALLBACK_TO_ACTIVE_SPEECH", True)
DEVICE = os.getenv("DEVICE", "cuda")
LIVE_SOURCE = os.getenv("LIVE_SOURCE", "mic").strip().lower()
LIVE_DEVICE = os.getenv("LIVE_DEVICE", "").strip()
LIVE_SAMPLE_RATE = int(os.getenv("LIVE_SAMPLE_RATE", "16000"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny pyannote speaker-memory demo.")
    parser.add_argument(
        "--mem-graph",
        nargs="?",
        const="artifacts/memory_profiles.png",
        default="",
        help="Save a dendrogram of global speaker-memory profiles.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Capture live microphone/desktop audio and run rolling realtime diarization.",
    )
    parser.add_argument(
        "--legacy-chunks",
        action="store_true",
        help="Use the old chunk/stride loop instead of the rolling realtime driver.",
    )
    return parser.parse_args()


def speech_seconds(annotation) -> dict[str, float]:
    totals: dict[str, float] = {}
    for turn, _, label in annotation.itertracks(yield_label=True):
        totals[label] = totals.get(label, 0.0) + turn.duration
    return totals


def print_global_turns(annotation, mapping: dict[str, str], offset: float) -> None:
    for turn, _, label in annotation.itertracks(yield_label=True):
        global_label = mapping.get(label, label)
        ts_start = offset + turn.start
        ts_end = offset + turn.end
        print(
            f"{(int(ts_start) // 60):02d}:{(int(ts_start) % 60):02d} -> {(int(ts_end) // 60):02d}:{(int(ts_end) % 60):02d}  "
            f"{global_label}  (local {label})"
        )


def load_wav(path: Path) -> tuple[torch.Tensor, int]:
    from scipy.io import wavfile

    sample_rate, samples = wavfile.read(path)
    samples = np.asarray(samples)

    if samples.ndim == 1:
        samples = samples[None, :]
    else:
        samples = samples.T

    if np.issubdtype(samples.dtype, np.integer):
        max_value = np.iinfo(samples.dtype).max
        samples = samples.astype(np.float32) / max_value
    else:
        samples = samples.astype(np.float32)

    return torch.from_numpy(samples), int(sample_rate)


def load_with_ffmpeg(path: Path, sample_rate: int = 16000) -> tuple[torch.Tensor, int]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    samples = np.frombuffer(completed.stdout, dtype=np.float32)
    return torch.from_numpy(samples.copy()).unsqueeze(0), sample_rate


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    if not path.exists():
        raise SystemExit(f"AUDIO_PATH does not exist: {path}")

    if path.suffix.lower() == ".wav":
        return load_wav(path)

    return load_with_ffmpeg(path)


def patch_clustering_empty_training_fallback(pipeline) -> None:
    clustering = getattr(pipeline, "clustering", None)
    if clustering is None or getattr(clustering, "_daiya_empty_training_fallback", False):
        return

    original_filter_embeddings = clustering.filter_embeddings

    def filter_embeddings_with_fallback(
        embeddings: np.ndarray,
        segmentations=None,
        min_active_ratio: float = 0.2,
    ):
        train_embeddings, chunk_idx, speaker_idx = original_filter_embeddings(
            embeddings,
            segmentations=segmentations,
            min_active_ratio=min_active_ratio,
        )
        if train_embeddings.shape[0] > 0 or segmentations is None:
            return train_embeddings, chunk_idx, speaker_idx

        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 3 or embeddings.size == 0:
            return train_embeddings, chunk_idx, speaker_idx

        segmentation_data = np.nan_to_num(
            np.asarray(segmentations.data),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if segmentation_data.ndim != 3:
            return train_embeddings, chunk_idx, speaker_idx

        _, num_frames, _ = segmentation_data.shape
        active_frames = np.sum(segmentation_data, axis=1)
        valid_embeddings = np.isfinite(embeddings).all(axis=2)

        # Pyannote filters clustering embeddings by clean non-overlap speech.
        # Short live chunks can be mostly overlap, leaving no train embeddings and
        # causing an empty centroid mean. Fall back to total active speech instead.
        enough_total_speech = active_frames >= min_active_ratio * num_frames
        chunk_idx, speaker_idx = np.where(valid_embeddings & enough_total_speech)

        if chunk_idx.size == 0:
            chunk_idx, speaker_idx = np.where(valid_embeddings & (active_frames > 0))

        if chunk_idx.size == 0:
            return train_embeddings, chunk_idx, speaker_idx

        return embeddings[chunk_idx, speaker_idx], chunk_idx, speaker_idx

    clustering.filter_embeddings = filter_embeddings_with_fallback
    clustering._daiya_empty_training_fallback = True


def load_pyannote_pipeline():
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"\s*torchcodec is not installed correctly.*",
        category=UserWarning,
    )
    from pyannote.audio import Pipeline

    try:
        try:
            pipeline = Pipeline.from_pretrained(MODEL_ID, token=HF_TOKEN)
        except TypeError as exc:
            if "token" not in str(exc):
                raise
            try:
                pipeline = Pipeline.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
            except TypeError as legacy_exc:
                if "use_auth_token" not in str(legacy_exc):
                    raise
                pipeline = Pipeline.from_pretrained(MODEL_ID)
    except Exception as exc:
        message = str(exc)
        if "gated repo" in message.lower() or "403" in message:
            raise SystemExit(
                "Hugging Face denied access to the pyannote model.\n"
                f"Model: {MODEL_ID}\n"
                "Open the model page, accept the user conditions, and make sure "
                "HF_TOKEN belongs to that authorized account."
            ) from exc
        raise

    if DEVICE == "cuda" and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    if hasattr(pipeline, "embedding_exclude_overlap"):
        pipeline.embedding_exclude_overlap = EMBEDDING_EXCLUDE_OVERLAP

    if CLUSTER_FALLBACK_TO_ACTIVE_SPEECH:
        patch_clustering_empty_training_fallback(pipeline)

    return pipeline


def make_memory() -> SpeakerMemory:
    return SpeakerMemory(
        match_threshold=MATCH_THRESHOLD,
        min_new_profile_seconds=MIN_NEW_PROFILE_SECONDS,
        candidate_promote_seconds=CANDIDATE_PROMOTE_SECONDS,
        candidate_promote_observations=CANDIDATE_PROMOTE_OBSERVATIONS,
    )


def make_realtime_config() -> RealtimeDiarizationConfig:
    config = RealtimeDiarizationConfig.for_profile(DIARIZATION_PROFILE)
    return RealtimeDiarizationConfig(
        window_seconds=float(DIARIZATION_WINDOW_SECONDS or config.window_seconds),
        hop_seconds=float(DIARIZATION_HOP_SECONDS or config.hop_seconds),
        latency_seconds=float(DIARIZATION_LATENCY_SECONDS or config.latency_seconds),
        commit_delay_seconds=float(
            DIARIZATION_COMMIT_DELAY_SECONDS or config.commit_delay_seconds
        ),
    )


def draw_memory_graph(memory: SpeakerMemory, output_path: str) -> None:
    profiles = list(memory.profiles.values())
    if len(profiles) < 2:
        print("Skipping memory graph: need at least two profiles.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    vectors = np.vstack([profile.centroid for profile in profiles])
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    distances = 1.0 - np.clip(vectors @ vectors.T, -1.0, 1.0)
    np.fill_diagonal(distances, 0.0)

    labels = [
        f"{profile.speaker_id} ({profile.observation_count} obs, {profile.total_speech_seconds:.1f}s)"
        for profile in profiles
    ]
    clustered = linkage(squareform(distances), method="average")

    height = max(6.0, len(profiles) * 0.32)
    fig, ax = plt.subplots(figsize=(12.0, height))
    dendrogram(clustered, labels=labels, orientation="right", leaf_font_size=8, ax=ax)
    ax.set_title("Speaker memory profile clustering")
    ax.set_xlabel("Cosine distance")
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved memory graph: {path}")


def parse_live_device(value: str):
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def device_label(sd, device) -> str:
    if device is None:
        info = sd.query_devices(kind="input")
        return f"default input: {info['name']}"

    info = sd.query_devices(device)
    host_api = sd.query_hostapis()[info["hostapi"]]["name"]
    return f"{device}: {info['name']} ({host_api})"


def choose_desktop_device(sd):
    if LIVE_DEVICE:
        return parse_live_device(LIVE_DEVICE)

    keywords = (
        "loopback",
        "stereo mix",
        "what u hear",
        "voicemeeter output",
        "cable output",
    )
    candidates = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] <= 0:
            continue

        name = info["name"].lower()
        if not any(keyword in name for keyword in keywords):
            continue

        host_api = sd.query_hostapis()[info["hostapi"]]["name"].lower()
        score = 0
        if "wasapi" in host_api:
            score += 3
        if "wdm-ks" in host_api:
            score += 2
        if "directsound" in host_api:
            score += 1
        candidates.append((score, index))

    if candidates:
        return sorted(candidates, reverse=True)[0][1]

    raise SystemExit(
        "No desktop-capture input device found. Set LIVE_DEVICE to a loopback, "
        "Stereo Mix, VoiceMeeter Output, or VB-CABLE Output device index/name.\n"
        "Tip: run `.venv\\Scripts\\python -m sounddevice` to list devices."
    )


def choose_live_device(sd):
    if LIVE_SOURCE == "desktop":
        device = choose_desktop_device(sd)
    elif LIVE_SOURCE == "mic":
        device = parse_live_device(LIVE_DEVICE)
    else:
        raise SystemExit("LIVE_SOURCE must be either 'mic' or 'desktop'.")

    info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
    channels = max(1, min(2, int(info["max_input_channels"])))
    return device, channels, device_label(sd, device)


def write_status_line(text: str) -> None:
    columns = shutil.get_terminal_size((120, 20)).columns
    sys.stdout.write("\r" + text[: columns - 1].ljust(columns - 1))
    sys.stdout.flush()


def print_bad_data(label: str, data, sample_size: int = 16) -> None:
    array = np.asarray(data)
    flat = array.reshape(-1) if array.ndim else array.reshape(1)
    finite = np.isfinite(flat) if np.issubdtype(array.dtype, np.number) else np.array([])
    finite_values = flat[finite] if finite.size else np.array([])

    if finite_values.size:
        value_range = f"min={float(np.min(finite_values)):.6g} max={float(np.max(finite_values)):.6g}"
    else:
        value_range = "min=n/a max=n/a"

    if np.issubdtype(array.dtype, np.number):
        nan_count = int(np.isnan(flat).sum())
        inf_count = int(np.isinf(flat).sum())
        finite_count = int(finite.sum())
    else:
        nan_count = 0
        inf_count = 0
        finite_count = 0

    sample = flat[:sample_size]
    sample_text = np.array2string(sample, precision=6, threshold=sample_size)
    print(
        "\n[BAD DATA] "
        f"{label}: shape={array.shape} dtype={array.dtype} "
        f"size={array.size} finite={finite_count} nan={nan_count} inf={inf_count} "
        f"{value_range} sample={sample_text}"
    )


def has_invalid_numeric_values(data) -> bool:
    if data is None:
        return False

    array = np.asarray(data)
    if array.size == 0 or 0 in array.shape:
        return False

    return np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()


def live_chunk_status(chunk_index: int, output, mapping: dict[str, str], memory: SpeakerMemory) -> str:
    turns = list(output.exclusive_speaker_diarization.itertracks(yield_label=True))
    if not turns:
        return (
            f"[Chunk {chunk_index}] no speech: "
            f"profiles={len(memory.profiles)} candidates={len(memory.candidates)}"
        )

    turn_index = len(turns) - 1
    turn, _, local_label = turns[-1]
    global_label = mapping.get(local_label, local_label)
    durations = speech_seconds(output.exclusive_speaker_diarization)
    return (
        f"[Chunk {chunk_index} Turn {turn_index}] "
        f"{local_label} -> {global_label}: "
        f"turn={turn.start:.1f}-{turn.end:.1f}s "
        f"speech={durations.get(local_label, 0.0):.1f}s "
        f"profiles={len(memory.profiles)} candidates={len(memory.candidates)}"
    )


def run_live() -> SpeakerMemory:
    import sounddevice as sd

    pipeline = load_pyannote_pipeline()
    memory = make_memory()
    sample_rate = LIVE_SAMPLE_RATE
    chunk_samples = int(CHUNK_SECONDS * sample_rate)
    stride_samples = int(STRIDE_SECONDS * sample_rate)
    device, channels, label = choose_live_device(sd)

    print(f"model={MODEL_ID}")
    print(f"live_source={LIVE_SOURCE} device={label}")
    print(f"chunk={CHUNK_SECONDS}s stride={STRIDE_SECONDS}s sample_rate={sample_rate}")
    print("Press Ctrl+C to stop and print speaker memory.")

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stream_warnings: list[str] = []

    def callback(indata, _frames, _time_info, status) -> None:
        if status:
            stream_warnings.append(str(status))
        audio_queue.put(indata.copy())

    buffer = np.empty((0, channels), dtype=np.float32)
    chunk_index = 0

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            device=device,
            channels=channels,
            dtype="float32",
            blocksize=max(1, int(sample_rate * 0.25)),
            callback=callback,
        ):
            while True:
                while buffer.shape[0] < chunk_samples:
                    block = np.asarray(audio_queue.get(), dtype=np.float32)
                    if block.ndim == 1:
                        block = block[:, None]
                    if block.ndim != 2 or block.shape[0] == 0 or block.shape[1] == 0:
                        print_bad_data(f"live audio block before chunk {chunk_index}", block)
                        write_status_line(
                            f"[Chunk {chunk_index}] waiting: skipped empty audio block"
                        )
                        continue

                    if not np.isfinite(block).all():
                        print_bad_data(f"live audio block before chunk {chunk_index}", block)
                    block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
                    block = block[:, :channels]
                    if block.shape[1] == 0:
                        print_bad_data(f"live zero-channel block before chunk {chunk_index}", block)
                        write_status_line(
                            f"[Chunk {chunk_index}] waiting: skipped zero-channel audio block"
                        )
                        continue

                    if block.shape[1] != buffer.shape[1]:
                        common_channels = min(block.shape[1], buffer.shape[1])
                        buffer = buffer[:, :common_channels]
                        block = block[:, :common_channels]

                    buffer = np.concatenate([buffer, block], axis=0)

                chunk = buffer[:chunk_samples]
                buffer = buffer[stride_samples:]
                if chunk.shape[0] == 0 or chunk.shape[1] == 0:
                    print_bad_data(f"live chunk {chunk_index}", chunk)
                    write_status_line(f"[Chunk {chunk_index}] skipped empty chunk")
                    chunk_index += 1
                    continue

                mono = np.mean(chunk, axis=1, dtype=np.float32)
                if mono.size == 0 or not np.isfinite(mono).all():
                    print_bad_data(f"live mono chunk {chunk_index}", mono)
                    write_status_line(f"[Chunk {chunk_index}] skipped invalid audio chunk")
                    chunk_index += 1
                    continue

                waveform = torch.from_numpy(mono.copy()).unsqueeze(0)
                output = pipeline(
                    {
                        "waveform": waveform,
                        "sample_rate": sample_rate,
                        "uri": f"live-{chunk_index}",
                    }
                )
                if has_invalid_numeric_values(output.speaker_embeddings):
                    print_bad_data(
                        f"pyannote speaker_embeddings for live chunk {chunk_index}",
                        output.speaker_embeddings,
                    )

                local_labels = list(output.speaker_diarization.labels())
                mapping = memory.assign(
                    local_labels,
                    output.speaker_embeddings,
                    speech_seconds=speech_seconds(output.exclusive_speaker_diarization),
                    segment_end=chunk_index * STRIDE_SECONDS + CHUNK_SECONDS,
                )

                status = live_chunk_status(chunk_index, output, mapping, memory)
                if stream_warnings:
                    status = f"{status} warn={stream_warnings[-1]}"
                    stream_warnings.clear()
                write_status_line(status)
                chunk_index += 1

    except KeyboardInterrupt:
        print()

    print("\n=== speaker memory ===")
    print(memory.debug_table())
    return memory


def run_live_realtime() -> SpeakerMemory:
    import sounddevice as sd

    pipeline = load_pyannote_pipeline()
    backend = PyannotePipelineBackend(pipeline)
    memory = make_memory()
    config = make_realtime_config()
    sample_rate = LIVE_SAMPLE_RATE
    device, channels, label = choose_live_device(sd)
    scheduler = RollingWindowScheduler(sample_rate, config, channels=1)
    metrics = MetricsRecorder(METRICS_PATH or None)
    driver = RealtimeDiarizationDriver(backend, memory, config, metrics=metrics)

    print(f"model={MODEL_ID}")
    print(f"live_source={LIVE_SOURCE} device={label}")
    print(
        "realtime="
        f"profile={DIARIZATION_PROFILE} window={config.window_seconds}s "
        f"hop={config.hop_seconds}s latency={config.latency_seconds}s "
        f"commit_delay={config.commit_delay_seconds}s sample_rate={sample_rate}"
    )
    print("Press Ctrl+C to stop and print speaker memory.")

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stream_warnings: list[str] = []

    def callback(indata, _frames, _time_info, status) -> None:
        if status:
            stream_warnings.append(str(status))
        audio_queue.put(indata.copy())

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            device=device,
            channels=channels,
            dtype="float32",
            blocksize=max(1, int(sample_rate * 0.25)),
            callback=callback,
        ):
            while True:
                block = np.asarray(audio_queue.get(), dtype=np.float32)
                if block.ndim == 1:
                    block = block[:, None]
                if block.ndim != 2 or block.shape[0] == 0 or block.shape[1] == 0:
                    print_bad_data("live realtime block", block)
                    continue
                block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
                mono = np.mean(block[:, :channels], axis=1, dtype=np.float32)
                for window in scheduler.append(mono):
                    hop = driver.process_window(window)
                    if stream_warnings:
                        print(f"warning={stream_warnings[-1]}")
                        stream_warnings.clear()
                    print_events(hop.events)
                    write_status_line(
                        f"[Window {window.index}] "
                        f"{window.start:.1f}-{window.end:.1f}s "
                        f"events={len(hop.events)} "
                        f"profiles={len(memory.profiles)} candidates={len(memory.candidates)}"
                    )
    except KeyboardInterrupt:
        print()

    print("\n=== metrics ===")
    print(metrics.summary())
    metrics.close()
    print("\n=== speaker memory ===")
    print(memory.debug_table())
    return memory


def run_real_audio() -> SpeakerMemory:
    if not AUDIO_PATH:
        raise SystemExit("Set AUDIO_PATH to a wav/mp3 file, or run without it for synthetic demo.")

    audio_path = Path(AUDIO_PATH)
    waveform, sample_rate = load_audio(audio_path)
    pipeline = load_pyannote_pipeline()
    memory = make_memory()
    chunk_samples = int(CHUNK_SECONDS * sample_rate)
    stride_samples = int(STRIDE_SECONDS * sample_rate)
    total_samples = waveform.shape[1]

    print(f"model={MODEL_ID}")
    print(f"audio={audio_path}")
    print(f"chunk={CHUNK_SECONDS}s stride={STRIDE_SECONDS}s threshold={MATCH_THRESHOLD}")
    print(
        "memory="
        f"min_new={MIN_NEW_PROFILE_SECONDS}s "
        f"promote={CANDIDATE_PROMOTE_SECONDS}s/{CANDIDATE_PROMOTE_OBSERVATIONS}obs "
        f"exclude_overlap_embeddings={EMBEDDING_EXCLUDE_OVERLAP}"
    )
    print()

    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        offset = start / sample_rate
        chunk = waveform[:, start:end]
        segment_end = end / sample_rate

        output = pipeline(
            {
                "waveform": chunk,
                "sample_rate": sample_rate,
                "uri": f"{audio_path.stem}-{offset:.1f}",
            }
        )

        local_labels = list(output.speaker_diarization.labels())
        mapping = memory.assign(
            local_labels,
            output.speaker_embeddings,
            speech_seconds=speech_seconds(output.exclusive_speaker_diarization),
            segment_end=segment_end,
        )

        print(f"\n=== chunk {offset:.1f}s -> {segment_end:.1f}s ===")
        print(f"local -> global: {mapping}")
        print_global_turns(output.exclusive_speaker_diarization, mapping, offset)

        if end == total_samples:
            break
        start += stride_samples

    print("\n=== speaker memory ===")
    print(memory.debug_table())
    return memory


def run_real_audio_realtime() -> SpeakerMemory:
    if not AUDIO_PATH:
        raise SystemExit("Set AUDIO_PATH to a wav/mp3 file, or run without it for synthetic demo.")

    audio_path = Path(AUDIO_PATH)
    waveform, sample_rate = load_audio(audio_path)
    pipeline = load_pyannote_pipeline()
    memory = make_memory()
    config = make_realtime_config()

    print(f"model={MODEL_ID}")
    print(f"audio={audio_path}")
    print(
        "realtime="
        f"profile={DIARIZATION_PROFILE} window={config.window_seconds}s "
        f"hop={config.hop_seconds}s latency={config.latency_seconds}s "
        f"commit_delay={config.commit_delay_seconds}s threshold={MATCH_THRESHOLD}"
    )
    print(
        "memory="
        f"min_new={MIN_NEW_PROFILE_SECONDS}s "
        f"promote={CANDIDATE_PROMOTE_SECONDS}s/{CANDIDATE_PROMOTE_OBSERVATIONS}obs "
        f"exclude_overlap_embeddings={EMBEDDING_EXCLUDE_OVERLAP}"
    )
    print()

    backend = PyannotePipelineBackend(pipeline)
    memory, timeline, metrics_summary = run_replay(
        waveform,
        sample_rate,
        backend,
        memory,
        config,
        metrics_path=METRICS_PATH or None,
    )

    print("\n=== realtime timeline ===")
    for turn in timeline.turns:
        final = "final" if turn.final else "provisional"
        print(
            f"{turn.turn_id} v{turn.version:<2} "
            f"{turn.start:7.2f}-{turn.end:7.2f}s {turn.speaker_id:<16} {final}"
        )

    print("\n=== metrics ===")
    print(metrics_summary)
    print("\n=== speaker memory ===")
    print(memory.debug_table())
    return memory


def run_synthetic() -> SpeakerMemory:
    """Tiny fake run showing identity retention across pyannote-like chunks."""

    rng = np.random.default_rng(7)
    base_a = rng.normal(size=192)
    base_b = rng.normal(size=192)
    base_a /= np.linalg.norm(base_a)
    base_b /= np.linalg.norm(base_b)

    memory = make_memory()
    chunks = [
        (["SPEAKER_00", "SPEAKER_01"], np.vstack([base_a, base_b])),
        # Next pyannote chunk flips local labels. Memory should keep global ids stable.
        (["SPEAKER_00", "SPEAKER_01"], np.vstack([base_b, base_a])),
        # New person appears.
        (["SPEAKER_00"], np.vstack([rng.normal(size=192)])),
    ]

    for index, (labels, centroids) in enumerate(chunks):
        noisy = centroids + rng.normal(scale=0.015, size=centroids.shape)
        mapping = memory.assign(
            labels,
            noisy,
            speech_seconds={label: 3.0 for label in labels},
            segment_end=(index + 1) * CHUNK_SECONDS,
        )
        print(f"chunk {index}: {mapping}")

    print("\n=== speaker memory ===")
    print(memory.debug_table())
    return memory

def main():
    args = parse_args()
    if args.live:
        speaker_memory = run_live() if args.legacy_chunks else run_live_realtime()
    elif AUDIO_PATH:
        speaker_memory = run_real_audio() if args.legacy_chunks else run_real_audio_realtime()
    else:
        speaker_memory = run_synthetic()

    if args.mem_graph:
        draw_memory_graph(speaker_memory, args.mem_graph)

if __name__ == "__main__":
    main()
