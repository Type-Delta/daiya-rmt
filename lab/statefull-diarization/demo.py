from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

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
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.38"))
MIN_NEW_PROFILE_SECONDS = float(os.getenv("MIN_NEW_PROFILE_SECONDS", "6.0"))
CANDIDATE_PROMOTE_SECONDS = float(os.getenv("CANDIDATE_PROMOTE_SECONDS", "3.0"))
CANDIDATE_PROMOTE_OBSERVATIONS = int(os.getenv("CANDIDATE_PROMOTE_OBSERVATIONS", "2"))
EMBEDDING_EXCLUDE_OVERLAP = env_bool("EMBEDDING_EXCLUDE_OVERLAP", True)
DEVICE = os.getenv("DEVICE", "cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny pyannote speaker-memory demo.")
    parser.add_argument(
        "--mem-graph",
        nargs="?",
        const="artifacts/memory_profiles.png",
        default="",
        help="Save a dendrogram of global speaker-memory profiles.",
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
        f"{profile.speaker_id} ({profile.observations} obs, {profile.total_speech_seconds:.1f}s)"
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


def run_real_audio() -> SpeakerMemory:
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"\s*torchcodec is not installed correctly.*",
        category=UserWarning,
    )
    from pyannote.audio import Pipeline

    if not AUDIO_PATH:
        raise SystemExit("Set AUDIO_PATH to a wav/mp3 file, or run without it for synthetic demo.")

    audio_path = Path(AUDIO_PATH)
    waveform, sample_rate = load_audio(audio_path)

    try:
        pipeline = Pipeline.from_pretrained(MODEL_ID, token=HF_TOKEN)
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

    memory = SpeakerMemory(
        match_threshold=MATCH_THRESHOLD,
        min_new_profile_seconds=MIN_NEW_PROFILE_SECONDS,
        candidate_promote_seconds=CANDIDATE_PROMOTE_SECONDS,
        candidate_promote_observations=CANDIDATE_PROMOTE_OBSERVATIONS,
    )
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


def run_synthetic() -> SpeakerMemory:
    """Tiny fake run showing identity retention across pyannote-like chunks."""

    rng = np.random.default_rng(7)
    base_a = rng.normal(size=192)
    base_b = rng.normal(size=192)
    base_a /= np.linalg.norm(base_a)
    base_b /= np.linalg.norm(base_b)

    memory = SpeakerMemory(
        match_threshold=MATCH_THRESHOLD,
        min_new_profile_seconds=MIN_NEW_PROFILE_SECONDS,
        candidate_promote_seconds=CANDIDATE_PROMOTE_SECONDS,
        candidate_promote_observations=CANDIDATE_PROMOTE_OBSERVATIONS,
    )
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


if __name__ == "__main__":
    args = parse_args()
    if AUDIO_PATH:
        speaker_memory = run_real_audio()
    else:
        speaker_memory = run_synthetic()

    if args.mem_graph:
        draw_memory_graph(speaker_memory, args.mem_graph)
