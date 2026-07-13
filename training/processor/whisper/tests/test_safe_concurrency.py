from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import threading
import time

import pytest

from daiya_whisper_pipeline.concurrency import bounded_ordered_map
from daiya_whisper_pipeline import export as export_module
from daiya_whisper_pipeline import ffmpeg as ffmpeg_module
from daiya_whisper_pipeline import llm as llm_module
from daiya_whisper_pipeline.types import Chunk, Interval, LabeledChunk, NormalizedAudio


def _config(**values: object) -> SimpleNamespace:
    defaults = {
        "channels": 1,
        "sample_rate": 16000,
        "audio_codec": "pcm_s16le",
        "ffmpeg_bin": "ffmpeg",
        "llm_context_max_chars": 100,
        "llm_max_workers": 4,
        "llm_max_in_flight": 2,
        "export_max_workers": 4,
        "export_max_in_flight": 2,
        "dataset_split": "train",
        "text_column": "text",
        "language_hint": "mixed",
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def _chunk(tmp_path: Path, source_index: int, chunk_index: int = 0) -> Chunk:
    source_path = tmp_path / f"source-{source_index}.wav"
    source = NormalizedAudio(
        source_path=source_path,
        normalized_path=tmp_path / f"normalized-{source_index}.wav",
        source_id=f"source-{source_index}",
    )
    chunk_path = tmp_path / "chunks" / f"{source_index}-{chunk_index}.wav"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_bytes(f"audio-{source_index}-{chunk_index}".encode())
    return Chunk(
        source=source,
        intervals=(Interval(0.0, 1.0),),
        chunk_path=chunk_path,
        index=chunk_index,
    )


def test_bounded_ordered_map_is_ordered_and_limits_active_work() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def work(value: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return value

    with ThreadPoolExecutor(max_workers=8) as executor:
        result = list(bounded_ordered_map(executor, work, range(20), max_in_flight=3))

    assert result == list(range(20))
    assert peak <= 3


def test_ffmpeg_failure_does_not_publish_partial_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(work_dir=tmp_path / "work")
    source_path = tmp_path / "input.wav"
    source_path.write_bytes(b"source")

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        ffmpeg_module.normalize_audio_file(source_path, config)

    normalized_dir = config.work_dir / "normalized"
    assert not list(normalized_dir.glob("*.wav"))
    assert not list(normalized_dir.glob(".*"))

    chunk = _chunk(tmp_path, 1)
    chunk.chunk_path.unlink()
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        ffmpeg_module.export_chunk(chunk, config)
    assert not chunk.chunk_path.exists()
    assert not list(chunk.chunk_path.parent.glob(".*"))


def test_ffmpeg_atomic_command_uses_wav_temporary_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(work_dir=tmp_path / "work")
    source_path = tmp_path / "input.wav"
    source_path.write_bytes(b"source")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"wav")

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", fake_run)
    normalized = ffmpeg_module.normalize_audio_file(source_path, config)
    normalized_again = ffmpeg_module.normalize_audio_file(source_path, config)

    temporary_output = Path(commands[0][-1])
    assert temporary_output.suffix == ".wav"
    assert temporary_output != normalized.normalized_path
    assert len(commands) == 2
    assert normalized.normalized_path.read_bytes() == b"wav"
    assert normalized_again.normalized_path == normalized.normalized_path


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_ffmpeg_real_wav_smoke(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    source_path = tmp_path / "generated.wav"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=0.1",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(source_path),
        ],
        check=True,
    )

    normalized = ffmpeg_module.normalize_audio_file(source_path, _config(work_dir=tmp_path / "work"))
    assert normalized.normalized_path.suffix == ".wav"
    assert normalized.normalized_path.stat().st_size > 44


def test_llm_source_jobs_are_bounded_and_return_in_source_order(tmp_path: Path) -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    class FakeTranscriber:
        def transcribe(
            self,
            chunk: Chunk,
            context: object = None,
            previous_text: str = "",
        ) -> LabeledChunk:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return LabeledChunk(
                chunk=chunk,
                transcript_text=f"{chunk.source.source_path.name}-{chunk.index}",
                language="mixed",
                extra={"context_after": ""},
            )

    chunks = [_chunk(tmp_path, source, index) for source in reversed(range(5)) for index in reversed(range(2))]
    labeled = llm_module.transcribe_chunks(chunks, FakeTranscriber(), _config())

    assert [(item.chunk.source.source_path.name, item.chunk.index) for item in labeled] == [
        (f"source-{source}.wav", index) for source in range(5) for index in range(2)
    ]
    assert peak <= 2


def test_export_copies_are_bounded_and_metadata_is_ordered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "output"
    chunks = [_chunk(tmp_path, source) for source in reversed(range(5))]
    labeled = [LabeledChunk(chunk=chunk, transcript_text=f"text-{chunk.source.source_id}") for chunk in chunks]
    config = _config(output_dir=output_dir)

    active = 0
    peak = 0
    lock = threading.Lock()
    original_copy2 = export_module.shutil.copy2

    def tracked_copy(source: str | Path, destination: str | Path, *args: object, **kwargs: object):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        try:
            return original_copy2(source, destination, *args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(export_module.shutil, "copy2", tracked_copy)
    metadata_path = export_module.export_audiofolder(labeled, config)

    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    assert [row["source_file"] for row in rows] == [
        str(tmp_path / f"source-{source}.wav") for source in range(5)
    ]
    assert peak <= 2
    assert not list(output_dir.rglob(".*.tmp"))


def test_export_failure_removes_partial_copy_and_metadata_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_chunk(tmp_path, 0), _chunk(tmp_path, 1)]
    output_dir = tmp_path / "output"
    config = _config(output_dir=output_dir)
    items = [LabeledChunk(chunk=chunk, transcript_text=f"text-{chunk.index}") for chunk in chunks]
    original_copy2 = export_module.shutil.copy2

    def partial_then_fail(source: str | Path, destination: str | Path, *args: object, **kwargs: object):
        if Path(source).name == "0-0.wav":
            return original_copy2(source, destination, *args, **kwargs)
        Path(destination).write_bytes(b"partial")
        raise RuntimeError("copy failed after another item was published to staging")

    monkeypatch.setattr(export_module.shutil, "copy2", partial_then_fail)
    with pytest.raises(RuntimeError, match="copy failed after another item"):
        export_module.export_audiofolder(items, config)

    assert not (output_dir / "metadata.jsonl").exists()
    assert not list(output_dir.rglob("*.wav"))
    assert not list(tmp_path.glob(".output.staging-*"))
    assert not (tmp_path / ".output.lock").exists()


def test_export_rejects_stale_target_without_touching_prior_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    metadata_path = output_dir / "metadata.jsonl"
    metadata_path.write_text('{"old": true}\n', encoding="utf-8")
    config = _config(output_dir=output_dir)

    with pytest.raises(FileExistsError, match="already exists"):
        export_module.export_audiofolder([LabeledChunk(chunk=_chunk(tmp_path, 0), transcript_text="new")], config)

    assert metadata_path.read_text(encoding="utf-8") == '{"old": true}\n'
    assert not (tmp_path / ".output.lock").exists()


def test_export_rejects_dangling_output_symlink(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    try:
        output_dir.symlink_to(tmp_path / "missing-output", target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(FileExistsError, match="already exists"):
        export_module.export_audiofolder([], _config(output_dir=output_dir))

    assert output_dir.is_symlink()
    assert not (tmp_path / ".output.lock").exists()


def test_export_rejects_concurrent_publication_lock(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = tmp_path / ".output.lock"
    lock_dir.mkdir()

    with pytest.raises(FileExistsError, match="Another export"):
        export_module.export_audiofolder([], _config(output_dir=output_dir))

    assert lock_dir.exists()
