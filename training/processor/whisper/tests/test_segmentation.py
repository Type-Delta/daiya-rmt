from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from daiya_whisper_pipeline import ffmpeg as ffmpeg_module
from daiya_whisper_pipeline import export as export_module
from daiya_whisper_pipeline.segmentation import SEGMENTATION_VERSION, build_chunks
from daiya_whisper_pipeline.types import Interval, LabeledChunk, NormalizedAudio


def _config(tmp_path: Path, **updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "work_dir": tmp_path / "work",
        "min_chunk_seconds": 1.0,
        "target_chunk_seconds": 18.0,
        "max_chunk_seconds": 25.0,
        "merge_gap_seconds": 0.8,
        "boundary_min_silence_seconds": 0.5,
        "boundary_search_seconds": 4.0,
        "fallback_context_seconds": 1.0,
        "overlap_mode": "preserve",
        "channels": 1,
        "sample_rate": 16000,
        "audio_codec": "pcm_s16le",
        "ffmpeg_bin": "ffmpeg",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _source(tmp_path: Path, duration: float = 60.0) -> NormalizedAudio:
    return NormalizedAudio(
        source_path=tmp_path / "source.wav",
        normalized_path=tmp_path / "normalized.wav",
        source_id="source",
        duration_seconds=duration,
    )


def _spans(chunks: list[object]) -> list[tuple[float, float]]:
    return [(chunk.start, chunk.end) for chunk in chunks]  # type: ignore[attr-defined]


def test_overlap_is_preserved_as_evidence_by_default(tmp_path: Path) -> None:
    chunks = build_chunks(
        _source(tmp_path, 10.0),
        [Interval(0.0, 10.0)],
        [Interval(3.0, 6.0)],
        _config(tmp_path),
    )

    assert _spans(chunks) == [(0.0, 10.0)]
    assert chunks[0].speech_intervals == (Interval(0.0, 10.0),)
    assert chunks[0].overlap_intervals == (Interval(3.0, 6.0),)
    assert chunks[0].segmentation_version == SEGMENTATION_VERSION


def test_legacy_overlap_exclusion_is_explicit_and_nondefault(tmp_path: Path) -> None:
    chunks = build_chunks(
        _source(tmp_path, 10.0),
        [Interval(0.0, 10.0)],
        [Interval(3.0, 6.0)],
        _config(tmp_path, overlap_mode="legacy-exclude"),
    )

    assert _spans(chunks) == [(0.0, 3.0), (6.0, 10.0)]


def test_bridged_vad_gap_is_kept_inside_one_wall_clock_window(tmp_path: Path) -> None:
    chunks = build_chunks(
        _source(tmp_path, 8.0),
        [Interval(1.0, 3.0), Interval(3.35, 5.0)],
        [],
        _config(tmp_path),
    )

    assert _spans(chunks) == [(1.0, 5.0)]
    assert chunks[0].speech_duration == pytest.approx(3.65)
    assert chunks[0].duration == 4.0


def test_overlapping_or_nested_vad_evidence_cannot_truncate_a_later_window(tmp_path: Path) -> None:
    chunks = build_chunks(
        _source(tmp_path, 15.0),
        [Interval(0.0, 10.0), Interval(1.0, 2.0), Interval(11.0, 12.0)],
        [],
        _config(tmp_path),
    )

    assert _spans(chunks) == [(0.0, 10.0), (11.0, 12.0)]


def test_export_chunk_trims_one_contiguous_window_not_vad_concat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk = build_chunks(
        _source(tmp_path, 8.0),
        [Interval(1.0, 3.0), Interval(3.35, 5.0)],
        [],
        _config(tmp_path),
    )[0]
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"wav")

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", fake_run)
    ffmpeg_module.export_chunk(chunk, _config(tmp_path))

    filter_complex = commands[0][commands[0].index("-filter_complex") + 1]
    assert "concat" not in filter_complex
    assert "start=1.000000:end=5.000000" in filter_complex


def test_long_group_prefers_actual_silence_near_target(tmp_path: Path) -> None:
    chunks = build_chunks(
        _source(tmp_path, 42.0),
        [Interval(0.0, 17.7), Interval(18.3, 40.0)],
        [],
        _config(tmp_path),
    )

    assert _spans(chunks) == [(0.0, 18.0), (18.0, 40.0)]
    assert all(chunk.training_eligible for chunk in chunks)


def test_no_silence_fallback_has_bounded_context_and_never_loses_coverage(tmp_path: Path) -> None:
    chunks = build_chunks(
        _source(tmp_path, 60.0),
        [Interval(0.0, 60.0)],
        [],
        _config(tmp_path),
    )

    assert _spans(chunks) == [(0.0, 25.0), (24.0, 49.0), (48.0, 60.0)]
    assert all(not chunk.training_eligible for chunk in chunks)
    assert chunks[0].context_overlap_after_seconds == 1.0
    assert chunks[1].context_overlap_before_seconds == 1.0
    # The union covers every VAD sample; the two seconds of overlap are explicit.
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 60.0
    assert sum(chunk.duration for chunk in chunks) - 60.0 == pytest.approx(2.0)


def test_silence_cut_never_creates_a_trailing_sliver(tmp_path: Path) -> None:
    chunks = build_chunks(
        _source(tmp_path, 26.0),
        [Interval(0.0, 24.5), Interval(25.0, 25.3)],
        [],
        _config(tmp_path, merge_gap_seconds=0.8),
    )

    assert _spans(chunks) == [(0.0, 25.0), (24.0, 25.3)]
    assert min(chunk.duration for chunk in chunks) >= 1.0


def test_source_bounds_clip_vad_evidence_and_export_windows(tmp_path: Path) -> None:
    chunks = build_chunks(
        _source(tmp_path, 30.0),
        [Interval(-2.0, 3.0), Interval(28.0, 35.0)],
        [Interval(29.0, 40.0)],
        _config(tmp_path),
    )

    assert _spans(chunks) == [(0.0, 3.0), (28.0, 30.0)]
    assert chunks[-1].overlap_intervals == (Interval(29.0, 30.0),)


def test_short_speech_is_retained_instead_of_being_silently_dropped(tmp_path: Path) -> None:
    chunks = build_chunks(
        _source(tmp_path, 2.0),
        [Interval(0.2, 0.45)],
        [],
        _config(tmp_path, min_chunk_seconds=1.0),
    )

    assert _spans(chunks) == [(0.2, 0.45)]


def test_metadata_carries_exact_timestamps_and_review_provenance(tmp_path: Path) -> None:
    chunk = build_chunks(
        _source(tmp_path, 60.0),
        [Interval(0.0, 60.0)],
        [Interval(3.0, 4.0)],
        _config(tmp_path),
    )[0]
    chunk.chunk_path.parent.mkdir(parents=True)
    chunk.chunk_path.write_bytes(b"contiguous-audio")
    config = _config(tmp_path, text_column="text", language_hint="mixed")

    row = export_module._row(LabeledChunk(chunk=chunk, transcript_text="label"), "train/clip.wav", config)

    assert row["source_start"] == 0.0
    assert row["source_end"] == 25.0
    assert row["audio_sha256"]
    assert row["overlap_detected"] is True
    assert row["training_eligible"] is False
    assert row["review_signals"] == [
        "overlapped_speech_detected",
        "no_silence_boundary_fallback",
        "adjacent_context_overlap",
    ]
    segmentation = row["segmentation"]
    assert isinstance(segmentation, dict)
    assert segmentation["window_is_contiguous"] is True
    assert segmentation["overlap_intervals"] == [{"start": 3.0, "end": 4.0}]
