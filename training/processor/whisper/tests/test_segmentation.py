from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from daiya_whisper_pipeline import ffmpeg as ffmpeg_module
from daiya_whisper_pipeline import export as export_module
from daiya_whisper_pipeline.evidence import TimestampEvidence, TimestampWord
from daiya_whisper_pipeline.llm import ownership_alignment_gate
from daiya_whisper_pipeline.segmentation import SEGMENTATION_VERSION, build_chunks
from daiya_whisper_pipeline.types import Interval, LabeledChunk, NormalizedAudio


def _config(tmp_path: Path, **updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "work_dir": tmp_path / "work",
        "min_chunk_seconds": 1.0,
        "target_chunk_seconds": 18.0,
        "max_chunk_seconds": 25.0,
        "hard_max_chunk_seconds": 30.0,
        "merge_gap_seconds": 0.8,
        "boundary_min_silence_seconds": 0.5,
        "boundary_search_seconds": 4.0,
        "fallback_context_seconds": 1.0,
        "boundary_candidate_tolerance_seconds": 0.35,
        "boundary_min_confidence": 0.55,
        "label_alignment_min_similarity": 0.45,
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

    assert _spans(chunks) == [(0.0, 25.0), (25.0, 50.0), (50.0, 60.0)]
    assert chunks[0].training_eligible is True
    assert all(not chunk.training_eligible for chunk in chunks[1:])
    assert chunks[0].context_overlap_after_seconds == 0.0
    assert chunks[1].context_overlap_before_seconds == 1.0
    assert (chunks[1].labeling_start, chunks[1].labeling_end) == (24.0, 50.0)
    # Owned source time never overlaps; duplicated seconds are labeling-only.
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 60.0
    assert sum(chunk.duration for chunk in chunks) == pytest.approx(60.0)
    assert sum(chunk.labeling_duration for chunk in chunks) - 60.0 == pytest.approx(2.0)


def test_silence_cut_never_creates_a_trailing_sliver(tmp_path: Path) -> None:
    chunks = build_chunks(
        _source(tmp_path, 26.0),
        [Interval(0.0, 24.5), Interval(25.0, 25.3)],
        [],
        _config(tmp_path, merge_gap_seconds=0.8),
    )

    assert _spans(chunks) == [(0.0, 25.3)]
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
    assert row["training_eligible"] is True
    assert row["review_signals"] == ["overlapped_speech_detected"]
    segmentation = row["segmentation"]
    assert isinstance(segmentation, dict)
    assert segmentation["window_is_contiguous"] is True
    assert segmentation["overlap_intervals"] == [{"start": 3.0, "end": 4.0}]


def _evidence(tmp_path: Path, words: tuple[TimestampWord, ...] = (), energy: tuple[Interval, ...] = ()) -> TimestampEvidence:
    return TimestampEvidence(
        source_id="source",
        source_audio_sha256="a" * 64,
        duration_seconds=60.0,
        status="ok" if words else "failed",
        model_identity={"family": "faster-whisper", "model": "test"},
        decoding_settings={"word_timestamps": True},
        words=words,
        energy_gaps=energy,
        failure="no model" if not words else "",
        cache_path=tmp_path / "evidence.json",
    )


def test_multisignal_boundary_prefers_agreement_without_language_tokenization(tmp_path: Path) -> None:
    evidence = _evidence(
        tmp_path,
        words=(
            TimestampWord(0.0, 17.75, "ภาษาไทย"),
            TimestampWord(18.25, 45.0, " English"),
        ),
        energy=(Interval(17.65, 18.35),),
    )
    chunks = build_chunks(
        _source(tmp_path, 45.0),
        [Interval(0.0, 17.7), Interval(18.3, 45.0)],
        [],
        _config(tmp_path),
        evidence,
    )

    assert _spans(chunks) == [(0.0, pytest.approx(18.0)), (pytest.approx(18.0), 45.0)]
    assert chunks[0].boundary_method == "low_energy_gap+silero_vad_gap+whisper_timestamp_gap"
    assert chunks[0].boundary_confidence >= 0.9
    assert chunks[0].training_eligible is True


def test_energy_only_is_review_safe_while_vad_only_can_cut(tmp_path: Path) -> None:
    energy_chunks = build_chunks(
        _source(tmp_path, 60.0),
        [Interval(0.0, 60.0)],
        [],
        _config(tmp_path),
        _evidence(tmp_path, energy=(Interval(17.7, 18.3),)),
    )
    vad_chunks = build_chunks(
        _source(tmp_path, 60.0),
        [Interval(0.0, 17.7), Interval(18.3, 60.0)],
        [],
        _config(tmp_path),
    )

    # A relative-energy dip is only a supporting acoustic signal: without VAD
    # or timestamp corroboration it deliberately falls back to the quarantined
    # one-sided pre-roll path instead of cutting continuous speech as eligible.
    assert energy_chunks[0].boundary_method == "continuous_speech_fallback"
    assert energy_chunks[0].end == 25.0
    assert energy_chunks[0].training_eligible is True
    assert vad_chunks[0].boundary_method == "silero_vad_gap"
    assert vad_chunks[0].training_eligible is True


def test_failed_timestamp_evidence_uses_one_sided_preroll_and_caps_owned_duration(tmp_path: Path) -> None:
    chunks = build_chunks(
        _source(tmp_path, 95.0),
        [Interval(0.0, 95.0)],
        [],
        _config(tmp_path),
        _evidence(tmp_path),
    )

    assert all(chunk.duration <= 30.0 for chunk in chunks)
    assert all(left.end == right.start for left, right in zip(chunks, chunks[1:]))
    assert chunks[0].training_eligible is True
    assert chunks[1].has_labeling_preroll is True
    assert chunks[1].training_eligible is False
    assert chunks[1].eligibility_reason == "pre_roll_alignment_required"


def test_preroll_alignment_gate_never_rewrites_and_quarantines_uncertain_rows(tmp_path: Path) -> None:
    chunks = build_chunks(
        _source(tmp_path, 60.0), [Interval(0.0, 60.0)], [], _config(tmp_path)
    )
    fallback = chunks[1]
    gate = ownership_alignment_gate(fallback, "unrelated label", _config(tmp_path))

    assert gate["status"] == "review_required"
    assert gate["eligible"] is False
    assert gate["reason"] == "no_owned_timestamp_evidence"


def test_preroll_alignment_can_explicitly_resolve_to_owned_target_only(tmp_path: Path) -> None:
    evidence = _evidence(
        tmp_path,
        words=(TimestampWord(24.1, 24.9, "context"), TimestampWord(25.0, 25.9, "target")),
    )
    chunks = build_chunks(_source(tmp_path, 60.0), [Interval(0.0, 60.0)], [], _config(tmp_path), evidence)
    fallback = chunks[1]
    gate = ownership_alignment_gate(fallback, "target", _config(tmp_path))

    assert fallback.has_labeling_preroll
    assert gate["status"] == "passed"
    assert gate["eligible"] is True


def test_preroll_alignment_rejects_context_plus_target(tmp_path: Path) -> None:
    evidence = _evidence(
        tmp_path,
        words=(
            TimestampWord(24.1, 24.9, "context"),
            TimestampWord(25.1, 26.0, "target"),
        ),
    )
    chunks = build_chunks(_source(tmp_path, 60.0), [Interval(0.0, 60.0)], [], _config(tmp_path), evidence)
    gate = ownership_alignment_gate(chunks[1], "context target", _config(tmp_path))

    assert gate["status"] == "review_required"
    assert gate["eligible"] is False
    assert gate["reason"] == "label_contains_or_matches_preroll_context"


def test_preroll_alignment_quarantines_boundary_straddling_timestamp(tmp_path: Path) -> None:
    evidence = _evidence(
        tmp_path,
        words=(
            TimestampWord(24.1, 24.9, "context"),
            TimestampWord(24.9, 25.1, "straddle"),
            TimestampWord(25.1, 26.0, "target"),
        ),
    )
    chunks = build_chunks(_source(tmp_path, 60.0), [Interval(0.0, 60.0)], [], _config(tmp_path), evidence)
    gate = ownership_alignment_gate(chunks[1], "target", _config(tmp_path))

    assert gate["status"] == "review_required"
    assert gate["eligible"] is False
    assert gate["reason"] == "timestamp_straddles_ownership_boundary"


def test_preroll_alignment_quarantines_timestamp_straddling_owned_end(tmp_path: Path) -> None:
    evidence = _evidence(
        tmp_path,
        words=(
            TimestampWord(24.1, 24.9, "context"),
            TimestampWord(25.1, 26.0, "target"),
            TimestampWord(49.9, 50.1, "spills_next"),
        ),
    )
    chunks = build_chunks(_source(tmp_path, 60.0), [Interval(0.0, 60.0)], [], _config(tmp_path), evidence)
    gate = ownership_alignment_gate(chunks[1], "target", _config(tmp_path))

    assert gate["status"] == "review_required"
    assert gate["eligible"] is False
    assert gate["reason"] == "timestamp_straddles_ownership_boundary"


def test_boundary_search_ignores_early_timestamp_gap_but_extends_for_multisignal_cue(tmp_path: Path) -> None:
    early_only = _evidence(
        tmp_path,
        words=(TimestampWord(0.0, 1.8, "thai"), TimestampWord(2.2, 60.0, "English")),
    )
    early_chunks = build_chunks(_source(tmp_path, 60.0), [Interval(0.0, 60.0)], [], _config(tmp_path), early_only)
    assert early_chunks[0].end == 25.0
    assert early_chunks[0].boundary_method == "continuous_speech_fallback"

    extended = _evidence(
        tmp_path,
        words=(TimestampWord(0.0, 27.7, "ภาษาไทย"), TimestampWord(28.3, 60.0, " English")),
        energy=(Interval(27.65, 28.35),),
    )
    extended_chunks = build_chunks(
        _source(tmp_path, 60.0),
        [Interval(0.0, 27.7), Interval(28.3, 60.0)],
        [],
        _config(tmp_path),
        extended,
    )
    assert extended_chunks[0].end == pytest.approx(28.0)
    assert extended_chunks[0].boundary_confidence >= 0.8


def test_preroll_exports_private_labeling_audio_but_dataset_row_owns_only_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = build_chunks(_source(tmp_path, 60.0), [Interval(0.0, 60.0)], [], _config(tmp_path))
    fallback = chunks[1]
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"wav")

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", fake_run)
    ffmpeg_module.export_chunk(fallback, _config(tmp_path))
    filters = [command[command.index("-filter_complex") + 1] for command in commands]
    assert len(filters) == 2
    assert f"start={fallback.start:.6f}:end={fallback.end:.6f}" in filters[0]
    assert f"start={fallback.labeling_start:.6f}:end={fallback.labeling_end:.6f}" in filters[1]
    fallback.chunk_path.write_bytes(b"owned-target")
    row = export_module._row(LabeledChunk(chunk=fallback, transcript_text="target"), "train/target.wav", _config(tmp_path, text_column="text", language_hint="mixed"))
    assert row["owned_source_start"] == 25.0
    assert row["labeling_audio_source_start"] == 24.0
    assert row["target_offset_seconds"] == 1.0
    assert row["training_eligible"] is False
    assert "labeling_preroll_context" in row["review_signals"]
