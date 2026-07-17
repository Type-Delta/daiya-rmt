from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from daiya_whisper_pipeline.evidence import TimestampEvidenceStage, TimestampWord, low_energy_gaps
from daiya_whisper_pipeline.types import NormalizedAudio


def _audio(tmp_path: Path, name: str = "source.wav") -> NormalizedAudio:
    path = tmp_path / name
    # quiet / voiced / quiet creates deterministic waveform-only evidence.
    samples = np.concatenate((np.zeros(1_600), np.full(1_600, 0.2, dtype=np.float32), np.zeros(1_600))).astype("float32")
    sf.write(path, samples, 16_000)
    return NormalizedAudio(path, path, "source", duration_seconds=0.3)


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp_evidence_cache_dir=tmp_path / "cache",
        timestamp_model="local-test-model",
        timestamp_device="cpu",
        timestamp_compute_type="int8",
        timestamp_beam_size=3,
        timestamp_language="",
        timestamp_condition_on_previous_text=False,
        energy_window_seconds=0.05,
        energy_low_percentile=30.0,
        energy_min_gap_seconds=0.05,
    )


def test_timestamp_evidence_cache_records_identity_settings_and_failure_safe_words(tmp_path: Path, monkeypatch) -> None:
    source = _audio(tmp_path)
    stage = TimestampEvidenceStage(_config(tmp_path))
    monkeypatch.setattr(stage, "_decode", lambda _: (TimestampWord(0.1, 0.2, "ไทย", 0.8),))

    first = stage.collect(source)
    assert first.status == "ok"
    assert first.cache_path and first.cache_path.is_file()
    assert first.provenance()["model"]["model"] == "local-test-model"
    assert first.provenance()["word_timestamp_count"] == 1
    assert "ไทย" not in str(first.provenance())

    cached = TimestampEvidenceStage(_config(tmp_path))
    monkeypatch.setattr(cached, "_decode", lambda _: (_ for _ in ()).throw(AssertionError("cache not used")))
    second = cached.collect(source)
    assert second.words == first.words
    assert second.source_audio_sha256 == first.source_audio_sha256


def test_timestamp_evidence_cache_invalidates_model_and_energy_changes(tmp_path: Path, monkeypatch) -> None:
    source = _audio(tmp_path)
    first_stage = TimestampEvidenceStage(_config(tmp_path))
    monkeypatch.setattr(first_stage, "_decode", lambda _: (TimestampWord(0.1, 0.2, "first"),))
    first_stage.collect(source)

    changed_model = _config(tmp_path)
    changed_model.timestamp_model = "different-local-model"
    model_stage = TimestampEvidenceStage(changed_model)
    monkeypatch.setattr(model_stage, "_decode", lambda _: (TimestampWord(0.1, 0.2, "model"),))
    assert model_stage.collect(source).words[0].text == "model"

    changed_energy = _config(tmp_path)
    changed_energy.energy_low_percentile = 10.0
    energy_stage = TimestampEvidenceStage(changed_energy)
    monkeypatch.setattr(energy_stage, "_decode", lambda _: (TimestampWord(0.1, 0.2, "energy"),))
    assert energy_stage.collect(source).words[0].text == "energy"


def test_timestamp_evidence_cache_invalidates_replaced_local_model_artifact(tmp_path: Path, monkeypatch) -> None:
    source = _audio(tmp_path)
    model = tmp_path / "local-model"
    model.mkdir()
    model_file = model / "model.bin"
    model_file.write_bytes(b"first-model")
    config = _config(tmp_path)
    config.timestamp_model = str(model)

    first_stage = TimestampEvidenceStage(config)
    monkeypatch.setattr(first_stage, "_decode", lambda _: (TimestampWord(0.1, 0.2, "first"),))
    first_stage.collect(source)

    model_file.write_bytes(b"replaced-model-content")
    replacement_stage = TimestampEvidenceStage(config)
    monkeypatch.setattr(replacement_stage, "_decode", lambda _: (TimestampWord(0.1, 0.2, "replacement"),))
    assert replacement_stage.collect(source).words[0].text == "replacement"


def test_low_energy_gaps_reject_flat_non_silent_waveform(tmp_path: Path) -> None:
    path = tmp_path / "flat.wav"
    sf.write(path, np.full(16_000, 0.2, dtype=np.float32), 16_000)
    assert low_energy_gaps(path, window_seconds=0.05, low_percentile=20.0, minimum_seconds=0.05) == []


def test_timestamp_failure_retains_energy_evidence_without_exporting_asr_text(tmp_path: Path, monkeypatch) -> None:
    source = _audio(tmp_path)
    stage = TimestampEvidenceStage(_config(tmp_path))
    monkeypatch.setattr(stage, "_decode", lambda _: (_ for _ in ()).throw(RuntimeError("model unavailable")))

    evidence = stage.collect(source)
    assert evidence.status == "failed"
    assert "whisper:RuntimeError:model unavailable" in evidence.failure
    assert evidence.energy_gaps
    assert evidence.provenance()["word_timestamp_count"] == 0


def test_low_energy_gaps_find_quiet_windows(tmp_path: Path) -> None:
    source = _audio(tmp_path)
    gaps = low_energy_gaps(source.normalized_path, window_seconds=0.05, low_percentile=30.0, minimum_seconds=0.05)
    assert gaps
    assert all(gap.duration >= 0.05 for gap in gaps)
