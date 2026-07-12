import pytest

from daiya_training_harness.provenance import (
    ProvenanceRecord,
    ProvenanceValidationError,
    read_provenance,
    write_provenance,
)


@pytest.fixture
def provenance():
    return ProvenanceRecord(
        dataset_version="thai-ja-mix@2026-07-12",
        conversion_settings={"sample_rate_hz": 16000, "channels": 1, "normalization": "peak"},
        base_model_revision="openai/whisper-large-v3@0123456789abcdef",
        evaluation_backend={"name": "whisper", "revision": "fedcba9876543210", "dtype": "float32"},
        split_manifest_sha256="a" * 64,
        metadata={"experiment": "baseline"},
    )


def test_provenance_round_trip_is_exact(tmp_path, provenance):
    path = tmp_path / "provenance.json"
    write_provenance(path, provenance)
    loaded = read_provenance(path)
    assert loaded == provenance
    assert loaded.sha256 == provenance.sha256
    assert loaded.to_dict() == provenance.to_dict()


def test_provenance_detaches_nested_settings(provenance):
    assert provenance.conversion_settings["sample_rate_hz"] == 16000
    with pytest.raises(TypeError):
        provenance.evaluation_backend["name"] = "changed"


@pytest.mark.parametrize("digest", ["", "abc", "g" * 64])
def test_provenance_rejects_invalid_manifest_digest(digest):
    with pytest.raises(ProvenanceValidationError, match="SHA-256"):
        ProvenanceRecord(
            dataset_version="v1",
            conversion_settings={},
            base_model_revision="model@revision",
            evaluation_backend={"name": "backend", "revision": "revision"},
            split_manifest_sha256=digest,
        )


def test_provenance_rejects_unknown_fields(provenance):
    value = provenance.to_dict()
    value["host_path"] = "C:/machine-specific"
    with pytest.raises(ProvenanceValidationError, match="unknown provenance fields"):
        ProvenanceRecord.from_dict(value)
