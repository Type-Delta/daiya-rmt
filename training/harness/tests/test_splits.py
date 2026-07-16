import json

import pytest

from daiya_training_harness.splits import (
    ManifestValidationError,
    SplitManifest,
    canonical_json_hash,
    read_manifest,
    write_manifest,
)


@pytest.fixture
def manifest():
    return SplitManifest(
        dataset_version="thai-ja-mix@2026-07-12",
        splits={"train": ["conv-03", "conv-01"], "validation": ["conv-02"], "test": []},
        metadata={"generator": "seeded-v1", "seed": 17},
    )


def test_canonical_hash_ignores_object_key_order():
    assert canonical_json_hash({"b": 2, "a": [1, "ไทย"]}) == canonical_json_hash(
        {"a": [1, "ไทย"], "b": 2}
    )


def test_manifest_round_trip_preserves_hash(tmp_path, manifest):
    path = tmp_path / "nested" / "splits.json"
    write_manifest(path, manifest)
    loaded = read_manifest(path)
    assert loaded == manifest
    assert loaded.sha256 == manifest.sha256
    assert json.loads(path.read_text(encoding="utf-8"))["dataset_version"] == manifest.dataset_version


@pytest.mark.parametrize(
    "splits, message",
    [
        ({"train": ["same"], "test": ["same"]}, "overlaps"),
        ({"train": ["same", "same"]}, "duplicated"),
        ({"train": [""]}, "non-empty"),
    ],
)
def test_manifest_rejects_invalid_or_overlapping_ids(splits, message):
    with pytest.raises(ManifestValidationError, match=message):
        SplitManifest(dataset_version="v1", splits=splits)


def test_manifest_is_frozen_and_detached_from_input():
    source = {"train": ["c1"]}
    manifest = SplitManifest(dataset_version="v1", splits=source)
    source["train"].append("c2")
    assert manifest.splits["train"] == ("c1",)
    with pytest.raises(TypeError):
        manifest.splits["test"] = ("c2",)
