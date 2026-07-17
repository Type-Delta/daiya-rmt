from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from daiya_whisper_lora.train import TrainingConfig, feature_cache_key, source_split_indices, select_training_rows


def test_trainer_excludes_review_only_rows_and_records_identities() -> None:
    rows = [
        (1, {"file_name": "train/a.wav", "training_eligible": True}),
        (2, {"file_name": "train/b.wav", "training_eligible": False}),
    ]
    selected, selection = select_training_rows(
        rows,
        include_ineligible_for_research=False,
        legacy_training_eligibility="error",
    )

    assert [row[1]["file_name"] for row in selected] == ["train/a.wav"]
    assert selection.provenance()["included_identities"] == ["train/a.wav"]
    assert selection.provenance()["excluded_identities"] == ["train/b.wav"]


def test_trainer_requires_explicit_legacy_compatibility_and_research_override() -> None:
    legacy = [(1, {"file_name": "train/legacy.wav"})]
    with pytest.raises(ValueError, match="lacks training_eligible"):
        select_training_rows(legacy, include_ineligible_for_research=False, legacy_training_eligibility="error")

    review_only = [(2, {"file_name": "train/review.wav", "training_eligible": False})]
    selected, selection = select_training_rows(
        review_only,
        include_ineligible_for_research=True,
        legacy_training_eligibility="error",
    )
    assert [row[1]["file_name"] for row in selected] == ["train/review.wav"]
    assert selection.provenance()["include_ineligible_for_research"] is True


def test_legacy_exclude_stays_excluded_under_research_override_and_identities_are_per_clip() -> None:
    rows = [
        (1, {"source_id": "same-source", "source_start": 0.0, "source_end": 10.0}),
        (2, {"source_id": "same-source", "source_start": 10.0, "source_end": 20.0}),
    ]
    selected, selection = select_training_rows(
        rows,
        include_ineligible_for_research=True,
        legacy_training_eligibility="exclude",
    )

    assert selected == []
    assert selection.provenance()["excluded_identities"] == ["same-source@0.0-10.0", "same-source@10.0-20.0"]


def test_validation_split_is_disjoint_by_source_recording() -> None:
    sources = ["recording-a", "recording-a", "recording-b", "recording-b", "recording-c"]
    train, validation = source_split_indices(sources, validation_size=0.34, seed=17)

    assert train and validation
    assert {sources[index] for index in train}.isdisjoint({sources[index] for index in validation})
    assert sorted(train + validation) == list(range(len(sources)))


def test_feature_cache_key_changes_with_selected_label_content() -> None:
    config = TrainingConfig(dataset_dir=Path("dataset"), model_name_or_path="model", output_dir=Path("output"))
    first = DatasetDict({"train": Dataset.from_dict({"file_name": ["a.wav"], "source_id": ["source"], "text": ["first"]})})
    second = DatasetDict({"train": Dataset.from_dict({"file_name": ["a.wav"], "source_id": ["source"], "text": ["changed"]})})

    assert feature_cache_key(first, config) != feature_cache_key(second, config)


def test_feature_cache_key_changes_with_exported_audio_identity() -> None:
    config = TrainingConfig(dataset_dir=Path("dataset"), model_name_or_path="model", output_dir=Path("output"))
    first = DatasetDict({"train": Dataset.from_dict({
        "file_name": ["a.wav"], "source_id": ["source"], "text": ["label"], "audio_sha256": ["a" * 64],
    })})
    replaced = DatasetDict({"train": Dataset.from_dict({
        "file_name": ["a.wav"], "source_id": ["source"], "text": ["label"], "audio_sha256": ["b" * 64],
    })})

    assert feature_cache_key(first, config) != feature_cache_key(replaced, config)
