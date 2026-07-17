from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from daiya_whisper_lora.split_manifest import apply_split_manifest, compile_manifest, partition_rows_by_manifest


class SplitManifestTests(unittest.TestCase):
    def test_manifest_partitions_by_group_without_row_random_split(self) -> None:
        rows = [
            {"sample_id": "a1", "source_file": "call-a.wav", "file_name": "a1.wav", "text": "a1"},
            {"sample_id": "a2", "source_file": "call-a.wav", "file_name": "a2.wav", "text": "a2"},
            {"sample_id": "b1", "source_file": "call-b.wav", "file_name": "b1.wav", "text": "b1"},
        ]
        entries = [
            {"source_file": "call-a.wav", "split": "train"},
            {"source_file": "call-b.wav", "split": "validation"},
        ]
        split = partition_rows_by_manifest(rows, entries)

        self.assertEqual([row["sample_id"] for row in split["train"]], ["a1", "a2"])
        self.assertEqual([row["sample_id"] for row in split["validation"]], ["b1"])
        sample_to_split, group_to_split = compile_manifest(entries)
        self.assertEqual(sample_to_split, {})
        self.assertEqual(group_to_split["call-a.wav"], "train")

    def test_group_overlap_is_rejected(self) -> None:
        rows = [
            {"sample_id": "a1", "source_file": "call-a.wav", "file_name": "a1.wav", "text": "a1"},
            {"sample_id": "a2", "source_file": "call-a.wav", "file_name": "a2.wav", "text": "a2"},
        ]
        entries = [
            {"sample_id": "a1", "split": "train"},
            {"sample_id": "a2", "split": "validation"},
        ]
        with self.assertRaisesRegex(ValueError, "leaks group"):
            partition_rows_by_manifest(rows, entries)

    def test_duplicate_dataset_sample_ids_are_rejected(self) -> None:
        rows = [
            {"sample_id": "dup", "source_file": "a.wav", "file_name": "a1.wav", "text": "a1"},
            {"sample_id": "dup", "source_file": "b.wav", "file_name": "b1.wav", "text": "b1"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
            partition_rows_by_manifest(rows, [{"sample_id": "dup", "split": "train"}])

    def test_source_file_manifest_is_portable_across_absolute_roots(self) -> None:
        rows = [
            {"sample_id": "a1", "source_file": r"C:\\local\\raw\\call-a.wav", "text": "a1"},
            {"sample_id": "b1", "source_file": "/mnt/data/raw/call-b.wav", "text": "b1"},
        ]
        split = partition_rows_by_manifest(
            rows,
            [
                {"source_file": "call-a.wav", "split": "train"},
                {"source_file": "call-b.wav", "split": "validation"},
            ],
        )
        self.assertEqual([row["sample_id"] for row in split["train"]], ["a1"])
        self.assertEqual([row["sample_id"] for row in split["validation"]], ["b1"])

    def test_apply_split_manifest_records_group_and_sample_hashes(self) -> None:
        rows = [
            {"sample_id": "a1", "source_file": "call-a.wav", "text": "a1"},
            {"sample_id": "a2", "source_file": "call-a.wav", "text": "a2"},
            {"sample_id": "b1", "source_file": "call-b.wav", "text": "b1"},
        ]

        class FakeDataset:
            def __init__(self, values: list[dict[str, str]]) -> None:
                self.values = values

            def __len__(self) -> int:
                return len(self.values)

            def __getitem__(self, index: int) -> dict[str, str]:
                return self.values[index]

            @classmethod
            def from_list(cls, values: list[dict[str, str]]) -> "FakeDataset":
                return cls(values)

        class FakeDatasetDict(dict):
            pass

        fake_datasets = SimpleNamespace(Dataset=FakeDataset, DatasetDict=FakeDatasetDict)
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "split.jsonl"
            manifest.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"source_file": "call-a.wav", "split": "train"},
                        {"source_file": "call-b.wav", "split": "validation"},
                    )
                ),
                encoding="utf-8",
            )
            with patch.dict("sys.modules", {"datasets": fake_datasets}):
                result, identity = apply_split_manifest(FakeDatasetDict(train=FakeDataset(rows)), manifest)

        self.assertEqual(len(result["train"]), 2)
        self.assertEqual(len(result["validation"]), 1)
        self.assertEqual(identity.group_count, 2)
        self.assertEqual(identity.splits, {"train": 2, "validation": 1})
        self.assertEqual(set(identity.split_sample_sha256), {"train", "validation"})


if __name__ == "__main__":
    unittest.main()
