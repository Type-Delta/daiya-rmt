from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from daiya_dataset_cleaning.decision import decide
from daiya_dataset_cleaning.io import write_csv, write_jsonl
from daiya_dataset_cleaning.models import Confidence, Disposition, ManifestRecord, ProposedLabel, ReasonCode, SourceIdentity
from daiya_dataset_cleaning.normalize import content_identity, normalize_text, source_identity
from daiya_dataset_cleaning.signals import evaluate_signals, script_profile


class ToolkitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SourceIdentity("source-1", "audio/example.wav", dataset="fixture")

    def test_normalization_and_identity_are_deterministic(self) -> None:
        self.assertEqual(normalize_text("Ａ  \n B"), "A B")
        self.assertEqual(content_identity("Hello"), content_identity("  hello "))
        self.assertEqual(source_identity(r"a\b.wav", record_id="1"), source_identity("a/b.wav", record_id="1"))

    def test_records_are_immutable_and_confidence_is_validated(self) -> None:
        record = decide(self.source, "hello", 1.0)
        with self.assertRaises(FrozenInstanceError):
            record.normalized_label = "changed"  # type: ignore[misc]
        with self.assertRaises(ValueError):
            Confidence(1.1, "test")
        with self.assertRaises(ValueError):
            Confidence(0.5, "")

    def test_missing_and_malformed_inputs_are_safe(self) -> None:
        record = decide(self.source, None, "not-a-duration")
        self.assertEqual(record.disposition, Disposition.DROP)
        self.assertIn(ReasonCode.EMPTY_LABEL, record.reasons)
        self.assertIn(ReasonCode.INVALID_DURATION, record.reasons)
        json.dumps(record.to_dict(), allow_nan=False)

    def test_duration_rate_and_language_neutral_script_signals(self) -> None:
        evidence, reasons = evaluate_signals("ภาษาไทย English", 0.1, expected_scripts=frozenset({"thai", "latin"}))
        self.assertIn(ReasonCode.TOO_SHORT, reasons)
        self.assertIn(ReasonCode.CHAR_RATE_HIGH, reasons)
        self.assertGreater(script_profile("ภาษาไทย English")["thai"], 0)
        self.assertGreater(script_profile("日本語")["han"], 0)
        self.assertTrue(any(item.name == "characters_per_second" for item in evidence))
        _, mismatch = evaluate_signals("ภาษาไทย", 2, expected_scripts=frozenset({"hiragana"}))
        self.assertIn(ReasonCode.SCRIPT_MISMATCH, mismatch)

    def test_low_adapter_confidence_causes_review(self) -> None:
        record = decide(self.source, "valid transcript", 2.0, confidence=Confidence(0.2, "asr logprob", calibrated=False))
        self.assertEqual(record.disposition, Disposition.REVIEW)
        self.assertIn(ReasonCode.LOW_CONFIDENCE, record.reasons)

    def test_correction_requires_grounded_provenance(self) -> None:
        with self.assertRaises(ValueError):
            ProposedLabel("replacement", "alignment", (), Confidence(0.9, "held-out calibration", True))
        proposal = ProposedLabel("replacement", "alignment consensus", ("alignment.word.0",), Confidence(0.9, "held-out calibration", True))
        record = decide(self.source, "original", 1.0, proposed_label=proposal)
        self.assertEqual(record.disposition, Disposition.CORRECT)
        self.assertIn(ReasonCode.GROUNDED_CORRECTION, record.reasons)
        with self.assertRaises(ValueError):
            ManifestRecord(self.source, "x", "x", Disposition.CORRECT)

    def test_jsonl_and_csv_round_trip_special_characters(self) -> None:
        import tempfile
        record = decide(self.source, 'ไทย, "quoted"\nEnglish', 2.0)
        with tempfile.TemporaryDirectory() as directory:
            json_path, csv_path = Path(directory) / "out.jsonl", Path(directory) / "out.csv"
            write_jsonl([record], json_path)
            write_csv([record], csv_path)
            parsed_json = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(encoding="utf-8", newline="") as handle:
                parsed_csv = next(csv.DictReader(handle))
            self.assertEqual(parsed_json["original_label"], record.original_label)
            self.assertEqual(parsed_csv["original_label"], record.original_label)

            formula = decide(self.source, "=1+1", 1.0)
            write_csv([formula], csv_path)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(next(csv.DictReader(handle))["original_label"], "'=1+1")


if __name__ == "__main__":
    unittest.main()
