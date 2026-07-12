from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_candidate_manifest import _spelling_signals, build_manifest  # noqa: E402
from daiya_dataset_cleaning.models import ReasonCode  # noqa: E402
from evaluate_gold import _metrics  # noqa: E402


class ExperimentScriptTests(unittest.TestCase):
    def test_metrics_use_token_edit_distance_for_wer_like(self) -> None:
        self.assertEqual(_metrics("a b", "a b")["wer_like"], 0)
        self.assertEqual(_metrics("a b", "a c")["wer_like"], 0.5)

    def test_spelling_results_are_bound_to_label_and_internally_consistent(self) -> None:
        stale_evidence, stale_reasons = _spelling_signals(
            {
                "label_sha256": "wrong",
                "checked_units": 1,
                "suspicious_units": 1,
                "suspicious_ratio": 1.0,
            },
            "label",
            review_threshold=0.2,
            min_issues=1,
        )
        self.assertEqual(stale_evidence[0].name, "spelling.stale_result")
        self.assertNotIn(ReasonCode.SPELLING_SUSPECT, stale_reasons)

        malformed_evidence, malformed_reasons = _spelling_signals(
            {
                "label_sha256": hashlib.sha256(b"label").hexdigest(),
                "checked_units": 100,
                "suspicious_units": 1,
                "suspicious_ratio": 0.9,
            },
            "label",
            review_threshold=0.2,
            min_issues=1,
        )
        self.assertEqual(malformed_evidence[0].name, "spelling.malformed_result")
        self.assertNotIn(ReasonCode.SPELLING_SUSPECT, malformed_reasons)

    def test_manifest_quarantines_protected_audio_without_mutating_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "dataset"
            train = audio / "train"
            gold = root / "gold"
            train.mkdir(parents=True)
            gold.mkdir()
            protected = train / "clip.wav"
            protected.write_bytes(b"protected-audio")
            (train / "other.wav").write_bytes(b"other-audio")
            (gold / "gold.wav").write_bytes(protected.read_bytes())
            metadata = root / "metadata.jsonl"
            metadata.write_text(
                json.dumps({"file_name": "train/clip.wav", "text": "label", "speech_duration": 1.0})
                + "\n"
                + json.dumps({"file_name": "train/other.wav", "text": "other", "speech_duration": 1.0})
                + "\n",
                encoding="utf-8",
            )
            output = root / "manifest.jsonl"
            summary = build_manifest(
                metadata,
                audio,
                output,
                dataset_version="fixture-v1",
                protected_gold_dir=gold,
            )
            self.assertEqual(summary["records"], 2)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["disposition"], "drop")
            self.assertIn("protected_gold_overlap", records[0]["reasons"])
            self.assertEqual(protected.read_bytes(), b"protected-audio")

            spelling = root / "spelling.jsonl"
            spelling.write_text(
                json.dumps({
                    "file_name": "train/other.wav",
                    "label_sha256": hashlib.sha256(b"other").hexdigest(),
                    "checked_units": 1,
                    "suspicious_units": 1,
                    "suspicious_ratio": 1.0,
                    "checker_names": ["fixture-checker"],
                    "routed_span_counts": {"english": 1},
                    "by_language": {
                        "english": {"checked_units": 1, "suspicious_units": 1, "suspicious_ratio": 1.0}
                    },
                }) + "\n",
                encoding="utf-8",
            )
            spelling_output = root / "spelling-manifest.jsonl"
            spelling_predictions = root / "spelling-predictions.jsonl"
            spelling_predictions.write_text(
                json.dumps({"file_name": "train/other.wav", "prediction": "replacement", "model_name": "m1"})
                + "\n"
                + json.dumps({"file_name": "train/other.wav", "prediction": "replacement", "model_name": "m2"})
                + "\n",
                encoding="utf-8",
            )
            build_manifest(
                metadata,
                audio,
                spelling_output,
                dataset_version="fixture-v1",
                protected_gold_dir=gold,
                predictions_path=spelling_predictions,
                spelling_results_path=spelling,
            )
            spelling_records = [json.loads(line) for line in spelling_output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(spelling_records[1]["disposition"], "review")
            self.assertIn("spelling_suspect", spelling_records[1]["reasons"])
            self.assertIsNone(spelling_records[1]["proposed_label"])

            missing_metadata = root / "missing-metadata.jsonl"
            missing_metadata.write_text(
                json.dumps({"file_name": "train/missing.wav", "text": "label", "speech_duration": 1.0}) + "\n",
                encoding="utf-8",
            )
            missing_output = root / "missing-manifest.jsonl"
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"file_name": "train/missing.wav", "prediction": "label", "model_name": "m1"})
                + "\n"
                + json.dumps({"file_name": "train/missing.wav", "prediction": "label", "model_name": "m2"})
                + "\n",
                encoding="utf-8",
            )
            build_manifest(missing_metadata, audio, missing_output, dataset_version="fixture-v1", predictions_path=predictions)
            missing_record = json.loads(missing_output.read_text(encoding="utf-8"))
            self.assertEqual(missing_record["disposition"], "review")
            self.assertIn("malformed_input", missing_record["reasons"])
            self.assertIsNone(missing_record["proposed_label"])


if __name__ == "__main__":
    unittest.main()
