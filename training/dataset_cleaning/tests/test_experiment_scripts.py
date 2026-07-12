from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_candidate_manifest import build_manifest  # noqa: E402
from evaluate_gold import _metrics  # noqa: E402


class ExperimentScriptTests(unittest.TestCase):
    def test_metrics_use_token_edit_distance_for_wer_like(self) -> None:
        self.assertEqual(_metrics("a b", "a b")["wer_like"], 0)
        self.assertEqual(_metrics("a b", "a c")["wer_like"], 0.5)

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
