from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "lab" / "vad_bakeoff.py"
SPEC = importlib.util.spec_from_file_location("vad_bakeoff_under_test", SCRIPT_PATH)
assert SPEC and SPEC.loader
vad_bakeoff = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = vad_bakeoff
SPEC.loader.exec_module(vad_bakeoff)


class VadBakeoffTest(unittest.TestCase):
    def test_default_thresholds_follow_backend_scale(self) -> None:
        args = SimpleNamespace(
            backend="energy,silero",
            threshold=None,
            min_speech_seconds="0.2",
            min_silence_seconds="0.3",
            speech_padding_seconds="0.1",
            max_utterance_seconds="8.0",
        )

        settings = list(vad_bakeoff.iter_settings(args))

        self.assertEqual(
            [(setting.backend, setting.threshold) for setting in settings],
            [("energy", 0.012), ("silero", 0.5)],
        )

    def test_text_reference_rows_are_concatenated_in_timestamp_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "refs.jsonl"
            rows = [
                {"file_name": "long.wav", "start": 5.0, "text": "second"},
                {"file_name": "long.wav", "start": 1.0, "text": "first"},
                {"file_name": "other.wav", "text": "other"},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            refs = vad_bakeoff.load_text_references(path)

        self.assertEqual(refs["long.wav"], "first second")
        self.assertEqual(refs["other.wav"], "other")

    def test_coverage_uses_unions_and_reference_span(self) -> None:
        audio_path = Path("long.wav").resolve()
        segments = [
            vad_bakeoff.Segment(audio_path, 1.0, 3.0, None),
            vad_bakeoff.Segment(audio_path, 2.0, 4.0, None),
        ]
        refs = {"long.wav": [(1.5, 2.5), (4.0, 5.0)]}

        metrics = vad_bakeoff.score_coverage(
            segments,
            refs,
            audio_paths=[audio_path],
        )

        self.assertAlmostEqual(metrics["unique_predicted_seconds"], 3.0)
        self.assertAlmostEqual(metrics["overlap_duplicated_seconds"], 1.0)
        self.assertAlmostEqual(metrics["scored_seconds"], 3.5)
        self.assertAlmostEqual(metrics["reference_speech_seconds"], 2.0)
        self.assertAlmostEqual(metrics["reference_speech_covered_seconds"], 1.0)
        self.assertAlmostEqual(metrics["reference_speech_missed_seconds"], 1.0)
        self.assertAlmostEqual(metrics["predicted_non_reference_seconds"], 1.5)

    def test_zero_predictions_are_counted_as_missed_reference_speech(self) -> None:
        audio_path = Path("silent-output.wav").resolve()
        metrics = vad_bakeoff.score_coverage(
            [],
            {"silent-output.wav": [(10.0, 12.0)]},
            audio_paths=[audio_path],
        )

        self.assertEqual(metrics["scored_seconds"], 2.0)
        self.assertEqual(metrics["reference_speech_seconds"], 2.0)
        self.assertEqual(metrics["reference_speech_covered_seconds"], 0.0)
        self.assertEqual(metrics["reference_speech_missed_seconds"], 2.0)

    def test_boundary_scope_does_not_invent_clipped_boundaries(self) -> None:
        audio_path = Path("long.wav").resolve()
        spanning_prediction = vad_bakeoff.Segment(audio_path, 0.0, 20.0, None)

        metrics = vad_bakeoff.score_boundaries(
            [spanning_prediction],
            {"long.wav": [(5.0, 10.0)]},
            collar=0.25,
            audio_paths=[audio_path],
        )

        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)

    def test_duration_buckets_have_stable_half_open_edges(self) -> None:
        buckets = vad_bakeoff.duration_bucket_counts(
            [0.99, 1.0, 1.99, 2.0, 4.99, 5.0, 7.99, 8.0]
        )
        self.assertEqual(
            buckets,
            {
                "duration_lt_1_count": 1,
                "duration_1_2_count": 2,
                "duration_2_5_count": 2,
                "duration_5_8_count": 2,
                "duration_8_plus_count": 1,
            },
        )

    def test_asr_scores_empty_segmentation_as_full_deletion(self) -> None:
        class UnusedModel:
            def transcribe(self, *_args, **_kwargs):
                raise AssertionError("no segments should be transcribed")

        audio_path = Path("dropped.wav").resolve()
        runner = vad_bakeoff.ASRRunner(UnusedModel(), None)
        cer, wer_like, status = vad_bakeoff.score_asr(
            [],
            runner,
            {"dropped.wav": "hello world"},
            audio_paths=[audio_path],
        )

        self.assertEqual(cer, 1.0)
        self.assertEqual(wer_like, 1.0)
        self.assertEqual(status, "asr ok")

    def test_explicit_silero_fallback_is_a_skipped_row(self) -> None:
        class EnergyFallback:
            pass

        setting = vad_bakeoff.Setting("silero", 0.5, 0.2, 0.5, 0.1, 8.0)
        audio = vad_bakeoff.AudioInput(Path("long.wav"), None, 16000, None, 30.0)
        args = SimpleNamespace(
            prefer_silero=False,
            chunk_seconds=0.25,
            boundary_collar_seconds=0.25,
            asr_model=None,
        )
        with mock.patch.object(
            vad_bakeoff,
            "create_segmenter",
            return_value=(EnergyFallback(), {"backend": "silero"}),
        ):
            row, details = vad_bakeoff.run_setting(setting, [audio], args, {}, None, {})

        self.assertEqual(row["status"], "skipped")
        self.assertEqual(row["segmenter_backend"], "energy")
        self.assertEqual(row["utterance_count"], 0)
        self.assertIn("requested Silero", row["notes"])
        self.assertEqual(details, [])


if __name__ == "__main__":
    unittest.main()
