from __future__ import annotations

import unittest
from pathlib import Path

from daiya_whisper_lora.checkpoint_probe import (
    ProbeConfig,
    audio_duration_seconds,
    discover_candidates,
    metric_delta,
    select_best_candidate,
    select_probe_indices,
)
from daiya_whisper_lora.metrics import (
    aggregate_metric_rows,
    english_terms_from_text,
    is_short_utterance,
    normalize_thai_spacing,
    text_metrics,
)


class GenerationMetricTests(unittest.TestCase):
    def test_text_metrics_include_cer_no_space_and_wer_like(self) -> None:
        metrics = text_metrics("relation กัน", "รีเอชันกัน")

        self.assertIn("cer", metrics)
        self.assertIn("cer_no_space", metrics)
        self.assertIn("wer_like", metrics)
        self.assertGreater(metrics["cer"], 0.0)

    def test_thai_spacing_normalizer_collapses_dense_thai_gaps(self) -> None:
        self.assertEqual(
            normalize_thai_spacing("ส วั ส ดี ค รั บ ส วั ส ดี"),
            "สวัสดีครับสวัสดี",
        )

    def test_aggregate_metric_rows_prefers_micro_corpus_rate(self) -> None:
        rows = [
            {"metrics": text_metrics("abcd", "abxd")},
            {"metrics": text_metrics("xy", "xy")},
        ]

        summary = aggregate_metric_rows(rows)

        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["micro_cer"], 1 / 6)

    def test_english_technical_terms_ignore_common_prompt_words(self) -> None:
        terms = english_terms_from_text("Topic Terms relation API status first")

        self.assertIn("API", terms)
        self.assertIn("relation", terms)
        self.assertIn("status", terms)
        self.assertNotIn("Topic", terms)
        self.assertNotIn("Terms", terms)
        self.assertNotIn("first", terms)


class CheckpointSelectionTests(unittest.TestCase):
    def test_probe_selection_includes_short_and_technical_rows_first(self) -> None:
        rows = [
            {"text": "ordinary long", "speech_duration": 8.0},
            {"text": "สั้น", "speech_duration": 1.2},
            {"text": "relation status", "speech_duration": 5.0},
            {"text": "another", "speech_duration": 4.0},
        ]

        indices = select_probe_indices(
            rows,
            max_samples=3,
            min_short_samples=1,
            min_technical_term_samples=1,
            short_utterance_seconds=3.0,
        )

        self.assertEqual(indices, [0, 1, 2])

    def test_short_utterance_accepts_audio_duration_fallback(self) -> None:
        self.assertTrue(is_short_utterance({"audio_duration_seconds": 1.5}, 3.0))
        self.assertEqual(audio_duration_seconds({"duration": 2.25}), 2.25)

    def test_best_candidate_uses_primary_metric_then_micro_cer_tiebreak(self) -> None:
        summaries = [
            {"name": "checkpoint-500", "overall": {"micro_cer": 0.22, "mean_cer": 0.20}},
            {"name": "checkpoint-400", "overall": {"micro_cer": 0.21, "mean_cer": 0.20}},
        ]

        best = select_best_candidate(summaries, "mean_cer")

        self.assertEqual(best["name"], "checkpoint-400")

    def test_metric_delta_reports_selected_minus_final(self) -> None:
        selected = {"overall": {"micro_cer": 0.21}}
        final = {"overall": {"micro_cer": 0.25}}

        delta = metric_delta(selected, final, "micro_cer")

        self.assertIsNotNone(delta)
        self.assertAlmostEqual(delta["selected_minus_final"], -0.04)  # type: ignore[index]

    def test_discover_candidates_sorts_checkpoints_and_includes_final(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_name:
            run_dir = Path(temp_name)
            for name in ("checkpoint-20", "checkpoint-10", "."):
                adapter_dir = run_dir if name == "." else run_dir / name
                adapter_dir.mkdir(exist_ok=True)
                (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

            candidates = discover_candidates(
                ProbeConfig(
                    run_dir=run_dir,
                    base_model="openai/whisper-large-v3",
                    dataset_dir=run_dir,
                    output_dir=run_dir / "out",
                )
            )

        self.assertEqual([candidate.name for candidate in candidates], ["checkpoint-10", "checkpoint-20", run_dir.name])


if __name__ == "__main__":
    unittest.main()
