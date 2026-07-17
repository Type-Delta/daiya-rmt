from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from daiya_whisper_lora.checkpoint_probe import (
    ProbeConfig,
    audio_duration_seconds,
    discover_candidates,
    metric_delta,
    run_probe,
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
            {"name": "checkpoint-500", "overall": {"count": 1, "micro_cer": 0.22, "mean_cer": 0.20}},
            {"name": "checkpoint-400", "overall": {"count": 1, "micro_cer": 0.21, "mean_cer": 0.20}},
        ]

        best = select_best_candidate(summaries, "mean_cer")

        self.assertEqual(best["name"], "checkpoint-400")

    def test_best_candidate_rejects_all_failed_candidates(self) -> None:
        summaries = [
            {
                "name": "checkpoint-400",
                "failed_count": 2,
                "overall": {"count": 0, "micro_cer": None},
            },
            {
                "name": "checkpoint-500",
                "failed_count": 2,
                "overall": {"count": 0, "micro_cer": None},
            },
        ]

        with self.assertRaisesRegex(ValueError, "No candidate has both"):
            select_best_candidate(summaries, "micro_cer")

    def test_best_candidate_ignores_failed_candidate_when_another_is_valid(self) -> None:
        summaries = [
            {"name": "checkpoint-400", "overall": {"count": 0, "micro_cer": 0.0}},
            {"name": "checkpoint-450", "overall": {"count": 1, "micro_cer": float("nan")}},
            {"name": "checkpoint-500", "overall": {"count": 2, "micro_cer": 0.25}},
        ]

        best = select_best_candidate(summaries, "micro_cer")

        self.assertEqual(best["name"], "checkpoint-500")

    def test_best_candidate_rejects_missing_and_non_finite_primary_metrics(self) -> None:
        summaries = [
            {"name": "missing", "overall": {"count": 1}},
            {"name": "nan", "overall": {"count": 1, "micro_cer": float("nan")}},
            {"name": "infinite", "overall": {"count": 1, "micro_cer": float("inf")}},
        ]

        with self.assertRaisesRegex(ValueError, "finite 'micro_cer'"):
            select_best_candidate(summaries, "micro_cer")

    def test_total_failure_writes_details_and_failed_summary_before_raising(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            candidate = root / "checkpoint-400"
            output_dir = root / "probes"
            config = ProbeConfig(
                run_dir=root,
                base_model="openai/whisper-large-v3",
                dataset_dir=root / "dataset",
                output_dir=output_dir,
            )
            failed_detail = {
                "candidate": candidate.name,
                "status": "failed",
                "error": "RuntimeError('out of memory')",
            }

            with (
                patch(
                    "daiya_whisper_lora.checkpoint_probe.discover_candidates",
                    return_value=[candidate],
                ),
                patch(
                    "daiya_whisper_lora.checkpoint_probe.load_probe_rows",
                    return_value=[{"reference": "hello"}],
                ),
                patch(
                    "daiya_whisper_lora.checkpoint_probe.score_candidate",
                    return_value=[failed_detail],
                ),
                patch("daiya_whisper_lora.checkpoint_probe.release_model_memory"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Inspect per-sample errors"):
                    run_probe(config)

            details_paths = list(output_dir.glob("details_*.jsonl"))
            summary_paths = list(output_dir.glob("summary_*.json"))
            self.assertEqual(len(details_paths), 1)
            self.assertEqual(len(summary_paths), 1)
            self.assertIn("out of memory", details_paths[0].read_text(encoding="utf-8"))
            summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(summary["selected_checkpoint"])
            self.assertEqual(summary["candidates"][0]["scored_count"], 0)
            self.assertEqual(summary["candidates"][0]["failed_count"], 1)

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
