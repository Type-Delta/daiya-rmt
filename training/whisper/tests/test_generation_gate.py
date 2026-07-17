from __future__ import annotations

import math
from pathlib import Path
import unittest

from daiya_whisper_lora.checkpoint_probe import (
    ProbeConfig,
    generation_prompt_text,
    generation_fingerprint,
    select_selector_indices,
    select_best_candidate,
    select_best_eval_loss_candidate,
)
from daiya_whisper_lora.metrics import aggregate_metric_rows, text_metrics


def summary(
    name: str,
    metric: float | None,
    count: int,
    *,
    freshness: dict[str, str] | None = None,
    eval_loss: float | None = None,
) -> dict[str, object]:
    overall = {"count": count, "micro_cer": metric}
    return {
        "name": name,
        "path": f"/tmp/{name}",
        "attempted_count": count,
        "scored_count": count,
        "failed_count": 0,
        "overall": overall,
        "freshness": freshness or {},
        "eval_loss": eval_loss,
    }


class GenerationGateTests(unittest.TestCase):
    def test_shared_metrics_compute_micro_cer_and_wer_like(self) -> None:
        rows = [
            {"metrics": text_metrics("Daiya RMT", "Daiya RMT")},
            {"metrics": text_metrics("hello", "hallo")},
        ]

        metrics = aggregate_metric_rows(rows)

        self.assertEqual(metrics["count"], 2)
        self.assertGreater(metrics["micro_cer"], 0.0)
        self.assertLess(metrics["micro_cer"], 0.1)
        self.assertGreater(metrics["micro_wer_like"], 0.0)

    def test_invalid_missing_nan_inf_and_zero_count_candidates_cannot_win(self) -> None:
        fresh = {
            "candidate_fingerprint": "ckpt-a",
            "dataset_fingerprint": "data",
            "generation_fingerprint": "gen",
        }
        expected = {
            "candidate_fingerprints": {"/tmp/a": "ckpt-a", "/tmp/b": "ckpt-b"},
            "dataset_fingerprint": "data",
            "generation_fingerprint": "gen",
        }
        candidates = [
            summary("zero", 0.0, 0, freshness=fresh),
            summary("nan", math.nan, 1, freshness=fresh),
            summary("inf", math.inf, 1, freshness=fresh),
            summary("missing", None, 1, freshness=fresh),
            summary(
                "b",
                0.2,
                1,
                freshness={
                    "candidate_fingerprint": "ckpt-b",
                    "dataset_fingerprint": "data",
                    "generation_fingerprint": "gen",
                },
            ),
        ]

        self.assertEqual(select_best_candidate(candidates, "micro_cer", expected)["name"], "b")

    def test_stale_dataset_or_prompt_fingerprint_is_rejected(self) -> None:
        stale = summary(
            "a",
            0.01,
            1,
            freshness={
                "candidate_fingerprint": "ckpt-a",
                "dataset_fingerprint": "old-data",
                "generation_fingerprint": "gen",
            },
        )
        fresh = summary(
            "b",
            0.2,
            1,
            freshness={
                "candidate_fingerprint": "ckpt-b",
                "dataset_fingerprint": "data",
                "generation_fingerprint": "gen",
            },
        )
        expected = {
            "candidate_fingerprints": {"/tmp/a": "ckpt-a", "/tmp/b": "ckpt-b"},
            "dataset_fingerprint": "data",
            "generation_fingerprint": "gen",
        }

        self.assertEqual(select_best_candidate([stale, fresh], "micro_cer", expected)["name"], "b")

    def test_eval_loss_fallback_is_explicit_and_uses_finite_fresh_loss(self) -> None:
        expected = {
            "candidate_fingerprints": {"/tmp/a": "ckpt-a", "/tmp/b": "ckpt-b"},
            "dataset_fingerprint": "data",
            "generation_fingerprint": "gen",
        }
        candidates = [
            summary(
                "a",
                None,
                0,
                freshness={
                    "candidate_fingerprint": "ckpt-a",
                    "dataset_fingerprint": "data",
                    "generation_fingerprint": "gen",
                },
                eval_loss=0.8,
            ),
            summary(
                "b",
                None,
                0,
                freshness={
                    "candidate_fingerprint": "ckpt-b",
                    "dataset_fingerprint": "data",
                    "generation_fingerprint": "gen",
                },
                eval_loss=0.4,
            ),
        ]

        with self.assertRaisesRegex(ValueError, "No fresh candidate"):
            select_best_candidate(candidates, "micro_cer", expected)
        best = select_best_eval_loss_candidate(candidates, expected)
        self.assertEqual(best["name"], "b")
        self.assertEqual(best["fallback_metric"], "eval_loss")

    def test_generation_fingerprint_includes_prompt_strategy(self) -> None:
        isolated = ProbeConfig(
            run_dir=Path("run"),
            base_model="base",
            dataset_dir=Path("data"),
            output_dir=Path("out"),
            prompt_strategy="isolated",
        )
        rolling = ProbeConfig(
            run_dir=Path("run"),
            base_model="base",
            dataset_dir=Path("data"),
            output_dir=Path("out"),
            prompt_strategy="rolling-initial-prompt",
        )

        self.assertNotEqual(generation_fingerprint(isolated), generation_fingerprint(rolling))

    def test_partial_generation_candidate_is_ineligible(self) -> None:
        candidate = summary("partial", 0.01, 1)
        candidate["attempted_count"] = 2
        candidate["required_count"] = 2
        candidate["scored_count"] = 1
        candidate["failed_count"] = 1
        with self.assertRaisesRegex(ValueError, "complete generation"):
            select_best_candidate([candidate], "micro_cer")

    def test_selector_indices_are_exact_and_returned_in_source_order(self) -> None:
        rows = [
            {"file_name": "train/a.wav"},
            {"file_name": "train/b.wav"},
            {"file_name": "train/c.wav"},
        ]
        self.assertEqual(select_selector_indices(rows, ["train/c.wav", "train/a.wav"]), [0, 2])
        with self.assertRaisesRegex(ValueError, "missing from probe split"):
            select_selector_indices(rows, ["train/missing.wav"])

    def test_rolling_prompt_uses_bounded_previous_hypotheses(self) -> None:
        config = ProbeConfig(
            run_dir=Path("run"),
            base_model="base",
            dataset_dir=Path("data"),
            output_dir=Path("out"),
            prompt_strategy="rolling-initial-prompt",
            prompt_include_row_context=False,
            rolling_prompt_turns=2,
            rolling_prompt_chars=12,
        )
        prompt = generation_prompt_text(
            {"context_before": "Terms: ignored"},
            ["first hypothesis", "second hypothesis", "third hypothesis"],
            config,
        )
        self.assertTrue(prompt.startswith("Previous transcript: "))
        self.assertTrue(prompt.endswith("d hypothesis"))
        self.assertNotIn("first", prompt)


if __name__ == "__main__":
    unittest.main()
