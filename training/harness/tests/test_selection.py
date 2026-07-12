import unittest

from daiya_training_harness.selection import (
    RankedCheckpoint,
    TopKValidationProtocol,
    ValidationRecord,
    select_checkpoint,
)


class CheckpointSelectionTests(unittest.TestCase):
    def setUp(self):
        self.ranking = [
            RankedCheckpoint("checkpoint-588", 0.10, "eval_loss"),
            RankedCheckpoint("checkpoint-400", 0.20, "eval_loss"),
            RankedCheckpoint("checkpoint-300", 0.30, "eval_loss"),
        ]
        self.protocol = TopKValidationProtocol("ct2_quantized", 2, "wer", "latency_ok")

    def record(self, checkpoint, wer, gate=True, backend="ct2_quantized"):
        return ValidationRecord(checkpoint, backend, {"wer": wer}, {"latency_ok": gate})

    def test_transformers_ranking_is_not_ct2_evidence(self):
        with self.assertRaisesRegex(ValueError, "requires CT2 deployment evidence"):
            select_checkpoint(self.ranking, self.protocol)

    def test_checkpoint_588_requires_its_own_ct2_record(self):
        with self.assertRaisesRegex(ValueError, "checkpoint-588"):
            select_checkpoint(
                self.ranking,
                self.protocol,
                validation_records=[self.record("checkpoint-400", 0.1)],
            )

    def test_callback_validates_top_k_and_ct2_metric_selects(self):
        called = []

        def validate(candidate, protocol):
            called.append(candidate.checkpoint)
            wer = {"checkpoint-588": 0.22, "checkpoint-400": 0.18}[candidate.checkpoint]
            return self.record(candidate.checkpoint, wer)

        result = select_checkpoint(self.ranking, self.protocol, validate=validate)
        self.assertEqual(called, ["checkpoint-588", "checkpoint-400"])
        self.assertEqual(result.selected_checkpoint, "checkpoint-400")
        self.assertEqual(result.metric_name, "wer")
        self.assertEqual(result.gate_name, "latency_ok")
        self.assertEqual(len(result.validation_records), 2)

    def test_gate_is_explicit_and_excludes_better_metric(self):
        result = select_checkpoint(
            self.ranking,
            self.protocol,
            validation_records=[
                self.record("checkpoint-588", 0.10, gate=False),
                self.record("checkpoint-400", 0.20),
            ],
        )
        self.assertEqual(result.selected_checkpoint, "checkpoint-400")

    def test_metric_tie_breaks_by_numeric_checkpoint(self):
        result = select_checkpoint(
            self.ranking,
            self.protocol,
            validation_records=[
                self.record("checkpoint-588", 0.20),
                self.record("checkpoint-400", 0.20),
            ],
        )
        self.assertEqual(result.selected_checkpoint, "checkpoint-400")

    def test_backend_metric_and_gate_must_match_protocol(self):
        bad_sets = [
            [self.record("checkpoint-588", 0.2, backend="transformers_peft"), self.record("checkpoint-400", 0.3)],
            [ValidationRecord("checkpoint-588", "ct2_quantized", {"cer": 0.2}, {"latency_ok": True}), self.record("checkpoint-400", 0.3)],
            [ValidationRecord("checkpoint-588", "ct2_quantized", {"wer": 0.2}, {"quality_ok": True}), self.record("checkpoint-400", 0.3)],
        ]
        for records in bad_sets:
            with self.subTest(records=records), self.assertRaises(ValueError):
                select_checkpoint(self.ranking, self.protocol, validation_records=records)


if __name__ == "__main__":
    unittest.main()
