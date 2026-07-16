import importlib.util
import json
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "src/daiya_training_harness/benchmark.py"
SPEC = importlib.util.spec_from_file_location("benchmark", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def fixtures(self):
        contract = {"name": "asr-v1", "metrics": ["wer", "cer"]}
        contract_hash = benchmark.canonical_hash(contract)
        manifest = {"contract_hash": contract_hash, "segments": ["a", "b"]}
        provenance = {
            "contract_hash": contract_hash,
            "recipe": "whisper-lora-v2",
            "base_revision": "model@abc",
            "data_hash": "sha256:data",
        }
        return contract, manifest, provenance

    def test_deterministic_summary_has_identity_and_no_raw_text(self):
        contract, manifest, provenance = self.fixtures()
        outputs = [
            {"segment_id": "b", "reference": "hello world", "prediction": "hello"},
            {"segment_id": "a", "reference": "สวัสดี", "prediction": "สวัสดี"},
        ]
        result = benchmark.run_benchmark(
            contract=contract, manifest=manifest, provenance=provenance,
            outputs=outputs, backend_identity={"name": "fixture", "revision": "1"},
        )
        reversed_result = benchmark.run_benchmark(
            contract=contract, manifest=manifest, provenance=provenance,
            outputs=reversed(outputs), backend_identity={"name": "fixture", "revision": "1"},
        )
        self.assertEqual(result, reversed_result)
        self.assertEqual(result["recipe"], "whisper-lora-v2")
        self.assertEqual(result["base_revision"], "model@abc")
        self.assertEqual(result["data_hash"], "sha256:data")
        self.assertEqual(result["metrics"]["wer"], 1 / 3)
        serialized = benchmark.compact_json(result)
        self.assertNotIn("hello world", serialized)
        self.assertNotIn("สวัสดี", serialized)
        self.assertNotIn(" ", serialized)
        self.assertEqual(json.loads(serialized), result)

    def test_accepts_json_record_paths(self):
        contract, manifest, provenance = self.fixtures()
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, record in enumerate((contract, manifest, provenance)):
                path = Path(directory) / f"{index}.json"
                path.write_text(json.dumps(record), encoding="utf-8")
                paths.append(path)
            result = benchmark.run_benchmark(
                contract=paths[0], manifest=paths[1], provenance=paths[2],
                outputs=[], backend_identity="fixture",
            )
        self.assertEqual(result["metrics"]["segments"], 0)

    def test_rejects_incompatible_linked_records_and_summaries(self):
        contract, manifest, provenance = self.fixtures()
        manifest["contract_hash"] = "different"
        with self.assertRaises(benchmark.IncompatibleContractError):
            benchmark.run_benchmark(
                contract=contract, manifest=manifest, provenance=provenance,
                outputs=[], backend_identity="fixture",
            )
        with self.assertRaises(benchmark.IncompatibleContractError):
            benchmark.assert_comparable({"contract_hash": "a"}, {"contract_hash": "b"})

    def test_duplicate_segment_ids_are_rejected(self):
        contract, manifest, provenance = self.fixtures()
        duplicate = {"segment_id": "a", "reference": "x", "prediction": "x"}
        with self.assertRaisesRegex(ValueError, "duplicate segment_id"):
            benchmark.run_benchmark(
                contract=contract, manifest=manifest, provenance=provenance,
                outputs=[duplicate, duplicate], backend_identity="fixture",
            )

    def test_consumes_shared_record_objects(self):
        class Record:
            def __init__(self, value):
                self.value = value
            def to_dict(self):
                return self.value

        contract = Record({"name": "recipe"})
        manifest_value = {"dataset_version": "data-v1", "splits": {"test": ["c1"]}}
        manifest_hash = benchmark.canonical_hash(manifest_value)
        provenance = Record({
            "dataset_version": "data-v1", "conversion_settings": {"rate": 16000},
            "base_model_revision": "revision-1", "split_manifest_sha256": manifest_hash,
        })
        result = benchmark.run_benchmark(
            contract=contract, manifest=Record(manifest_value), provenance=provenance,
            outputs=[], backend_identity="fixture",
        )
        self.assertEqual(result["base_revision"], "revision-1")
        self.assertEqual(result["manifest_hash"], manifest_hash)
        self.assertEqual(len(result["recipe"]), 64)
        self.assertEqual(len(result["data_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
