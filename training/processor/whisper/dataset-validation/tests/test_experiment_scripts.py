from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_candidate_manifest import _spelling_signals, build_manifest  # noqa: E402
import build_candidate_manifest  # noqa: E402
from daiya_dataset_validation.models import ReasonCode  # noqa: E402
from daiya_dataset_validation.path_safety import path_is_within, paths_alias  # noqa: E402
from evaluate_gold import _metrics  # noqa: E402
import run_spelling_validation  # noqa: E402


_fixture_worker_config: dict[str, object] = {}


def _fixture_worker_init(config: tuple[str, ...]) -> None:
    global _fixture_worker_config
    _fixture_worker_config = {"config": config}


def _fixture_worker(payload: tuple[int, dict[str, object]]) -> dict[str, object]:
    row_number, row = payload
    time.sleep((5 - row_number) * 0.01)
    return {"row_number": row_number, "file_name": row["file_name"], "config": _fixture_worker_config["config"]}


class _FakeChecker:
    language = "english"

    def __init__(self, name: str) -> None:
        self.name = name


class ExperimentScriptTests(unittest.TestCase):
    def test_resolved_path_equality_handles_symlink_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "metadata.jsonl"
            target.write_text("source\n", encoding="utf-8")
            alias = root / "alias.jsonl"
            try:
                alias.symlink_to(target)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            self.assertTrue(paths_alias(target, target))
            self.assertTrue(paths_alias(alias, target))
            self.assertTrue(path_is_within(alias, root))

    def test_spelling_rejects_output_aliases_before_mutating_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.jsonl"
            metadata.write_text(json.dumps({"file_name": "clip.wav", "text": "label"}) + "\n", encoding="utf-8")
            allowlist = root / "allowlist.txt"
            allowlist.write_text("label\n", encoding="utf-8")
            dictionary = root / "dictionary.txt"
            dictionary.write_text("label 10\n", encoding="utf-8")
            before = {
                path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                for path in (metadata, allowlist, dictionary)
            }
            for resource, option in ((metadata, None), (allowlist, "--allowlist"), (dictionary, "--english-dictionary")):
                alias = root / f"{resource.stem}-alias.jsonl"
                try:
                    alias.symlink_to(resource)
                except (OSError, NotImplementedError) as exc:
                    self.skipTest(f"symlink creation is unavailable: {exc}")
                arguments = [str(metadata), str(alias), "--thai-engine", "pn"]
                if option:
                    arguments.extend((option, str(resource)))
                with self.assertRaisesRegex(ValueError, rf"aliases (metadata|allowlist|English dictionary)"):
                    run_spelling_validation.main(arguments)
            after = {
                path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                for path in (metadata, allowlist, dictionary)
            }
            self.assertEqual(before, after)

    def test_manifest_rejects_symlink_aliases_and_preserves_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio"
            audio.mkdir()
            source = audio / "clip.wav"
            source.write_bytes(b"source")
            metadata = root / "metadata.jsonl"
            metadata.write_text(json.dumps({"file_name": "clip.wav", "text": "label", "duration_seconds": 1.0}) + "\n", encoding="utf-8")
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({"file_name": "clip.wav", "prediction": "label", "model_name": "one"}) + "\n", encoding="utf-8")
            spelling = root / "spelling.jsonl"
            spelling.write_text("{}\n", encoding="utf-8")
            protected = root / "gold"
            protected.mkdir()
            gold_source = protected / "gold.wav"
            gold_source.write_bytes(b"gold")
            resources = (
                (metadata, {"predictions_path": None, "spelling_results_path": None, "protected_gold_dir": None}),
                (predictions, {"predictions_path": predictions, "spelling_results_path": None, "protected_gold_dir": None}),
                (spelling, {"predictions_path": None, "spelling_results_path": spelling, "protected_gold_dir": None}),
                (gold_source, {"predictions_path": None, "spelling_results_path": None, "protected_gold_dir": protected}),
                (source, {"predictions_path": None, "spelling_results_path": None, "protected_gold_dir": None}),
                (audio, {"predictions_path": None, "spelling_results_path": None, "protected_gold_dir": None}),
            )
            fingerprints = {
                path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                for path in (metadata, predictions, spelling, gold_source, source)
            }
            for resource, options in resources:
                alias = root / f"alias-{resource.name}.jsonl"
                try:
                    alias.symlink_to(resource, target_is_directory=resource.is_dir())
                except (OSError, NotImplementedError) as exc:
                    self.skipTest(f"symlink creation is unavailable: {exc}")
                output = alias
                kwargs = dict(options)
                metadata_arg = metadata
                with self.assertRaisesRegex(ValueError, "output path aliases"):
                    build_manifest(metadata_arg, kwargs.pop("audio_root", audio), output, dataset_version="fixture", **kwargs)
                self.assertTrue(paths_alias(output, resource))
            after = {
                path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                for path in fingerprints
            }
            self.assertEqual(fingerprints, after)

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

    def test_spelling_parallel_results_match_ordered_single_worker_results(self) -> None:
        rows = [{"file_name": f"clip-{index}.wav", "text": f"label-{index}"} for index in range(1, 5)]
        config = ("fixture-checker",)
        sequential = [
            {"row_number": row_number, "file_name": row["file_name"], "config": config}
            for row_number, row in enumerate(rows, 1)
        ]
        parallel = list(
            run_spelling_validation._ordered_parallel_results(
                rows,
                config=(),
                allowlist=frozenset(),
                review_threshold=0.2,
                min_issues=1,
                include_issue_text=False,
                workers=2,
                max_in_flight=2,
                initializer=_fixture_worker_init,
                worker=_fixture_worker,
                initargs=(config,),
            )
        )
        self.assertEqual(parallel, sequential)

    def test_spelling_worker_initialization_rebuilds_checker_configuration(self) -> None:
        created: list[tuple[str, object]] = []

        def thai(engine: str) -> _FakeChecker:
            checker = _FakeChecker(f"thai-{engine}")
            created.append(("thai", engine))
            return checker

        with patch.object(run_spelling_validation, "PyThaiNLPChecker", side_effect=thai), patch.object(
            run_spelling_validation,
            "SudachiJapaneseChecker",
            side_effect=lambda dictionary: _FakeChecker(f"japanese-{dictionary}"),
        ), patch.object(
            run_spelling_validation,
            "SymSpellEnglishChecker",
            side_effect=lambda dictionary: _FakeChecker(f"english-{dictionary}"),
        ):
            run_spelling_validation._initialize_worker(
                (("pn",), "core", "terms.txt"),
                frozenset({"Daiya"}),
                0.4,
                2,
                False,
            )
            first = run_spelling_validation._worker_checkers
            run_spelling_validation._initialize_worker(
                (("phunspell",), "small", "other-terms.txt"),
                frozenset({"other"}),
                0.1,
                1,
                True,
            )
            second = run_spelling_validation._worker_checkers

        self.assertIsNot(first, second)
        self.assertEqual(created, [("thai", "pn"), ("thai", "phunspell")])
        self.assertEqual(run_spelling_validation._worker_allowlist, frozenset({"other"}))
        self.assertEqual(run_spelling_validation._worker_review_threshold, 0.1)
        self.assertEqual(run_spelling_validation._worker_min_issues, 1)
        self.assertTrue(run_spelling_validation._worker_include_issue_text)

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

    def test_manifest_hashing_preserves_metadata_order_across_worker_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio"
            audio.mkdir()
            paths = [audio / f"{name}.wav" for name in ("slow", "fast", "middle")]
            for path, data in zip(paths, (b"slow", b"fast", b"middle"), strict=True):
                path.write_bytes(data)
            metadata = root / "metadata.jsonl"
            metadata.write_text(
                "".join(json.dumps({"file_name": f"{path.stem}.wav", "text": path.stem, "duration_seconds": 1.0}) + "\n" for path in paths),
                encoding="utf-8",
            )
            output_one = root / "manifest-one.jsonl"
            output_many = root / "manifest-many.jsonl"
            build_manifest(metadata, audio, output_one, dataset_version="fixture", hash_workers=1)
            build_manifest(metadata, audio, output_many, dataset_version="fixture", hash_workers=3)
            rows_one = [json.loads(line) for line in output_one.read_text(encoding="utf-8").splitlines()]
            rows_many = [json.loads(line) for line in output_many.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows_one, rows_many)
            self.assertEqual([row["source"]["uri"] for row in rows_many], [path.name for path in paths])

    def test_hash_completion_order_does_not_change_source_hash_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("slow.wav", "fast.wav", "middle.wav")]
            for path in paths:
                path.write_bytes(path.name.encode("utf-8"))

            original_hash = build_candidate_manifest._sha256

            def delayed_hash(path: Path) -> str:
                if path.name == "slow.wav":
                    time.sleep(0.03)
                return original_hash(path)

            with patch.object(build_candidate_manifest, "_sha256", side_effect=delayed_hash):
                actual = build_candidate_manifest._parallel_hashes(paths, hash_workers=3)
            expected = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            self.assertEqual(actual, expected)

    def test_manifest_rejects_metadata_path_traversal_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio"
            audio.mkdir()
            outside = root / "outside.wav"
            outside.write_bytes(b"outside")
            metadata = root / "metadata.jsonl"
            metadata.write_text(
                json.dumps({"file_name": "../outside.wav", "text": "label", "duration_seconds": 1.0}) + "\n",
                encoding="utf-8",
            )
            output = root / "manifest.jsonl"
            output.write_text("sentinel\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes audio_root"):
                build_manifest(metadata, audio, output, dataset_version="fixture")
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")
            self.assertEqual(outside.read_bytes(), b"outside")


if __name__ == "__main__":
    unittest.main()
