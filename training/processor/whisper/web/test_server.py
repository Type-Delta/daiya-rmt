from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from server import (
    AppState,
    RequestError,
    apply_import,
    as_path,
    build_auto_label_job,
    build_validation_job,
    create_session,
    is_loopback_host,
    normalise_rows,
    legacy_row_chunk_key,
    row_chunk_key,
    save_review,
    session_reviews,
    import_preview,
    web_configuration,
)


def loaded_row(row_id: str = "row-1") -> dict[str, object]:
    return {
        "id": row_id,
        "sourceUri": "train/clip.wav",
        "audioPath": "C:/audio/clip.wav",
        "audioSha256": "0" * 64,
        "disposition": "keep",
        "originalLabel": "original",
        "proposedLabel": None,
        "reasons": [],
    }


def active_session(directory: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "directory": str(directory),
        "reviewer": "tester",
        "rows": {str(row["id"]): row for row in rows},
        "rows_by_key": {row_chunk_key(row): str(row["id"]) for row in rows if row_chunk_key(row)},
        "legacy_rows_by_key": {legacy_row_chunk_key(row): str(row["id"]) for row in rows if legacy_row_chunk_key(row)},
        "rows_by_uri": {str(row["sourceUri"]): str(row["id"]) for row in rows},
        "lock": threading.Lock(),
    }


class ServerTests(unittest.TestCase):
    def test_normalise_manifest_keeps_source_label_and_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train").mkdir()
            (root / "train" / "clip.wav").write_bytes(b"RIFF")
            metadata = root / "metadata.jsonl"
            metadata.write_text(json.dumps({"file_name": "train/clip.wav", "text": "ต้นฉบับ", "speech_duration": 2.5}, ensure_ascii=False) + "\n", encoding="utf-8")
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({"source": {"source_id": "stable-1", "uri": "train/clip.wav"}, "original_label": "ต้นฉบับ", "disposition": "correct", "reasons": ["spelling_suspect"], "proposed_label": {"text": "แก้ไขแล้ว"}}, ensure_ascii=False) + "\n", encoding="utf-8")
            row = normalise_rows(metadata, manifest, root)[0]
            self.assertEqual(row["id"], "stable-1")
            self.assertEqual(row["originalLabel"], "ต้นฉบับ")
            self.assertEqual(row["proposedLabel"], "แก้ไขแล้ว")
            self.assertTrue(row["audioAvailable"])

    def test_normalise_rows_rejects_duplicate_canonical_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = root / "metadata.jsonl"
            metadata.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"file_name": "a.wav", "id": "a", "text": "one"},
                        {"file_name": "b.wav", "id": "b", "text": "two"},
                    )
                ) + "\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"source": {"source_id": "duplicate", "uri": "a.wav"}},
                        {"source": {"source_id": "duplicate", "uri": "b.wav"}},
                    )
                ) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RequestError, "Duplicate canonical row identity"):
                normalise_rows(metadata, manifest, root)

    def test_normalise_rows_rejects_duplicate_metadata_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = root / "metadata.jsonl"
            metadata.write_text(
                "\n".join(json.dumps({"file_name": "same.wav", "text": text}) for text in ("one", "two")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RequestError, "Duplicate canonical metadata identity"):
                normalise_rows(metadata, None, root)

    def test_auto_label_rejects_all_path_overlaps_and_nonempty_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            with self.assertRaisesRegex(RequestError, "must not overlap"):
                build_auto_label_job({"inputDir": str(source), "outputDir": str(source / "output"), "workDir": str(root / "work")})
            with self.assertRaisesRegex(RequestError, "must not overlap"):
                build_auto_label_job({"inputDir": str(source), "outputDir": str(root), "workDir": str(root / "work")})
            with self.assertRaisesRegex(RequestError, "must not overlap"):
                build_auto_label_job({"inputDir": str(source), "outputDir": str(root / "output"), "workDir": str(root / "output")})
            nonempty = root / "nonempty"
            nonempty.mkdir()
            (nonempty / "already-there").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(RequestError, "must not exist yet"):
                build_auto_label_job({"inputDir": str(source), "outputDir": str(nonempty), "workDir": str(root / "work")})
            empty_output = root / "empty-output"
            empty_output.mkdir()
            with self.assertRaisesRegex(RequestError, "must not exist yet"):
                build_auto_label_job({"inputDir": str(source), "outputDir": str(empty_output), "workDir": str(root / "work")})

    def test_job_commands_use_the_unified_processor_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            _, commands, _ = build_auto_label_job(
                {"inputDir": str(source), "outputDir": str(root / "output"), "workDir": str(root / "work")}
            )
            self.assertEqual(commands[0][0:4], ["uv", "run", "--directory", str(Path(__file__).resolve().parent.parent)])
            self.assertNotIn("--with-editable", commands[0])

    def test_web_configuration_prefills_the_workflow(self) -> None:
        configuration = web_configuration()
        self.assertTrue(configuration["autoLabel"]["inputDir"])
        self.assertTrue(configuration["autoLabel"]["outputDir"])
        self.assertFalse(Path(configuration["autoLabel"]["inputDir"]).is_absolute())
        self.assertFalse(Path(configuration["autoLabel"]["outputDir"]).is_absolute())
        self.assertEqual(configuration["validation"]["metadataPath"], f"{configuration['autoLabel']['outputDir']}/metadata.jsonl")
        self.assertEqual(configuration["review"]["audioRoot"], configuration["autoLabel"]["outputDir"])

    def test_relative_ui_paths_are_resolved_from_the_project_root(self) -> None:
        self.assertEqual(as_path("AGENTS.md", label="fixture", directory=False), Path(__file__).resolve().parents[4] / "AGENTS.md")

    def test_validation_and_review_targets_must_be_separate_and_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "audio"
            audio.mkdir()
            metadata = audio / "metadata.jsonl"
            metadata.write_text(json.dumps({"file_name": "clip.wav", "text": "label"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RequestError, "must not overlap"):
                build_validation_job({"metadataPath": str(metadata), "audioRoot": str(audio), "outputRoot": str(root)})
            validation = root / "validation"
            validation.mkdir()
            (validation / "old-run").mkdir()
            with self.assertRaisesRegex(RequestError, "absent or an empty"):
                build_validation_job({"metadataPath": str(metadata), "audioRoot": str(audio), "outputRoot": str(validation)})
            with self.assertRaisesRegex(RequestError, "must not overlap"):
                create_session({"reviewRoot": str(root)}, metadata, None, audio, [loaded_row()])
            reviews = root / "reviews"
            reviews.mkdir()
            (reviews / "old-review").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(RequestError, "does not contain a resumable"):
                create_session({"reviewRoot": str(reviews)}, metadata, None, audio, [loaded_row()])

    def test_existing_review_session_restores_saved_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "audio"
            audio.mkdir()
            metadata = audio / "metadata.jsonl"
            metadata.write_text(json.dumps({"file_name": "clip.wav", "text": "label"}) + "\n", encoding="utf-8")
            review_directory = root / "review"
            session = create_session({"reviewRoot": str(review_directory), "reviewer": "first"}, metadata, None, audio, [loaded_row()])
            save_review(session, {"rowId": "row-1", "text": "saved revision", "action": "edited"})
            resumed = create_session({"reviewRoot": str(review_directory), "reviewer": "second"}, metadata, None, audio, [loaded_row()])
            self.assertTrue(resumed["resumed"])
            self.assertEqual(resumed["reviewer"], "first")
            self.assertEqual(session_reviews(resumed)["row-1"]["human"]["label"], "saved revision")

    def test_running_job_reports_progress_and_can_be_cancelled(self) -> None:
        state = AppState()
        job = state.add_job(
            "fixture",
            [[sys.executable, "-c", "import time; print('25%', flush=True); time.sleep(5)"]],
            {},
        )
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            active = state.job_list()[0]
            if active["status"] == "running" and active["progress"]["fraction"] > 0:
                break
            time.sleep(0.03)
        self.assertEqual(active["status"], "running")
        self.assertGreater(active["progress"]["fraction"], 0)
        cancelled = state.cancel_job(job["id"])
        self.assertTrue(cancelled["cancelRequested"])
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and state.job_list()[0]["status"] != "cancelled":
            time.sleep(0.03)
        self.assertEqual(state.job_list()[0]["status"], "cancelled")

    def test_save_review_derives_provenance_from_the_canonical_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            session = active_session(directory, [loaded_row()])
            event = save_review(
                session,
                {
                    "rowId": "row-1",
                    "text": "corrected",
                    "action": "edited",
                    "row": {"id": "row-1", "sourceUri": "spoofed.wav", "originalLabel": "spoofed"},
                },
            )
            self.assertEqual(event["chunk"]["audio_sha256"], "0" * 64)
            self.assertEqual(event["human"]["label"], "corrected")
            with self.assertRaisesRegex(RequestError, "not part of this active session"):
                save_review(session, {"rowId": "unknown", "text": "x", "action": "edited"})

    def test_concurrent_saves_preserve_every_review_in_the_jsonl_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            rows = [loaded_row(f"row-{index}") for index in range(24)]
            for index, row in enumerate(rows):
                row["audioSha256"] = f"{index:064x}"
            session = active_session(directory, rows)
            with ThreadPoolExecutor(max_workers=8) as executor:
                events = list(
                    executor.map(
                        lambda row: save_review(session, {"rowId": row["id"], "text": f"edit-{row['id']}", "action": "edited"}),
                        rows,
                    )
            )
            saved_reviews = [json.loads(line) for line in (directory / "reviews.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(saved_reviews), len(rows))
            self.assertTrue(all(review["schema_version"] == "daiya-human-review-3" for review in saved_reviews))
            self.assertFalse(any(path.suffix == ".tmp" for path in directory.iterdir()))

    def test_identical_audio_at_different_source_uris_keeps_both_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            first, second = loaded_row("row-1"), loaded_row("row-2")
            second["sourceUri"] = "train/duplicate-clip.wav"
            session = active_session(directory, [first, second])

            save_review(session, {"rowId": "row-1", "text": "first", "action": "edited"})
            save_review(session, {"rowId": "row-2", "text": "second", "action": "edited"})

            saved_reviews = [json.loads(line) for line in (directory / "reviews.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(saved_reviews), 2)
            self.assertTrue(all(review["schema_version"] == "daiya-human-review-3" for review in saved_reviews))

    def test_import_normalizes_v1_and_v2_reviews_to_v3(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            row = loaded_row()
            session = active_session(directory, [row])
            legacy_v1 = {
                "schema_version": "daiya-human-review-1",
                "saved_at": "2026-07-13T13:48:59+07:00",
                "reviewer": "tester",
                "source": {"row_id": "row-1", "source_uri": row["sourceUri"], "audio_path": row["audioPath"]},
                "human": {"action": "edited", "label": "from v1"},
            }
            _, preview = import_preview(session, json.dumps(legacy_v1))
            self.assertEqual(preview["new"], 1)
            reviews, _ = apply_import(session, json.dumps(legacy_v1), "ours")
            self.assertEqual(reviews["row-1"]["schema_version"], "daiya-human-review-3")

            v2 = dict(reviews["row-1"])
            v2["schema_version"] = "daiya-human-review-2"
            v2["chunk"] = dict(v2["chunk"])
            v2["chunk"].pop("source_uri")
            second_directory = Path(temp) / "second"
            second_directory.mkdir()
            second_session = active_session(second_directory, [row])
            _, preview = import_preview(second_session, json.dumps(v2))
            self.assertEqual(preview["new"], 1)
            imported, _ = apply_import(second_session, json.dumps(v2), "ours")
            self.assertEqual(imported["row-1"]["schema_version"], "daiya-human-review-3")

    def test_save_uses_reviews_jsonl_as_the_current_review_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            first, second = loaded_row("row-1"), loaded_row("row-2")
            first["audioSha256"] = "1" * 64
            second["audioSha256"] = "2" * 64
            session = active_session(directory, [first, second])
            saved = save_review(session, {"rowId": "row-1", "text": "initial", "action": "edited"})
            saved["human"]["label"] = "external edit"
            (directory / "reviews.jsonl").write_text(json.dumps(saved) + "\n", encoding="utf-8")

            save_review(session, {"rowId": "row-2", "text": "second", "action": "edited"})

            saved_reviews = [json.loads(line) for line in (directory / "reviews.jsonl").read_text(encoding="utf-8").splitlines()]
            labels = {review["human"]["label"] for review in saved_reviews}
            self.assertEqual(labels, {"external edit", "second"})

    def test_loopback_hosts_require_no_unsafe_opt_in(self) -> None:
        self.assertTrue(is_loopback_host("127.0.0.1"))
        self.assertTrue(is_loopback_host("::1"))
        self.assertTrue(is_loopback_host("localhost"))
        self.assertFalse(is_loopback_host("0.0.0.0"))
        self.assertFalse(is_loopback_host("192.168.1.20"))


if __name__ == "__main__":
    unittest.main()
