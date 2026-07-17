from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from daiya_whisper_pipeline.review_mapping import build_mapping_report, write_mapping_report


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(file_name: str, start: float, end: float, audio_hash: str) -> dict[str, object]:
    return {
        "file_name": file_name,
        "source_file": r"C:\dataset\raw\Th-En_sample_11.m4a",
        "source_start": start,
        "source_end": end,
        "audio_sha256": audio_hash,
    }


def test_mapping_only_reuses_byte_identical_audio_and_reviews(tmp_path: Path) -> None:
    digest = sha256(b"same-audio").hexdigest()
    old = tmp_path / "old.jsonl"
    new = tmp_path / "new.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    _write_jsonl(old, [_row("train/old.wav", 1.0, 2.0, digest)])
    _write_jsonl(new, [_row("train/new.wav", 1.0, 2.0, digest)])
    _write_jsonl(
        reviews,
        [
            {
                "chunk": {"source_uri": "train/old.wav", "audio_sha256": digest},
                "human": {"action": "edited", "label": "human label"},
            }
        ],
    )

    report = build_mapping_report(old, new, old_reviews=reviews)

    assert report["clips"][0]["status"] == "unchanged"
    assert report["clips"][0]["new_file_name"] == "train/new.wav"
    assert report["reviews"][0]["status"] == "safely_reusable_review"
    assert "human label" not in json.dumps(report)


def test_mapping_requires_relabel_for_changed_boundary_and_marks_ambiguous_fanout(tmp_path: Path) -> None:
    digest = sha256(b"old-audio").hexdigest()
    old = tmp_path / "old.jsonl"
    changed = tmp_path / "changed.jsonl"
    fanout = tmp_path / "fanout.jsonl"
    _write_jsonl(old, [_row("train/old.wav", 1.0, 3.0, digest)])
    _write_jsonl(changed, [_row("train/new.wav", 1.0, 3.2, sha256(b"new").hexdigest())])
    _write_jsonl(
        fanout,
        [
            _row("train/new-a.wav", 1.0, 2.0, sha256(b"a").hexdigest()),
            _row("train/new-b.wav", 2.0, 3.0, sha256(b"b").hexdigest()),
        ],
    )

    assert build_mapping_report(old, changed)["clips"][0]["status"] == "changed_boundary_needs_relabel"
    assert build_mapping_report(old, fanout)["clips"][0]["status"] == "ambiguous"


def test_mapping_report_never_overwrites_an_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_mapping_report({"schema_version": "test"}, output)
    assert output.read_text(encoding="utf-8") == "keep"
