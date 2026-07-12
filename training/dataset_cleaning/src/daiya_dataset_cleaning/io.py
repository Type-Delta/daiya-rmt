"""Atomic manifest writers. Source inputs are never opened for writing."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

from .models import ManifestRecord


def _csv_cell(value: object) -> object:
    """Keep spreadsheet programs from treating label text as a formula."""
    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@"}:
        return "'" + value
    return value


def _atomic_write(path: Path, writer: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer(handle)  # type: ignore[operator]
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_jsonl(records: Iterable[ManifestRecord], path: str | Path) -> None:
    def emit(handle: object) -> None:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")  # type: ignore[attr-defined]
    _atomic_write(Path(path), emit)


def write_csv(records: Iterable[ManifestRecord], path: str | Path) -> None:
    fields = ("schema_version", "source_id", "source_uri", "original_label", "normalized_label", "disposition", "reasons", "confidence", "proposed_label", "evidence_json")
    def emit(handle: object) -> None:
        writer = csv.DictWriter(handle, fieldnames=fields)  # type: ignore[arg-type]
        writer.writeheader()
        for record in records:
            data = record.to_dict()
            writer.writerow({
                "schema_version": record.schema_version, "source_id": record.source.source_id,
                "source_uri": _csv_cell(record.source.uri), "original_label": _csv_cell(record.original_label),
                "normalized_label": _csv_cell(record.normalized_label), "disposition": record.disposition.value,
                "reasons": json.dumps(data["reasons"], ensure_ascii=False),
                "confidence": "" if record.confidence is None else json.dumps(data["confidence"]),
                "proposed_label": "" if record.proposed_label is None else json.dumps(data["proposed_label"], ensure_ascii=False),
                "evidence_json": json.dumps(data["evidence"], ensure_ascii=False, allow_nan=False),
            })
    _atomic_write(Path(path), emit)
