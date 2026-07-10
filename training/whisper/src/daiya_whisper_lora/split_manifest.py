from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from datasets import DatasetDict


SPLIT_ALIASES = {
    "dev": "validation",
    "eval": "validation",
    "valid": "validation",
    "val": "validation",
}
ALLOWED_SPLITS = {"train", "validation", "test", "benchmark"}
SAMPLE_ID_FIELDS = ("sample_id", "id", "uid")
GROUP_FIELDS = ("source_file", "conversation", "conversation_id", "group_id", "session_id")


@dataclass(frozen=True)
class SplitManifestIdentity:
    path: str
    sha256: str
    entry_count: int
    assigned_sample_count: int
    group_count: int
    splits: dict[str, int]
    split_sample_sha256: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_split_manifest(dataset: DatasetDict, manifest_path: Path) -> tuple[DatasetDict, SplitManifestIdentity]:
    from datasets import Dataset, DatasetDict

    entries = load_manifest_entries(manifest_path)
    if not entries:
        raise ValueError(f"Split manifest has no entries: {manifest_path}")

    rows = flatten_dataset(dataset)
    split_rows = partition_rows_by_manifest(rows, entries)
    assigned_group_ids = {
        group_id_for_row(row)
        for rows_for_split in split_rows.values()
        for row in rows_for_split
    }

    used_split_rows = {split: Dataset.from_list(rows) for split, rows in split_rows.items() if rows}
    identity = SplitManifestIdentity(
        path=str(manifest_path.resolve()),
        sha256=file_sha256(manifest_path),
        entry_count=len(entries),
        assigned_sample_count=sum(len(rows) for rows in split_rows.values()),
        group_count=len(assigned_group_ids),
        splits={split: len(rows) for split, rows in sorted(split_rows.items()) if rows},
        split_sample_sha256={
            split: sample_ids_sha256(rows)
            for split, rows in sorted(split_rows.items())
            if rows
        },
    )
    return DatasetDict(used_split_rows), identity


def partition_rows_by_manifest(
    rows: list[dict[str, Any]],
    entries: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    sample_to_split, group_to_split = compile_manifest(entries)
    validate_dataset_sample_ids(rows)

    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in ALLOWED_SPLITS}
    assigned_groups: dict[str, str] = {}
    missing: list[str] = []
    for row in rows:
        sample_id = sample_id_for_row(row)
        group_id = group_id_for_row(row)
        split = sample_to_split.get(sample_id) or group_to_split.get(group_id)
        if split is None:
            missing.append(sample_id)
            continue
        existing_group_split = assigned_groups.setdefault(group_id, split)
        if existing_group_split != split:
            raise ValueError(
                f"Split manifest leaks group {group_id!r} across {existing_group_split!r} and {split!r}."
            )
        split_rows[split].append(row)

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            f"Split manifest does not assign {len(missing)} dataset samples. "
            f"First missing sample IDs: {preview}{suffix}"
        )
    return split_rows


def load_manifest_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return entries_from_json_payload(payload)
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in split manifest {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Split manifest row must be an object at {path}:{line_number}")
            entries.append(payload)
    return entries


def entries_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if not all(isinstance(row, dict) for row in payload):
            raise ValueError("JSON split manifest lists must contain only objects.")
        return list(payload)
    if not isinstance(payload, dict):
        raise ValueError("JSON split manifest must be an object or list of objects.")
    if isinstance(payload.get("samples"), list):
        rows = payload["samples"]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("JSON split manifest samples must contain only objects.")
        return list(rows)
    if isinstance(payload.get("splits"), dict):
        entries: list[dict[str, Any]] = []
        for split, values in payload["splits"].items():
            if not isinstance(values, list):
                raise ValueError("JSON split manifest split values must be lists.")
            for value in values:
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault("split", split)
                else:
                    row = {"split": split, "sample_id": value}
                entries.append(row)
        return entries
    raise ValueError("JSON split manifest must contain samples or splits.")


def compile_manifest(entries: Iterable[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    sample_to_split: dict[str, str] = {}
    group_to_split: dict[str, str] = {}
    for index, entry in enumerate(entries, start=1):
        split = normalize_split(entry.get("split") or entry.get("partition") or entry.get("dataset_split"))
        sample_id = first_non_empty(entry, *SAMPLE_ID_FIELDS)
        group_id = manifest_group_id(entry)
        if sample_id is None and group_id is None:
            raise ValueError(f"Split manifest entry {index} must contain a sample ID or group identity.")
        if sample_id is not None:
            previous = sample_to_split.setdefault(sample_id, split)
            if previous != split:
                raise ValueError(f"Duplicate sample ID {sample_id!r} maps to both {previous!r} and {split!r}.")
        if group_id is not None:
            previous = group_to_split.setdefault(group_id, split)
            if previous != split:
                raise ValueError(f"Group {group_id!r} maps to both {previous!r} and {split!r}.")
    return sample_to_split, group_to_split


def normalize_split(value: Any) -> str:
    if value is None:
        raise ValueError("Split manifest entry is missing split.")
    split = SPLIT_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
    if split not in ALLOWED_SPLITS:
        valid = ", ".join(sorted(ALLOWED_SPLITS))
        raise ValueError(f"Unsupported split {value!r}; expected one of: {valid}.")
    return split


def flatten_dataset(dataset: DatasetDict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, split in dataset.items():
        for index in range(len(split)):
            row = dict(split[index])
            row.setdefault("original_split", split_name)
            rows.append(row)
    return rows


def validate_dataset_sample_ids(rows: list[dict[str, Any]]) -> None:
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for index, row in enumerate(rows):
        sample_id = sample_id_for_row(row)
        previous = seen.setdefault(sample_id, index)
        if previous != index:
            duplicates.append(sample_id)
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"Dataset has duplicate sample IDs; first duplicates: {preview}")


def sample_id_for_row(row: dict[str, Any]) -> str:
    sample_id = first_non_empty(row, *SAMPLE_ID_FIELDS)
    if sample_id is not None:
        return sample_id
    file_name = first_non_empty(row, "file_name", "path")
    if file_name is not None:
        return file_name
    audio = row.get("audio")
    if isinstance(audio, dict):
        audio_path = audio.get("path")
        if audio_path:
            return str(audio_path)
    if audio:
        return str(audio)
    raise ValueError("Dataset row is missing sample_id/id/uid/file_name/audio path.")


def group_id_for_row(row: dict[str, Any]) -> str:
    group_id = first_non_empty(row, "conversation", "conversation_id", "group_id", "session_id")
    if group_id is not None:
        return group_id
    source_file = first_non_empty(row, "source_file")
    if source_file is not None:
        return portable_source_id(source_file)
    file_name = first_non_empty(row, "file_name", "path")
    if file_name is None:
        return sample_id_for_row(row)
    path = Path(file_name)
    if len(path.parts) > 1:
        return str(path.parent)
    return path.stem or str(file_name)


def manifest_group_id(entry: dict[str, Any]) -> str | None:
    explicit = first_non_empty(entry, "conversation", "conversation_id", "group_id", "session_id")
    if explicit is not None:
        return explicit
    source_file = first_non_empty(entry, "source_file")
    return portable_source_id(source_file) if source_file is not None else None


def portable_source_id(value: str) -> str:
    """Normalize local absolute source paths to a stable cross-machine basename."""
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def sample_ids_sha256(rows: Iterable[dict[str, Any]]) -> str:
    payload = "\n".join(sorted(sample_id_for_row(row) for row in rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def first_non_empty(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
