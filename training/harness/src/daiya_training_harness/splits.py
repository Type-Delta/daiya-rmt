"""Portable, frozen conversation-level dataset split manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class ManifestValidationError(ValueError):
    """Raised when a split manifest is malformed or contains overlap."""


def _json_value(value: Any, location: str = "$") -> Any:
    """Return a JSON-compatible copy, rejecting ambiguous/non-portable values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} has a non-string object key")
            result[key] = _json_value(item, f"{location}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{location}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{location} contains unsupported type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode *value* to deterministic UTF-8 JSON suitable for hashing."""
    portable = _json_value(value)
    return json.dumps(
        portable,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    """Return the SHA-256 hex digest of canonical JSON for *value*."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class SplitManifest:
    """Immutable assignment of conversation identifiers to named splits."""

    dataset_version: str
    splits: Mapping[str, Sequence[str]]
    manifest_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_splits = {
            name: tuple(conversation_ids) for name, conversation_ids in self.splits.items()
        }
        if not isinstance(self.metadata, Mapping):
            raise ManifestValidationError("metadata must be an object")
        normalized_metadata = _json_value(self.metadata, "$.metadata")
        object.__setattr__(self, "splits", MappingProxyType(normalized_splits))
        object.__setattr__(self, "metadata", MappingProxyType(normalized_metadata))
        validate_manifest(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "dataset_version": self.dataset_version,
            "splits": {name: list(ids) for name, ids in self.splits.items()},
            "metadata": dict(self.metadata),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SplitManifest":
        expected = {"manifest_version", "dataset_version", "splits", "metadata"}
        unknown = set(value) - expected
        if unknown:
            raise ManifestValidationError(f"unknown manifest fields: {sorted(unknown)}")
        try:
            return cls(
                manifest_version=value.get("manifest_version", 1),
                dataset_version=value["dataset_version"],
                splits=value["splits"],
                metadata=value.get("metadata", {}),
            )
        except KeyError as exc:
            raise ManifestValidationError(f"missing manifest field: {exc.args[0]}") from exc


def validate_manifest(manifest: SplitManifest) -> None:
    """Validate identifiers and guarantee each conversation occurs exactly once."""
    if not isinstance(manifest.manifest_version, int) or isinstance(manifest.manifest_version, bool):
        raise ManifestValidationError("manifest_version must be an integer")
    if manifest.manifest_version != 1:
        raise ManifestValidationError(f"unsupported manifest_version: {manifest.manifest_version}")
    if not isinstance(manifest.dataset_version, str) or not manifest.dataset_version.strip():
        raise ManifestValidationError("dataset_version must be a non-empty string")
    if not manifest.splits:
        raise ManifestValidationError("splits must not be empty")

    owners: dict[str, str] = {}
    for split_name, conversation_ids in manifest.splits.items():
        if not isinstance(split_name, str) or not split_name.strip():
            raise ManifestValidationError("split names must be non-empty strings")
        if isinstance(conversation_ids, (str, bytes)):
            raise ManifestValidationError(f"split {split_name!r} must contain a sequence of IDs")
        local: set[str] = set()
        for conversation_id in conversation_ids:
            if not isinstance(conversation_id, str) or not conversation_id.strip():
                raise ManifestValidationError("conversation IDs must be non-empty strings")
            if conversation_id in local:
                raise ManifestValidationError(
                    f"conversation {conversation_id!r} is duplicated in split {split_name!r}"
                )
            if conversation_id in owners:
                raise ManifestValidationError(
                    f"conversation {conversation_id!r} overlaps splits "
                    f"{owners[conversation_id]!r} and {split_name!r}"
                )
            local.add(conversation_id)
            owners[conversation_id] = split_name


def write_manifest(path: str | os.PathLike[str], manifest: SplitManifest) -> None:
    """Atomically write a manifest as readable canonical JSON."""
    _write_json(path, manifest.to_dict())


def read_manifest(path: str | os.PathLike[str]) -> SplitManifest:
    """Read and validate a manifest from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ManifestValidationError("manifest root must be a JSON object")
    return SplitManifest.from_dict(value)


def _write_json(path: str | os.PathLike[str], value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_value(value), ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    ) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
