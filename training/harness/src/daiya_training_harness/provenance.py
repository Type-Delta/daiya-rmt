"""Exact, serializable provenance records for training and evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .splits import _json_value, _write_json, canonical_json_hash


class ProvenanceValidationError(ValueError):
    """Raised when required provenance is absent or malformed."""


@dataclass(frozen=True)
class ProvenanceRecord:
    """Reproducibility-critical inputs, without machine-specific runtime state."""

    dataset_version: str
    conversion_settings: Mapping[str, Any]
    base_model_revision: str
    evaluation_backend: Mapping[str, Any]
    split_manifest_sha256: str
    provenance_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("dataset_version", "base_model_revision"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ProvenanceValidationError(f"{field_name} must be a non-empty string")
        if self.provenance_version != 1 or isinstance(self.provenance_version, bool):
            raise ProvenanceValidationError(
                f"unsupported provenance_version: {self.provenance_version}"
            )
        digest = self.split_manifest_sha256
        if not isinstance(digest, str) or len(digest) != 64:
            raise ProvenanceValidationError("split_manifest_sha256 must be a SHA-256 hex digest")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ProvenanceValidationError(
                "split_manifest_sha256 must be a SHA-256 hex digest"
            ) from exc
        for field_name in ("conversion_settings", "evaluation_backend", "metadata"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise ProvenanceValidationError(f"{field_name} must be an object")
            normalized = _json_value(value, f"$.{field_name}")
            object.__setattr__(self, field_name, MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance_version": self.provenance_version,
            "dataset_version": self.dataset_version,
            "conversion_settings": dict(self.conversion_settings),
            "base_model_revision": self.base_model_revision,
            "evaluation_backend": dict(self.evaluation_backend),
            "split_manifest_sha256": self.split_manifest_sha256.lower(),
            "metadata": dict(self.metadata),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProvenanceRecord":
        expected = {
            "provenance_version", "dataset_version", "conversion_settings",
            "base_model_revision", "evaluation_backend", "split_manifest_sha256", "metadata",
        }
        unknown = set(value) - expected
        if unknown:
            raise ProvenanceValidationError(f"unknown provenance fields: {sorted(unknown)}")
        try:
            return cls(
                provenance_version=value.get("provenance_version", 1),
                dataset_version=value["dataset_version"],
                conversion_settings=value["conversion_settings"],
                base_model_revision=value["base_model_revision"],
                evaluation_backend=value["evaluation_backend"],
                split_manifest_sha256=value["split_manifest_sha256"],
                metadata=value.get("metadata", {}),
            )
        except KeyError as exc:
            raise ProvenanceValidationError(f"missing provenance field: {exc.args[0]}") from exc


def write_provenance(path: str | os.PathLike[str], record: ProvenanceRecord) -> None:
    _write_json(path, record.to_dict())


def read_provenance(path: str | os.PathLike[str]) -> ProvenanceRecord:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ProvenanceValidationError("provenance root must be a JSON object")
    return ProvenanceRecord.from_dict(value)
