"""Immutable public data model for audit-friendly cleaning manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any


class Disposition(StrEnum):
    KEEP = "keep"
    DROP = "drop"
    REVIEW = "review"
    CORRECT = "correct"


class ReasonCode(StrEnum):
    EMPTY_LABEL = "empty_label"
    INVALID_DURATION = "invalid_duration"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    CHAR_RATE_LOW = "char_rate_low"
    CHAR_RATE_HIGH = "char_rate_high"
    SCRIPT_MISMATCH = "script_mismatch"
    DUPLICATE_CONTENT = "duplicate_content"
    LOW_CONFIDENCE = "low_confidence"
    SIGNAL_CONFLICT = "signal_conflict"
    GROUNDED_CORRECTION = "grounded_correction"
    PROTECTED_GOLD_OVERLAP = "protected_gold_overlap"
    PREDICTION_DISAGREEMENT = "prediction_disagreement"
    MALFORMED_INPUT = "malformed_input"


@dataclass(frozen=True, slots=True)
class Confidence:
    """A score in [0, 1] with an explicit calibration description."""

    value: float
    method: str
    calibrated: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or not 0 <= self.value <= 1:
            raise ValueError("confidence value must be finite and in [0, 1]")
        if not self.method.strip():
            raise ValueError("confidence method must be explicit")


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Stable source identity and provenance; paths are recorded, never modified."""

    source_id: str
    uri: str
    content_sha256: str | None = None
    dataset: str | None = None
    record_id: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id or not self.uri:
            raise ValueError("source_id and uri are required")


@dataclass(frozen=True, slots=True)
class Evidence:
    """Adapter-friendly scalar or categorical evidence."""

    name: str
    value: float | int | str | bool | None
    source: str = "baseline"
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.source.strip():
            raise ValueError("evidence name and source are required")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("evidence values must be finite")


@dataclass(frozen=True, slots=True)
class ProposedLabel:
    """A proposed correction tied to reproducible, non-generative evidence."""

    text: str
    method: str
    evidence_refs: tuple[str, ...]
    confidence: Confidence

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.method.strip() or not self.evidence_refs:
            raise ValueError("proposed labels require text, method, and evidence references")


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    source: SourceIdentity
    original_label: str | None
    normalized_label: str
    disposition: Disposition
    reasons: tuple[ReasonCode, ...] = ()
    confidence: Confidence | None = None
    evidence: tuple[Evidence, ...] = ()
    proposed_label: ProposedLabel | None = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.disposition is Disposition.CORRECT and self.proposed_label is None:
            raise ValueError("correct disposition requires a grounded proposed label")
        if self.proposed_label is not None and self.disposition is not Disposition.CORRECT:
            raise ValueError("proposed label is only valid for correct disposition")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping without exposing mutable internal state."""
        from dataclasses import asdict
        data = asdict(self)
        data["disposition"] = self.disposition.value
        data["reasons"] = [reason.value for reason in self.reasons]
        return data
