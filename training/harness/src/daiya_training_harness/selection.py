"""Deployment-aware checkpoint selection.

The training ranking is only a way to choose which checkpoints to validate.  For
CT2 deployments (including quantized CT2), selection is made exclusively from
CT2 validation records produced for the top-K training candidates.  In
particular, a Transformers/PEFT score is never deployment evidence.

The intended protocol is::

    result = select_checkpoint(ranking, protocol, validate=validate_with_ct2)

``validate_with_ct2`` is called once for every top-K candidate and must return a
``ValidationRecord`` carrying the protocol's deployment backend, metric, and
gate.  Records can instead be supplied explicitly, which keeps the interface
easy to drive from persisted JSON fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Callable, Iterable, Mapping, Sequence


CT2_BACKENDS = frozenset({"ct2", "ct2_quantized", "quantized"})


def _checkpoint_key(checkpoint: str) -> tuple[int, int | str, str]:
    """Order numeric checkpoint identifiers numerically, then other IDs."""
    text = str(checkpoint)
    suffix = text.rsplit("-", 1)[-1]
    if suffix.isdigit():
        return (0, int(suffix), text)
    return (1, text, text)


@dataclass(frozen=True)
class RankedCheckpoint:
    """A training-side ranking entry, not deployment validation evidence."""

    checkpoint: str
    score: float
    metric_name: str
    source_backend: str = "transformers_peft"

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", str(self.checkpoint))
        if not self.checkpoint or not self.metric_name or not self.source_backend:
            raise ValueError("checkpoint, metric_name, and source_backend are required")
        if not isfinite(self.score):
            raise ValueError("ranking score must be finite")


@dataclass(frozen=True)
class ValidationRecord:
    """Auditable deployment validation evidence for one checkpoint."""

    checkpoint: str
    backend: str
    metrics: Mapping[str, float]
    gates: Mapping[str, bool]
    details: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", str(self.checkpoint))
        object.__setattr__(self, "backend", self.backend.lower())
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "gates", dict(self.gates))
        object.__setattr__(self, "details", dict(self.details))
        if not self.checkpoint or not self.backend:
            raise ValueError("checkpoint and backend are required")
        if any(not name or not isfinite(value) for name, value in self.metrics.items()):
            raise ValueError("metric names must be non-empty and values finite")
        if any(not name or type(value) is not bool for name, value in self.gates.items()):
            raise ValueError("gate names must be non-empty and values boolean")


@dataclass(frozen=True)
class TopKValidationProtocol:
    """Names and ordering rules for deployment validation of training top-K."""

    deployment_backend: str
    top_k: int
    metric_name: str
    gate_name: str
    higher_is_better: bool = False
    ranking_higher_is_better: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "deployment_backend", self.deployment_backend.lower())
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.deployment_backend or not self.metric_name or not self.gate_name:
            raise ValueError("deployment_backend, metric_name, and gate_name are required")


@dataclass(frozen=True)
class SelectionResult:
    """Selection plus all considered validation evidence."""

    selected_checkpoint: str | None
    deployment_backend: str
    metric_name: str
    gate_name: str
    top_k_checkpoints: tuple[str, ...]
    validation_records: tuple[ValidationRecord, ...]

    @property
    def valid(self) -> bool:
        return self.selected_checkpoint is not None


ValidationCallback = Callable[[RankedCheckpoint, TopKValidationProtocol], ValidationRecord]


def rank_top_k(
    ranking: Iterable[RankedCheckpoint], protocol: TopKValidationProtocol
) -> tuple[RankedCheckpoint, ...]:
    """Return training candidates with deterministic checkpoint-ID tie breaks."""
    entries = tuple(ranking)
    if not entries:
        raise ValueError("ranking must contain at least one checkpoint")
    seen: set[str] = set()
    for entry in entries:
        if entry.checkpoint in seen:
            raise ValueError(f"duplicate ranked checkpoint: {entry.checkpoint}")
        seen.add(entry.checkpoint)
    ordered = sorted(entries, key=lambda item: _checkpoint_key(item.checkpoint))
    ordered.sort(key=lambda item: item.score, reverse=protocol.ranking_higher_is_better)
    return tuple(ordered[: protocol.top_k])


def select_checkpoint(
    ranking: Iterable[RankedCheckpoint],
    protocol: TopKValidationProtocol,
    *,
    validate: ValidationCallback | None = None,
    validation_records: Sequence[ValidationRecord] = (),
) -> SelectionResult:
    """Select from passing top-K deployment records.

    For a CT2/quantized target, absent CT2 evidence is an error rather than a
    fallback to training scores.  This means checkpoint 588 (like every other
    checkpoint) cannot be reported valid until a matching CT2 record exists.
    Other backends return ``selected_checkpoint=None`` when no validation
    records pass; callers must treat that result as not selected, not as
    deployment evidence.
    """
    top = rank_top_k(ranking, protocol)
    top_ids = tuple(item.checkpoint for item in top)
    if validate is not None and validation_records:
        raise ValueError("provide validate or validation_records, not both")
    records = tuple(validate(item, protocol) for item in top) if validate else tuple(validation_records)

    if protocol.deployment_backend in CT2_BACKENDS and not records:
        raise ValueError("CT2/quantized selection requires CT2 deployment evidence")

    by_checkpoint: dict[str, ValidationRecord] = {}
    for record in records:
        if record.checkpoint not in top_ids:
            raise ValueError(f"validation record is outside training top-K: {record.checkpoint}")
        if record.checkpoint in by_checkpoint:
            raise ValueError(f"duplicate validation record: {record.checkpoint}")
        if record.backend != protocol.deployment_backend:
            raise ValueError(
                f"backend mismatch for {record.checkpoint}: expected "
                f"{protocol.deployment_backend}, got {record.backend}"
            )
        if protocol.metric_name not in record.metrics:
            raise ValueError(f"missing metric {protocol.metric_name!r} for {record.checkpoint}")
        if protocol.gate_name not in record.gates:
            raise ValueError(f"missing gate {protocol.gate_name!r} for {record.checkpoint}")
        by_checkpoint[record.checkpoint] = record

    if protocol.deployment_backend in CT2_BACKENDS and set(by_checkpoint) != set(top_ids):
        missing = sorted(set(top_ids) - set(by_checkpoint), key=_checkpoint_key)
        raise ValueError(f"missing CT2 evidence for top-K checkpoints: {', '.join(missing)}")

    passing = [record for record in records if record.gates[protocol.gate_name]]
    passing.sort(key=lambda record: _checkpoint_key(record.checkpoint))
    passing.sort(
        key=lambda record: record.metrics[protocol.metric_name],
        reverse=protocol.higher_is_better,
    )
    selected = passing[0].checkpoint if passing else None
    return SelectionResult(
        selected_checkpoint=selected,
        deployment_backend=protocol.deployment_backend,
        metric_name=protocol.metric_name,
        gate_name=protocol.gate_name,
        top_k_checkpoints=top_ids,
        validation_records=records,
    )
