"""Conservative multi-signal decision policy."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Confidence, Disposition, Evidence, ManifestRecord, ProposedLabel, ReasonCode, SourceIdentity
from .normalize import normalize_text
from .signals import evaluate_signals


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    drop_reasons: frozenset[ReasonCode] = frozenset({ReasonCode.EMPTY_LABEL})
    review_reasons: frozenset[ReasonCode] = frozenset({
        ReasonCode.INVALID_DURATION, ReasonCode.TOO_SHORT, ReasonCode.TOO_LONG,
        ReasonCode.CHAR_RATE_LOW, ReasonCode.CHAR_RATE_HIGH, ReasonCode.SCRIPT_MISMATCH,
        ReasonCode.DUPLICATE_CONTENT, ReasonCode.LOW_CONFIDENCE, ReasonCode.SIGNAL_CONFLICT,
        ReasonCode.PROTECTED_GOLD_OVERLAP, ReasonCode.PREDICTION_DISAGREEMENT,
        ReasonCode.MALFORMED_INPUT,
    })
    min_adapter_confidence: float = 0.5


def disposition_for_reasons(
    reasons: tuple[ReasonCode, ...] | list[ReasonCode],
    *,
    policy: DecisionPolicy = DecisionPolicy(),
    proposed_label: ProposedLabel | None = None,
) -> Disposition:
    """Map the complete reason set to a disposition after all adapters run."""
    if proposed_label is not None:
        return Disposition.CORRECT
    if any(reason in policy.drop_reasons for reason in reasons):
        return Disposition.DROP
    if any(reason in policy.review_reasons for reason in reasons):
        return Disposition.REVIEW
    return Disposition.KEEP


def decide(
    source: SourceIdentity,
    label: object,
    duration_seconds: object,
    *,
    policy: DecisionPolicy = DecisionPolicy(),
    expected_scripts: frozenset[str] | None = None,
    adapter_evidence: tuple[Evidence, ...] = (),
    confidence: Confidence | None = None,
    proposed_label: ProposedLabel | None = None,
) -> ManifestRecord:
    """Combine baseline and adapter evidence into a deterministic disposition."""
    evidence, reasons = evaluate_signals(label, duration_seconds, expected_scripts=expected_scripts)
    reason_list = list(reasons)
    if confidence is not None and confidence.value < policy.min_adapter_confidence:
        reason_list.append(ReasonCode.LOW_CONFIDENCE)
    reason_list = list(dict.fromkeys(reason_list))
    if proposed_label is not None:
        reason_list.append(ReasonCode.GROUNDED_CORRECTION)
    disposition = disposition_for_reasons(reason_list, policy=policy, proposed_label=proposed_label)
    original = None if label is None else str(label)
    return ManifestRecord(
        source=source, original_label=original, normalized_label=normalize_text(label),
        disposition=disposition, reasons=tuple(reason_list), confidence=confidence,
        evidence=evidence + tuple(adapter_evidence), proposed_label=proposed_label,
    )
