"""Non-mutating baseline tools for evaluating ASR dataset records."""

from .decision import DecisionPolicy, decide, disposition_for_reasons
from .models import (
    Confidence,
    Disposition,
    Evidence,
    ManifestRecord,
    ProposedLabel,
    ReasonCode,
    SourceIdentity,
)
from .normalize import content_identity, normalize_text, source_identity
from .signals import evaluate_signals
from .spelling import (
    ScriptSpan,
    SpanCheckResult,
    SpellingIssue,
    SpellingValidation,
    route_script_spans,
    validate_spelling,
)

__all__ = [
    "Confidence", "DecisionPolicy", "Disposition", "Evidence", "ManifestRecord",
    "ProposedLabel", "ReasonCode", "SourceIdentity", "content_identity", "decide",
    "ScriptSpan", "SpanCheckResult", "SpellingIssue", "SpellingValidation",
    "disposition_for_reasons", "evaluate_signals", "normalize_text",
    "route_script_spans", "source_identity", "validate_spelling",
]
