"""Build a versioned, non-mutating candidate-cleaning manifest from metadata."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from daiya_dataset_cleaning.decision import DecisionPolicy, decide, disposition_for_reasons
from daiya_dataset_cleaning.io import write_jsonl
from daiya_dataset_cleaning.models import (
    Confidence,
    Disposition,
    Evidence,
    ManifestRecord,
    ProposedLabel,
    ReasonCode,
    SourceIdentity,
)
from daiya_dataset_cleaning.normalize import normalize_text, source_identity


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_hashes(directory: Path | None) -> set[str]:
    if directory is None:
        return set()
    return {_sha256(path) for path in directory.glob("*.wav") if path.is_file()}


def _prediction_groups(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    if path is None:
        return groups
    for row in _rows(path):
        key = row.get("file_name") or row.get("sample_id") or row.get("audio_path")
        text = row.get("prediction", row.get("hypothesis"))
        if not key or not isinstance(text, str):
            continue
        key = str(key).replace("\\", "/")
        groups.setdefault(key, []).append(row)
    return groups


def _spelling_groups(path: Path | None) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    if path is None:
        return groups
    for row in _rows(path):
        key = row.get("file_name") or row.get("uri")
        if key:
            groups[str(key).replace("\\", "/")] = row
    return groups


def _spelling_signals(
    row: dict[str, Any] | None,
    label: object,
    *,
    review_threshold: float,
    min_issues: int,
) -> tuple[tuple[Evidence, ...], tuple[ReasonCode, ...]]:
    if row is None:
        return (), ()
    label_text = normalize_text(label)
    expected_hash = hashlib.sha256(label_text.encode("utf-8")).hexdigest()
    if row.get("label_sha256") != expected_hash:
        return (Evidence("spelling.stale_result", True, source="spelling-validation"),), ()
    try:
        checked = int(row.get("checked_units", 0))
        suspicious = int(row.get("suspicious_units", 0))
        ratio = float(row.get("suspicious_ratio", 0.0))
    except (TypeError, ValueError, OverflowError):
        return (Evidence("spelling.malformed_result", True, source="spelling-validation"),), ()
    if checked < 0 or suspicious < 0 or suspicious > checked or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        return (Evidence("spelling.malformed_result", True, source="spelling-validation"),), ()
    expected_ratio = suspicious / checked if checked else 0.0
    if not math.isclose(ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-9):
        return (Evidence("spelling.malformed_result", True, source="spelling-validation"),), ()
    names = row.get("checker_names")
    checker_names = [str(name) for name in names] if isinstance(names, list) else []
    evidence: list[Evidence] = [
        Evidence("spelling.checked_units", checked, source="spelling-validation"),
        Evidence("spelling.suspicious_units", suspicious, source="spelling-validation"),
        Evidence("spelling.suspicious_ratio", ratio, source="spelling-validation"),
        Evidence("spelling.checkers", ",".join(checker_names), source="spelling-validation"),
    ]
    routed = row.get("routed_span_counts")
    if isinstance(routed, dict):
        for language, count in sorted(routed.items()):
            if isinstance(count, int) and count >= 0:
                evidence.append(Evidence(f"spelling.routed_spans.{language}", count, source="spelling-validation"))
    language_trigger = False
    by_language = row.get("by_language")
    if isinstance(by_language, dict):
        for language, values in sorted(by_language.items()):
            if not isinstance(values, dict):
                continue
            try:
                language_checked = int(values.get("checked_units", 0))
                language_suspicious = int(values.get("suspicious_units", 0))
                language_ratio = float(values.get("suspicious_ratio", 0.0))
            except (TypeError, ValueError, OverflowError):
                return (Evidence("spelling.malformed_result", True, source="spelling-validation"),), ()
            expected_language_ratio = language_suspicious / language_checked if language_checked else 0.0
            if (
                language_checked < 0
                or language_suspicious < 0
                or language_suspicious > language_checked
                or not math.isfinite(language_ratio)
                or not math.isclose(language_ratio, expected_language_ratio, rel_tol=0.0, abs_tol=1e-9)
            ):
                return (Evidence("spelling.malformed_result", True, source="spelling-validation"),), ()
            evidence.extend((
                Evidence(f"spelling.checked_units.{language}", language_checked, source="spelling-validation"),
                Evidence(f"spelling.suspicious_units.{language}", language_suspicious, source="spelling-validation"),
                Evidence(f"spelling.suspicious_ratio.{language}", language_ratio, source="spelling-validation"),
            ))
            language_trigger = language_trigger or (
                language_suspicious >= min_issues and language_ratio >= review_threshold
            )
    reasons = (
        (ReasonCode.SPELLING_SUSPECT,)
        if language_trigger or (not isinstance(by_language, dict) and suspicious >= min_issues and ratio >= review_threshold)
        else ()
    )
    return tuple(evidence), reasons


def _consensus(rows: list[dict[str, Any]], min_views: int) -> tuple[ProposedLabel | None, tuple[Evidence, ...], bool]:
    if not rows:
        return None, (), False
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        text = row.get("prediction", row.get("hypothesis"))
        if isinstance(text, str) and normalize_text(text):
            buckets.setdefault(normalize_text(text, casefold=True), []).append(row)
    if not buckets:
        return None, (), False
    ordered = sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0]))
    winner_key, winner_rows = ordered[0]
    distinct_views = {str(r.get("model_name", r.get("model", r.get("strategy", "unknown")))) for r in winner_rows}
    evidence: list[Evidence] = [
        Evidence("prediction_view_count", len(rows), source="prediction-artifact"),
        Evidence("prediction_consensus_count", len(winner_rows), source="prediction-artifact"),
        Evidence("prediction_consensus_fraction", len(winner_rows) / len(rows), source="prediction-artifact"),
        Evidence("prediction_distinct_view_count", len(distinct_views), source="prediction-artifact"),
    ]
    disagreement = len(buckets) > 1
    if disagreement:
        evidence.append(Evidence("prediction_disagreement", True, source="prediction-artifact"))
    if len(winner_rows) < min_views or len(distinct_views) < 2:
        return None, tuple(evidence), disagreement
    text = str(winner_rows[0].get("prediction", winner_rows[0].get("hypothesis", ""))).strip()
    refs = tuple(f"prediction.{i}" for i in range(len(winner_rows)))
    proposal = ProposedLabel(
        text=text,
        method="multi_view_consensus",
        evidence_refs=refs,
        confidence=Confidence(
            len(winner_rows) / len(rows),
            "raw multi-view consensus fraction; not calibrated",
            calibrated=False,
        ),
    )
    return proposal, tuple(evidence), disagreement


def build_manifest(
    metadata_path: Path,
    audio_root: Path,
    output_path: Path,
    *,
    dataset_version: str,
    protected_gold_dir: Path | None = None,
    predictions_path: Path | None = None,
    spelling_results_path: Path | None = None,
    expected_scripts: frozenset[str] | None = None,
    min_consensus_views: int = 2,
    spelling_review_threshold: float = 0.2,
    spelling_min_issues: int = 1,
) -> dict[str, int | str]:
    if not 0 <= spelling_review_threshold <= 1:
        raise ValueError("spelling_review_threshold must be in [0, 1]")
    if spelling_min_issues < 1:
        raise ValueError("spelling_min_issues must be at least 1")
    protected = _protected_hashes(protected_gold_dir)
    predictions = _prediction_groups(predictions_path)
    spelling_results = _spelling_groups(spelling_results_path)
    input_rows = list(_rows(metadata_path))
    content_hashes: list[str | None] = []
    for row in input_rows:
        uri = str(row.get("file_name") or row.get("uri") or "").replace("\\", "/")
        path = audio_root / uri
        content_hashes.append(_sha256(path) if path.is_file() else None)
    duplicate_counts = Counter(value for value in content_hashes if value is not None)
    records: list[ManifestRecord] = []
    counts: dict[str, int] = {}
    for row_number, (row, content_sha) in enumerate(zip(input_rows, content_hashes, strict=True), 1):
        uri = str(row.get("file_name") or row.get("uri") or f"metadata.jsonl#line={row_number}").replace("\\", "/")
        stable_record_id = str(row.get("id")) if row.get("id") is not None else None
        source = SourceIdentity(
            source_id=source_identity(uri, record_id=stable_record_id),
            uri=uri,
            content_sha256=content_sha,
            dataset=dataset_version,
            record_id=stable_record_id,
        )
        label = row.get("text", row.get("label"))
        duration = row.get("speech_duration", row.get("duration_seconds"))
        key = uri
        proposal, prediction_evidence, disagreement = _consensus(predictions.get(key, []), min_consensus_views)
        record = decide(source, label, duration, expected_scripts=expected_scripts, proposed_label=proposal)
        extra_reasons = list(record.reasons)
        extra_evidence = list(record.evidence) + list(prediction_evidence)
        spelling_evidence, spelling_reasons = _spelling_signals(
            spelling_results.get(key),
            label,
            review_threshold=spelling_review_threshold,
            min_issues=spelling_min_issues,
        )
        extra_evidence.extend(spelling_evidence)
        extra_reasons.extend(spelling_reasons)
        if disagreement and proposal is None:
            extra_reasons.append(ReasonCode.PREDICTION_DISAGREEMENT)
        if content_sha is not None and duplicate_counts[content_sha] > 1:
            extra_reasons.append(ReasonCode.DUPLICATE_CONTENT)
            extra_evidence.append(Evidence("duplicate_group_size", duplicate_counts[content_sha], source="manifest-builder"))
        if content_sha is None:
            extra_reasons.append(ReasonCode.MALFORMED_INPUT)
            extra_evidence.append(Evidence("audio_present", False, source="manifest-builder"))
        else:
            extra_evidence.append(Evidence("audio_present", True, source="manifest-builder"))
        if content_sha is None or duplicate_counts.get(content_sha, 0) > 1:
            proposal = None
        if proposal is None:
            extra_reasons = [reason for reason in extra_reasons if reason is not ReasonCode.GROUNDED_CORRECTION]
        disposition = disposition_for_reasons(
            extra_reasons,
            policy=DecisionPolicy(),
            proposed_label=proposal,
        )
        if content_sha in protected:
            extra_reasons = [reason for reason in extra_reasons if reason is not ReasonCode.GROUNDED_CORRECTION]
            extra_reasons.append(ReasonCode.PROTECTED_GOLD_OVERLAP)
            extra_evidence.append(Evidence("protected_gold_overlap", True, source="protected-gold-sha256"))
            record = ManifestRecord(
                source=source,
                original_label=record.original_label,
                normalized_label=record.normalized_label,
                disposition=Disposition.DROP,
                reasons=tuple(dict.fromkeys(extra_reasons)),
                confidence=record.confidence,
                evidence=tuple(extra_evidence),
                proposed_label=None,
            )
        else:
            record = ManifestRecord(
                source=record.source,
                original_label=record.original_label,
                normalized_label=record.normalized_label,
                disposition=disposition,
                reasons=tuple(dict.fromkeys(extra_reasons)),
                confidence=record.confidence,
                evidence=tuple(extra_evidence),
                proposed_label=proposal,
            )
        records.append(record)
        counts[record.disposition.value] = counts.get(record.disposition.value, 0) + 1
    write_jsonl(records, output_path)
    return {"dataset_version": dataset_version, "records": len(records), **counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("audio_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--protected-gold-dir", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--spelling-results", type=Path)
    parser.add_argument("--expected-script", action="append", default=[])
    parser.add_argument("--min-consensus-views", type=int, default=2)
    parser.add_argument("--spelling-review-threshold", type=float, default=0.2)
    parser.add_argument("--spelling-min-issues", type=int, default=1)
    args = parser.parse_args(argv)
    summary = build_manifest(
        args.metadata,
        args.audio_root,
        args.output,
        dataset_version=args.dataset_version,
        protected_gold_dir=args.protected_gold_dir,
        predictions_path=args.predictions,
        spelling_results_path=args.spelling_results,
        expected_scripts=frozenset(args.expected_script) or None,
        min_consensus_views=args.min_consensus_views,
        spelling_review_threshold=args.spelling_review_threshold,
        spelling_min_issues=args.spelling_min_issues,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
