"""Build a versioned, non-mutating candidate-cleaning manifest from metadata."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
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
    expected_scripts: frozenset[str] | None = None,
    min_consensus_views: int = 2,
) -> dict[str, int | str]:
    protected = _protected_hashes(protected_gold_dir)
    predictions = _prediction_groups(predictions_path)
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
    parser.add_argument("--expected-script", action="append", default=[])
    parser.add_argument("--min-consensus-views", type=int, default=2)
    args = parser.parse_args(argv)
    summary = build_manifest(
        args.metadata,
        args.audio_root,
        args.output,
        dataset_version=args.dataset_version,
        protected_gold_dir=args.protected_gold_dir,
        predictions_path=args.predictions,
        expected_scripts=frozenset(args.expected_script) or None,
        min_consensus_views=args.min_consensus_views,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
