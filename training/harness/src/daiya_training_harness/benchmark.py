"""Deterministic, dependency-free benchmark evaluation.

The harness deliberately accepts plain mappings (or JSON paths).  This keeps it
fixture friendly and allows the experiment contract package to own validation.
Raw references and predictions are hashed, but never included in summaries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class IncompatibleContractError(ValueError):
    """Raised when benchmark results cannot be compared safely."""


def _load(record: Mapping[str, Any] | str | Path | Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if not isinstance(value, Mapping):
            raise TypeError("record to_dict() must return a mapping")
        return value
    value = json.loads(Path(record).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {record}")
    return value


def canonical_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for a JSON-compatible value."""
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _words(text: str) -> list[str]:
    return text.casefold().split()


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, 1):
        current = [row]
        for column, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _segment_metrics(reference: str, prediction: str) -> dict[str, Any]:
    ref_words, predicted_words = _words(reference), _words(prediction)
    ref_chars, predicted_chars = list(reference), list(prediction)
    word_errors = _edit_distance(ref_words, predicted_words)
    char_errors = _edit_distance(ref_chars, predicted_chars)
    return {
        "exact_match": reference == prediction,
        "word_errors": word_errors,
        "reference_words": len(ref_words),
        "character_errors": char_errors,
        "reference_characters": len(ref_chars),
    }


def _contract_hash(contract: Mapping[str, Any]) -> str:
    supplied = _identity(contract, "contract_hash", "hash")
    return str(supplied) if supplied is not None else canonical_hash(contract)


def run_benchmark(
    *,
    contract: Mapping[str, Any] | str | Path | Any,
    manifest: Mapping[str, Any] | str | Path | Any,
    provenance: Mapping[str, Any] | str | Path | Any,
    outputs: Iterable[Mapping[str, Any]],
    backend_identity: str | Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate deterministic segment outputs and return a compact summary.

    Each output needs ``segment_id``, ``reference`` and ``prediction``.  Output
    ordering does not affect the result.  Text is represented only by hashes.
    """
    contract_record = _load(contract)
    manifest_record = _load(manifest)
    provenance_record = _load(provenance)
    contract_hash = _contract_hash(contract_record)

    for label, record in (("manifest", manifest_record), ("provenance", provenance_record)):
        linked = _identity(record, "contract_hash", "experiment_contract_hash")
        if linked is not None and str(linked) != contract_hash:
            raise IncompatibleContractError(
                f"{label} contract hash {linked!r} does not match {contract_hash!r}"
            )

    segments: list[dict[str, Any]] = []
    totals = {"word_errors": 0, "reference_words": 0, "character_errors": 0,
              "reference_characters": 0, "exact_matches": 0}
    seen: set[str] = set()
    for output in outputs:
        segment_id = str(output["segment_id"])
        if segment_id in seen:
            raise ValueError(f"duplicate segment_id: {segment_id}")
        seen.add(segment_id)
        reference, prediction = str(output["reference"]), str(output["prediction"])
        metrics = _segment_metrics(reference, prediction)
        totals["exact_matches"] += int(metrics["exact_match"])
        for key in ("word_errors", "reference_words", "character_errors", "reference_characters"):
            totals[key] += metrics[key]
        segments.append({
            "segment_id": segment_id,
            "output_hash": canonical_hash({"reference": reference, "prediction": prediction}),
            "metrics": metrics,
        })
    segments.sort(key=lambda item: item["segment_id"])
    count = len(segments)

    recipe = _identity(provenance_record, "recipe", "recipe_id", "recipe_hash")
    if recipe is None:
        recipe = canonical_hash(contract_record)
    base_revision = _identity(
        provenance_record, "base_revision", "base_model_revision", "model_revision"
    )
    data_hash = _identity(provenance_record, "data_hash", "dataset_hash")
    if data_hash is None:
        data_identity = {
            "dataset_version": provenance_record.get("dataset_version"),
            "conversion_settings": provenance_record.get("conversion_settings"),
        }
        data_hash = canonical_hash(data_identity)
    manifest_hash = _identity(manifest_record, "manifest_hash", "hash") or canonical_hash(manifest_record)
    recorded_manifest_hash = provenance_record.get("split_manifest_sha256")
    if recorded_manifest_hash is not None and str(recorded_manifest_hash) != manifest_hash:
        raise ValueError("provenance split_manifest_sha256 does not match manifest")
    return {
        "schema_version": 1,
        "contract_hash": contract_hash,
        "backend": backend_identity,
        "recipe": recipe,
        "base_revision": base_revision,
        "data_hash": data_hash,
        "manifest_hash": str(manifest_hash),
        "metrics": {
            "segments": count,
            "exact_match_rate": totals["exact_matches"] / count if count else 0.0,
            "wer": totals["word_errors"] / totals["reference_words"] if totals["reference_words"] else 0.0,
            "cer": totals["character_errors"] / totals["reference_characters"] if totals["reference_characters"] else 0.0,
        },
        "segments": segments,
    }


def compact_json(summary: Mapping[str, Any]) -> str:
    """Serialize a summary deterministically without insignificant whitespace."""
    return json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def assert_comparable(*summaries: Mapping[str, Any]) -> None:
    """Reject comparison across differing experiment contracts."""
    hashes = {summary.get("contract_hash") for summary in summaries}
    if len(hashes) > 1 or None in hashes:
        raise IncompatibleContractError("benchmark summaries use incompatible contracts")
