"""Evaluate original labels and supplied ASR candidates against protected gold.

The evaluator writes aggregate metrics only. It never writes protected label text,
audio, or model-output rows, and marks exact audio-overlap scores as contaminated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
import re
from typing import Any

from daiya_dataset_cleaning.normalize import normalize_text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def _metrics(reference: str, hypothesis: str) -> dict[str, float | int]:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    no_space_ref, no_space_hyp = ref.replace(" ", ""), hyp.replace(" ", "")
    edit = _distance(list(hyp), list(ref))
    no_space_edit = _distance(list(no_space_hyp), list(no_space_ref))
    ref_words, hyp_words = ref.split(), hyp.split()
    word_edit = _distance(hyp_words, ref_words)
    return {
        "cer": edit / max(1, len(ref)),
        "cer_edit": edit,
        "cer_reference_length": len(ref),
        "cer_no_space": no_space_edit / max(1, len(no_space_ref)),
        "cer_no_space_edit": no_space_edit,
        "cer_no_space_reference_length": len(no_space_ref),
        "wer_like": word_edit / max(1, len(ref_words)),
        "wer_like_edit": word_edit,
        "wer_like_reference_length": len(ref_words),
    }


def _labels(path: Path) -> dict[int, str]:
    values: dict[int, str] = {}
    current: int | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"#(\d+)\s+\|", line)
        if match:
            current = int(match.group(1))
        elif current is not None and line.strip() and not line.startswith("#"):
            values[current] = line.strip()
            current = None
    return values


def _prediction_groups(path: Path | None) -> dict[str, dict[str, list[str]]]:
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    if path is None:
        return groups
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("file_name") or row.get("sample_id") or row.get("audio_path")
            text = row.get("prediction", row.get("hypothesis"))
            if not key or not isinstance(text, str):
                continue
            key = str(key).replace("\\", "/")
            candidate = str(row.get("candidate") or row.get("model_name") or row.get("model") or "candidate")
            candidate = candidate.replace("\\", "/").rstrip("/").split("/")[-1]
            strategy = str(row.get("strategy") or "default")
            groups[key][f"{candidate}|{strategy}"].append(text)
    return groups


def evaluate(metadata: Path, audio_root: Path, gold_audio: Path, gold_labels: Path, predictions: Path | None = None) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    malformed_metadata = 0
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            file_name = row.get("file_name")
            if not file_name:
                malformed_metadata += 1
                continue
            rows[str(file_name).replace("\\", "/")] = row
    train_hashes: dict[str, str] = {}
    for path in (audio_root / "train").glob("*.wav"):
        train_hashes[_sha256(path)] = f"train/{path.name}"
    labels = _labels(gold_labels)
    mapped: list[dict[str, Any]] = []
    unmatched = 0
    malformed = 0
    for path in sorted(gold_audio.glob("*.wav")):
        match = re.search(r"_(\d+)\.wav$", path.name)
        if match is None:
            malformed += 1
            continue
        index = int(match.group(1))
        if index not in labels:
            malformed += 1
            continue
        train_key = train_hashes.get(_sha256(path))
        if train_key is None:
            unmatched += 1
            continue
        if train_key not in rows:
            malformed += 1
            continue
        mapped.append({"gold_index": index, "train_key": train_key, "reference": labels[index], "row": rows[train_key]})
    candidate_groups = _prediction_groups(predictions)
    metrics: dict[str, list[dict[str, Any]]] = {"original_label": []}
    for item in mapped:
        metrics["original_label"].append(_metrics(item["reference"], item["row"].get("text", "")))
        for candidate_key, values in candidate_groups.get(item["train_key"], {}).items():
            metrics.setdefault(candidate_key, []).append(_metrics(item["reference"], values[0]))

    def aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "coverage": 0.0}
        return {
            "count": len(values),
            "coverage": len(values) / len(labels),
            "mean_cer": sum(float(v["cer"]) for v in values) / len(values),
            "micro_cer": sum(int(v["cer_edit"]) for v in values) / max(1, sum(int(v["cer_reference_length"]) for v in values)),
            "mean_cer_no_space": sum(float(v["cer_no_space"]) for v in values) / len(values),
            "micro_cer_no_space": sum(int(v["cer_no_space_edit"]) for v in values) / max(1, sum(int(v["cer_no_space_reference_length"]) for v in values)),
            "mean_wer_like": sum(float(v["wer_like"]) for v in values) / len(values),
        }

    source_groups = sorted({str(item["row"].get("source_file", "unknown")) for item in mapped})
    return {
        "schema_version": "cleaning-eval-1",
        "gold_count": len(labels),
        "exact_audio_overlap_count": len(mapped),
        "unmatched_gold_count": unmatched,
        "malformed_gold_count": malformed,
        "malformed_metadata_count": malformed_metadata,
        "overlap_is_contaminated": bool(mapped),
        "source_group_count_in_mapped_overlap": len(source_groups),
        "source_group_split_status": "blocked_single_group" if len(source_groups) < 2 else "groups_available",
        "candidate_metrics": {key: aggregate(value) for key, value in sorted(metrics.items())},
        "interpretation": "Exact-overlap metrics are diagnostic only; no uncontaminated original-label comparison is available from this mapping.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--gold-audio", type=Path, required=True)
    parser.add_argument("--gold-labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.metadata, args.audio_root, args.gold_audio, args.gold_labels, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
