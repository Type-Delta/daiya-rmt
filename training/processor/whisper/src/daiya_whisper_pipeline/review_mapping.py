"""Conservative mapping for migrating labels across segmentation generations."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "daiya-segmentation-mapping-2"
_TIMESTAMP_TOLERANCE_SECONDS = 0.000_001


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            rows.append(value)
    return rows


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_key(row: dict[str, Any]) -> str:
    source = row.get("source_file")
    if not isinstance(source, str) or not source:
        return "\0missing-source"
    # A source path is provenance, not display text.  Do not collapse two
    # different recordings with the same basename into one match group.
    return source.replace("\\", "/").casefold()


def _file_key(row: dict[str, Any]) -> str:
    value = row.get("file_name")
    return str(value).replace("\\", "/").casefold() if value else ""


def _number(row: dict[str, Any], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _span(row: dict[str, Any]) -> tuple[float, float] | None:
    # Timestamp-ownership exports keep source_start/source_end as legacy
    # aliases, but prefer the explicit owned range so a labeler pre-roll can
    # never make a review look reusable through approximate overlap.
    start = _number(row, "owned_source_start")
    end = _number(row, "owned_source_end")
    if start is None or end is None:
        start = _number(row, "source_start")
        end = _number(row, "source_end")
    if start is None or end is None or end <= start:
        return None
    return start, end


def _same_span(left: dict[str, Any], right: dict[str, Any]) -> bool:
    first = _span(left)
    second = _span(right)
    if first is None or second is None:
        return False
    return all(abs(a - b) <= _TIMESTAMP_TOLERANCE_SECONDS for a, b in zip(first, second))


def _overlap_seconds(left: dict[str, Any], right: dict[str, Any]) -> float:
    first = _span(left)
    second = _span(right)
    if first is None or second is None:
        return 0.0
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _row_hash(row: dict[str, Any], audio_root: Path | None) -> str | None:
    value = row.get("audio_sha256")
    if isinstance(value, str) and value:
        return value.lower()
    if audio_root is None:
        return None
    file_name = row.get("file_name")
    if not isinstance(file_name, str) or not file_name:
        return None
    return _hash_file(audio_root / Path(file_name))


def _clip_mapping(
    old_row: dict[str, Any],
    new_candidates: list[dict[str, Any]],
    old_audio_root: Path | None,
    new_audio_root: Path | None,
) -> dict[str, Any]:
    exact = [candidate for candidate in new_candidates if _same_span(old_row, candidate)]
    old_hash = _row_hash(old_row, old_audio_root)
    if len(exact) == 1:
        new_row = exact[0]
        new_hash = _row_hash(new_row, new_audio_root)
        if old_hash and new_hash and old_hash == new_hash:
            return {
                "status": "unchanged",
                "new_file_name": new_row.get("file_name"),
                "reason": "source span and exported audio SHA-256 are identical",
            }
        return {
            "status": "ambiguous",
            "new_file_name": new_row.get("file_name"),
            "reason": "matching timestamps without identical exported-audio hashes are not safe to reuse",
        }
    if len(exact) > 1:
        return {
            "status": "ambiguous",
            "new_file_name": None,
            "reason": "multiple new clips share the old source span",
        }

    overlaps = [candidate for candidate in new_candidates if _overlap_seconds(old_row, candidate) > 0]
    if len(overlaps) == 1:
        return {
            "status": "changed_boundary_needs_relabel",
            "new_file_name": overlaps[0].get("file_name"),
            "reason": "source windows overlap but boundaries or audio changed",
        }
    if len(overlaps) > 1:
        return {
            "status": "ambiguous",
            "new_file_name": None,
            "reason": "old audio overlaps multiple new clips",
        }
    return {
        "status": "changed_boundary_needs_relabel",
        "new_file_name": None,
        "reason": "no matching source window exists in the regenerated metadata",
    }


def build_mapping_report(
    old_metadata: Path,
    new_metadata: Path,
    *,
    old_reviews: Path | None = None,
    old_audio_root: Path | None = None,
    new_audio_root: Path | None = None,
) -> dict[str, Any]:
    """Compare generations without copying a label or review into the new set."""
    old_rows = _read_jsonl(old_metadata)
    new_rows = _read_jsonl(new_metadata)
    new_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in new_rows:
        new_by_source[_source_key(row)].append(row)
    for rows in new_by_source.values():
        rows.sort(key=lambda item: (_number(item, "source_start") or -1.0, _file_key(item)))

    clips: list[dict[str, Any]] = []
    old_by_file = {_file_key(row): row for row in old_rows if _file_key(row)}
    for old_row in sorted(old_rows, key=lambda item: (_source_key(item), _number(item, "source_start") or -1.0, _file_key(item))):
        result = _clip_mapping(
            old_row,
            new_by_source.get(_source_key(old_row), []),
            old_audio_root,
            new_audio_root,
        )
        clips.append(
            {
                "old_file_name": old_row.get("file_name"),
                "source_file": old_row.get("source_file"),
                "source_start": old_row.get("source_start"),
                "source_end": old_row.get("source_end"),
                **result,
            }
        )

    reviews: list[dict[str, Any]] = []
    if old_reviews is not None:
        for line_number, review in enumerate(_read_jsonl(old_reviews), start=1):
            chunk = review.get("chunk")
            if not isinstance(chunk, dict):
                reviews.append(
                    {"review_line": line_number, "status": "ambiguous", "reason": "review has no chunk provenance"}
                )
                continue
            source_uri = str(chunk.get("source_uri", "")).replace("\\", "/").casefold()
            old_row = old_by_file.get(source_uri)
            review_hash = chunk.get("audio_sha256")
            if not old_row:
                reviews.append(
                    {
                        "review_line": line_number,
                        "source_uri": source_uri,
                        "status": "ambiguous",
                        "reason": "review source URI is absent from old metadata",
                    }
                )
                continue
            candidate_rows = new_by_source.get(_source_key(old_row), [])
            matching_hashes = [
                candidate
                for candidate in candidate_rows
                if isinstance(review_hash, str)
                and review_hash.lower() == (_row_hash(candidate, new_audio_root) or "").lower()
            ]
            if len(matching_hashes) == 1:
                reviews.append(
                    {
                        "review_line": line_number,
                        "source_uri": source_uri,
                        "new_file_name": matching_hashes[0].get("file_name"),
                        "status": "safely_reusable_review",
                        "reason": "review and regenerated exported audio SHA-256 are identical",
                    }
                )
                continue
            clip = _clip_mapping(old_row, candidate_rows, old_audio_root, new_audio_root)
            status = "changed_boundary_needs_relabel" if clip["status"] != "ambiguous" else "ambiguous"
            reviews.append(
                {
                    "review_line": line_number,
                    "source_uri": source_uri,
                    "new_file_name": clip.get("new_file_name"),
                    "status": status,
                    "reason": "review was not copied; " + str(clip["reason"]),
                }
            )

    counts = Counter(item["status"] for item in clips)
    review_counts = Counter(item["status"] for item in reviews)
    return {
        "schema_version": SCHEMA_VERSION,
        "old_metadata": str(old_metadata),
        "new_metadata": str(new_metadata),
        "old_reviews": str(old_reviews) if old_reviews else None,
        "summary": {
            "clips": dict(sorted(counts.items())),
            "reviews": dict(sorted(review_counts.items())),
            "safety": "This report never copies labels or review decisions into regenerated metadata.",
        },
        "clips": clips,
        "reviews": reviews,
    }


def write_mapping_report(report: dict[str, Any], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite mapping report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
