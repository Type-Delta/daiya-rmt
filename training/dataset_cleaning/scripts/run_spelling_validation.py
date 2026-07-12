"""Generate script-aware spelling evidence without modifying source labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from daiya_dataset_cleaning.normalize import normalize_text
from daiya_dataset_cleaning.spelling import (
    PyThaiNLPChecker,
    SudachiJapaneseChecker,
    SymSpellEnglishChecker,
    validate_spelling,
)


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield row


def _allowlist(path: Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    return frozenset(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _atomic_jsonl(rows: Iterable[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="JSONL containing file_name and text/label")
    parser.add_argument("output", type=Path)
    parser.add_argument("--thai-engine", action="append", choices=("pn", "symspellpy", "phunspell"), default=[])
    parser.add_argument("--japanese-dictionary", choices=("small", "core", "full"))
    parser.add_argument("--english-dictionary", type=Path, help="SymSpell term/count frequency dictionary")
    parser.add_argument("--allowlist", type=Path, help="UTF-8 known terms, one per line")
    parser.add_argument("--review-threshold", type=float, default=0.2)
    parser.add_argument("--min-issues", type=int, default=1)
    parser.add_argument("--include-issue-text", action="store_true", help="Include suspicious units/suggestions in output")
    args = parser.parse_args(argv)
    if not 0 <= args.review_threshold <= 1:
        parser.error("--review-threshold must be in [0, 1]")
    if args.min_issues < 1:
        parser.error("--min-issues must be at least 1")
    if len(args.thai_engine) > 1:
        parser.error("use one --thai-engine per run so issue ratios remain comparable")

    checkers = [PyThaiNLPChecker(engine) for engine in args.thai_engine]
    if args.japanese_dictionary:
        checkers.append(SudachiJapaneseChecker(args.japanese_dictionary))
    if args.english_dictionary:
        checkers.append(SymSpellEnglishChecker(str(args.english_dictionary)))
    if not checkers:
        parser.error("enable at least one checker")
    allowlist = _allowlist(args.allowlist)

    output_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(_rows(args.metadata), 1):
        key = str(row.get("file_name") or row.get("uri") or f"metadata.jsonl#line={row_number}").replace("\\", "/")
        label = row.get("text", row.get("label"))
        text = normalize_text(label)
        validation = validate_spelling(text, checkers, allowlist=allowlist)
        result = validation.to_dict(include_issue_text=args.include_issue_text)
        by_language = result["by_language"]
        result.update({
            "schema_version": "spelling-validation-1",
            "file_name": key,
            "label_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "review_recommended": (
                validation.suspicious_units >= args.min_issues
                and any(
                    language_result["suspicious_units"] >= args.min_issues
                    and language_result["suspicious_ratio"] >= args.review_threshold
                    for language_result in by_language.values()
                )
            ),
        })
        output_rows.append(result)
    _atomic_jsonl(output_rows, args.output)
    print(json.dumps({"records": len(output_rows), "checkers": [checker.name for checker in checkers]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
