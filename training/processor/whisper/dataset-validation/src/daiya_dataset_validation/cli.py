"""CLI for evaluating generic JSONL metadata without touching source media."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .decision import decide
from .io import write_csv, write_jsonl
from .models import SourceIdentity
from .normalize import source_identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL metadata with uri, label, and duration_seconds")
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--expected-script", action="append", default=[])
    args = parser.parse_args(argv)
    records = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record must be an object")
            except (json.JSONDecodeError, ValueError) as exc:
                parser.error(f"line {line_number}: {exc}")
            uri = str(row.get("uri") or f"{args.input}#line={line_number}")
            identity = SourceIdentity(source_identity(uri, record_id=str(row.get("id", ""))), uri, record_id=str(row.get("id")) if row.get("id") is not None else None)
            records.append(decide(identity, row.get("label"), row.get("duration_seconds"), expected_scripts=frozenset(args.expected_script) or None))
    (write_jsonl if args.format == "jsonl" else write_csv)(records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

